"""P1.7.B.3 — per-rollout metadata for case-2/3 raw-container acquires.

Sketch::

    import xrlenv

    client = xrlenv.from_env()    # the only xrlenv-specific line
                                  # in the audience's harness

    # smoke driver — wraps each per-instance call:
    with xrlenv.rollout_metadata(
        artifact_path="/repo/tmp/job-id/logs/run_evaluation/...",
        displayed_name="astropy__astropy-7166",
    ):
        run_instance(test_spec, pred, ..., client=client)
        # ↑ unmodified upstream swebench code; the contextvar
        #   propagates through threads / asyncio Tasks into the
        #   docker-py drop-in's create_container override.

The drop-in's cluster-mode ``create_container`` reads the contextvar
and emits ``xrlenv.rollout.artifact_path`` /
``xrlenv.rollout.displayed_name`` docker labels on the outgoing
``AcquireContainerCommand``. The control plane's
``RawContainerCoordinator`` parses those keys off the labels dict
and writes them to the typed columns on ``RawRolloutRecord``.

Why typed kwargs (not free-form dict)
======================================

Two recognized fields today; future fields land additively as
new kwargs + new columns + new parse cases. The typed-kwargs API:

- IDE auto-completes the field names.
- Static type-checkers reject typos at the call site.
- Consumers can't accidentally dump arbitrary metadata into the
  cluster's record (the cluster reserves no namespace beyond
  what it explicitly recognizes).
- ``rollout_metadata(unknown_key=...)`` raises ``TypeError`` via
  Python's normal kwarg-validation path.

Concurrency
===========

``ContextVar`` (not ``threading.local``) so the metadata
propagates correctly across:

- ``concurrent.futures.ThreadPoolExecutor`` — workers inherit a
  copy of the submitter's context (Python 3.7+ default).
- ``asyncio.Task`` — each task carries its own context, with
  copy-on-write inheritance from the parent.
- Plain serial / single-thread — set + restore in-frame.

Multiprocessing-with-spawn does NOT inherit ContextVar values
(the child starts with a fresh interpreter). For multiprocessing
harnesses, the consumer sets the contextvar inside each child
process. Documented in the README's concurrency section.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass

from xrlenv.backends.base import CpuIsolation


@dataclass(frozen=True)
class RolloutMetadata:
    """Frozen typed snapshot of the per-rollout metadata fields.

    The drop-in's cluster-mode ``create_container`` override reads
    :data:`_ROLLOUT_METADATA_VAR` and converts each non-None ``str``
    field to a ``xrlenv.rollout.<field>`` docker label on the outgoing
    ``AcquireContainerCommand``.

    ``cpu_isolation`` is the one field that is NOT a label — it's an
    acquire-time scheduling hint (P2/P6 cpuset pinning) the drop-in
    forwards straight to ``acquire_container(cpu_isolation=...)``. It
    lets a docker-py-drop-in harness (e.g. swebench) opt a *specific*
    rollout into cpuset pinning per-task — the drop-in equivalent of
    the ``XRLENV_CPU_PINNING`` task-env marker the harbor/import_path
    plugins use — without turning pinning on globally. ``OFF`` (the
    default) is a no-op: the acquire behaves exactly as before.
    """

    artifact_path: str | None = None
    displayed_name: str | None = None
    group_id: str | None = None
    cpu_isolation: CpuIsolation = CpuIsolation.OFF


# Default = empty metadata (both fields None). The drop-in skips the
# label injection entirely when nothing's set, so the wire shape for
# uninstrumented harnesses stays byte-identical to what it was before
# this slice.
_EMPTY = RolloutMetadata()

_ROLLOUT_METADATA_VAR: contextvars.ContextVar[RolloutMetadata] = (
    contextvars.ContextVar("xrlenv_rollout_metadata", default=_EMPTY)
)


@contextlib.contextmanager
def rollout_metadata(
    *,
    artifact_path: str | None = None,
    displayed_name: str | None = None,
    group_id: str | None = None,
    cpu_isolation: CpuIsolation = CpuIsolation.OFF,
) -> Iterator[None]:
    """Set per-rollout metadata for the duration of the block.

    The drop-in's cluster-mode ``create_container`` reads the
    metadata on every ``client.containers.create(...)`` call inside
    the block, emits the corresponding ``xrlenv.rollout.*`` docker
    labels, and the control plane's RawContainerCoordinator parses
    them onto the persistent ``RawRolloutRecord``.

    Args:
        artifact_path: Absolute filesystem path on the consumer
            machine where the harness's per-instance artifacts
            live (e.g. ``<repo>/tmp/<job-id>/logs/run_evaluation/
            <run-id>/<model>/<instance-id>/``). Admin's per-rollout
            detail page renders this directory's contents inline if
            the path resolves on the control-plane host's
            filesystem; otherwise shows the path as a string for
            the operator to navigate to externally. Never uploaded
            over the wire.
        displayed_name: Operator-friendly name for the admin's
            ``/rollouts`` row (e.g. ``"astropy__astropy-7166"``
            instead of the synthetic uuid). Optional; admin falls
            back to a short prefix of ``rollout_id`` when None.
        group_id: Cancel-cohort tag emitted as the ``xrlenv.group_id``
            label on every acquire in the block, so a consumer can tear
            the whole cohort down in one call (``Client.terminate_raw_group``
            / the drop-in's ``terminate_raw_group``) — e.g. on Ctrl-C. The
            contextvar route means a harness that acquires through the
            drop-in (harbor, swebench) gets its containers tagged without
            passing an explicit ``labels={"xrlenv.group_id": ...}`` per call.
        cpu_isolation: Per-rollout cpuset-pinning hint (P2/P6). The
            drop-in forwards it to ``acquire_container(cpu_isolation=)``
            so a docker-py-drop-in harness can pin a *specific* task's
            container to whole cores (``nproc``/affinity == the CPU
            budget) — the fix for OpenMP/BLAS workloads that otherwise
            size their thread pools to the host core count and thrash
            against the CFS quota. ``OFF`` (default) is a no-op.

    Yields:
        None. The contextvar is restored on exit (normal or
        exceptional) via the standard contextvars reset pattern.

    Example:

    .. code-block:: python

        with xrlenv.rollout_metadata(
            artifact_path=str(artifact_root / instance_id),
            displayed_name=instance_id,
        ):
            run_instance(test_spec, pred, ..., client=client)
    """
    snapshot = RolloutMetadata(
        artifact_path=artifact_path,
        displayed_name=displayed_name,
        group_id=group_id,
        cpu_isolation=cpu_isolation,
    )
    token = _ROLLOUT_METADATA_VAR.set(snapshot)
    try:
        yield
    finally:
        _ROLLOUT_METADATA_VAR.reset(token)


def current_rollout_metadata() -> RolloutMetadata:
    """Return the currently-scoped rollout metadata.

    Returns the empty :class:`RolloutMetadata` (both fields None)
    when no ``rollout_metadata`` context is active. The drop-in's
    cluster-mode ``create_container`` calls this and skips label
    injection when nothing's set.
    """
    return _ROLLOUT_METADATA_VAR.get()


# Reserved docker-label keys the drop-in emits when the
# corresponding metadata field is set. Mirrored on the cluster
# side by ``RawContainerCoordinator`` to populate the typed
# columns on ``RawRolloutRecord``.
LABEL_ARTIFACT_PATH = "xrlenv.rollout.artifact_path"
LABEL_DISPLAYED_NAME = "xrlenv.rollout.displayed_name"
# Operator-supplied label keys the drop-in PROMOTES off the
# incoming labels dict — not emitted by xrlenv itself. Two
# conventions intentionally live in different namespaces:
# ``xrlenv.rollout.*`` for metadata flowing through the
# ``rollout_metadata()`` contextvar (above); bare ``xrlenv.*``
# for caller-passed scheduler hints + grouping (below).
LABEL_TASK_KEY = "xrlenv.task_key"
LABEL_GROUP_ID = "xrlenv.group_id"
# Fleet-reservation declaration (phase 1, opt-in). A consumer that
# schedules a *fleet* of containers per logical task (one long-lived
# lead + one or more heavier companions) declares it to the control
# plane with these three generic labels on the FIRST (fleet-opening)
# ``containers.run``; companions carry only ``xrlenv.fleet_id``. The
# control plane reserves the declared peak footprint on one node so the
# companions can't be starved by other work admitted in between. Core is
# generic — it never learns *why* a fleet has its shape; it only reads
# these labels (see ``RawContainerCoordinator`` + spec 03/10/21). The
# CPU / mem values are the fleet's *peak* footprint (a task-level number),
# NOT any single container's own request.
LABEL_FLEET_ID = "xrlenv.fleet_id"
LABEL_FLEET_CPU_REQUEST = "xrlenv.fleet_cpu_request"
LABEL_FLEET_MEM_REQUEST = "xrlenv.fleet_mem_request"


def metadata_to_labels(meta: RolloutMetadata) -> dict[str, str]:
    """Convert a :class:`RolloutMetadata` to the docker-label dict
    the drop-in merges into outgoing ``containers.create`` calls.

    Only sets keys for fields that are not None — the empty
    metadata case yields ``{}`` so uninstrumented harnesses don't
    pay any wire cost.
    """
    out: dict[str, str] = {}
    if meta.artifact_path is not None:
        out[LABEL_ARTIFACT_PATH] = meta.artifact_path
    if meta.displayed_name is not None:
        out[LABEL_DISPLAYED_NAME] = meta.displayed_name
    if meta.group_id is not None:
        # Emitted under the bare ``xrlenv.group_id`` cancel-cohort key (not the
        # ``xrlenv.rollout.*`` metadata namespace) — the same key an operator would pass
        # explicitly, so ``terminate_raw_group`` finds these containers by group.
        out[LABEL_GROUP_ID] = meta.group_id
    return out


__all__ = [
    "LABEL_ARTIFACT_PATH",
    "LABEL_DISPLAYED_NAME",
    "LABEL_FLEET_CPU_REQUEST",
    "LABEL_FLEET_ID",
    "LABEL_FLEET_MEM_REQUEST",
    "LABEL_GROUP_ID",
    "LABEL_TASK_KEY",
    "RolloutMetadata",
    "current_rollout_metadata",
    "metadata_to_labels",
    "rollout_metadata",
]
