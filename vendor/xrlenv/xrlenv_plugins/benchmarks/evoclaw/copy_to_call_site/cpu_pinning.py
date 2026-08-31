"""Opt-in: turn on xrlenv's cpuset-pinning for EvoClaw's containers (``--cpu-pinning``).

xrlenv **already** implements per-container cpuset pinning: an acquire whose
``RuntimeLimits.cpu_pinning=True`` makes the node allocate ``ceil(cpu_limit)`` dedicated
cores from its per-node core ledger and set ``cpuset_cpus`` (released on destroy). The
**harbor** plugin turns it on by passing that ``RuntimeLimits`` straight to
``client.acquire_container`` (``xrlenv_plugins/harbor/environment.py``). We do **not** change
any xrlenv-core file to get the same thing here.

EvoClaw's adapter is the ``docker_shim``, which rides the docker-py compat
``.containers.run()``; its ``create_container`` builds the ``RuntimeLimits`` *internally*
from ``host_config`` (``_resolve_runtime_limits``) and exposes no ``cpu_pinning`` channel to
the caller. So — exactly like ``yd_fixes`` does for the harness — this module runtime
monkey-patches that ONE assembler function, from the onboarding, so every container the shim
acquires carries ``cpu_pinning=True``. It is in-process, gated by ``--cpu-pinning``, touches
no file under ``xrlenv/``, and leaves harbor and every other compat user untouched.

Why it matters (Table A): without pinning, ``nproc`` inside a ``--cpus N`` container reports
the *host* core count (e.g. 192), so ``go test`` / ``cargo`` / ``jest`` size their worker
pools to 192 and, under co-located load, oversubscribe the scheduler — tests near a timeout
flip and ``go -bench`` output is dropped (the FLIP/VARIES contention signature). Pinning
sizes ``nproc`` to ``N`` so the pools match the budget. See
``notes/table-a-resource-contention.md``.
"""
from __future__ import annotations

_APPLIED = False


def apply_cpu_pinning() -> None:
    """Install the cpuset-pinning patch (idempotent). Call before any acquire."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    import xrlenv.compat.docker_client as dc
    from xrlenv.backends.base import RuntimeLimits

    orig = dc._resolve_runtime_limits

    def _resolve_runtime_limits(host_config):  # type: ignore[no-untyped-def]
        rl = orig(host_config)
        if rl is None:
            return RuntimeLimits(cpu_pinning=True)
        if rl.cpu_pinning:
            return rl
        # RuntimeLimits is frozen (pydantic v2) -> copy with the flag flipped.
        return rl.model_copy(update={"cpu_pinning": True})

    dc._resolve_runtime_limits = _resolve_runtime_limits  # type: ignore[assignment]
    print(
        "🧵 cpu-pinning ON (--cpu-pinning): each acquired container gets ceil(cpus) "
        "dedicated cores (nproc == cpus), via xrlenv's existing core-ledger cpuset "
        "(RuntimeLimits.cpu_pinning). No xrlenv-core change.",
        flush=True,
    )
