"""Build-plan generator for swebench-verified.

Emits a per-image-ref ``build-plan.yaml`` with one ``BuildEntry``
per instance. swebench's upstream namespace publishes prebuilt
``swebench/sweb.eval.x86_64.<key>:latest`` images for every Verified
instance, so every entry uses ``context_source: type: registry``.

Sizes are probed from Docker Hub's v2 manifest API
(``size_hint_source: registry-probe``). The ``cluster-reported``
upgrade path via ``xrlenv build calibrate`` is planned for
B.1.next.b; until then sizes stay registry-probe.

Usage::

    .venv/bin/python -m xrlenv_plugins.images_build.swebench_verified.build_plan_gen \\
        --smoke --output build_plan.yaml

    .venv/bin/python -m xrlenv_plugins.images_build.swebench_verified.build_plan_gen \\
        --instances django__django-11099,sympy__sympy-13615 \\
        --output -

    # Full Verified sweep (500 instances). Authenticated + 8-way
    # concurrent probes runs in ~30-60s on a fast link; serial+unauth
    # takes ~10 minutes and may rate-limit (set $DOCKERHUB_USER +
    # $DOCKERHUB_TOKEN to lift the limit — see docs/technical_details/
    # images/build_plan.md "Docker Hub probing and rate limits"):
    .venv/bin/python -m xrlenv_plugins.images_build.swebench_verified.build_plan_gen \\
        --all --max-workers 8 --output build_plan.yaml

The committed ``build_plan.yaml`` next to this generator is the
phase-0 smoke set (8 instances). Operators driving a Verified sweep
should regenerate with ``--all``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from xrlenv_plugins.images_build._dockerhub_probe import (
    announce_auth_status,
    print_probe_summary,
    probe_image_size,
)

# tqdm ships transitively via the ``datasets`` package (the same
# dep that loads SWE-bench's HF dataset). Importing it here keeps
# the long-running ``--all`` run from looking hung: per-instance
# probes against Docker Hub take 200ms-10s each, and 500 of them
# without a progress bar is the user-found pain point. Fall back
# to a no-op iterator when tqdm isn't available (custom installs
# that strip the dep) so we don't break the CLI.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable

# Phase-0 smoke list. Mirrors ``examples/benchmarks-onboarding/
# swebench-verified/smoke.py::SMOKE_INSTANCES``; the list is
# committed in two places so an operator can drive the generator
# without standing up the smoke driver.
SMOKE_INSTANCES: tuple[str, ...] = (
    "astropy__astropy-7166",
    "django__django-11099",
    "sympy__sympy-18189",
    "astropy__astropy-12907",
    "astropy__astropy-14182",
    "sympy__sympy-13615",
    "django__django-11138",
    "sympy__sympy-12489",
)

# Default upstream image namespace + tag. The harness flips
# ``TestSpec.is_remote_image`` to True when ``namespace="swebench"``
# is passed to ``make_test_spec`` (see swebench/harness/test_spec.py).
DEFAULT_NAMESPACE = "swebench"
DEFAULT_TAG = "latest"

# Conservative default when the Docker Hub probe fails; swebench
# instance images run ~1.5-3 GiB compressed depending on the repo.
DEFAULT_SIZE_HINT_BYTES = 2_500_000_000  # 2.5 GiB


def _instance_to_image_ref(
    instance_id: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    tag: str = DEFAULT_TAG,
) -> str:
    """Mirror swebench's image-key derivation. Upstream replaces
    ``__`` with ``1776`` and lowercases the rest:
    ``astropy__astropy-7166`` -> ``swebench/sweb.eval.x86_64.astropy_1776_astropy-7166:latest``.

    See ``swebench.harness.test_spec.test_spec.TestSpec.instance_image_key``.
    """
    key = instance_id.replace("__", "_1776_").lower()
    return f"{namespace}/sweb.eval.x86_64.{key}:{tag}"


def _load_verified_instance_ids() -> list[str]:
    """All 500 Verified instance ids for the ``--all`` sweep.

    Reads the benchmark-local VALIDATED manifest (count/uniqueness/order/MANDATORY digest —
    audit M11) as the SOLE authority. Fails CLOSED if the manifest is missing/corrupt: there
    is deliberately no dataset-loader fallback (an unvalidated 499-row upstream result must
    never become the authority and green a 499-entry plan), and never the 8-instance smoke
    set. Regenerating the manifest is a separate, revision-pinned maintenance step."""
    from xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen import (
        read_verified_manifest,
    )
    ids = read_verified_manifest()
    if ids is None:
        raise SystemExit(
            "cannot resolve the Verified corpus for --all: the vendored manifest is "
            "missing/invalid (count/uniqueness/order/digest) — refusing to fall back to an "
            "unvalidated dataset load or the smoke set (audit M11). Restore/regenerate "
            "verified_instance_ids.txt.",
        )
    return ids


def generate_plan(
    instance_ids: list[str],
    *,
    namespace: str = DEFAULT_NAMESPACE,
    tag: str = DEFAULT_TAG,
    preferred_home_count: int = 1,
    pinned: bool = False,
    probe_sizes: bool = True,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Build a YAML-shaped ``BuildPlan`` dict (per-image-ref
    schema) with one entry per instance.

    ``max_workers`` chooses how many Docker Hub probes run
    concurrently. ``1`` (default) is serial. Pool sizes 8-16 are
    typical for the 500-entry ``--all`` sweep when authenticated
    (``$DOCKERHUB_USER`` + ``$DOCKERHUB_TOKEN`` set); each probe is
    network-bound so threads work better than processes.
    """
    if probe_sizes:
        announce_auth_status()
    # Per-instance Docker Hub probes can take 200ms-10s each (and
    # stall silently when DH rate-limits unauth requests past
    # ~100/6h), so a 500-instance ``--all`` run looks hung without a
    # heartbeat. Pass desc/unit so the line is informative even when
    # piped to a log file (tqdm degrades cleanly there).
    bar_desc = "probing image sizes" if probe_sizes else "building plan"

    def _build_entry(inst: str) -> dict[str, Any]:
        image_ref = _instance_to_image_ref(inst, namespace=namespace, tag=tag)
        repo = image_ref.rsplit(":", 1)[0]
        size_bytes: int | None = None
        size_source = "heuristic"
        if probe_sizes:
            size_bytes = probe_image_size(repo, tag)
            if size_bytes is not None:
                size_source = "registry-probe"
        if size_bytes is None:
            size_bytes = DEFAULT_SIZE_HINT_BYTES
        return {
            "image_ref": image_ref,
            "context_source": {"type": "registry"},
            "placement": {
                "preferred_home_count": preferred_home_count,
                "size_hint_bytes": size_bytes,
                "size_hint_source": size_source,
            },
            "pinned": pinned,
            "priority": 0,
            "labels": {
                "xrlenv.benchmark": "swebench-verified",
                "xrlenv.instance_id": inst,
            },
        }

    entries: list[dict[str, Any]] = []
    if max_workers <= 1:
        bar = tqdm(
            instance_ids, desc=bar_desc, unit="img",
            total=len(instance_ids), dynamic_ncols=True,
        )
        for inst in bar:
            if probe_sizes and hasattr(bar, "set_postfix_str"):
                # Update the bar's postfix with the current instance
                # so operators see where a stall happened if they
                # have to ^C mid-run.
                bar.set_postfix_str(inst[:40])
            entries.append(_build_entry(inst))
    else:
        # Threads (not processes): each probe is network-bound, so
        # the GIL doesn't hurt and threads share the helper's JWT
        # cache cheaply. ``ThreadPoolExecutor.map`` preserves input
        # order, so the resulting ``entries`` list matches
        # ``instance_ids`` order — important for the
        # content-addressed plan_id staying stable across re-runs.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="dh-probe",
        ) as pool:
            for entry in tqdm(
                pool.map(_build_entry, instance_ids),
                desc=f"{bar_desc} (x{max_workers})",
                unit="img",
                total=len(instance_ids),
                dynamic_ncols=True,
            ):
                entries.append(entry)

    if probe_sizes:
        print_probe_summary(DEFAULT_SIZE_HINT_BYTES)
    suffix = (
        "smoke-8" if list(instance_ids) == list(SMOKE_INSTANCES)
        else f"{len(instance_ids)}-instance"
    )
    return {
        "version": 1,
        "name": f"swebench-verified-{suffix}",
        "replication": 1,
        "budget": {
            "reserved_runtime_gb": reserved_runtime_gb,
            "buffer_gb": buffer_gb,
        },
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    # Auto-load .env from cwd / ancestors so operators who set
    # DOCKERHUB_USER + DOCKERHUB_TOKEN there get authenticated probes
    # without needing to shell-export. This generator runs as
    # ``python -m xrlenv_plugins...`` and never imports ``xrlenv``,
    # so the package-level import hook in ``xrlenv/__init__.py``
    # doesn't fire on its own (caught in 2026-05-11 audit). The
    # helper module is stdlib-only — no docker/grpc/prometheus
    # fan-out on this side-channel.
    from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
    _maybe_auto_load_dotenv()

    p = argparse.ArgumentParser(
        prog="swebench-verified-build-plan-gen",
        description=(
            "Generate a build-plan.yaml for swebench-verified. All "
            "instances have prebuilt registry images under "
            "docker.io/swebench/; the plan uses type: registry "
            "entries throughout."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true",
                     help="Every instance in the Verified split (500).")
    sel.add_argument("--smoke", action="store_true",
                     help="The 8 phase-0 smoke instances.")
    sel.add_argument("--instances", default=None,
                     help="Comma-separated explicit instance ids.")
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE,
                   help=f"Docker Hub namespace (default {DEFAULT_NAMESPACE}).")
    p.add_argument("--tag", default=DEFAULT_TAG,
                   help=f"Image tag (default {DEFAULT_TAG}).")
    p.add_argument("--preferred-home-count", type=int, default=1,
                   help="Per-image preferred_home_count (default 1).")
    p.add_argument("--pinned", action="store_true",
                   help="Pin entries so eviction skips them.")
    p.add_argument("--no-probe", action="store_true",
                   help="Skip Docker Hub size probes; use the conservative "
                        "default hint (2.5 GiB) for every entry.")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Number of concurrent Docker Hub probes (default "
                        "1 = serial). Probes are network-bound; pool sizes "
                        "8-16 cut a 500-entry --all sweep from ~5 min to "
                        "under a minute when authenticated. The output "
                        "entry order is preserved regardless of pool size.")
    p.add_argument("--output", default="-",
                   help="Output path (default '-' for stdout).")
    args = p.parse_args(argv)

    if args.all:
        instances = _load_verified_instance_ids()
    elif args.smoke:
        instances = list(SMOKE_INSTANCES)
    elif args.instances:
        instances = [s.strip() for s in args.instances.split(",") if s.strip()]
    else:
        return 2

    plan = generate_plan(
        instances,
        namespace=args.namespace,
        tag=args.tag,
        preferred_home_count=args.preferred_home_count,
        pinned=args.pinned,
        probe_sizes=not args.no_probe,
        max_workers=max(1, args.max_workers),
    )

    import yaml
    text = yaml.safe_dump(plan, sort_keys=False, default_flow_style=False)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
