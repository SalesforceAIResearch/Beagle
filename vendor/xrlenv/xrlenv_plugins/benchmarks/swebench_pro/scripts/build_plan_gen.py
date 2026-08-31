"""build_plan_gen.py — image warm-up plan generator for swebench-pro (ScaleAI/SWE-bench_Pro).

Emits a per-image-ref ``build-plan.yaml`` with one ``BuildEntry``
per instance. Pro's upstream publishes a prebuilt image for every
one of the 731 test instances, so every entry uses
``context_source: type: registry``.

The image story diverges from swebench-verified in two ways that
shape this generator:

  - **One shared repo, per-instance tag.** Verified publishes a
    distinct repo per instance (``swebench/sweb.eval.x86_64.<key>``)
    tagged ``latest``. Pro publishes *every* instance into the
    single repo ``jefzda/sweap-images`` and distinguishes them by
    tag.
  - **The tag is not derivable from the instance id.** Verified's
    image key is a deterministic ``__`` -> ``1776`` rewrite of the
    id. Pro's tag lives only in the dataset row's ``dockerhub_tag``
    column (it is an environment-hashed string, truncated to Docker
    Hub's 128-char tag limit). So this generator must load the
    dataset to learn every tag — there is no offline id->tag map.

Sizes are probed from Docker Hub's v2 manifest API
(``size_hint_source: registry-probe``); ``xrlenv build calibrate``
can later replace them with true on-disk sizes.

Usage::

    .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen \\
        --smoke --output build_plan.yaml

    # the filtered / subset-100 plans, sizes carried over from the committed full plan (no probing):
    .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen \\
        --subset-100 --resume ../build_plan_full.yaml --no-probe --output build_plan_subset_100.yaml
    .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen \\
        --filtered --resume ../build_plan_full.yaml --no-probe --output build_plan_filtered.yaml

    .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen \\
        --instances instance_ansible__ansible-f327e65d...-vba6da65a... \\
        --output -

    # Full Pro sweep (731 instances). Authenticated + 8-way
    # concurrent probes runs in ~1-2 min on a fast link; serial+unauth
    # is much slower and may rate-limit (set $DOCKERHUB_USER +
    # $DOCKERHUB_TOKEN to lift the limit — see docs/technical_details/
    # images/build_plan.md "Docker Hub probing and rate limits"):
    .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen \\
        --all --max-workers 8 --output build_plan.yaml

Docker Hub's ``hub.docker.com/v2`` metadata API enforces a
per-account request quota on a rolling ~6h window that a token
*raises but does not remove* — a 731-entry ``--all`` sweep can
exhaust it partway through and 429 the rest. Re-running with
``--resume`` carries forward every ``registry-probe`` size from a
prior plan and only re-probes the entries that fell back to the
heuristic, so a sweep can be completed across several windows:

    .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen \\
        --all --max-workers 8 --resume build_plan.yaml --output build_plan.yaml

Three plans are committed, one per shipped
configuration: ``../build_plan_full.yaml`` (731, every size registry-
probed), ``build_plan_filtered.yaml`` (478) and
``build_plan_subset_100.yaml`` (100) — the latter two derived from
the full plan with ``--resume … --no-probe`` as above.

The dataset rows come from ``$SWEBENCH_PRO_PARQUET`` (the parquet, or
a snapshot directory — see build_cache.py) when it is set, else from
the HF hub (``datasets.load_dataset``, network).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from xrlenv_plugins.benchmarks._dockerhub_probe import (
    announce_auth_status,
    print_probe_summary,
    probe_image_size,
)

# tqdm ships transitively via the ``datasets`` package (the same
# dep that loads Pro's HF dataset). Importing it here keeps the
# long-running ``--all`` run from looking hung: per-instance probes
# against Docker Hub take 200ms-10s each, and 731 of them without a
# progress bar is the user-found pain point. Fall back to a no-op
# iterator when tqdm isn't available (custom installs that strip
# the dep) so we don't break the CLI.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable

# Pro's HF dataset. 731 instances on the public test split.
DATASET_PATH = "ScaleAI/SWE-bench_Pro"
DEFAULT_SPLIT = "test"

# All Pro instance images live in this single Docker Hub repo,
# distinguished by tag (upstream's evaluator pulls the same repo).
DEFAULT_REPO = "jefzda/sweap-images"

# Number of instances the ``--smoke`` selection takes off the front
# of the test split. Unlike swebench-verified, Pro has no separate
# onboarding smoke driver to mirror, so the smoke set is simply the
# first N rows of the dataset in its natural (stable) order.
SMOKE_COUNT = 8

# Conservative default when the Docker Hub probe fails; Pro
# instance images run ~0.5-1 GiB compressed (registry-probe
# observed range), so 2.5 GiB leaves generous bin-packing headroom.
DEFAULT_SIZE_HINT_BYTES = 2_500_000_000  # 2.5 GiB


KIT = Path(__file__).resolve().parent
FILTERED_IDS = KIT / "filtered_instance_ids.txt"          # the quality-filtered set (478)
SUBSET_100_IDS = KIT / "subset_100_instance_ids.txt"      # the 100-task sample of it (sample_subset.py)


def _local_parquet() -> Path | None:
    """The local dataset named by ``$SWEBENCH_PRO_PARQUET`` (file or snapshot directory; resolved by
    build_cache.parquet_path, which fails loud on a bad value); ``None`` when unset → the HF hub."""
    import os
    if not os.environ.get("SWEBENCH_PRO_PARQUET"):
        return None
    from xrlenv_plugins.benchmarks.swebench_pro.build_cache import parquet_path
    return parquet_path()


def _load_pro_instances() -> list[dict[str, str]]:
    """Load every Pro instance as a ``{instance_id, dockerhub_tag}``
    dict, in dataset order.

    Pro's tag is not a function of the id, so the full dataset is
    the only source of truth for the id->tag mapping. The local
    snapshot named by ``$SWEBENCH_PRO_PARQUET`` is preferred — offline
    and identical to the hub split; the hub download is the fallback.
    """
    local = _local_parquet()
    if local is not None:
        import pyarrow.parquet as pq
        table = pq.read_table(local, columns=["instance_id", "dockerhub_tag"])
        return [{"instance_id": r["instance_id"], "dockerhub_tag": r["dockerhub_tag"]} for r in table.to_pylist()]
    from datasets import load_dataset

    ds = load_dataset(DATASET_PATH, split=DEFAULT_SPLIT)
    rows: list[dict[str, str]] = []
    for row in tqdm(
        ds, desc="loading Pro dataset", unit="row",
        total=len(ds), dynamic_ncols=True,
    ):
        rows.append({
            "instance_id": row["instance_id"],
            "dockerhub_tag": row["dockerhub_tag"],
        })
    return rows


def _load_probed_sizes(path: Path) -> dict[str, int]:
    """Read a prior build-plan.yaml and return ``{image_ref:
    size_bytes}`` for every entry whose size was *registry-probed*.

    Entries that fell back to the heuristic are deliberately omitted
    so a ``--resume`` run re-probes exactly those — the Docker Hub
    metadata-API quota is per-rolling-window, so a sweep that 429s
    partway can be finished across several windows by re-running
    with the partial plan as ``--resume``.
    """
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    cache: dict[str, int] = {}
    for entry in data.get("entries", []):
        placement = entry.get("placement") or {}
        if placement.get("size_hint_source") != "registry-probe":
            continue
        ref = entry.get("image_ref")
        size = placement.get("size_hint_bytes")
        if isinstance(ref, str) and isinstance(size, int) and size > 0:
            cache[ref] = size
    return cache


def _read_ids_file(path: Path) -> list[str]:
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if not ids:
        raise SystemExit(f"{path} lists no instance ids")
    return ids


def _select_instances(
    rows: list[dict[str, str]],
    *,
    all_: bool,
    smoke: bool,
    instances: str | None,
    ids_file: Path | None = None,
    filtered: bool = False,
    subset_100: bool = False,
) -> list[dict[str, str]]:
    """Resolve the CLI selection flags to an ordered list of rows.

    ``ids_file`` (an id manifest, one per line, ``#`` comments),
    ``filtered`` (``filtered_instance_ids.txt``) and ``subset_100``
    (``subset_100_instance_ids.txt``, written by ``sample_subset.py``)
    keep the manifest's order.
    """
    if all_:
        return rows
    if smoke:
        return rows[:SMOKE_COUNT]
    if filtered and ids_file is None:
        ids_file = FILTERED_IDS
    if subset_100 and ids_file is None:
        ids_file = SUBSET_100_IDS
    if ids_file is not None:
        if not ids_file.is_file():
            raise SystemExit(f"id manifest not found: {ids_file}"
                             + (" — regenerate it with sample_subset.py" if subset_100 else ""))
        instances = ",".join(_read_ids_file(ids_file))
    if instances:
        wanted = [s.strip() for s in instances.split(",") if s.strip()]
        by_id = {r["instance_id"]: r for r in rows}
        picked: list[dict[str, str]] = []
        for inst in wanted:
            if inst not in by_id:
                raise KeyError(
                    f"instance_id {inst!r} not found in {DATASET_PATH} "
                    f"split={DEFAULT_SPLIT!r}"
                )
            picked.append(by_id[inst])
        return picked
    # Argparse mutex group guarantees this is unreachable.
    raise AssertionError("no selection flag set")


def generate_plan(
    instances: list[dict[str, str]],
    *,
    repo: str = DEFAULT_REPO,
    preferred_home_count: int = 1,
    pinned: bool = False,
    probe_sizes: bool = True,
    probed_cache: dict[str, int] | None = None,
    is_smoke: bool = False,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
    max_workers: int = 1,
    plan_label: str | None = None,
) -> dict[str, Any]:
    """Build a YAML-shaped ``BuildPlan`` dict (per-image-ref
    schema) with one entry per instance.

    ``instances`` is a list of ``{instance_id, dockerhub_tag}``
    dicts (see :func:`_load_pro_instances`). Every entry's image
    ref is ``<repo>:<dockerhub_tag>`` and uses
    ``context_source: { type: registry }`` since Pro publishes a
    prebuilt image per instance.

    ``probed_cache`` maps ``image_ref`` to a previously
    registry-probed size (see :func:`_load_probed_sizes`). An entry
    found in the cache skips its network probe entirely and reuses
    the cached size — this is what makes ``--resume`` idempotent:
    re-running only spends Docker Hub quota on the entries that
    still lack a probed size.

    ``max_workers`` chooses how many Docker Hub probes run
    concurrently. ``1`` (default) is serial. Pool sizes 8-16 are
    typical for the 731-entry ``--all`` sweep when authenticated
    (``$DOCKERHUB_USER`` + ``$DOCKERHUB_TOKEN`` set); each probe is
    network-bound so threads work better than processes.
    """
    if probe_sizes:
        announce_auth_status()
    bar_desc = "probing image sizes" if probe_sizes else "building plan"
    cache = probed_cache or {}

    def _build_entry(row: dict[str, str]) -> dict[str, Any]:
        inst = row["instance_id"]
        tag = row["dockerhub_tag"]
        image_ref = f"{repo}:{tag}"
        size_bytes: int | None = None
        size_source = "heuristic"
        cached = cache.get(image_ref)
        if cached is not None:
            # Carried over from a prior plan via --resume; no probe.
            size_bytes = cached
            size_source = "registry-probe"
        elif probe_sizes:
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
                "xrlenv.benchmark": "swebench-pro",
                "xrlenv.instance_id": inst,
            },
        }

    entries: list[dict[str, Any]] = []
    if max_workers <= 1:
        bar = tqdm(
            instances, desc=bar_desc, unit="img",
            total=len(instances), dynamic_ncols=True,
        )
        for row in bar:
            if probe_sizes and hasattr(bar, "set_postfix_str"):
                # Surface the current instance so operators see where
                # a stall happened if they have to ^C mid-run.
                bar.set_postfix_str(row["instance_id"][:40])
            entries.append(_build_entry(row))
    else:
        # Threads (not processes): each probe is network-bound, so
        # the GIL doesn't hurt and threads share the helper's JWT
        # cache cheaply. ``ThreadPoolExecutor.map`` preserves input
        # order, so ``entries`` matches ``instances`` order — keeps
        # the content-addressed plan_id stable across re-runs.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="dh-probe",
        ) as pool:
            for entry in tqdm(
                pool.map(_build_entry, instances),
                desc=f"{bar_desc} (x{max_workers})",
                unit="img",
                total=len(instances),
                dynamic_ncols=True,
            ):
                entries.append(entry)

    if probe_sizes:
        print_probe_summary(DEFAULT_SIZE_HINT_BYTES)
    # plan names: swebench-pro-full-731 / -filtered-478 / -subset-100 (label + count), smoke-8, else N-instance
    suffix = f"smoke-{SMOKE_COUNT}" if is_smoke else f"{len(instances)}-instance"
    if plan_label:
        suffix = f"{plan_label}-{len(instances)}"
    return {
        "version": 1,
        "name": f"swebench-pro-{suffix}",
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
    # doesn't fire on its own. The helper module is stdlib-only — no
    # docker/grpc/prometheus fan-out on this side-channel.
    from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
    _maybe_auto_load_dotenv()

    p = argparse.ArgumentParser(
        prog="swebench-pro-build-plan-gen",
        description=(
            "Generate a build-plan.yaml for swebench-pro. All "
            "instances have prebuilt registry images under the "
            f"docker.io/{DEFAULT_REPO} repo (one tag per instance); "
            "the plan uses type: registry entries throughout."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true",
                     help="Every instance in the Pro test split (731).")
    sel.add_argument("--smoke", action="store_true",
                     help=f"The first {SMOKE_COUNT} instances of the split.")
    sel.add_argument("--instances", default=None,
                     help="Comma-separated explicit instance ids.")
    sel.add_argument("--ids-file", default=None, metavar="PATH",
                     help="An id manifest (one instance id per line, '#' comments).")
    sel.add_argument("--filtered", action="store_true",
                     help=f"The quality-filtered set ({FILTERED_IDS.name}, 478).")
    sel.add_argument("--subset-100", action="store_true",
                     help="The 100-task sample of the filtered set "
                          f"({SUBSET_100_IDS.name}, written by sample_subset.py).")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"Docker Hub repo (default {DEFAULT_REPO}).")
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
                        "8-16 cut a 731-entry --all sweep substantially when "
                        "authenticated. The output entry order is preserved "
                        "regardless of pool size.")
    p.add_argument("--resume", default=None, metavar="PATH",
                   help="Path to a prior build-plan.yaml. Entries already "
                        "carrying a registry-probed size are reused as-is "
                        "(no network probe); only heuristic-fallback entries "
                        "are re-probed. Lets a quota-throttled --all sweep "
                        "be finished across several windows. Safe to pass "
                        "the same file as --output.")
    p.add_argument("--output", default="-",
                   help="Output path (default '-' for stdout).")
    args = p.parse_args(argv)

    probed_cache: dict[str, int] | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            probed_cache = _load_probed_sizes(resume_path)
            print(
                f"resume: carried {len(probed_cache)} registry-probe "
                f"size(s) from {resume_path}; remaining entries will be "
                f"probed.",
                file=sys.stderr,
            )
        else:
            print(
                f"resume: {resume_path} not found — probing every entry.",
                file=sys.stderr,
            )

    rows = _load_pro_instances()
    instances = _select_instances(
        rows, all_=args.all, smoke=args.smoke, instances=args.instances,
        ids_file=Path(args.ids_file).expanduser() if args.ids_file else None,
        filtered=args.filtered, subset_100=args.subset_100,
    )
    plan_label = ("full" if args.all else "filtered" if args.filtered else "subset" if args.subset_100
                  else "ids" if args.ids_file else None)

    plan = generate_plan(
        instances,
        repo=args.repo,
        preferred_home_count=args.preferred_home_count,
        pinned=args.pinned,
        probe_sizes=not args.no_probe,
        probed_cache=probed_cache,
        is_smoke=args.smoke,
        max_workers=max(1, args.max_workers),
        plan_label=plan_label,
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
