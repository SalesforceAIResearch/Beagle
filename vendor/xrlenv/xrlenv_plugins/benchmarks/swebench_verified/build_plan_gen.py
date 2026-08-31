"""build_plan_gen.py — emit the swebench-verified image plan.

One ``registry``-type entry per instance: swebench publishes prebuilt
``swebench/sweb.eval.x86_64.<key>:latest`` images on Docker Hub for every Verified
instance, so nothing is built — the node's ImageCacheManager pulls on first
acquire. Sizes are probed from Docker Hub's v2 manifest API
(``size_hint_source: registry-probe``); ``xrlenv build calibrate`` upgrades them
to ``cluster-reported`` after the first warm.

The committed ``swebench_verified_build_plan.yaml`` next to this generator is the
8-instance smoke plan; regenerate with ``--all`` for the full Verified sweep.

Usage::

    python -m xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen \\
        --smoke --output xrlenv_plugins/benchmarks/swebench_verified/swebench_verified_build_plan.yaml
    python -m xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen \\
        --all --max-workers 8 --output -            # full 500, 8-way concurrent probes
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

from xrlenv_plugins.benchmarks._dockerhub_probe import (
    announce_auth_status,
    print_probe_summary,
    probe_image_size,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Any, **_kwargs: Any) -> Any:
        return iterable

# Kept in sync with build_cache.SMOKE_INSTANCES.
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

DEFAULT_NAMESPACE = "swebench"
DEFAULT_TAG = "latest"
# Conservative fallback when the Docker Hub probe fails (~1.5-3 GiB compressed).
DEFAULT_SIZE_HINT_BYTES = 2_500_000_000  # 2.5 GiB

SHARD = "swebench-verified"

# SWE-bench Verified is exactly 500 instances. `--all` prefers the cache shard, so a
# partial populate would silently yield a smaller plan than the CLI promises (audit M6).
VERIFIED_TOTAL = 500


def _instance_to_image_ref(
    instance_id: str, *, namespace: str = DEFAULT_NAMESPACE, tag: str = DEFAULT_TAG,
) -> str:
    """Mirror swebench's image-key derivation: ``__`` -> ``1776``, lowercased.
    ``astropy__astropy-7166`` -> ``swebench/sweb.eval.x86_64.astropy_1776_astropy-7166:latest``.
    (See ``swebench.harness.test_spec.TestSpec.instance_image_key``.)"""
    key = instance_id.replace("__", "_1776_").lower()
    return f"{namespace}/sweb.eval.x86_64.{key}:{tag}"


def _load_verified_instance_ids() -> list[str]:
    """Instance ids to plan for ``--all``.

    Prefer only SEMANTICALLY-COMPLETE cache entries (``build_cache.list_complete`` — a valid
    anchor whose id agrees with the dir name + matching extracts), so the plan matches the
    prepared corpus and a bare/corrupt anchor never inflates it. If no complete entry exists,
    fall back to the VALIDATED vendored manifest (offline-reliable, count/uniqueness/order/
    digest checked). NEVER the smoke set and NEVER an unrevisioned dataset load — an ``--all``
    plan must be authoritative; the caller validates the result against the manifest and only
    ``--allow-partial`` accepts a smaller complete subset (audit Low/M11)."""
    # Reject the retired cache env/path BEFORE reading it (audit: cache env renamed —
    # XRLENV_HARBOR_CACHE / .../xrlenv_harbor_cache retired -> unreliable results). Lazy imports
    # match plugin style.
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    from xrlenv_plugins.benchmarks.swebench_verified.build_cache import list_complete
    guard_legacy_cache_env()
    cache = os.environ.get("XRLENV_BENCHMARK_CACHE")
    if cache:
        ids = list_complete(cache)   # only complete entries; skips dot temp/lock siblings
        if ids:
            return ids
    manifest = read_verified_manifest()
    if manifest is not None:
        return manifest
    raise SystemExit(
        "cannot resolve the Verified corpus for --all: no complete cache entries and the "
        "vendored manifest is missing/invalid (audit M11). Populate the cache "
        "(build_cache.py --stage all --all) or restore verified_instance_ids.txt.",
    )


_MANIFEST = Path(__file__).with_name("verified_instance_ids.txt")


def read_verified_manifest() -> list[str] | None:
    """Read + VALIDATE the vendored 500-id manifest (audit M11). Returns the sorted id list
    iff the file is INTACT — id count == ``VERIFIED_TOTAL``, all unique, sorted, AND a
    MANDATORY ``sha256(ids)`` header digest that matches the id body — else None.

    An accidental edit, packaging damage, or a regeneration that dropped/added an id (a
    499- or 501-line file) must NOT silently become the authority; on any mismatch callers
    fail CLOSED rather than trusting a damaged pin. The digest is REQUIRED (not optional):
    a manifest with no ``sha256(ids)`` header is rejected, so a hand-edited body without a
    refreshed digest can't pass (audit M11)."""
    try:
        text = _MANIFEST.read_text()
    except OSError:
        return None
    ids: list[str] = []
    declared: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            m = re.match(r"#\s*([\w()]+)\s*:\s*(\S+)", s)   # header "# key : value"
            if m:
                declared[m.group(1).lower()] = m.group(2)
            continue
        if s:
            ids.append(s)
    if (len(ids) != VERIFIED_TOTAL or len(set(ids)) != len(ids) or ids != sorted(ids)):
        return None
    want = declared.get("sha256(ids)")
    if not want or want != hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest():
        return None   # digest missing OR mismatched -> not authoritative
    return ids


def _authoritative_ids() -> set[str] | None:
    """The authoritative 500 Verified instance ids from the VALIDATED vendored manifest, or
    None if it is missing/corrupt.

    The vendored pin is the SOLE authority (audit M11): there is deliberately NO network /
    dataset-loader fallback here. An unvalidated 499-row upstream result must never become the
    authority (a damaged manifest would then silently green a 499-entry plan); callers treat
    None as fail-closed. Regenerating the manifest is a separate, revision-pinned maintenance
    step — not something the gate does implicitly."""
    ids = read_verified_manifest()
    return set(ids) if ids is not None else None


def _authoritative_mismatch(instances: list[str]) -> str | None:
    """None if ``instances`` IS the full authoritative Verified corpus, else a one-line
    reason. Checks MEMBERSHIP against the VALIDATED vendored manifest — so 500 WRONG ids fail
    too, not only a short count. FAILS CLOSED when the manifest is unavailable/invalid: there
    is no count-only degradation (the old "accept any 500 unique ids" path is removed — audit
    M11), so a direct caller can't be tricked into accepting an unverifiable set."""
    have = set(instances)
    auth = _authoritative_ids()
    if auth is None:
        return ("authoritative Verified manifest unavailable/invalid — cannot verify "
                "membership (audit M11); regenerate verified_instance_ids.txt")
    missing, extra = auth - have, have - auth
    if missing or extra:
        return (f"cache does not match the authoritative Verified corpus: {len(missing)} "
                f"missing, {len(extra)} unexpected (have {len(have)}, want {len(auth)})")
    # Exact cardinality: the plan is generated from the ORIGINAL list, so 500 authoritative
    # ids PLUS a duplicate would pass membership but emit 501 entries (audit Low).
    if len(instances) != len(auth):
        return (f"{len(instances)} instance(s) but {len(auth)} unique authoritative ids "
                f"— duplicate id(s) in the requested set")
    return None


def generate_plan(
    instance_ids: list[str], *,
    namespace: str = DEFAULT_NAMESPACE, tag: str = DEFAULT_TAG,
    preferred_home_count: int = 1, pinned: bool = False, probe_sizes: bool = True,
    reserved_runtime_gb: int = 30, buffer_gb: int = 10, max_workers: int = 1,
) -> dict[str, Any]:
    """A YAML-shaped ``BuildPlan`` dict, one ``registry`` entry per instance."""
    if probe_sizes:
        announce_auth_status()
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
        bar = tqdm(instance_ids, desc=bar_desc, unit="img",
                   total=len(instance_ids), dynamic_ncols=True)
        for inst in bar:
            if probe_sizes and hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(inst[:40])
            entries.append(_build_entry(inst))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dh-probe") as pool:
            for entry in tqdm(pool.map(_build_entry, instance_ids),
                              desc=f"{bar_desc} (x{max_workers})", unit="img",
                              total=len(instance_ids), dynamic_ncols=True):
                entries.append(entry)

    if probe_sizes:
        print_probe_summary(DEFAULT_SIZE_HINT_BYTES)
    suffix = ("smoke-8" if list(instance_ids) == list(SMOKE_INSTANCES)
              else f"{len(instance_ids)}-instance")
    return {
        "version": 1,
        "name": f"swebench-verified-{suffix}",
        "replication": 1,
        "budget": {"reserved_runtime_gb": reserved_runtime_gb, "buffer_gb": buffer_gb},
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    # Auto-load .env so DOCKERHUB_USER/DOCKERHUB_TOKEN lift the probe rate limit.
    from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
    _maybe_auto_load_dotenv()

    p = argparse.ArgumentParser(
        prog="swebench-verified-build-plan-gen",
        description=(
            "Generate the swebench-verified image plan. All instances have "
            "prebuilt registry images under docker.io/swebench/ — the plan uses "
            "type: registry entries throughout."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true", help="Every Verified instance (500).")
    sel.add_argument("--smoke", action="store_true", help="The 8 smoke instances.")
    sel.add_argument("--instances", default=None, help="Comma-separated explicit ids.")
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--preferred-home-count", type=int, default=1)
    p.add_argument("--pinned", action="store_true")
    p.add_argument("--no-probe", action="store_true",
                   help="Skip Docker Hub size probes; use the 2.5 GiB default hint.")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Concurrent Docker Hub probes (default 1). Output order is preserved.")
    p.add_argument("--output", default="-", help="Output path (default '-' = stdout).")
    p.add_argument("--allow-partial", action="store_true",
                   help="Permit --all to emit a plan for fewer than the full 500 Verified "
                        "instances (intentional partial cache). Without this, a partial/"
                        "wrong-membership corpus FAILS closed rather than a partial plan.")
    args = p.parse_args(argv)

    if args.all:
        instances = _load_verified_instance_ids()
        # Duplicates ALWAYS fail — even under --allow-partial. A subset is a smaller SET;
        # repeated ids emit repeated image entries, which is never intentional (audit Low).
        dups = sorted({i for i in instances if instances.count(i) > 1})
        if dups:
            print(f"ERROR: --all has duplicate instance id(s): {', '.join(dups[:5])}"
                  f"{' …' if len(dups) > 5 else ''}", file=sys.stderr)
            return 1
        auth = _authoritative_ids()
        if auth is None:
            # Fail CLOSED (audit M11): with no VALIDATED authority we cannot verify Verified
            # membership. The old count-only degradation (accept any 500 unique ids) is gone —
            # --allow-partial does NOT relax this, it only relaxes a MISSING subset.
            print("ERROR: authoritative Verified manifest is missing/invalid (count/uniqueness/"
                  "order/digest) — refusing to generate a plan we cannot verify. Regenerate "
                  "verified_instance_ids.txt.", file=sys.stderr)
            return 1
        extra = sorted(set(instances) - auth)
        if extra:
            # Unexpected ids (not in the Verified corpus) are NEVER a subset — always fatal,
            # even under --allow-partial, which relaxes only MISSING ids (audit Low).
            print(f"ERROR: --all includes id(s) NOT in the authoritative Verified corpus: "
                  f"{', '.join(extra[:5])}{' …' if len(extra) > 5 else ''}", file=sys.stderr)
            return 1
        err = _authoritative_mismatch(instances)   # here: MISSING-subset only (dups/extras gone)
        if err and not args.allow_partial:
            # Fail closed: --all must emit the authoritative 500-instance Verified plan, not
            # a partial plan off a smoke or drifted cache (audit M6). --allow-partial is the
            # explicit opt-in for an intentional SUBSET (missing ids only).
            print(f"ERROR: {err}\n  Run 'build_cache.py --stage all --all' to materialize "
                  f"the full corpus, or pass --allow-partial for an intentional subset.",
                  file=sys.stderr)
            return 1
        if err:
            print(f"WARNING (--allow-partial): {err}", file=sys.stderr)
    elif args.smoke:
        instances = list(SMOKE_INSTANCES)
    else:
        instances = [s.strip() for s in args.instances.split(",") if s.strip()]

    plan = generate_plan(
        instances, namespace=args.namespace, tag=args.tag,
        preferred_home_count=args.preferred_home_count, pinned=args.pinned,
        probe_sizes=not args.no_probe, max_workers=max(1, args.max_workers),
    )

    import yaml
    text = yaml.safe_dump(plan, sort_keys=False, default_flow_style=False)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
