#!/usr/bin/env python3
"""sample_subset.py — a per-repo-balanced sample of the quality-filtered set (the shipped subset-100).

SWE-bench Pro's 731 public instances come from 11 repositories; the quality filter keeps 478 of
them (``filtered_instance_ids.txt``) and every repo still holds dozens. This script draws a
smaller, repo-balanced evaluation set from the kept ids. The defaults reproduce the committed
``subset_100_instance_ids.txt``: ``--total 100 --policy random --seed 0`` — 100 instances spread
over the 11 repos proportionally to their kept counts, every repo ≥ 1.

Quota: ``--total K`` spreads K proportionally (largest remainders first, each repo ≥ 1, capped by
availability); ``--per-repo N`` takes up to N per repo (``--per-repo 1`` is the cheapest set that
still exercises every repo's image family, run script and parser — images of one repo share a base).
Policy: ``random`` (seeded; default) | ``first`` (dataset order) | ``smallest-image`` (registry-probed
sizes from ``--sizes-plan``, default ``build_plan_full.yaml``; ties → id order).

Outputs: the id manifest (``--out``, default ``subset_100_instance_ids.txt``) and a JSON report
(``--report``, default ``subset_100.json``: per pick repo / language / image / size). Regenerate the
matching image plan afterwards:
``build_plan_gen.py --subset-100 --resume ../build_plan_full.yaml --no-probe``.
Input: ``$SWEBENCH_PRO_PARQUET`` (see build_cache.py). ``--dry-run`` prints the selection only.

    .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.sample_subset                      # the shipped subset-100
    .venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.sample_subset --per-repo 1 --policy smallest-image --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from xrlenv_plugins.benchmarks.swebench_pro.build_cache import (
    load_dotenv,
    parquet_path,
    read_ids_file,
)

HERE = Path(__file__).resolve().parent
DEFAULT_KEPT = HERE / "filtered_instance_ids.txt"
DEFAULT_OUT = HERE / "subset_100_instance_ids.txt"
DEFAULT_REPORT = HERE / "subset_100.json"
DEFAULT_SIZES_PLAN = HERE.parent / "build_plan_full.yaml"      # registry-probed sizes for every public instance
DEFAULT_TOTAL = 100
DEFAULT_SEED = 0
IMAGE_REPO = "jefzda/sweap-images"
POLICIES = ("random", "first", "smallest-image")


def load_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=["instance_id", "repo", "repo_language", "dockerhub_tag"]).to_pylist()


def load_sizes(plan: Path | None) -> dict[str, int]:
    """``{image_ref: size_bytes}`` for registry-probed entries of a plan (missing plan → empty)."""
    if plan is None or not plan.is_file():
        return {}
    import yaml
    out: dict[str, int] = {}
    for e in (yaml.safe_load(plan.read_text()) or {}).get("entries", []):
        pl = e.get("placement") or {}
        if pl.get("size_hint_source") == "registry-probe" and isinstance(pl.get("size_hint_bytes"), int):
            out[e["image_ref"]] = pl["size_hint_bytes"]
    return out


def _quota(groups: dict[str, list[Any]], per_repo: int | None, total: int | None) -> dict[str, int]:
    """How many instances each repo contributes. ``per_repo`` caps every repo at N (default 1);
    ``total`` instead spreads K over the repos proportionally to their kept counts (every repo ≥ 1,
    largest remainders first), capped by availability."""
    if total is not None:
        n = sum(len(c) for c in groups.values())
        if total >= n:
            return {r: len(c) for r, c in groups.items()}
        base = {r: max(1, min(len(c), (total * len(c)) // n)) for r, c in groups.items()}
        rem = total - sum(base.values())
        order = sorted(groups, key=lambda r: ((total * len(groups[r])) % n, len(groups[r])), reverse=True)
        i = 0
        while rem > 0 and any(base[r] < len(groups[r]) for r in groups):
            r = order[i % len(order)]
            i += 1
            if base[r] < len(groups[r]):
                base[r] += 1
                rem -= 1
        while rem < 0:
            r = max(groups, key=lambda r: base[r])
            base[r] -= 1
            rem += 1
        return base
    k = max(1, int(per_repo or 1))
    return {r: min(k, len(c)) for r, c in groups.items()}


def select_subset(rows: list[dict[str, Any]], kept: list[str], *, policy: str = "random",
                  sizes: dict[str, int] | None = None, seed: int = DEFAULT_SEED, per_repo: int | None = None,
                  total: int | None = None) -> list[dict[str, Any]]:
    """Up to ``per_repo`` rows per repo, or ``total`` rows spread proportionally over the repos — always drawn
    from ``kept`` only, repos in dataset order (``per_repo`` and ``total`` both unset → one per repo). Pure."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    by_id = {r["instance_id"]: r for r in rows}
    missing = [k for k in kept if k not in by_id]
    if missing:
        raise SystemExit(f"kept ids not in the dataset: {missing[:3]}{' …' if len(missing) > 3 else ''}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for k in kept:                                   # manifest order == dataset order for the kept set
        groups[by_id[k]["repo"]].append(by_id[k])
    sizes = sizes or {}
    rng = random.Random(seed)
    quota = _quota(groups, per_repo, total)
    picks: list[dict[str, Any]] = []
    for repo, cands in groups.items():
        n = quota[repo]
        if policy == "first":
            chosen = cands[:n]
        elif policy == "random":
            chosen = rng.sample(cands, n)
        else:
            def key(r: dict[str, Any]) -> tuple[int, str]:
                return (sizes.get(f"{IMAGE_REPO}:{r['dockerhub_tag']}", 1 << 62), r["instance_id"])
            chosen = sorted(cands, key=key)[:n]
        for pick in chosen:
            picks.append({"repo": repo, "instance_id": pick["instance_id"], "language": pick["repo_language"],
                          "image_ref": f"{IMAGE_REPO}:{pick['dockerhub_tag']}",
                          "size_hint_bytes": sizes.get(f"{IMAGE_REPO}:{pick['dockerhub_tag']}"), "kept_in_repo": len(cands)})
    return picks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swebench-pro sample_subset", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kept", default=str(DEFAULT_KEPT), help=f"the quality-filtered id manifest (default: {DEFAULT_KEPT.name})")
    p.add_argument("--parquet", default=None, help="dataset parquet or snapshot directory (default $SWEBENCH_PRO_PARQUET)")
    p.add_argument("--sizes-plan", default=str(DEFAULT_SIZES_PLAN), help=f"plan with registry-probed sizes (default: {DEFAULT_SIZES_PLAN.name})")
    p.add_argument("--policy", choices=POLICIES, default="random")
    q = p.add_mutually_exclusive_group()
    q.add_argument("--total", type=int, default=None, metavar="K", help=f"K instances spread over the repos proportionally to their kept counts, each repo ≥ 1 (default {DEFAULT_TOTAL})")
    q.add_argument("--per-repo", type=int, default=None, metavar="N", help="up to N instances per repo instead")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"id manifest to write (default: {DEFAULT_OUT.name})")
    p.add_argument("--report", default=str(DEFAULT_REPORT), help=f"JSON report to write (default: {DEFAULT_REPORT.name})")
    p.add_argument("--dry-run", action="store_true", help="print the selection, write nothing")
    a = p.parse_args(argv)
    load_dotenv()
    total = a.total if (a.total is not None or a.per_repo is not None) else DEFAULT_TOTAL
    kept_path = Path(a.kept).expanduser()
    rows = load_rows(parquet_path(a.parquet))
    kept = read_ids_file(kept_path)
    picks = select_subset(rows, kept, policy=a.policy, sizes=load_sizes(Path(a.sizes_plan).expanduser() if a.sizes_plan else None), seed=a.seed,
                          per_repo=a.per_repo, total=total)
    size_total = sum(x["size_hint_bytes"] or 0 for x in picks)
    per = defaultdict(list)
    for x in picks:
        per[x["repo"]].append(x)
    for repo, xs in per.items():
        gb = sum(x["size_hint_bytes"] or 0 for x in xs) / 1e9
        print(f"  {repo:36s} {xs[0]['language']:6s} {len(xs):3d} of {xs[0]['kept_in_repo']:3d} kept  {gb:6.2f} GB" + (f"  {xs[0]['instance_id']}" if len(xs) == 1 else ""), file=sys.stderr)
    print(f"{len(per)} repos -> {len(picks)} instances, {size_total / 1e9:.1f} GB of images (policy {a.policy}, from {len(kept)} kept)", file=sys.stderr)
    if a.dry_run:
        return 0
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    how = f"--total {total}" if total is not None else f"--per-repo {a.per_repo}"
    out.write_text(f"# {len(picks)} instances ({how}: {'proportional per repo' if total is not None else 'per-repo cap'}), from the quality-filtered set ({len(kept)} kept); generated by\n"
                   f"# xrlenv_plugins/benchmarks/swebench_pro/scripts/sample_subset.py --policy {a.policy} --seed {a.seed} {how}\n"
                   + "\n".join(x["instance_id"] for x in picks) + "\n", encoding="utf-8")
    kept_label = kept_path.name if kept_path.resolve().parent == HERE else str(a.kept)    # the shipped kit names no absolute paths
    Path(a.report).expanduser().write_text(json.dumps({"policy": a.policy, "seed": a.seed, "per_repo": a.per_repo, "total": total, "kept_manifest": kept_label,
                                                     "n_kept": len(kept), "picks": picks, "total_size_hint_bytes": size_total}, indent=1), encoding="utf-8")
    print(f"wrote {out} and {a.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
