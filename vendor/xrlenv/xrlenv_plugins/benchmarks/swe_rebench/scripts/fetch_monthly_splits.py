#!/usr/bin/env python3
"""Fetch upstream's monthly-split membership and write ``monthly_splits.json``.

SWE-rebench is organised into monthly splits (``2025_01`` … ), added
continuously, whose union is the ``test`` split. The Harbor Hub package
``build_cache.py`` downloads **is** that ``test`` split, flattened: every task's
``[metadata] source`` reads ``…::test/<id>`` and records no month. This script
restores the grouping so ``run_full_sweep.sh --split`` can select one.

The mapping is **not derivable from the cache**. The nearest local signal,
``tests/config.json``'s ``created_at``, is the upstream *PR* date: it agrees with
the split name for all but the newest split, which absorbs every
recently-collected task — as of 2026-09-01 that misfiles 64 of 860 tasks into
``2026_04`` / ``2026_05``, months that are not splits at all. So the HF
datasets-server is the authority, and this script is the only way to refresh.

Run it after a corpus refresh (``tests/test_monthly_splits.py`` pins the current
counts and fails until you do)::

    .venv/bin/python -m xrlenv_plugins.benchmarks.swe_rebench.scripts.fetch_monthly_splits
    .venv/bin/python -m …fetch_monthly_splits --check      # verify, write nothing

Output shape (ids sorted, so a regeneration diffs cleanly)::

    {"dataset": …, "task_count": 860, "splits": {"2025_01": ["id", …], …}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "nebius/SWE-rebench-leaderboard"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "monthly_splits.json"
_SERVER = "https://datasets-server.huggingface.co"
# The union split — excluded from the per-month mapping (it IS the union).
_UNION_SPLIT = "test"
_PAGE = 100          # datasets-server caps a /rows page at 100
_RETRIES = 5
# The datasets-server rate-limits an 860-row walk (HTTP 429). A plain linear
# backoff is not enough — back off HARD on 429 specifically, and honour
# Retry-After when the server sends one.
_RATE_LIMIT_BACKOFF_S = 20.0


def _get(url: str, *, timeout: float = 90.0) -> Any:
    """GET + parse JSON, retrying transient failures with a linear backoff.

    The datasets-server rate-limits and occasionally 5xxs on a cold cache; a
    partial fetch would silently produce an index missing tasks, so every
    failure either retries or raises.
    """
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429 and attempt < _RETRIES - 1:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                delay = max(delay, _RATE_LIMIT_BACKOFF_S * (attempt + 1))
                print(
                    f"   rate-limited (429); sleeping {delay:.0f}s "
                    f"[{attempt + 1}/{_RETRIES - 1}]",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue
            if attempt < _RETRIES - 1:
                time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if attempt < _RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    hint = ""
    if isinstance(last, urllib.error.HTTPError) and last.code == 429:
        hint = (
            " — the HF datasets-server rate limit is per-IP and resets after a "
            "few minutes; re-run then. A partial index is never written."
        )
    raise SystemExit(
        f"ERROR: giving up on {url}: {type(last).__name__}: {last}{hint}",
    )


def list_splits(dataset: str) -> list[str]:
    """Every monthly split name, sorted. Excludes the ``test`` union split."""
    query = urllib.parse.urlencode({"dataset": dataset})
    payload = _get(f"{_SERVER}/splits?{query}")
    names = sorted(
        s["split"] for s in payload.get("splits", []) if s["split"] != _UNION_SPLIT
    )
    if not names:
        raise SystemExit(
            f"ERROR: {dataset} exposes no split besides {_UNION_SPLIT!r} — refusing "
            f"to write an empty index.",
        )
    return names


def fetch_split_ids(dataset: str, split: str) -> list[str]:
    """Every ``instance_id`` in one split, sorted, paging until exhausted."""
    ids: list[str] = []
    offset = 0
    total: int | None = None
    while True:
        query = urllib.parse.urlencode({
            "dataset": dataset, "config": "default", "split": split,
            "offset": offset, "length": _PAGE,
        })
        payload = _get(f"{_SERVER}/rows?{query}")
        if isinstance(payload.get("num_rows_total"), int):
            total = int(payload["num_rows_total"])
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            instance_id = row.get("row", {}).get("instance_id")
            if not instance_id:
                raise SystemExit(
                    f"ERROR: {dataset}:{split} row {offset} has no 'instance_id' — "
                    f"the upstream schema changed; this script needs updating.",
                )
            ids.append(str(instance_id))
        offset += len(rows)
        if total is not None and offset >= total:
            break
    # Verify AFTER the loop, so an early empty page (the server truncating the
    # walk) is caught too — checking inside the `offset >= total` branch could
    # never fire, since offset and len(ids) advance in lockstep.
    if total is not None and len(ids) != total:
        raise SystemExit(
            f"ERROR: {dataset}:{split} returned {len(ids)} row(s), expected "
            f"{total} — refusing to write a partial split. Re-run.",
        )
    return sorted(ids)


def build_index(dataset: str) -> dict[str, Any]:
    splits = list_splits(dataset)
    mapping: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for split in splits:
        ids = fetch_split_ids(dataset, split)
        for instance_id in ids:
            if instance_id in seen:
                raise SystemExit(
                    f"ERROR: {instance_id} appears in both {seen[instance_id]} and "
                    f"{split} — the splits are expected to be disjoint.",
                )
            seen[instance_id] = split
        mapping[split] = ids
        print(f"  {split}: {len(ids)}", file=sys.stderr)
    return {
        "dataset": dataset,
        "task_count": len(seen),
        "splits": mapping,
    }


def render(index: dict[str, Any]) -> str:
    """Deterministic JSON (sorted keys, trailing newline) so a regeneration that
    changes nothing produces a zero-line diff."""
    return json.dumps(index, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_monthly_splits",
        description=(
            "Fetch SWE-rebench's monthly-split membership from the HF "
            "datasets-server and write monthly_splits.json."
        ),
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"HF dataset (default {DEFAULT_DATASET}).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Index path (default monthly_splits.json beside this "
                             "script). '-' writes to stdout.")
    parser.add_argument("--check", action="store_true",
                        help="Compare upstream against the committed index and exit "
                             "non-zero if they differ. Writes nothing.")
    args = parser.parse_args(argv)

    # --check reads the committed index first: a missing file is answerable
    # without spending a rate-limited walk of the whole dataset.
    if args.check and args.output != "-" and not Path(args.output).is_file():
        print(f"MISSING: {args.output} — run without --check to create it.",
              file=sys.stderr)
        return 1

    print(f">> fetching splits for {args.dataset}", file=sys.stderr)
    index = build_index(args.dataset)
    text = render(index)

    if args.check:
        path = Path(args.output)
        if path.read_text() == text:
            print(
                f"OK: {path.name} matches upstream "
                f"({index['task_count']} tasks, {len(index['splits'])} splits).",
                file=sys.stderr,
            )
            return 0
        current = json.loads(path.read_text())
        print(
            f"STALE: {path.name} has {current.get('task_count')} tasks / "
            f"{len(current.get('splits', {}))} splits; upstream now has "
            f"{index['task_count']} / {len(index['splits'])}. Re-run without "
            f"--check to refresh, then re-pin tests/test_monthly_splits.py.",
            file=sys.stderr,
        )
        return 1

    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
        print(
            f"\nwrote {args.output}: {index['task_count']} task(s) across "
            f"{len(index['splits'])} split(s)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
