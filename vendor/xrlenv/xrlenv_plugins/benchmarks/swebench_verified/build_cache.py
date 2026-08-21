"""build_cache.py — materialize SWE-bench Verified into the shared cache.

Unlike the harbor/pier benchmarks, swebench-verified has no ``task.toml`` +
``environment/`` task dirs: the corpus is upstream's HF dataset
(``SWE-bench/SWE-bench_Verified``) and the per-instance images are prebuilt on
Docker Hub (``swebench/sweb.eval.x86_64.<key>:latest``). But we still cache the
**task data** so the corpus is self-contained and offline:

    <cache_root>/swebench-verified/<instance_id>/
    ├── instance.json        # the full upstream row (the anchor file)
    ├── problem_statement.md  # the task PROMPT — what a real agent reads
    └── gold_patch.diff       # the GOLD patch — what run_oracle_sweep applies

The oracle sweep reads the gold patch from ``instance.json["patch"]`` (mirrored in
``gold_patch.diff``) as its prediction; a future real-agent run reads
``problem_statement.md``. The image is
NOT cached here (it's pulled on first acquire by the node's ImageCacheManager);
``build_plan_gen.py`` emits the registry image plan.

Contract (mirrors GUIDELINE_onboard_benchmarks.md §3.1, adapted to the HF-dataset
shape — there is no ``patch`` stage because swebench content is clean upstream):

* CLI: ``--stage {all,populate}`` (idempotent; ``all`` == ``populate``),
  ``--dest <cache root>`` (default ``$XRLENV_BENCHMARK_CACHE``), ``--all`` /
  ``--smoke`` / ``--instances <csv>`` to bound the set (default: the whole
  Verified split).
* Idempotent: a COMPLETE instance (all files present, a valid anchor whose id
  agrees with the dir name, and extracts matching the anchor — see ``_is_complete``)
  is skipped; an incomplete/corrupt one is re-materialized; re-runs are safe.
* ``_row_to_cache`` (row -> files-to-write) is a pure, unit-tested function; the
  HF ``populate`` fetch is network-dependent and not unit-tested.
* Fail loud on a bad upstream row (missing ``instance_id`` / ``patch`` /
  ``problem_statement``) — never write a half-populated instance dir.

Usage::

    export XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache
    python xrlenv_plugins/benchmarks/swebench_verified/build_cache.py --stage all --all
    python .../build_cache.py --stage all --smoke          # just the 8 smoke instances
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

# The cache sub-directory (shard). Doubles as the namespace every consumer
# enumerates: ``<cache>/swebench-verified/<instance_id>/``.
SHARD = "swebench-verified"

DATASET_NAME = os.environ.get("SWEBENCH_VERIFIED_DATASET", "SWE-bench/SWE-bench_Verified")
DATASET_SPLIT = os.environ.get("SWEBENCH_VERIFIED_SPLIT", "test")

# The anchor file — written LAST, so its presence marks a complete dir.
ANCHOR = "instance.json"
# Every file a complete instance dir must carry (keep in sync with _row_to_cache).
_EXPECTED_FILES: tuple[str, ...] = (ANCHOR, "problem_statement.md", "gold_patch.diff")

# 8-instance smoke set: 3 repos x 3 difficulty bands. Kept in sync with
# build_plan_gen.SMOKE_INSTANCES.
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

# Fields required on every upstream row; a row missing any is a corpus defect.
_REQUIRED = ("instance_id", "patch", "problem_statement")

# The EXACT crash-orphan grammar left by an interrupted _materialize:
# ``tempfile.mkdtemp(prefix=f".{iid}.tmp-", dir=shard)`` -> ``.<id>.tmp-<8 chars [a-z0-9_]>``.
# Matching this precisely (not a loose ``".tmp-" in name``) keeps a legitimately-named dir
# (``.legitimate.tmp-data``) from being reclaimed (audit Low).
_MKDTEMP_ORPHAN_RE = re.compile(r"\..+\.tmp-[a-z0-9_]{8}")


def _cache_root(dest: str | None) -> Path:
    # Delegate to the shared guard+resolver: the cache env/path were renamed (audit: retired
    # XRLENV_HARBOR_CACHE / .../xrlenv_harbor_cache -> unreliable results); benchmark_cache_root
    # HARD-REJECTS the legacy var/path before any cache read. Lazy import matches plugin style.
    from xrlenv_plugins.benchmarks._benchmark_cache import benchmark_cache_root
    return Path(benchmark_cache_root(dest)).expanduser()


def _row_to_cache(row: dict[str, Any]) -> dict[str, str]:
    """Pure: an upstream Verified row -> {relative_filename: text} to write under
    the instance dir. Fail loud on a row missing a required field.

    ``instance.json`` keeps the WHOLE row (faithful, self-contained);
    ``problem_statement.md`` / ``gold_patch.diff`` are convenience extracts the
    prompt- and oracle-readers use directly.
    """
    missing = [k for k in _REQUIRED if not row.get(k)]
    if missing:
        raise SystemExit(
            f"swebench row {row.get('instance_id', '<no id>')!r} missing "
            f"required field(s): {missing}",
        )
    return {
        ANCHOR: json.dumps(row, indent=2, sort_keys=True, default=str),
        "problem_statement.md": str(row["problem_statement"]),
        "gold_patch.diff": str(row["patch"]),
    }


def _safe_instance_id(iid: str) -> str:
    """Reject a dataset row whose ``instance_id`` is not a BARE path component. The builder
    interpolates it into the destination / lock / temp / stale / rmtree paths, so a
    ``../victim`` id would escape the shard and destroy a sibling with the builder's fs
    authority (audit H6) — reachable via the SWEBENCH_VERIFIED_DATASET/_SPLIT env vars."""
    if not iid or iid in (".", "..") or iid != Path(iid).name:
        raise SystemExit(f"unsafe instance id {iid!r} in dataset row: must be a bare name")
    return iid


def _is_complete(inst_dir: Path) -> bool:
    """True iff ``inst_dir`` is a COMPLETE, SEMANTICALLY-VALID instance dir (audit M7):
    every expected file present; a parseable anchor carrying all required fields whose
    ``instance_id`` AGREES with the dir name; AND the convenience extracts
    (``problem_statement.md`` / ``gold_patch.diff``) match the anchor's values. A bare
    ``{}`` anchor, a missing field, an id mismatch, or a corrupt extract is NOT complete —
    so it re-materializes rather than being skipped forever."""
    # A SYMLINKED instance dir is not trusted — following it would read/replace out of the
    # shard (audit Low; defense-in-depth for the operator-owned cache). is_symlink() uses
    # lstat, so it doesn't follow.
    if inst_dir.is_symlink() or not inst_dir.is_dir():
        return False
    # Each required child must be a real regular file, NOT a symlink: a planted child symlink
    # would be followed on read/consume out of the shard (audit Low). is_symlink() (lstat) is
    # checked BEFORE is_file() (which follows) so an out-of-shard target can't satisfy it.
    for name in _EXPECTED_FILES:
        f = inst_dir / name
        if f.is_symlink() or not f.is_file():
            return False
    try:
        anchor_row = json.loads((inst_dir / ANCHOR).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(anchor_row, dict):
        return False
    if any(not anchor_row.get(k) for k in _REQUIRED):
        return False
    if str(anchor_row.get("instance_id")) != inst_dir.name:
        return False
    # Extracts must match their anchor (a corrupt extract is not "complete" — this matters
    # when a real agent consumes problem_statement.md as the prompt).
    try:
        expected = _row_to_cache(anchor_row)
    except SystemExit:
        return False
    for name, want in expected.items():
        if name == ANCHOR:
            continue
        try:
            # Compare BYTES, not ``read_text()``: text mode applies universal-newline
            # translation (``\r\n`` -> ``\n``), so an instance whose ``problem_statement`` /
            # ``patch`` carries CRLF — common in GitHub issue text; 252 of the 500 Verified
            # rows — would falsely fail this check even though the stored bytes are the
            # verbatim dataset value. ``_materialize`` writes ``want`` with newline translation
            # DISABLED, so the on-disk bytes are exactly ``want.encode()``; verify that
            # byte-for-byte (audit: cache-completeness CRLF false-negative).
            if (inst_dir / name).read_bytes() != want.encode("utf-8"):
                return False
        except OSError:
            return False
    return True


def _materialize(row: dict[str, Any], shard: Path) -> bool:
    """Write one instance's cache dir ATOMICALLY + self-healingly. Returns False if the dir
    is already COMPLETE (skipped), True if freshly written (audit M7).

    Guarantees:
    * completeness-checked skip — a bare/corrupt anchor or a missing sibling re-materializes
      (not skipped on the anchor's mere existence);
    * unique temp sibling (``mkdtemp``) — two concurrent writers for one instance don't race
      on a PID-only name;
    * anchor written LAST inside the temp, then an atomic ``os.replace`` — an interrupted
      write never exposes a half-populated final dir carrying the anchor;
    * an incomplete leftover destination is removed, then a single atomic ``os.replace``
      installs the new dir (no move-aside, so no ``FileNotFoundError`` under a competing
      writer);
    * the temp siblings are dot-prefixed and cleaned up (and enumeration skips dot-dirs),
      so a crash can't leave a temp dir that later reads as a real instance;
    * a per-instance file lock (``flock``) serializes SAME-HOST writers; cross-host writers
      (``/shared-fs`` Lustre is ``localflock``, so flock doesn't coordinate across nodes) converge
      via an IDEMPOTENT atomic install — same dataset row -> identical content (audit M7);
    * only an INCOMPLETE destination is replaced (a complete one returns early), so a reader
      using ``_is_complete`` never sees a *complete* dir vanish. A crash mid-replace can
      leave the dir absent, which the next run simply re-materializes — no half-populated
      complete dir is ever exposed.

    The ``instance_id`` is validated as a bare component BEFORE it touches any path (audit
    H6): the alternate-dataset env vars make a ``../victim`` row attacker-reachable."""
    iid = _safe_instance_id(str(row["instance_id"]))
    inst_dir = shard / iid
    if _is_complete(inst_dir):
        return False
    files = _row_to_cache(row)  # validates before we create anything
    shard.mkdir(parents=True, exist_ok=True)
    lock_path = shard / f".{iid}.lock"     # dot-prefixed (enumeration skips it); persists
    # O_NOFOLLOW: a planted lock SYMLINK must not let the create/flock truncate an
    # out-of-shard file (audit Low; defense-in-depth for the operator-owned cache).
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if _is_complete(inst_dir):    # double-checked: a peer may have finished
            return False
        tmp = Path(tempfile.mkdtemp(prefix=f".{iid}.tmp-", dir=shard))
        try:
            # ``newline=""`` DISABLES newline translation on write so the on-disk bytes are
            # EXACTLY ``text.encode()`` — verbatim including any CRLF (audit: cache-completeness
            # CRLF). Symmetric with the byte-exact ``_is_complete`` check, and platform-stable
            # (a non-Linux builder's ``\n`` -> os.linesep translation can't corrupt a stored
            # patch / statement).
            for name, text in files.items():
                if name == ANCHOR:
                    continue               # anchor written LAST (below), never mid-dir
                (tmp / name).write_text(text, encoding="utf-8", newline="")
            (tmp / ANCHOR).write_text(files[ANCHOR], encoding="utf-8", newline="")
            # Replace only an incomplete leftover: remove it (a reader never trusted it —
            # it was incomplete), then a single atomic rename installs the new dir. No
            # move-aside, so no FileNotFoundError under a competing writer.
            # A SYMLINKED destination (rejected as incomplete by _is_complete) must be
            # unlinked, not rmtree'd: rmtree raises NotADirectoryError on a symlink and
            # os.replace would then fail onto it. The link is inside the shard (iid is a
            # bare component), so unlinking removes only the link, never its target (audit Low).
            if inst_dir.is_symlink():
                inst_dir.unlink()
            elif inst_dir.exists():
                shutil.rmtree(inst_dir, ignore_errors=True)
            try:
                os.replace(tmp, inst_dir)
            except OSError:
                # A cross-host peer installed it between our rmtree and rename. Its content
                # is identical (same row), so accept the peer's dir; only re-raise if what
                # landed is NOT actually complete.
                if not _is_complete(inst_dir):
                    raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)   # no-op once os.replace moved it out
    finally:
        os.close(lock_fd)
    return True


def list_complete(dest: str | None) -> list[str]:
    """The ids of COMPLETE, semantically-valid instance dirs under the shard (audit M13).

    Pure filesystem (no network): enumerates the shard and returns only the ids whose dir
    passes ``_is_complete`` — a valid anchor with all required fields, id agreeing with the
    dir name, and matching extracts. The sweep wrapper's green-set gate lists via THIS, so a
    dir carrying only a bare ``{}`` / corrupt ``instance.json`` (which would satisfy a mere
    "instance.json exists" check) is NOT counted as a prepared instance and cannot pass
    membership as the authoritative Verified corpus."""
    shard = _cache_root(dest) / SHARD
    if not shard.is_dir():
        return []
    return sorted(d.name for d in shard.iterdir()
                  if not d.name.startswith(".") and _is_complete(d))


def _load_rows() -> list[dict[str, Any]]:
    """Load the Verified split via swebench's own dataset loader (network)."""
    from swebench.harness.run_evaluation import load_swebench_dataset
    rows = load_swebench_dataset(DATASET_NAME, DATASET_SPLIT)
    if not rows:
        raise SystemExit(f"{DATASET_NAME}:{DATASET_SPLIT} loaded 0 rows — check the dataset id")
    return list(rows)


def _reclaim_orphan_temps(shard: Path) -> list[str]:
    """Remove crash-orphan temp siblings from a prior interrupted build; return their names.

    Safe in the single-builder model: no concurrent build owns them (each _materialize cleans
    only its OWN temp, so a killed process leaves a ``.<id>.tmp-<rand>`` dir behind). Matches
    ONLY the EXACT mkdtemp grammar ``.<id>.tmp-<8 chars from [a-z0-9_]>`` (audit Low): a dir
    like ``.legitimate.tmp-data`` (4-char suffix) or ``x.old`` is never reclaimed. Under-
    matching (e.g. a future mkdtemp suffix length) is safe — an orphan just lingers as a
    skipped dot-dir; over-matching would delete a real dir, so we bias to the known grammar."""
    removed: list[str] = []
    for p in shard.iterdir():
        if p.is_dir() and _MKDTEMP_ORPHAN_RE.fullmatch(p.name):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p.name)
    return removed


def populate(dest: str | None, instance_ids: list[str] | None) -> tuple[int, int]:
    """Materialize the requested instances (all if ``instance_ids`` is None).
    Returns ``(written, skipped)``."""
    shard = _cache_root(dest) / SHARD
    shard.mkdir(parents=True, exist_ok=True)
    _reclaim_orphan_temps(shard)
    rows = _load_rows()
    if instance_ids is not None:
        by_id = {r["instance_id"]: r for r in rows}
        unknown = [i for i in instance_ids if i not in by_id]
        if unknown:
            raise SystemExit(f"unknown instance_id(s): {unknown}")
        rows = [by_id[i] for i in instance_ids]
    written = skipped = 0
    for row in rows:
        if _materialize(row, shard):
            written += 1
        else:
            skipped += 1
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="swebench-verified-build-cache",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", choices=("all", "populate"), default="all",
                   help="all == populate (no patch stage — swebench content is clean).")
    p.add_argument("--dest", default=None,
                   help="cache root (default: $XRLENV_BENCHMARK_CACHE).")
    p.add_argument("--list-complete", action="store_true",
                   help="Print the ids of COMPLETE instance dirs (passing the same semantic "
                        "check as the builder) under the shard, one per line, and exit. The "
                        "sweep wrapper's green-set gate lists via this so a bare/corrupt "
                        "anchor can't pass as a prepared instance (audit M13). No network.")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="the whole Verified split (default).")
    sel.add_argument("--smoke", action="store_true", help="just the 8 smoke instances.")
    sel.add_argument("--instances", default=None, help="comma-separated instance ids.")
    args = p.parse_args(argv)

    if args.list_complete:
        for iid in list_complete(args.dest):
            print(iid)
        return 0

    if args.smoke:
        ids: list[str] | None = list(SMOKE_INSTANCES)
    elif args.instances:
        ids = [s.strip() for s in args.instances.split(",") if s.strip()]
    else:  # --all or default
        ids = None

    written, skipped = populate(args.dest, ids)
    shard = _cache_root(args.dest) / SHARD
    print(f"==> swebench-verified cache @ {shard}: {written} written, {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
