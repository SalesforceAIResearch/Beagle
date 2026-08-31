#!/usr/bin/env python3
"""notebook-compression submission — xrlenv-authored solution.

NOT the upstream oracle: FrontierSWE withholds the reference solution for this task
(it ships no solution/solve.sh). This is an xrlenv-authored *best-effort* solution,
used to prove the task is solvable end-to-end (plumbing works + a reachable positive
reward ceiling), clearly labelled as such.

Implements the task's `/app/run {fit,compress,decompress}` submission contract with a
lossless per-file compressor built only on the Python standard library (`lzma`, i.e.
xz preset 9 | EXTREME) — no network, no third-party packages, guaranteed available in
any Python image. Notebooks are canonicalized JSON, which lzma compresses to roughly
a quarter of their size, so score = (artifact_bytes + compressed_bytes)/original_bytes
lands well below 1.0 → reward = 1 - score > 0. Correctness is the hard gate (any
byte-for-byte mismatch is an instant fail), so this deliberately favours a simple,
provably-lossless codec over a fancier lower-ratio one. A dictionary-trained codec
(zstd --train / zlib zdict) would lower the ratio further — noted as a follow-up.
"""
from __future__ import annotations

import lzma
import os
import shutil
import sys

SUFFIX = ".xz"
_PRESET = 9 | lzma.PRESET_EXTREME


def _iter_regular_files(root: str):
    """Yield (abs_path, rel_path) for every regular file under ``root`` (recursively),
    skipping symlinks / sockets / pipes / device files — the contract says only
    regular files are scored and non-regular entries are ignored."""
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            ap = os.path.join(dirpath, name)
            if os.path.islink(ap) or not os.path.isfile(ap):
                continue
            yield ap, os.path.relpath(ap, root)


def _write(path: str, data: bytes) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def do_fit(visible_dir: str, artifact_dir: str) -> None:
    """Build whatever decompress needs. This baseline codec needs no trained
    dictionary (lzma is self-describing), so the artifact is just a tiny version
    marker — keeping artifact_bytes ~0 so it barely counts toward the score."""
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "VERSION"), "w") as fh:
        fh.write("xrlenv-lzma-1\n")


def do_compress(artifact_dir: str, input_dir: str, compressed_dir: str) -> None:
    """One compressed output per input file at the same relative path + ``.xz``."""
    for src, rel in _iter_regular_files(input_dir):
        with open(src, "rb") as fh:
            data = fh.read()
        _write(os.path.join(compressed_dir, rel + SUFFIX),
               lzma.compress(data, preset=_PRESET))


def do_decompress(artifact_dir: str, compressed_dir: str, recovered_dir: str) -> None:
    """Recover each file byte-for-byte to its original relative path (strip ``.xz``)."""
    for src, rel in _iter_regular_files(compressed_dir):
        with open(src, "rb") as fh:
            blob = fh.read()
        if rel.endswith(SUFFIX):
            _write(os.path.join(recovered_dir, rel[: -len(SUFFIX)]),
                   lzma.decompress(blob))
        else:
            # Not one of ours — pass through unchanged (defensive; shouldn't happen).
            dst = os.path.join(recovered_dir, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copyfile(src, dst)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: run {fit|compress|decompress} <args...>", file=sys.stderr)
        return 2
    stage, rest = argv[1], argv[2:]
    # Per-stage arity from the contract: `fit <visible> <artifact>` (2);
    # `compress <artifact> <input> <compressed>` and
    # `decompress <artifact> <compressed> <recovered>` (3 each).
    handlers = {
        "fit": (do_fit, 2),
        "compress": (do_compress, 3),
        "decompress": (do_decompress, 3),
    }
    entry = handlers.get(stage)
    if entry is None:
        print(f"unknown stage {stage!r}", file=sys.stderr)
        return 2
    fn, need = entry
    if len(rest) < need:
        print(f"stage {stage} needs {need} path args, got {len(rest)}", file=sys.stderr)
        return 2
    fn(*rest[:need])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
