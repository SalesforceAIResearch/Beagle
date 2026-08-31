#!/usr/bin/env bash
set -euo pipefail

# ── xrlenv-AUTHORED solution (frontier-swe / notebook-compression) ────────────
# NOT the upstream oracle. FrontierSWE withholds the reference solution for this
# task (it ships NO solution/solve.sh), so this is an xrlenv-authored best-effort
# solution used to prove the task is solvable end-to-end (plumbing + a reachable
# positive-reward ceiling). It installs a lossless per-file compressor (Python
# stdlib lzma) as the task's required /app/run submission. See patches/README.md.
echo "=== xrlenv-authored solution: notebook-compression (lossless lzma) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install -m 0755 "$SCRIPT_DIR/run.py" /app/run
echo "Installed /app/run (lzma preset 9|EXTREME, byte-for-byte lossless)"
