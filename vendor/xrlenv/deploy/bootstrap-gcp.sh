#!/usr/bin/env bash
# bootstrap-gcp.sh — thin wrapper around ``xrlenv bootstrap --target gcp``.
#
# The bash bootstrap logic was replaced with a Python subcommand at
# ``xrlenv/cli/bootstrap.py``. This
# wrapper preserves the historical operator interface:
#
#     sudo -E bash deploy/bootstrap-gcp.sh [<control-plane>] [<node-id>]
#
# All knobs documented under ``xrlenv bootstrap --help`` are also
# settable via env var (XRLENV_CONTROL_PLANE / XRLENV_NODE_ID /
# XRLENV_WHEEL / XRLENV_REPO / XRLENV_VERSION / XRLENV_NODE_TOKEN).
# The wrapper picks up positional args as a back-compat convenience.

set -euo pipefail

if (( $# >= 1 )); then export XRLENV_CONTROL_PLANE="$1"; shift; fi
if (( $# >= 1 )); then export XRLENV_NODE_ID="$1"; shift; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Hard CPU isolation is opt-in and OFF by default (a stock node advertises
# isolation_capable=false and behaves as today). To make a node capable, run
# `sudo bash deploy/node/enable_cpu_isolation.sh` on the worker (or set
# XRLENV_ENABLE_CPU_ISOLATION=1 in the bootstrap env) — see the "CPU isolation
# (opt-in)" note in deploy/bootstrap-aws.sh.

# Invoke the bootstrap module as a flat script (NOT
# ``python3 -m xrlenv.cli.bootstrap``) because importing
# ``xrlenv.__init__`` pulls in pydantic — which isn't installed yet
# on a fresh VM. The flat-script path uses only Python stdlib +
# subprocess, runs under any system Python 3.10+, and bootstraps the
# 3.12 venv as part of its sequence. --xrlenv-repo points the
# subsequent pip install at the same checkout we ran from.
exec python3 "${REPO_ROOT}/xrlenv/cli/bootstrap.py" \
    --target gcp \
    --xrlenv-repo "${REPO_ROOT}" \
    "$@"
