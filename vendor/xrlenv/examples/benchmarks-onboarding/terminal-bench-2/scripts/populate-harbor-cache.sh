#!/usr/bin/env bash
# populate-harbor-cache.sh — clone the upstream terminal-bench-2
# task catalog into the harbor cache, idempotently.
#
# The terminal-bench-2 task catalog is NOT vendored in the XRLEnv
# repo; it lives at https://github.com/harbor-framework/terminal-bench-2.
# The plug-in's resolver expects to find ``<task>/task.toml`` under
# the path identified by ``$XRLENV_BENCHMARK_CACHE`` (default
# ``$HOME/.cache/harbor/tasks``).
#
# This script:
#   1. Resolves the cache path (env override > $HOME default).
#   2. Skips silently if any ``*/task.toml`` is already present
#      (idempotent — safe to re-run on every VM bring-up).
#   3. Otherwise ``git clone --depth 1`` the upstream repo into
#      the cache path (flat layout: ``<cache>/<task>/task.toml``).
#
# Operator overrides:
#   XRLENV_BENCHMARK_CACHE          where the cache lives (default
#                                ``$HOME/.cache/harbor/tasks``; the
#                                bootstrap-managed path is
#                                ``/var/cache/xrlenv/harbor/tasks``)
#   XRLENV_TB2_UPSTREAM_REPO     git URL (default upstream)
#   XRLENV_TB2_UPSTREAM_REF      branch / tag / sha (default ``main``)
#
# After this script returns OK, the daemon's resolver and the
# build-task-images.sh script both find the same task data.

set -euo pipefail

CACHE="${XRLENV_BENCHMARK_CACHE:-$HOME/.cache/harbor/tasks}"

# audit M17: the old XRLENV_HARBOR_CACHE var + .../xrlenv_harbor_cache path are RETIRED. Fail
# loud so a stale env can't populate/read the wrong cache — migrate to XRLENV_BENCHMARK_CACHE.
if [ -n "${XRLENV_HARBOR_CACHE+x}" ]; then
    echo "ERROR: XRLENV_HARBOR_CACHE is retired — unset it and use XRLENV_BENCHMARK_CACHE." >&2
    exit 1
fi
case "$CACHE" in
    *xrlenv_harbor_cache*)
        echo "ERROR: retired cache path '$CACHE' — the .../xrlenv_harbor_cache path is retired." >&2
        exit 1 ;;
esac

UPSTREAM_REPO="${XRLENV_TB2_UPSTREAM_REPO:-https://github.com/harbor-framework/terminal-bench-2.git}"
UPSTREAM_REF="${XRLENV_TB2_UPSTREAM_REF:-main}"

# Idempotency check — accept either layout the resolver supports
# (flat ``<cache>/<task>/task.toml`` or content-addressable
# ``<cache>/<hash>/<task>/task.toml``). ``compgen -G`` returns 0
# when a glob matches at least one file.
if compgen -G "$CACHE/*/task.toml" > /dev/null \
   || compgen -G "$CACHE/*/*/task.toml" > /dev/null; then
    echo "OK: harbor cache already populated at $CACHE — nothing to do." >&2
    exit 0
fi

# Refuse to overwrite a non-empty cache directory that has files
# but no task.toml — that's an unexpected layout the operator
# should investigate first.
if [[ -d "$CACHE" ]] && [[ -n "$(ls -A "$CACHE" 2>/dev/null)" ]]; then
    echo "ERROR: $CACHE exists and is non-empty but contains no task.toml." >&2
    echo "       Either point XRLENV_BENCHMARK_CACHE at a different path, or" >&2
    echo "       remove the directory and re-run." >&2
    exit 1
fi

mkdir -p "$(dirname "$CACHE")"
echo ">> Cloning $UPSTREAM_REPO ($UPSTREAM_REF) → $CACHE" >&2

# ``git clone <repo> <dest>`` requires <dest> to either not exist
# or be empty. We just verified emptiness above.
if [[ -d "$CACHE" ]]; then
    rmdir "$CACHE"
fi
git clone --branch "$UPSTREAM_REF" --depth 1 "$UPSTREAM_REPO" "$CACHE"

echo "OK: terminal-bench-2 task catalog populated at $CACHE." >&2
echo "    Tasks present:" >&2
# Portable across BSD find (macOS) and GNU find (Linux): list each
# task.toml's parent dir, strip the cache prefix, indent for output.
find "$CACHE" -maxdepth 2 -name task.toml 2>/dev/null \
    | sed -E "s|/task.toml$||; s|^${CACHE}/||" \
    | sort \
    | head -20 \
    | sed 's/^/      /' >&2
