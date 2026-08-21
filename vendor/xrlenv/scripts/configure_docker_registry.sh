#!/usr/bin/env bash
# configure_docker_registry.sh — CLIENT-side config. Run this ON A WORKER (or a
# build host) to point its Docker daemon at the xrlenv registries, by merging
# `registry-mirrors` + `insecure-registries` into /etc/docker/daemon.json without
# clobbering existing keys (data-root etc.). Same merge pattern as
# set_docker_data_root.sh.
#
# It configures up to two things, each optional (set at least one):
#
#   MIRROR_URL=http://<host>:5010        the PROXY mirror (run-registry-proxy.sh).
#     Added to `registry-mirrors` (+ its host:port to `insecure-registries` for
#     plain HTTP). Applies ONLY to docker.io pulls; no image refs change; if the
#     mirror is down, dockerd falls back to Docker Hub automatically.
#
#   PRIVATE_REGISTRY=<host>:5011         the PRIVATE writable registry
#     (run-registry-private.sh). Added to `insecure-registries` only (NOT a mirror —
#     named refs like `<host>:5011/xrlenv-seta-env/0:main` are addressed directly).
#     Lets this host `docker push` to / `docker pull` from the private registry
#     over plain HTTP. Set this on build hosts (to push) and on workers (to pull).
#     A named-ref miss has NO Docker-Hub fallback, unlike a mirror miss.
#
# It does NOT run a registry. The registry SERVERS are separate things
# (deploy/registry/run-registry-{proxy,local}.sh) that run on the control-plane
# box or a dedicated registry node — never on a worker.
#
# Idempotent + self-healing: re-running is a no-op once the config is live in the
# daemon, but it (re)loads a running daemon that has the config on disk yet never
# re-read it — so re-running fixes a node left in the "configured but not active"
# state.
#
# Usage (on each worker / build host, via sudo):
#   sudo MIRROR_URL=http://internal-ip:5010 bash scripts/configure_docker_registry.sh [--restart]
#   sudo PRIVATE_REGISTRY=internal-ip:5011 bash scripts/configure_docker_registry.sh [--restart]
#   sudo MIRROR_URL=http://internal-ip:5010 PRIVATE_REGISTRY=internal-ip:5011 \
#        bash scripts/configure_docker_registry.sh [--restart]
#
# Applying the merged config is automatic and safe in both bootstrap directions:
#   * Docker not installed / not running yet (fresh node): the config is just
#     written; dockerd reads it on its FIRST start. Nothing is (re)started.
#   * Docker already running (re-bootstrap, or a node with Docker up): the merge
#     is applied LIVE via `systemctl reload docker` — registry-mirrors and
#     insecure-registries are SIGHUP-reloadable — so no containers bounce and the
#     node agent keeps its daemon connection.
#   * --restart: force a full `systemctl restart docker` instead of the live
#     reload (also applies non-reloadable keys, e.g. a same-pass data-root
#     change), then restart xrlenv-node, which the restart disconnects.
set -euo pipefail

MIRROR_URL="${MIRROR_URL:-}"
PRIVATE_REGISTRY="${PRIVATE_REGISTRY:-}"
DAEMON_JSON="${DAEMON_JSON:-/etc/docker/daemon.json}"
DO_RESTART=0
for arg in "$@"; do
    case "$arg" in
        --restart) DO_RESTART=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [ -z "${MIRROR_URL}" ] && [ -z "${PRIVATE_REGISTRY}" ]; then
    echo "ERROR: set at least one of:" >&2
    echo "  MIRROR_URL=http://internal-ip:5010        (proxy mirror; docker.io pulls)" >&2
    echo "  PRIVATE_REGISTRY=internal-ip:5011         (private writable registry)" >&2
    exit 2
fi
# host:port for insecure-registries (strip scheme). Empty when no mirror.
MIRROR_HOSTPORT=""
if [ -n "${MIRROR_URL}" ]; then
    MIRROR_HOSTPORT="${MIRROR_URL#http://}"
    MIRROR_HOSTPORT="${MIRROR_HOSTPORT#https://}"
    MIRROR_HOSTPORT="${MIRROR_HOSTPORT%%/*}"
fi

echo "==> CLIENT config (this host -> registries; no registry runs here)"
[ -n "${MIRROR_URL}" ]        && echo "==> mirror   : ${MIRROR_URL} (insecure ${MIRROR_HOSTPORT})"
[ -n "${PRIVATE_REGISTRY}" ]  && echo "==> private  : ${PRIVATE_REGISTRY} (insecure; push/pull named refs)"

python3 - "$DAEMON_JSON" "$MIRROR_URL" "$MIRROR_HOSTPORT" "$PRIVATE_REGISTRY" <<'PY'
import json, os, sys
from urllib.parse import urlparse

path, mirror_url, hostport, private_registry = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
)
cfg = {}
if os.path.isfile(path):
    with open(path) as f:
        cfg = json.load(f)

def _clean_registry_mirrors(values):
    if not isinstance(values, list):
        raise SystemExit("daemon.json 'registry-mirrors' is not a list; refusing to clobber")
    clean = []
    for value in values:
        if not isinstance(value, str):
            continue
        parsed = urlparse(value.strip())
        if parsed.scheme in ("http", "https") and parsed.netloc:
            clean.append(value.strip())
    return clean

def _clean_insecure_registries(values):
    if not isinstance(values, list):
        raise SystemExit("daemon.json 'insecure-registries' is not a list; refusing to clobber")
    return [
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip() not in ("", ":")
    ]

cleaned_mirrors = _clean_registry_mirrors(cfg.get("registry-mirrors") or [])
cfg["insecure-registries"] = _clean_insecure_registries(cfg.get("insecure-registries") or [])

# registry-mirrors: xrlenv routes docker.io through exactly ONE canonical
# pull-through proxy per node, so when a mirror is provided we REPLACE the list
# rather than append. Appending stranded a dead mirror (a decommissioned CP /
# registry host) AHEAD of the live one across a host migration — docker then
# tried the dead host first on every docker.io pull and timed out before falling
# through. Reconcile to exactly the desired mirror. With no MIRROR_URL, keep the
# cleaned existing list untouched (don't clobber a node's config to empty).
if mirror_url:
    cfg["registry-mirrors"] = [mirror_url]
else:
    cfg["registry-mirrors"] = cleaned_mirrors

def _add(key, val):
    if not val:
        return
    lst = cfg.get(key) or []
    if val not in lst:
        lst.append(val)
    cfg[key] = lst

# The mirror's host:port is also an insecure entry (plain HTTP). The private
# registry is insecure-only — addressed by named ref, not via registry-mirrors,
# so it must NOT go in registry-mirrors. insecure-registries stays additive: a
# stale host here is inert (it only whitelists HTTP; it doesn't route traffic).
_add("insecure-registries", hostport)
_add("insecure-registries", private_registry)
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
echo "==> ${DAEMON_JSON}:"; cat "$DAEMON_JSON"

# Validate JSON parses (defensive — a malformed daemon.json wedges docker start).
python3 -c "import json; json.load(open('$DAEMON_JSON'))" \
    && echo "==> daemon.json is valid JSON" \
    || { echo "ERROR: daemon.json is not valid JSON; NOT restarting" >&2; exit 1; }

# Full restart + node-agent restart. Used for --restart, and as the fallback
# when a running daemon's unit has no ExecReload (SIGHUP) support.
restart_docker_and_node() {
    if command -v docker >/dev/null 2>&1; then
        echo "==> running containers this restart will bounce:"
        docker ps --format '    {{.Names}} {{.Image}}' 2>/dev/null || true
    fi
    echo "==> restarting docker..."
    systemctl restart docker
    sleep 2
    if systemctl is-enabled --quiet xrlenv-node 2>/dev/null; then
        echo "==> restarting xrlenv-node (docker restart dropped its connections)..."
        systemctl restart xrlenv-node || true
    fi
}

# Is the desired config already LIVE in the running daemon? We drive the apply
# decision off the daemon's *actual* state (docker info), not whether the file
# changed — the bug being fixed is exactly "config is on disk but the daemon never
# re-read it." Keying off live state also makes the script self-healing:
# re-running on a node stuck in that state reloads it. Both checks are vacuously
# true when their value is unset (grep of an empty pattern matches), so a
# mirror-only or private-only invocation only gates on the thing it configured.
mirror_live() {
    [ -n "${MIRROR_URL}" ] || return 0
    # EXACT match: the live mirror list must be exactly [MIRROR_URL]
    # (trailing-slash-insensitive — docker normalises). A grep-for-presence would
    # treat a stale `[dead, desired]` live list as reconciled, leaving the dead
    # mirror tried FIRST on every pull (a per-pull timeout) — the very failure
    # this reconciliation removes. So compare the whole list, not just presence.
    docker info --format '{{json .RegistryConfig.Mirrors}}' 2>/dev/null \
        | python3 -c 'import sys, json
want = sys.argv[1].rstrip("/")
try:
    live = [str(m).rstrip("/") for m in (json.load(sys.stdin) or [])]
except Exception:
    sys.exit(1)
sys.exit(0 if live == [want] else 1)' "${MIRROR_URL}"
}
# Insecure registries (the private registry) are SIGHUP-reloadable like mirrors.
# `docker info` lists them under "Insecure Registries:"; grep the plain output.
private_live() {
    [ -n "${PRIVATE_REGISTRY}" ] || return 0
    docker info 2>/dev/null | grep -q -- "${PRIVATE_REGISTRY}"
}
config_live() { mirror_live && private_live; }

# Fresh node: Docker isn't running yet, so don't try to (re)start it — the merged
# daemon.json is read on dockerd's first start. This is the normal
# bootstrap-before-install ordering, and keeps this script safe to call early.
if ! systemctl is-active --quiet docker 2>/dev/null; then
    echo "==> docker not running yet; merged config will be read on first dockerd start."
    exit 0
fi

# Already effective — nothing to do (idempotent).
if config_live; then
    echo "==> registry config already live in the running daemon; nothing to do."
    exit 0
fi

# Docker is running but the mirror isn't active yet (just-merged, or a prior merge
# that was never reloaded). Apply it now.
if [ "$DO_RESTART" -eq 1 ]; then
    echo "==> --restart: forcing a full docker restart..."
    restart_docker_and_node
elif systemctl reload docker 2>/dev/null; then
    # registry-mirrors / insecure-registries apply on SIGHUP: live, no container
    # bounce, and the running xrlenv-node keeps its daemon connection.
    echo "==> docker reloaded live (no container bounce)."
    sleep 1
else
    echo "==> reload not supported by this docker.service; falling back to restart..."
    restart_docker_and_node
fi

# Confirm the config is EXACTLY live in the daemon now (not just present on disk).
# Fail closed: a still-stale live mirror (e.g. `[dead, desired]` that a reload
# didn't reset) means the dead mirror is still tried first on every pull — the
# exact failure this reconciliation removes. Set XRLENV_REGISTRY_FORCE=1 to
# downgrade to a warning if you must proceed anyway.
if config_live; then
    echo "==> OK: registry config is exactly live in the running daemon."
elif [ "${XRLENV_REGISTRY_FORCE:-}" = 1 ]; then
    echo "WARN: registry config still not exactly live after apply; proceeding" >&2
    echo "      (XRLENV_REGISTRY_FORCE=1). Try 'sudo systemctl restart docker'." >&2
else
    echo "ERROR: registry config is still not exactly live after apply — a stale/" >&2
    echo "       dead mirror may remain active in the daemon. Run 'sudo systemctl" >&2
    echo "       restart docker' and re-run (or XRLENV_REGISTRY_FORCE=1 to override)." >&2
    exit 1
fi
