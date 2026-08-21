#!/usr/bin/env bash
# enable_cpu_isolation.sh — make THIS node P6-CPU-isolation-capable (opt-in).
#
# P6 (notes/cluster-resource-isolation-plan.md §8) confines unpinned runc
# containers to the complement of the pinned cores via a shared-parent cpuset
# cgroup (``xrlenv-shared``), so an unpinned workload can't trample a pinned
# container's cores (the duckdb-under-load regression). A node advertises
# ``isolation_capable=true`` only after a real self-test — gated (P6 v1) on the
# ``cgroupfs`` docker cgroup driver. AL2023 / Ubuntu default to ``systemd``, so a
# stock node is NON-capable and behaves exactly as today.
#
# WHY THIS SCRIPT (and not the agent) SETS THINGS UP — §8.13. The node agent
# runs as a NON-ROOT user (spec 19: ``User=xrlenv``); it cannot ``mkdir`` / write
# under ``/sys/fs/cgroup`` (a DAC check ``CAP_SYS_ADMIN`` does NOT override). So
# the privileged one-time setup lives here (root), and we hand the agent exactly
# what it needs via cgroup DELEGATION:
#   1. flip docker to the ``cgroupfs`` cgroup driver (idempotent daemon.json merge);
#   2. build a tiny long-lived probe image (``xrlenv-selftest:1``) the probe runs;
#   3. run the REAL container probe (as root) — proves docker+cgroup honor
#      ``cgroup_parent`` cpuset propagation on this node. If it fails, we stop:
#      the node stays non-capable (no false ``true``);
#   4. create ``/sys/fs/cgroup/xrlenv-shared`` and ``chown`` its ``cpuset.cpus`` /
#      ``cgroup.procs`` / ``cgroup.subtree_control`` / ``cgroup.threads`` to the
#      agent user — DELEGATION, so the non-root agent can write the complement +
#      place containers under it, and NOTHING else;
#   5. restart docker (a cgroup-driver change is not SIGHUP-reloadable) + the node
#      agent. This BOUNCES running containers — run it in a maintenance window /
#      on an idle node.
#
# Usage (as root, on the worker node):
#   sudo bash scripts/enable_cpu_isolation.sh
#
# Env knobs:
#   XRLENV_SELFTEST_IMAGE  probe image tag to build/use (default xrlenv-selftest:1)
#   XRLENV_VENV_PY         python that can import xrlenv + docker for the probe
#                          (default <repo>/.venv/bin/python, derived from this
#                          script's path)
#   DAEMON_JSON            path to daemon.json (default /etc/docker/daemon.json)
#   CPU_ISOLATION_ENV      env drop-in path (default /etc/xrlenv/cpu_isolation.env)
#   SKIP_RESTART=1         configure only (merge daemon.json + build image + write
#                          env); do NOT restart docker, run the probe, delegate,
#                          or restart the agent. The full path runs on the next
#                          non-SKIP invocation.
#   XRLENV_FORCE_PROBE=1   re-run the container probe even on an already-capable
#                          node. Default: SKIP the probe when the node is already
#                          cgroupfs + has the probe image + xrlenv-shared exists
#                          (propagation can't change without a reboot) — this is
#                          what makes a redeploy of a known node near-instant.
#
# REBOOT PERSISTENCE (§8.13): cgroups don't survive a node reboot, so
# ``xrlenv-shared`` + its delegation would be lost. This script installs
# scripts/setup_shared_cpuset.sh locally + enables a systemd oneshot
# (deploy/systemd/xrlenv-cpu-isolation.service) that re-runs the delegation at
# boot (After=docker, Before=xrlenv-node), so a rebooted node comes back capable
# without a redeploy.
#
# To REVERT: ``systemctl disable --now xrlenv-cpu-isolation.service`` + remove
# /etc/systemd/system/xrlenv-cpu-isolation.service and /etc/xrlenv/setup_shared_cpuset.sh;
# remove the "exec-opts" entry from daemon.json, delete ${CPU_ISOLATION_ENV},
# ``rmdir /sys/fs/cgroup/xrlenv-shared`` (once idle), and restart docker — the
# node returns to non-capable / today's behavior.
set -euo pipefail

SELFTEST_IMAGE="${XRLENV_SELFTEST_IMAGE:-xrlenv-selftest:1}"
DAEMON_JSON="${DAEMON_JSON:-/etc/docker/daemon.json}"
CPU_ISOLATION_ENV="${CPU_ISOLATION_ENV:-/etc/xrlenv/cpu_isolation.env}"
SKIP_RESTART="${SKIP_RESTART:-0}"
CGROUP_ROOT="/sys/fs/cgroup"
SHARED="${CGROUP_ROOT}/xrlenv-shared"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PY="${XRLENV_VENV_PY:-${REPO_ROOT}/.venv/bin/python}"

echo "==> CPU isolation: making this node capable (probe image=${SELFTEST_IMAGE})"

# ── 1. merge native.cgroupdriver=cgroupfs into daemon.json (preserve keys) ────
python3 - "$DAEMON_JSON" <<'PY'
import json, os, sys
path = sys.argv[1]
cfg = {}
if os.path.isfile(path):
    with open(path) as f:
        cfg = json.load(f)
opts = cfg.get("exec-opts") or []
if not isinstance(opts, list):
    raise SystemExit("daemon.json 'exec-opts' is not a list; refusing to clobber")
# Drop any existing native.cgroupdriver=... then set cgroupfs (idempotent).
opts = [o for o in opts if isinstance(o, str) and not o.startswith("native.cgroupdriver=")]
opts.append("native.cgroupdriver=cgroupfs")
cfg["exec-opts"] = opts
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
echo "==> ${DAEMON_JSON}:"; cat "$DAEMON_JSON"
python3 -c "import json; json.load(open('$DAEMON_JSON'))" \
    && echo "==> daemon.json is valid JSON" \
    || { echo "ERROR: daemon.json is not valid JSON; NOT restarting" >&2; exit 1; }

# ── 2. build the long-lived probe image (idempotent) ──────────────────────────
if docker image inspect "${SELFTEST_IMAGE}" >/dev/null 2>&1; then
    echo "==> probe image ${SELFTEST_IMAGE} already present"
else
    echo "==> building probe image ${SELFTEST_IMAGE} (FROM busybox, long-lived default)"
    # A pause-style image: the probe runs it with NO command override, so its
    # default CMD must keep PID 1 alive. 2147483647s ≈ 68y — effectively forever.
    printf 'FROM busybox\nCMD ["sleep", "2147483647"]\n' \
        | docker build -q -t "${SELFTEST_IMAGE}" - >/dev/null
    echo "==> built ${SELFTEST_IMAGE}"
fi

# ── 3. persist XRLENV_SELFTEST_IMAGE in a bootstrap-durable env drop-in ────────
# The enable-time probe (step 5) reads it; kept as a durable marker of the opt-in
# (the agent detects capability from the delegated cgroup, not this file).
mkdir -p "$(dirname "${CPU_ISOLATION_ENV}")"
cat >"${CPU_ISOLATION_ENV}" <<EOF
# CPU isolation opt-in (scripts/enable_cpu_isolation.sh). Loaded by the
# xrlenv-node systemd unit as an OPTIONAL EnvironmentFile so it survives the
# node.env rewrite on every bootstrap/refresh. Delete this file + restart docker
# to make the node non-capable again.
XRLENV_SELFTEST_IMAGE=${SELFTEST_IMAGE}
EOF
echo "==> wrote ${CPU_ISOLATION_ENV} (XRLENV_SELFTEST_IMAGE=${SELFTEST_IMAGE})"

# ── SKIP_RESTART: stage config only (no restart / probe / delegation) ─────────
if [ "${SKIP_RESTART}" = "1" ]; then
    LIVE_DRIVER="$(docker info --format '{{.CgroupDriver}}' 2>/dev/null || echo unknown)"
    echo "==> SKIP_RESTART=1: staged daemon.json + probe image + env only."
    echo "    NOT restarting docker, running the probe, delegating xrlenv-shared,"
    echo "    or restarting the agent (current live driver: ${LIVE_DRIVER})."
    echo "    Re-run WITHOUT SKIP_RESTART to complete enablement."
    exit 0
fi

# ── 4. make the cgroupfs driver live (probe + delegation need it) ─────────────
LIVE_DRIVER="$(docker info --format '{{.CgroupDriver}}' 2>/dev/null || echo unknown)"
if [ "${LIVE_DRIVER}" = "cgroupfs" ]; then
    echo "==> docker cgroup driver already cgroupfs live — no docker bounce"
else
    echo "==> running containers this docker restart will bounce:"
    docker ps --format '    {{.Names}} {{.Image}}' 2>/dev/null || true
    echo "==> restarting docker (cgroup-driver change → cgroupfs) ..."
    systemctl restart docker
    sleep 3
fi
NOW_DRIVER="$(docker info --format '{{.CgroupDriver}}' 2>/dev/null || echo unknown)"
echo "==> docker cgroup driver now: ${NOW_DRIVER}"
if [ "${NOW_DRIVER}" != "cgroupfs" ]; then
    echo "WARN: driver is not cgroupfs after restart — the node will stay" >&2
    echo "      non-capable. Check ${DAEMON_JSON} + 'journalctl -u docker'." >&2
    exit 1
fi

# ── 5. REAL container probe (root) — the capability gate (§8.10) ──────────────
# The probe proves docker+cgroup honor cgroup_parent cpuset propagation on this
# node, using the authoritative Python self-test (one source of truth).
#
# FAST-PATH (like the sysbox pool's "already registered — skip"): on an ALREADY-
# capable node — the driver is cgroupfs (confirmed above) + the probe image is
# present + the delegated xrlenv-shared is already there — the propagation the
# probe checks cannot have changed without a reboot (which loses xrlenv-shared and
# forces a fresh probe on the next enable), so SKIP the probe and its slow venv
# import. This is what makes a redeploy of a known node near-instant. Set
# XRLENV_FORCE_PROBE=1 to re-prove regardless (e.g. after a kernel change on a
# node that stayed up).
RUN_PROBE=1
if [ "${XRLENV_FORCE_PROBE:-0}" != "1" ] \
        && docker image inspect "${SELFTEST_IMAGE}" >/dev/null 2>&1 \
        && [ -d "${SHARED}" ]; then
    echo "==> already capable (cgroupfs + probe image + ${SHARED} present) —"
    echo "    skipping the container probe (set XRLENV_FORCE_PROBE=1 to re-prove)."
    RUN_PROBE=0
fi

# When the probe DOES run: the non-root agent can't run it, which is exactly why
# it runs here (root). If the venv python is missing we can't prove it → refuse
# to delegate (fail loud, not a silent false-capable). NB: the ``if``/``fi`` and
# heredoc below stay at column 0 — an indented ``<<'PY'`` body would corrupt the
# Python. The RUN_PROBE guard gates execution without nesting.
if [ "${RUN_PROBE}" = "1" ]; then
if [ ! -x "${VENV_PY}" ]; then
    echo "ERROR: probe interpreter not found/executable: ${VENV_PY}" >&2
    echo "       set XRLENV_VENV_PY to a python that can import xrlenv + docker." >&2
    exit 1
fi
echo "==> running the CPU-isolation container probe as root (${VENV_PY}) ..."
if "${VENV_PY}" - "${SELFTEST_IMAGE}" <<'PY'
import sys
import docker
from xrlenv.node.raw_container import _run_cgroup_isolation_selftest
ok = _run_cgroup_isolation_selftest(docker.from_env(), image=sys.argv[1])
print(f"    probe verdict: {ok}")
sys.exit(0 if ok else 1)
PY
then
    echo "==> probe PASSED — this node honors cgroup_parent cpuset propagation"
else
    echo "WARN: CPU-isolation container probe FAILED — NOT delegating xrlenv-shared; the" >&2
    echo "      node stays non-capable (falls back to per-container pinning)." >&2
    echo "      Check 'journalctl -u docker' + that ${SELFTEST_IMAGE} is present." >&2
    # Bounce the agent so it re-advertises the (non-)capability cleanly, then
    # exit 0: a non-capable node is a valid state, not a script failure.
    systemctl is-enabled --quiet xrlenv-node 2>/dev/null && systemctl restart xrlenv-node || true
    exit 0
fi
fi

# ── 6. create + DELEGATE xrlenv-shared to the (non-root) agent user (§8.13) ───
# Install the setup script LOCALLY (so the boot unit below doesn't depend on the
# shared /shared-fs mount being up), then run it to (re)create + delegate the cgroup.
LOCAL_SETUP="/etc/xrlenv/setup_shared_cpuset.sh"
install -m 0755 "${SCRIPT_DIR}/setup_shared_cpuset.sh" "${LOCAL_SETUP}"
echo "==> installed ${LOCAL_SETUP}; delegating ${SHARED} ..."
bash "${LOCAL_SETUP}"

# ── 6b. reboot persistence — a boot-time oneshot re-runs the delegation ───────
# cgroups don't survive a reboot, so without this the node reverts to non-capable
# after every reboot until a redeploy. Install + enable a systemd oneshot that
# re-runs the setup After=docker Before=xrlenv-node on each boot (§8.13).
UNIT_SRC="${REPO_ROOT}/deploy/systemd/xrlenv-cpu-isolation.service"
UNIT_DST="/etc/systemd/system/xrlenv-cpu-isolation.service"
if [ -f "${UNIT_SRC}" ]; then
    install -m 0644 "${UNIT_SRC}" "${UNIT_DST}"
    systemctl daemon-reload
    systemctl enable xrlenv-cpu-isolation.service >/dev/null 2>&1 \
        && echo "==> enabled xrlenv-cpu-isolation.service (re-establishes xrlenv-shared on reboot)" \
        || echo "WARN: could not enable xrlenv-cpu-isolation.service — reboot won't re-establish CPU isolation" >&2
else
    echo "WARN: ${UNIT_SRC} missing — reboot persistence NOT installed" >&2
fi

# ── 7. restart the agent so it re-runs its (delegation) capability check ──────
if systemctl is-enabled --quiet xrlenv-node 2>/dev/null; then
    echo "==> restarting xrlenv-node (picks up the delegated xrlenv-shared) ..."
    systemctl restart xrlenv-node || true
fi

# ── verify ────────────────────────────────────────────────────────────────────
docker image inspect "${SELFTEST_IMAGE}" >/dev/null 2>&1 \
    && echo "==> probe image present: ${SELFTEST_IMAGE}" \
    || echo "WARN: probe image ${SELFTEST_IMAGE} missing"
echo "==> done. The node agent re-runs its delegation check at NodeHello; check"
echo "    'xrlenv nodes' for the CPU_ISOLATION column (expect 'yes …') once it reconnects."
