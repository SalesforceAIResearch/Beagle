#!/usr/bin/env bash
# _deploy_gates.sh — the fail-closed pre/post-deploy gates, shared by
# deploy_example.sh and deploy_example.sh and unit-tested in tests/unit/deploy/.
#
# Every gate returns NONZERO on an unsatisfied safety condition (the callers run
# `set -e`, so a nonzero return aborts the deploy) UNLESS XRLENV_DEPLOY_FORCE=1,
# which downgrades to a warning + return 0. The gates are written to be robust
# regardless of the caller's `set -e` (all fallible commands are inside `if`/`||`
# conditions or capture their status via `if x=$(cmd)`), so a failed scheduler
# query is never silently read as success.
#
# Callers must have defined the array ``SSH_OPTS``. Loop bounds are trailing
# parameters (defaults match the historical inline values) so tests run fast.

# Wait for NODE_JOB to reach RUNNING. Fail closed on timeout: a control plane
# with no workers is a broken deploy.
deploy_wait_node_running() {
    local job="$1" tries="${2:-60}" sleep_s="${3:-5}" i running=0
    for ((i = 0; i < tries; i++)); do
        if squeue --noheader --name="${job}" -o '%T' 2>/dev/null | grep -q RUNNING; then
            running=1; break
        fi
        sleep "${sleep_s}"
    done
    if [ "${running}" != 1 ] && [ "${XRLENV_DEPLOY_FORCE:-}" != 1 ]; then
        echo "    ERROR: ${job} did not reach RUNNING within the wait window." >&2
        echo "    Aborting (investigate with squeue/sacct, then re-run; or set" >&2
        echo "    XRLENV_DEPLOY_FORCE=1 to proceed without workers)." >&2
        return 1
    fi
    return 0
}

# Wait for CONTROL_JOB to FULLY exit (leave the queue AND release the DB on
# CP_NODE) before a new control plane starts. Two `xrlenv up` on one state.db
# corrupt the SQLite -shm / race the journal. Fail closed on: still-queued,
# persistent squeue failure, a still-present process, OR an SSH error.
deploy_wait_control_gone() {
    local job="$1" cp_node="$2" tries="${3:-40}" sleep_s="${4:-3}"
    local i gone=0 q qrc rc
    for ((i = 0; i < tries; i++)); do
        # Capture squeue's OWN exit code without letting a failed query abort us
        # (an `if x=$(cmd)` assignment is exempt from set -e). A query error must
        # NOT be read as "job gone" (that would start an overlapping CP).
        if q=$(squeue --noheader --name="${job}" -o '%T' 2>/dev/null); then qrc=0; else qrc=$?; fi
        if [ "${qrc}" != 0 ]; then sleep "${sleep_s}"; continue; fi
        printf '%s\n' "${q}" | grep -qE 'RUNNING|COMPLETING|PENDING|CONFIGURING' \
            || { gone=1; break; }
        sleep "${sleep_s}"
    done
    if [ "${gone}" != 1 ] && [ "${XRLENV_DEPLOY_FORCE:-}" != 1 ]; then
        echo "    ERROR: ${job} not confirmed gone within the wait window" >&2
        echo "    (still queued, or squeue kept failing). Refusing to start a second" >&2
        echo "    control plane (overlapping state.db opens corrupt it). Investigate," >&2
        echo "    then re-run, or set XRLENV_DEPLOY_FORCE=1 if you know it's gone." >&2
        return 1
    fi
    # ssh rc: 0=released, 3=still present, other=ssh failure.
    if ssh "${SSH_OPTS[@]}" "${cp_node}" \
            'for _ in $(seq 1 20); do pgrep -f "[x]rlenv up" >/dev/null 2>&1 || exit 0; sleep 1; done; exit 3' \
            2>/dev/null; then
        return 0
    else
        rc=$?  # MUST capture inside the else — after `fi`, $? is the if's status (0)
    fi
    if [ "${XRLENV_DEPLOY_FORCE:-}" != 1 ]; then
        if [ "${rc}" = 3 ]; then
            echo "    ERROR: an 'xrlenv up' process still holds the DB on ${cp_node}." >&2
        else
            echo "    ERROR: could not verify the old CP is gone on ${cp_node} (ssh" >&2
            echo "    rc=${rc}) — refusing to risk an overlapping state.db open." >&2
        fi
        echo "    Fix it / confirm the CP is down, then re-run (or XRLENV_DEPLOY_FORCE=1)." >&2
        return 1
    fi
    echo "    WARN: proceeding despite unverified CP-process state on ${cp_node} (rc=${rc}, FORCE)."
    return 0
}

# Post-bootstrap verification, two fail-closed proofs:
#  (a) config — every allocated node's /etc/xrlenv/node.env points at EXPECT_CP;
#  (b) LIVE   — every allocated node shows 'connected' in the CP registry (a
#      current NodeHello), polled read-only via XRLENV_BIN (never mutating).
# Fatal on: unresolved nodelist, EMPTY host expansion (would loop zero times and
# vacuously pass), a config mismatch, or an unregistered host.
deploy_verify_fleet() {
    local node_job="$1" expect_cp="$2" state_db="$3" xrlenv_bin="$4"
    local reg_tries="${5:-24}" reg_sleep="${6:-5}"
    local nodelist hosts host got bad="" unreg="" view i
    # When state.db is CP-box-LOCAL (not on the shared FS), it isn't readable from here, so
    # read it over ssh ON the CP box. Set XRLENV_VERIFY_CP_SSH_HOST=<cp host> (deploy_example.sh
    # does this). Empty (secondary cluster, state.db still on /shared-fs) → read the file directly as before.
    local cp_ssh_host="${XRLENV_VERIFY_CP_SSH_HOST:-}"

    if nodelist=$(squeue --noheader --name="${node_job}" -o '%N' 2>/dev/null | head -1); then :; fi
    if [ -z "${nodelist}" ] && [ "${XRLENV_DEPLOY_FORCE:-}" != 1 ]; then
        echo "    ERROR: could not resolve the ${node_job} nodelist from squeue —" >&2
        echo "    cannot verify the deploy. Investigate, then re-run (or XRLENV_DEPLOY_FORCE=1)." >&2
        return 1
    fi
    hosts="$(scontrol show hostnames "${nodelist}" 2>/dev/null || true)"
    if [ -z "${hosts}" ] && [ "${XRLENV_DEPLOY_FORCE:-}" != 1 ]; then
        echo "    ERROR: host expansion of '${nodelist}' produced no hosts — the" >&2
        echo "    verification loops would run zero times and vacuously 'pass'." >&2
        echo "    Investigate, then re-run (or XRLENV_DEPLOY_FORCE=1)." >&2
        return 1
    fi
    # FORCE bypassed the empty checks above; guard the vacuous-pass here so we
    # don't print the misleading "all nodes connected" success for zero hosts.
    if [ -z "${hosts}" ]; then
        echo "    WARN: no hosts to verify (XRLENV_DEPLOY_FORCE=1) — skipping config" \
             "+ registration checks (NOT a proof the fleet is up)."
        return 0
    fi

    for host in ${hosts}; do
        got="$(ssh "${SSH_OPTS[@]}" "${host}" \
            'sudo grep -oE "XRLENV_CONTROL_PLANE=\S+" /etc/xrlenv/node.env 2>/dev/null | cut -d= -f2' \
            2>/dev/null || echo '')"
        if [ "${got}" = "${expect_cp}" ]; then
            echo "    ${host}: config OK (${got})"
        else
            echo "    ${host}: node.env CP='${got:-<missing>}' != '${expect_cp}' — bootstrap FAILED here."
            bad="${bad} ${host}"
        fi
    done
    if [ -n "${bad}" ] && [ "${XRLENV_DEPLOY_FORCE:-}" != 1 ]; then
        echo "    ERROR: stale/missing agent config on:${bad}" >&2
        echo "    Re-bootstrap the node job then re-run (or XRLENV_DEPLOY_FORCE=1)." >&2
        return 1
    fi

    for ((i = 0; i < reg_tries; i++)); do
        unreg=""
        if [ -n "${cp_ssh_host}" ]; then
            # state.db is local to the CP box → run the read there (--state-db is explicit,
            # so `nodes` needs no .env; the repo/venv are on the shared FS, same path there).
            view="$(ssh "${SSH_OPTS[@]}" "${cp_ssh_host}" \
                "${xrlenv_bin} --state-db ${state_db} nodes" 2>/dev/null || true)"
        else
            view="$("${xrlenv_bin}" --state-db "${state_db}" nodes 2>/dev/null || true)"
        fi
        for host in ${hosts}; do
            # node_id = aws-<hostname -s> (deploy/bootstrap-aws.sh)
            printf '%s\n' "${view}" | grep -qE "aws-${host}[[:space:]].*connected" \
                || unreg="${unreg} ${host}"
        done
        [ -z "${unreg}" ] && break
        sleep "${reg_sleep}"
    done
    if [ -z "${unreg}" ]; then
        echo "    OK: all ${node_job} nodes are 'connected' in the CP registry."
    elif [ "${XRLENV_DEPLOY_FORCE:-}" = 1 ]; then
        echo "    WARN: not registered (XRLENV_DEPLOY_FORCE=1):${unreg}"
    else
        echo "    ERROR: these nodes never registered 'connected' with the CP:${unreg}" >&2
        echo "    (no current NodeHello — the deploy is NOT complete). Investigate the" >&2
        echo "    agent(s), then re-run (or XRLENV_DEPLOY_FORCE=1 to accept a partial fleet)." >&2
        return 1
    fi
    return 0
}
