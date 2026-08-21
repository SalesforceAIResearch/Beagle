# Cluster bring-up smokes

← back to [smoke runbook index](../README.md)

These are the **first thing to run after a fresh deploy or any
control-plane change**. They confirm the basic xrlenv path works
end-to-end before you trust higher-tier smokes against the same
cluster.

| Smoke | What it validates |
|---|---|
| [`single_rollout.py`](#single_rolloutpy) | One rollout end-to-end through `LocalRuntime` (no gRPC). The minimal "does xrlenv work at all" check. |
| [`cluster_smoke.py`](#cluster_smokepy) | Multi-rollout acceptance against a live (embedded or already-running) `xrlenv up` with one or more node agents attached. The standard cluster bring-up gate. |

See [Conventions shared across smokes](../README.md#conventions-shared-across-smokes)
for invocation patterns, the three-mode structure, artifact
output, and cleanup recipes that apply across all groups.

---

## `single_rollout.py`

**Group**: Cluster bring-up. **Wall-clock**: ~30 s. **Mode**:
script-only (no pytest test functions; the script is the smoke).

**What it validates.** The minimum-viable xrlenv flow: build a
`LocalRuntime`, construct a `Client.in_process(...)`, run one
hello-shell rollout for 3 steps, assert it seals to
`status=finished`. No gRPC, no remote nodes, no benchmark harness.
If this fails, nothing else will. If this passes but a
higher-tier smoke fails, the wire layer is the suspect.

**Prerequisites.**
- Docker daemon reachable from the local user.
- `xrlenv/hello-shell:0.1` image built locally:
  `docker build -t xrlenv/hello-shell:0.1 xrlenv/templates/hello_shell`.

**Invocation.**

```bash
uv run python tests/smoke/single_rollout.py
# or:
.venv/bin/python tests/smoke/single_rollout.py
```

**Output.** A sequence of step results from `echo` commands inside
the sandbox, then a sealed `Trajectory` with `status=finished`
pretty-printed. No artifact tree under `tmp/`.

**What "pass" means.** The rollout reaches a sealed terminal state
without raising. If the hello-shell image is missing, the script
fails fast at acquire time with a clear `ImageNotFound` — that's
expected behavior, not a smoke regression.

---

## `cluster_smoke.py`

**Group**: Cluster bring-up. **Wall-clock**: ~1-3 min depending on
`--rollouts`. **Modes**: embedded (default), `--connect-host`.

**What it validates.** A live `xrlenv up` control plane plus one or
more node agents (cloud VMs or local ones) attached and accepting
spec-21 commands. Submits `--rollouts` hello-shell rollouts and
prints one-line summaries. The standard gate after deploying or
upgrading the control plane / node binary.

**Prerequisites.**
- Docker daemon on the node-side machines (cloud VMs or local).
- For embedded mode: stop any existing `xrlenv up` first; cloud VMs'
  systemd unit will reconnect to the embedded control plane on the
  same port.
- For `--connect-host`: the existing `xrlenv up` must be running
  with `--grpc-port` reachable from this host. Consumer token via
  `$XRLENV_CONSUMER_TOKEN` or `--consumer-token`.

**Invocation.**

```bash
# Embedded mode (replaces xrlenv up for the duration of the run):
python tests/smoke/cluster_smoke.py \
    --grpc-port 50051 --min-nodes 2 --rollouts 4 --spread

# Connect mode (leaves xrlenv up running, dials it):
python tests/smoke/cluster_smoke.py \
    --connect-host 127.0.0.1 --connect-port 50051 \
    --consumer-token "$XRLENV_CONSUMER_TOKEN" \
    --min-nodes 2 --rollouts 4 --spread
```

**Output.** Per-rollout summary lines (rollout id, node, status,
final reward) and a final aggregate. No durable artifact tree.

**What "pass" means.** Every requested rollout reaches a terminal
status (`finished` / `failed` / `truncated`) without the smoke
itself raising. A nonzero rollout `failed` count usually points at
template-layer bugs, not the cluster wiring — investigate the
specific rollout's `coordinator.log` under `~/.xrlenv/runs/<id>/`.
