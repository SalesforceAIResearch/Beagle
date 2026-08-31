# API surface smokes (case-2 primitives)

← back to [smoke runbook index](../README.md)

These pin the **consumer-facing wire contract** before any harness
layer is involved. If a benchmark integration smoke regresses, run
these first to localize: a primitive that still passes here means
the bug is in the harness adaptation layer above; a primitive that
also regresses means fix the wire path before debugging the
harness.

| Smoke | What it validates |
|---|---|
| [`raw_container_smoke.py`](#raw_container_smokepy) | `Client.acquire_container` → spec-21 `AcquireContainerCommand` → node `RawContainerManager`. The lowest-level case-2/3 evaluation primitive. |
| [`dropin_cluster_smoke.py`](#dropin_cluster_smokepy) | docker-py drop-in's manager surface (`containers.run` / `put_archive` / `exec_run` / `remove`) end-to-end against the cluster, so the `XrlenvAPIClient` translation layer is itself under test. |
| [`egress_restriction_smoke.py`](#egress_restriction_smokepy) | `ContainerSession.apply_egress` (spec 07) → node `nsenter`+iptables actually restricts a live container's egress to an IP/CIDR allowlist. The deployed-node gate the unit tests can only fake. |

See [Conventions shared across smokes](../README.md#conventions-shared-across-smokes)
for invocation patterns, the three-mode structure, artifact
output, and cleanup recipes that apply across all groups.

---

## `raw_container_smoke.py`

**Group**: API surface (case-2 primitives). **Wall-clock**: ~30 s.
**Modes**: `--in-process`, embedded, `--connect-host`.

**What it validates.** The `Client.acquire_container` surface
end-to-end: SDK → control plane scheduler → node → spec-21
`AcquireContainerCommand` → `RawContainerManager` → real Docker
container → `ContainerExecCommand` → destroy. **No EnvAdapter,
no in-sandbox stub** — the bare evaluation primitive case-2 and
case-3 harnesses build on. If `swebench drop-in` or any case-3
adapter regresses, run this first to localize whether the bug is
in the primitive or in the harness adaptation layer above.

**Prerequisites.**
- Embedded / connect modes: at least one node-agent attached AND
  the smoke image (`busybox:latest` by default; `--image` to
  override) already locally pulled on that node. **Phase-1
  contract: no implicit pull on the raw-container path** — operator
  is responsible for staging images via the build plan / image
  warmup.
- `--in-process` mode: Docker daemon reachable on this host;
  `busybox:latest` (or `--image`) pulled locally.

**Invocation.**

```bash
# In-process (no gRPC, single host) — fastest:
python tests/smoke/raw_container_smoke.py --in-process

# Embedded (replaces xrlenv up; cloud nodes reconnect):
python tests/smoke/raw_container_smoke.py \
    --grpc-port 50051 --min-nodes 1

# Connect (dials existing xrlenv up):
python tests/smoke/raw_container_smoke.py \
    --connect-host 127.0.0.1 --connect-port 50051 \
    --consumer-token "$XRLENV_CONSUMER_TOKEN"
```

**Output.** Stdout-only: acquire confirmation, exec stdout/stderr,
destroy confirmation. No durable artifact tree.

**What "pass" means.** Acquire returns a container session, exec
returns the expected stdout, destroy succeeds. A failure on
acquire usually means the image isn't on the node (phase-1
contract violation, not a smoke bug); a failure on exec points at
the spec-21 `ContainerExecCommand` wire path or the node's exec
plumbing.

---

## `dropin_cluster_smoke.py`

**Group**: API surface (case-2 primitives). **Wall-clock**: ~1 min.
**Modes**: `--in-process`, embedded, `--connect-host`.

**What it validates.** The docker-py drop-in's manager surface
end-to-end:

```python
client = xrlenv.from_env(client=...)
container = client.containers.run(image, command, detach=True)
container.put_archive("/tmp", tar_bytes)
result = container.exec_run(["echo", "hi"])
container.remove(force=True)
```

This is what swebench, harbor, and any docker-py-using harness see
when they swap `docker.from_env()` for `xrlenv.from_env(client=...)`.
**Distinct from `raw_container_smoke.py`**: that one drives the SDK
directly; this one drives the docker-py manager surface so the
`XrlenvAPIClient` cluster-mode translation layer is itself under
test. A swebench drop-in regression that doesn't surface here
points at swebench-specific behavior; one that does surface here
is a translation-layer bug.

**Prerequisites.** Same as `raw_container_smoke.py`. Phase-1 "no
implicit pull" contract applies to the drop-in too.

**Invocation.**

```bash
python tests/smoke/dropin_cluster_smoke.py --in-process
python tests/smoke/dropin_cluster_smoke.py --grpc-port 50051 --min-nodes 1
python tests/smoke/dropin_cluster_smoke.py \
    --connect-host 127.0.0.1 --connect-port 50051 \
    --consumer-token "$XRLENV_CONSUMER_TOKEN"
```

**Output.** Stdout: per-step confirmations (run / put_archive /
exec_run / remove) plus the exec result body. No durable artifact
tree.

**What "pass" means.** Every drop-in call returns the docker-py-shape
object the upstream harness expects, and the underlying spec-21
`AcquireContainerCommand` / `ContainerExecCommand` flow round-trips
cleanly. If swebench's drop-in smoke regresses but this one passes,
the bug is swebench-specific (likely a docker-py method call we
haven't translated yet); if both regress, fix the translation layer
first.

---

## `egress_restriction_smoke.py`

**Group**: API surface (case-2 primitives). **Wall-clock**: ~1-2 min
(acquire + curl install + probes). **Modes**: `--in-process`,
embedded, `--connect-host`.

**What it validates.** `ContainerSession.apply_egress` (spec 07)
end-to-end, and — uniquely — that the rules actually *enforce* on a
real node: SDK → rollout-control `ApplyEgress` → node-control
`ApplyEgressCommand` → `RawContainerManager.apply_egress` →
`nsenter` + `iptables`/`ip6tables` install an OUTPUT chain in the
container's netns. The unit tests pin the rule compiler, safety
guards, fail-closed teardown, and the gRPC encode/decode with a
*fake* enforcer; only this smoke proves the kernel-level apply works
on a deployed node. **This is the residual-risk gate** before
treating the egress restriction as a trusted anti-cheat boundary.

It mirrors the intended open-setup→tighten flow: acquire on the open
bridge → install curl → **BEFORE**: confirm the gateway *and* an
arbitrary external IP (`1.1.1.1`) are reachable (the net is genuinely
open) → resolve the gateway host to IPv4s and pin them in the
container's `/etc/hosts` → `apply_egress(gateway /32s, gateway port)`
→ **AFTER**: gateway still reachable, but the external IP and cloud
metadata (`169.254.169.254`) are now blocked → optional
`--check-block-all`: an empty allowlist blocks even the gateway.

**Prerequisites.**
- A node whose **host** has `nsenter`, `iptables`, `ip6tables`, the
  relevant kernel modules, and whose node-agent process holds
  `CAP_NET_ADMIN` + `CAP_SYS_ADMIN` + `CAP_SYS_PTRACE` (granted by the
  `deploy/systemd/xrlenv-node.service` unit, so a properly bootstrapped
  node has them; `CAP_SYS_PTRACE` is what lets `nsenter` open the
  root-owned container's `/proc/<pid>/ns/net`). If any are missing the
  `apply_egress` call fails loudly (fail-closed → the container is
  destroyed) — that *is* the environment check landing where it should.
- The `--image` (default `alpine:latest`) already present on the node
  (**phase-1 "no implicit pull"** on the raw path) and carrying a
  package manager for the curl bootstrap (alpine/apk or debian/apt),
  or pass `--no-install-curl` for an image that ships curl.
- During the BEFORE phase the node must reach the gateway and the
  external probe IP over the open bridge.

**Invocation.**

```bash
# Dev cluster (the realistic mode) — dials an existing xrlenv up:
python tests/smoke/api_surface/egress_restriction_smoke.py \
    --connect-host <dev-control-plane> --connect-port 50051 \
    --consumer-token "$XRLENV_CONSUMER_TOKEN" \
    --gateway-url "$SFR_GATEWAY_OPENAI_URL" \
    --image alpine:latest --check-block-all

# Embedded (replaces xrlenv up; cloud nodes reconnect):
python tests/smoke/api_surface/egress_restriction_smoke.py \
    --grpc-port 50051 --min-nodes 1 --gateway-url "$SFR_GATEWAY_OPENAI_URL"

# In-process (single host with a local Docker daemon):
python tests/smoke/api_surface/egress_restriction_smoke.py --in-process
```

Key flags: `--gateway-url` (defaults to `$SFR_GATEWAY_OPENAI_URL`) is
the endpoint allowlisted; `--image` must be on the node;
`--no-install-curl` skips the curl bootstrap; `--check-block-all`
adds the empty-allowlist block-all check.

**Output.** Stdout-only, one line per probe under `[egress-smoke]`:
the resolved gateway IPs, the BEFORE reachability of the gateway +
external IP, the `apply_egress` call, the AFTER reachability of the
gateway / external IP / metadata, then a final `RESULT: PASS` or
`RESULT: FAIL` (with the specific failed expectations). No durable
artifact tree.

**What "pass" means.** The BEFORE→AFTER contrast holds: `external
1.1.1.1 reachable=True` and gateway reachable *before*; after the
tighten the gateway is still reachable but the external IP and
metadata are `reachable=False`. A pass means the iptables program
genuinely enforces on the node. A failure where everything stays
reachable after `apply_egress` points at the node lacking
`nsenter`/`iptables`/`CAP_NET_ADMIN` or the rules not landing in the
right netns; a failure where the gateway becomes unreachable points
at the allowlist resolution / `/etc/hosts` pin (the client couldn't
resolve the gateway once DNS egress was cut).
