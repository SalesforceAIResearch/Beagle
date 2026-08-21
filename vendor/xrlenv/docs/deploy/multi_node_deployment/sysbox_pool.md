# Sysbox node pool (Docker-in-Docker / systemd sandbox)

Certain benchmarks run workloads that require **Docker-in-Docker, `systemd` as
PID 1, or privileged-looking kernel namespaces** (`ip netns`, `iptables`,
`ptrace`) inside the sandbox. Examples:

- **EvoClaw / element-web** — the E2E suite uses `testcontainers` to launch
  Matrix homeservers; without an inner Docker runtime the suite dies in seconds.
- **harbor-terminalworld-verified** — tasks that exercise `dockerd`, `systemctl`,
  `ip netns add`, and `gdb` ptrace.

The default Docker runtime (`runc`) does not support these capabilities in
unprivileged containers. The correct approach is [Sysbox](https://github.com/nestybox/sysbox)
(`sysbox-runc`), an OCI runtime that provides inner Docker, `systemd` as PID 1,
and kernel namespace operations — all **unprivileged**: the container's "root"
maps to an unprivileged host subuid, so inner root is not host root.

This page is the operator guide for building, installing, and declaring a Sysbox
node pool. Consumer API changes (the `container_runtime` parameter) are covered in
{doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/direct_api`.

---

## When to set up this pool

You need this pool only if you are running benchmarks that require inner Docker,
`systemd`, or privileged kernel namespaces. Standard benchmarks (Terminal-Bench,
SWE-bench Verified) run on the normal runc path and are unaffected.

Keep the pool to **dev or single-tenant nodes**. Sysbox expands the
kernel/userns/inner-daemon attack surface. Do not place sysbox nodes in a pool
shared with other tenants' normal containers.

---

## Source-build requirement (load-bearing caveat)

The packaged `sysbox-ce 0.7.0` release **cannot run any container** on our Docker
29.x / containerd 2.x stack. Every `docker run --runtime=sysbox-runc …` fails at
task-create with:

```
OCI runtime create failed: namespace {"time" ""} does not exist
```

Root cause ([nestybox/sysbox#1011](https://github.com/nestybox/sysbox/issues/1011)):
containerd ≥ 2.0 introspects the runtime via `sysbox-runc features`; 0.7.0's
output doesn't declare user-namespace handling, so containerd strips the userns
and sysbox dies on the time namespace.

The fix — [sysbox-runc PR #106](https://github.com/nestybox/sysbox-runc/pull/106),
commit `cf83133d` — is merged to `sysbox-runc` main but is **not in any packaged
`sysbox-ce` release**. Until a packaged release ships it, the toolkit builds
the patched binaries from source and installs them as a checksum-pinned, vendored
artifact.

When nestybox ships a packaged release carrying PR #106: bump `SYSBOX_CE_VERSION`
in `pin.env`, drop the binary overlay from `install_sysbox_node.sh`, and install
straight from the release `.deb`.

---

## Toolkit files

The toolkit lives in `xrlenv_plugins/sysbox/`:

| File | Role |
|---|---|
| `pin.env` | Single source of truth: the `sysbox-ce` release + the `sysbox-runc` fix commit to build. Edit here, then rebuild. |
| `build_sysbox.sh` | Builds the patched static binaries via the official containerized `make sysbox-static`, emits `vendor/<commit>/{binaries,SHA256SUMS,PROVENANCE.txt}`. |
| `install_sysbox_node.sh` | Installs Sysbox on one node: packaged `.deb` (units/sysctls/subuid) + checksum-verified patched-binary overlay + `daemon.json` assertions + docker restart + `xrlenv-node` restart. |
| `deploy_sysbox_pool.sh` | Reads `nodes.yaml`, runs `install_sysbox_node.sh` on every node marked `sysbox: true`. |

The built binaries live **out of the git tree** on shared storage —
`SYSBOX_VENDOR_ROOT/<commit>/` (default `/path/to/sysbox-vendor`).
A ~40 MB unofficial build does not belong in-tree, and duplicating it across the
dev and prod repos would invite drift; one canonical copy on shared `/shared-fs` serves
both.

### The `XRLENV_SYSBOX_VENDOR_DIR` variable

`XRLENV_SYSBOX_VENDOR_DIR` overrides where the toolkit reads/writes the vendored
binaries. It is a **build-tooling knob, not a runtime setting** — nothing in the
running `xrlenv` control plane or node agent reads it.

| Role | Sets it? |
|---|---|
| **Sysbox-pool operator** (runs build/install/deploy) | Only to point at storage other than the default `/shared-fs` path; it has a default, so it is *not required* on this cluster. |
| Operator of a non-sysbox cluster | No — never runs the toolkit. |
| **Consumer** (calls `acquire_container(container_runtime="sysbox-runc")`) | **Never** — consumers select the runtime by name and never see the binaries or this variable. |

---

## Step-by-step: building and installing

```{note}
**Install order — no hard dependency.** Installing Sysbox and declaring the pool
are decoupled on purpose; do **not** bake the install into node bootstrap
(`dev_xrlenv_node.sh`) or couple it to roster generation
(`dev_xrlenv_control.sh`). Placement is gated by the **live**
`NodeHello.supported_runtimes` advertisement, not by the `nodes.yaml` marker — a
node attracts `sysbox-runc` work only once its Docker actually advertises the
runtime. So declaring the pool before the install finishes (or after) is
harmless; it self-corrects on reconnect. Install is a separate, pool-scoped
operator step (`deploy_sysbox_pool.sh`) — baking it into the node job would run
on *every* node in that job's nodelist and force it to read the
control-generated `nodes.yaml` (a circular dependency), besides widening the
escape surface without a deliberate gate.
```

### Step 1 — build the patched binaries (once)

Run this on any host that has Docker available. The build takes 15–30 minutes on
first run; subsequent runs use the Docker layer cache.

```bash
bash xrlenv_plugins/sysbox/build_sysbox.sh
```

Output lands on shared storage at `SYSBOX_VENDOR_ROOT/<commit>/` (default
`/path/to/sysbox-vendor/<commit>/`):

```
sysbox-runc
sysbox-mgr
sysbox-fs
SHA256SUMS
PROVENANCE.txt
```

Verify `SHA256SUMS` before distributing the binaries to nodes. On this cluster
the current pinned build is already present — skip this step unless you bump
`pin.env`.

### Step 2 — declare the pool in `nodes.yaml`

Two independent knobs are required:

```yaml
version: 1
nodes:
  - id: aws-node-host
    address: internal-ip
    backends: [docker]
    sysbox: true          # (1) pool membership — install + advertise this runtime
  - id: aws-node-host
    address: internal-ip
    backends: [docker]
    # no sysbox: stays a normal Docker node

policy:
  allowed_runtimes: [sysbox-runc]   # (2) cluster policy — permit the override
```

Both knobs are required to run sysbox work:

- **`sysbox: true`** (per node) — declares pool membership. The deploy script
  installs Sysbox on these nodes; the control plane uses this as an operator
  assertion, but placement routing is driven by what each node's Docker actually
  advertises at connect time (`NodeHello.supported_runtimes`), so a
  mis-marked node simply won't attract sysbox work.
- **`policy.allowed_runtimes: [sysbox-runc]`** — the cluster-wide `KwargsPolicy`
  gate. **Empty by default — every runtime override is rejected.** Add
  `sysbox-runc` here to permit acquires that request it. This is the operator
  opt-in for the expanded security surface.

### Step 3 — install on pool nodes

Install on all nodes marked `sysbox: true` in one command:

```bash
bash xrlenv_plugins/sysbox/deploy_sysbox_pool.sh nodes.yaml
```

Or install a single node directly (run **on that node**, as root):

```bash
sudo bash xrlenv_plugins/sysbox/install_sysbox_node.sh
```

The install script:

1. Installs the `sysbox-ce` `.deb` (registers systemd units, kernel sysctls,
   subuid entries).
2. Overlays the checksum-verified patched `sysbox-runc` binary.
3. Merges `daemon.json` non-destructively — preserves any existing `data-root`,
   `registry-mirrors`, and `insecure-registries` keys; adds
   `runtimes.sysbox-runc`.
4. Asserts that `default-runtime` is unset or `runc`. An absent
   `default-runtime` key in `daemon.json` means `runc` (Docker's built-in
   default); the script accepts that. **A non-runc default would silently bypass
   `allowed_runtimes` for every acquire** — the script fails loudly if it detects
   this.
5. Restarts Docker (`systemctl restart docker`) and then explicitly restarts
   `xrlenv-node.service`. The node service stops when Docker restarts (it is
   bound to `docker.service`) and does not auto-revive — the install script
   handles the explicit restart.

```{note}
The Docker restart kills all running containers on the node. Run this only during
a drain window (no active rollouts). Use `xrlenv up --drain` or wait for the
node's active session count to reach zero before installing.
```

### Step 4 — restart the control plane

After updating `nodes.yaml` (adding `allowed_runtimes`), restart `xrlenv up` so
the updated `KwargsPolicy` is picked up. The nodes re-connect automatically and
advertise their `supported_runtimes` on reconnect.

---

## Verification

After install and control-plane restart, confirm the pool is active:

```bash
# Check the node advertises the runtime:
xrlenv nodes --nodes-yaml nodes.yaml
# The sysbox pool node should show sysbox-runc in its runtimes column.

# Quick end-to-end smoke (requires the cluster to be up):
python - <<'EOF'
import asyncio
from xrlenv import Client

async def main():
    client = Client.grpc(host="127.0.0.1", port=50051)
    async with await client.acquire_container(
        image="ubuntu:22.04",
        command=["sleep", "10"],
        container_runtime="sysbox-runc",
    ) as session:
        result = await session.exec(["id"])
        print(result.stdout.decode())   # uid=0(root) inside, subuid on the host
asyncio.run(main())
EOF
```

---

## Security posture

Sysbox is a **surface reducer** relative to `--privileged` DinD, but it is not
equivalent to a plain sandbox:

| Property | Plain runc sandbox | Sysbox sandbox |
|---|:---:|:---:|
| Inner root = host root | no | no (user-ns mapped) |
| Inner `dockerd` supported | no | yes |
| `systemd` as PID 1 | no | yes |
| `ip netns` / `iptables` | no | yes (own netns) |
| Egress allowlist trustworthy | yes | **no** (see below) |

**Egress restriction is incompatible with sysbox.** Because the inner root
controls its own network namespace, it can flush `iptables` rules installed by
xrlenv's `apply_egress`. For this reason, `apply_egress` refuses to install its
allowlist on a sysbox container — the request raises `XRLEnvError`. This means
**sysbox containers are always internet-on**. Do not use sysbox for
offline/egress-restricted grading tasks.

See {doc}`/developer_guide/security` for the full egress model.

**`allowed_runtimes` stays empty by default.** Nothing can request `sysbox-runc`
until an operator explicitly opts in via `nodes.yaml`.

**`default-runtime` must stay `runc`.** Both the install script and the control
plane assert this. A non-runc daemon default would silently bypass the opt-in for
every acquire.

**Treat the vendored binary as a pinned third-party artifact.** Verify
`SHA256SUMS` on every node. Retire it for a packaged release as soon as nestybox
ships one carrying PR #106.

---

## See also

- {doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/direct_api` —
  `container_runtime` parameter on `acquire_container`.
- {doc}`/developer_guide/security` — egress model and why sysbox + offline is
  incompatible.
- {doc}`inventory` — full `nodes.yaml` field reference.
- `xrlenv_plugins/sysbox/README.md` in the source tree — the "why source-build"
  story and full toolkit file descriptions.
