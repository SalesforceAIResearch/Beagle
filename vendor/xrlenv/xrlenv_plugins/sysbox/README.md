# Sysbox node pool — secure nested-container runtime for xrlenv

This directory is the **self-contained toolkit** for building, vendoring, and
installing [Sysbox](https://github.com/nestybox/sysbox) (`sysbox-runc`) on a
dedicated xrlenv node pool. Sysbox lets a sandbox run **Docker-in-Docker,
systemd-as-PID 1, and `ip netns` / kernel namespaces — all unprivileged**, under
a user namespace where the container's "root" maps to an unprivileged host
subuid. That is what several benchmarks need and what plain `--privileged` DinD
gives up (privileged DinD's inner root *is* host root).

- **EvoClaw / element-web** needs Docker-in-Docker for `testcontainers`.
- **harbor-terminalworld-verified** needs DinD + systemd + `NET_ADMIN`/netns.

The design rationale, the xrlenv-core integration, and the security analysis live
in `notes/sysbox-dind-runner.md` (the proposal). This README is the operational
"how to build + install + run the pool" companion.

---

## Why we build from source (the load-bearing caveat)

The latest **packaged** release, `sysbox-ce 0.7.0`, **cannot run any container**
on our nodes' Docker 29.x / containerd 2.x stack. Every
`docker run --runtime=sysbox-runc …` fails at task-create with:

```
OCI runtime create failed: namespace {"time" ""} does not exist
```

Root cause ([nestybox/sysbox#1011](https://github.com/nestybox/sysbox/issues/1011)):
containerd ≥ 2.0 introspects the runtime via `sysbox-runc features`; 0.7.0's
output doesn't declare user-namespace handling, so containerd strips the userns
from the OCI spec and sysbox then dies on the time namespace. Downgrading
`sysbox-runc` to 0.6.7 does **not** help — it's the containerd-side interaction.

The fix — [sysbox-runc PR #106](https://github.com/nestybox/sysbox-runc/pull/106),
commit `cf83133d` "add time namespace to sys container by default" — is merged to
`sysbox-runc` `main` but **is not in any packaged `sysbox-ce` release**. So we
build the patched binaries from source and install them as a **checksum-pinned,
vendored** artifact.

This was validated end-to-end on 2026-07-05 (dev node, Ubuntu 22.04, kernel 6.8,
Docker 29.5.3, containerd 2.2.4): with the patched build, an **unprivileged**
sysbox container ran inner `dockerd` (`Hello from Docker!`), booted `systemd` as
PID 1, did `ip netns` add/list/delete, and — the security payoff — its inner
`root` mapped to host subuid `11765408`, **not** host root.

### When can this be retired?

When nestybox ships a packaged `sysbox-ce` release that carries PR #106. At that
point: bump `SYSBOX_CE_VERSION` in `pin.env`, drop the binary overlay from
`install_sysbox_node.sh`, and install straight from the release `.deb`. Until
then, the vendored binary is the only way to run sysbox on our current Docker.

---

## Files

| File | Role |
|---|---|
| `pin.env` | Single source of truth: the `sysbox-ce` release + the `sysbox-runc` fix commit to build. Edit here, then rebuild. |
| `build_sysbox.sh` | Builds the patched static binaries via the official containerized `make sysbox-static`, emits `vendor/<commit>/{binaries,SHA256SUMS,PROVENANCE.txt}`. |
| `install_sysbox_node.sh` | Installs Sysbox on ONE node: packaged `.deb` (units/sysctls/user) + checksum-verified patched-binary overlay + daemon.json assertions + docker/xrlenv-node restart. |
| `deploy_sysbox_pool.sh` | Reads `nodes.yaml`, and runs `install_sysbox_node.sh` on every node marked `sysbox: true`. |

The build output (`{sysbox-runc,sysbox-mgr,sysbox-fs,SHA256SUMS}`) lives **out of
the git tree** on shared storage — `SYSBOX_VENDOR_ROOT/<commit>/` (default
`/path/to/sysbox-vendor`, override with
`XRLENV_SYSBOX_VENDOR_DIR`). A ~40 MB unofficial build has no business in-tree,
and holding it in both the dev and prod repos would duplicate it and invite
drift; one canonical copy on shared `/shared-fs` serves both. The current pinned build
is already there.

### Who sets `XRLENV_SYSBOX_VENDOR_DIR`?

| Role | Sets it? | Why |
|---|---|---|
| **Sysbox-pool operator** (runs build/install/deploy) | **Only if the vendored binaries live somewhere other than the default `/shared-fs` path.** It has a default, so it is *not required* on this cluster. | It's a build-tooling knob read by `pin.env` → `build_sysbox.sh` / `install_sysbox_node.sh` / `deploy_sysbox_pool.sh`. |
| **Operator of a non-sysbox cluster** | No | Never runs the sysbox toolkit. |
| **Consumer** (RL trainer calling `acquire_container(container_runtime="sysbox-runc")`) | **Never** | Consumers select the runtime by *name*; they never see the binaries, the vendor path, or this variable. |

It is **not** a runtime / control-plane / node-agent env var — nothing in
`xrlenv` reads it at run time. It only tells the build/install shell scripts
where the vendored artifacts are. Consumers and the running services are
completely unaware of it.

---

## Quickstart

```bash
# 1) Build the patched binaries once (needs docker; ~15-30 min first run).
#    Writes to shared storage (SYSBOX_VENDOR_ROOT/<commit>/), NOT the repo tree.
#    The current pinned build is already there — skip this unless you bump pin.env.
bash xrlenv_plugins/sysbox/build_sysbox.sh
#    → $SYSBOX_VENDOR_ROOT/<commit>/{sysbox-runc,sysbox-mgr,sysbox-fs,SHA256SUMS}

# 2) Mark the pool in nodes.yaml (see below), then install on all pool nodes:
bash xrlenv_plugins/sysbox/deploy_sysbox_pool.sh nodes.yaml
#    (or install one node directly, ON that node:)
#    sudo bash xrlenv_plugins/sysbox/install_sysbox_node.sh

# 3) Allow the runtime in nodes.yaml policy (see below) and restart the control
#    plane so the KwargsPolicy picks up allowed_runtimes.
```

---

## Declaring the pool in `nodes.yaml`

Two independent knobs:

```yaml
version: 1
nodes:
  - id: aws-node-host
    address: internal-ip
    backends: [docker]
    sysbox: true          # ← (1) pool membership: deploy installs sysbox here
  - id: aws-node-host
    address: internal-ip
    backends: [docker]
    # no sysbox: this node stays a normal docker node

policy:
  allowed_runtimes: [sysbox-runc]   # ← (2) cluster policy: permit the override
```

1. **`sysbox: true`** (per node) — declarative pool membership. It drives
   `deploy_sysbox_pool.sh` (which nodes to install on) and operator validation.
   It is *intent*; the **ground truth** is what each node's docker advertises on
   connect (`NodeHello.supported_runtimes`). The control-plane placement filter
   routes `container_runtime='sysbox-runc'` acquires only to nodes that actually
   advertise the runtime — so a mis-marked node simply won't attract sysbox work.
2. **`policy.allowed_runtimes: [sysbox-runc]`** — the cluster-wide `KwargsPolicy`
   gate. **Empty by default → every runtime override is rejected.** You must add
   `sysbox-runc` here for any acquire to be allowed to request it. This is the
   operator opt-in for the escape surface.

Both are required to actually run sysbox work: the pool nodes must have it
installed (advertisement), and the policy must permit the override (admission).

**Generated rosters (HyperPod):** if your `nodes.yaml` is generated from a Slurm
script by `xrlenv nodes-from-slurm`, mark pool members with
`--sysbox-node <hostname-or-id>` (repeatable). A marker already present in the
destination file is **preserved across regeneration** — same contract as
`policy:` — so the pool survives a `nodes-from-slurm` re-run without re-passing
the flag. `policy.allowed_runtimes` is likewise preserved.

## Install order (there is no hard ordering dependency)

Installing Sysbox and declaring the pool are **decoupled on purpose** — do NOT
bake the install into the node bootstrap (`dev_xrlenv_node.sh`) or couple it to
roster generation (`dev_xrlenv_control.sh`):

- **Placement is gated by the live advertisement, not the marker.** A node
  attracts `sysbox-runc` work *only* if its Docker actually advertises the
  runtime on `NodeHello` (i.e. Sysbox is installed and Docker restarted). The
  `nodes.yaml` `sysbox: true` marker is operator *intent* / visibility only. So
  declaring the pool before the install completes (or after) is harmless — it
  self-corrects the moment the node reconnects advertising `sysbox-runc`.
- **Install is a separate, pool-scoped step.** `deploy_sysbox_pool.sh` (or
  `install_sysbox_node.sh` per node) is an explicit operator action, not part of
  the normal per-node bootstrap. Baking it into the node job would install on
  *every* node in that job's nodelist (the pool is a subset) and would force the
  node job to read the control-generated `nodes.yaml` — the exact
  circular/ordering dependency to avoid. Keeping it separate also keeps the
  container-escape surface behind a deliberate operator gate.

Recommended sequence (order between 1 and 2 does not matter for correctness):
1. Bring up the cluster normally (node job + control job).
2. Declare the pool: add `sysbox: true` to the pool nodes (via `--sysbox-node`
   on the control job's `nodes-from-slurm`, or by hand) and `sysbox-runc` to
   `policy.allowed_runtimes`.
3. Install: `deploy_sysbox_pool.sh nodes.yaml` (reads the pool, installs +
   restarts Docker + bounces `xrlenv-node` on each pool node).
4. Restart the control plane so the updated `allowed_runtimes` policy loads.

---

## How xrlenv core consumes this (the runtime plumbing)

The `container_runtime` field is threaded end-to-end (design `notes/sysbox-dind-runner.md`
§5). Consumers select it per-acquire:

```python
await client.acquire_container(image="...", container_runtime="sysbox-runc")
# or via the docker-py compat drop-in:
docker.from_env(...).containers.run("...", runtime="sysbox-runc")
```

What core does with it:

- **Policy (§5.2)** — `KwargsPolicy.allowed_runtimes` rejects any non-`runc`
  runtime unless the operator opted in. `runc` / `None` are the default and
  always allowed.
- **Placement (§5.3)** — the scheduler filters candidates to nodes advertising
  the runtime (`supported_runtimes()`), *before* reservation; the admission queue
  re-passes it on every drain retry; image-affinity narrows to runtime-eligible
  nodes; fleet companions inherit + are validated against the opener's runtime.
- **Advertisement (§5.3/§9)** — each node advertises `supported_runtimes` + its
  daemon `default_runtime` on `NodeHello`; the control plane WARNs at connect if
  a node's `default-runtime` is not `runc` (a non-runc default would silently
  bypass `allowed_runtimes`).
- **Node verify (§5.5)** — before `containers.run`, the node fails loud if the
  requested runtime isn't registered in its docker (no silent fall-back to runc).
- **Init (§5.6)** — a sysbox acquire skips the injected `tini` (`--init`) so
  systemd / inner-init is PID 1; the normal path keeps `init=True` (zombie
  reaping) unchanged.
- **Egress (§6)** — `container_can_escape_egress()` returns True for a
  `sysbox-runc` container, so `apply_egress()` refuses to install its
  `nsenter`-based allowlist on one (the inner root could flush it). A sysbox
  container therefore cannot be used for an offline/egress-restricted task —
  fail-loud, by design.

The normal `runc` raw-container path is byte-for-byte unchanged — proven by the
`§10.x` regression tests (`tests/unit/node/test_raw_container.py`,
`tests/unit/control/test_scheduler.py`, `tests/unit/control/test_kwargs_policy.py`,
`tests/unit/backends/test_egress.py`).

---

## Security

Sysbox is a **container-escape / cross-tenant surface reducer** relative to
privileged DinD, but it is still a nested-runtime with more attack surface than a
plain sandbox, and its inner root controls its own network namespace:

- **Dev / single-tenant only.** Keep the pool to nodes that do not co-host other
  tenants' sandboxes. Do **not** put a shared multi-tenant node in the pool.
- **Egress is not trusted inside sysbox.** The inner root can rewrite its netns
  iptables, so xrlenv refuses the post-install egress allowlist on a sysbox
  container (§6). Use sysbox for internet-on tasks; don't rely on it for
  offline/egress-restricted grading.
- **`allowed_runtimes` stays empty by default.** Nothing can request sysbox until
  an operator explicitly opts in per cluster.
- **`default-runtime` must stay `runc`.** The install script and the control
  plane both assert/verify this — a non-runc default silently bypasses the
  opt-in.
- **Unofficial binary.** The vendored build is an unreleased upstream fix. Treat
  it like any pinned third-party artifact: verify `SHA256SUMS`, and retire it for
  a packaged release as soon as one ships PR #106.
