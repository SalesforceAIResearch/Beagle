# WebArena-Infinity

WebArena-Infinity (WAI) is a web-browsing benchmark where agents
operate a full browser inside a container. XRLEnv runs WAI in
**raw-container mode**: the benchmark harness talks directly to the
container via the standard `acquire_container` API, with no Harbor
adapter layer between them.

This page covers the image lifecycle — how the substrate image is built
and distributed, what channel tags mean, how downstream consumer code
pins the image ref, and how to launch WAI evaluation jobs against the
cluster from the WAI repo.

## The substrate image

WAI's benchmark environment lives in a single container image called
the **substrate**. The substrate bundles Chromium, the WAI Python
package, and a stripped-down server set (answer-free). It is built
from:

```
xrlenv_plugins/benchmarks/webarena_infinity/
├── Dockerfile          # WEBARENA_REF pins the WAI git commit
└── build_plan.yaml     # declares image_ref: …/substrate:dev
```

The Dockerfile pins the WAI source commit in the `WEBARENA_REF` build
argument. Updating WAI means bumping `WEBARENA_REF` in the Dockerfile
and rebuilding — not changing the `image_ref` tag.

## Channel tags

The `image_ref` in `build_plan.yaml` is a **channel tag** — a stable
distribution channel, not a version pin:

| Channel tag | Intended use |
|---|---|
| `xrlenv-webarena-infinity/substrate:dev` | Development and CI — rebuilt on each WAI source update |
| `xrlenv-webarena-infinity/substrate:stable` | Production runs — promoted manually from `:dev` after validation |

Channel tags are **mutable**: the same tag name refers to different
image content across rebuilds. The key behaviors that make this safe:

- **Consumers pin the channel tag forever** and never change it when
  WAI is updated. The tag is a stable handle; its byte content evolves
  beneath it.
- **The control plane resolves the tag to a digest at acquire time.**
  The {doc}`registry freshness model <../technical_details/images/registry_freshness>`
  performs `HEAD /v2/<repo>/manifests/<tag>` → `Docker-Content-Digest`
  before dispatching to the node. Each rollout is pinned to the exact
  digest that was current when the acquire was processed, and the
  digest is recorded on the session for auditability.
- **A rebuilt+re-pushed `:dev` reaches nodes automatically** — no
  consumer config change needed. The next acquire after the resolver's
  TTL (default 60 s) dispatches the new digest.

### Why not `:latest` or a git-ref tag?

| Scheme | Problem |
|---|---|
| `:latest` | No promote gate; every push immediately becomes the "stable" surface. |
| `:1ca77813` (WAI git ref) | Conflates source identity with distribution channel. Adding a bug-fix on the same WAI commit requires a tag rename; the tag itself carries no semantic of "dev" vs "stable". |
| `:dev` / `:stable` (channel tag) | Stable distribution channel. Source commit lives in `WEBARENA_REF` (the Dockerfile); the tag announces readiness, not provenance. |

## Rebuild and deploy workflow

To update the substrate image to a new WAI commit or fix a bug:

```bash
# 1. Edit the Dockerfile's WEBARENA_REF to the new WAI commit.
#    Build and push under the same channel tag.
.venv/bin/python deploy/registry/build_and_push_images.py \
    --plan xrlenv_plugins/benchmarks/webarena_infinity/build_plan.yaml \
    --registry <REGISTRY_HOST>:5011 \
    --registry-scheme http \
    --force           # --force is required: the tag already exists on the registry

# 2. Optional: evict the old image from nodes immediately.
#    Without this, nodes serve the old digest for in-flight rollouts
#    and pull the new one on the next acquire.
xrlenv images evict xrlenv-webarena-infinity/substrate:dev \
    --connect-host <admin-host>

# 3. No consumer config changes needed.
#    The next acquire dispatches the new digest automatically.
```

`--force` is needed because `deploy/registry/build_and_push_images.py` skips
the build when the tag already exists on the registry by default (the
idempotent re-run behavior for immutable tags). For a mutable channel
tag, `--force` overrides this check.

After step 1, the control plane's resolver cache expires within
`XRLENV_REGISTRY_RESOLVE_TTL_S` (default 60 s), and subsequent acquires
pick up the new digest without any restart.

```{note}
The freshness model and `xrlenv images evict` are **control-plane and
node features**. A cluster running old control-plane code uses the legacy
mutable-tag behavior (no automatic digest resolution). Update the
control plane with `xrlenv up` (new code) before relying on automatic
digest propagation.
```

## Downstream consumer config

Benchmark code stores the **bare** channel ref in a named constant
(portable across registries) and **prepends the private-registry host
from `.env`** to produce the registry-qualified ref it dispatches to
`acquire()`. This is the exact pattern coding-bench's `loader.py` and
WAI's `xrlenv_config.py` use:

```python
import os

# Bare channel ref — portable; the registry host:port lives in .env, not here.
DEFAULT_SUBSTRATE_REF = "xrlenv-webarena-infinity/substrate:dev"

def substrate_image() -> str:
    # A per-run override is already a full ref — use it verbatim.
    override = os.environ.get("WAI_SUBSTRATE_IMAGE")
    if override:
        return override
    # Otherwise prepend the private-registry host:port → the
    # REGISTRY-QUALIFIED ref the control plane resolves to a digest.
    host = os.environ["XRLENV_PRIVATE_REGISTRY_HOST"]
    port = os.environ.get("XRLENV_PRIVATE_REGISTRY_PORT", "5011")
    return f"{host}:{port}/{DEFAULT_SUBSTRATE_REF}"
```

Key rules:
- **Dispatch a registry-qualified ref.** The control plane's tag→digest
  resolution runs on `host:port/repo:tag` — it *parses* the registry
  host from the ref, it does **not** add one. A bare ref (`repo:tag`,
  no host) is **not** resolved and won't reach the private registry, so
  always prepend the host (from `.env`) before calling `acquire()`.
- **Keep the registry host in `.env`, not in the constant.** Store only
  the bare channel ref in code (portable across registries / clusters);
  read `XRLENV_PRIVATE_REGISTRY_HOST` (and `_PORT`) at runtime.
- **Never change `DEFAULT_SUBSTRATE_REF` when WAI is rebuilt.** Only
  change it when the channel itself changes (promoting `:dev` → `:stable`,
  or creating a new channel).
- **Override per run** (not in code) to test a non-default channel, via
  `WAI_SUBSTRATE_IMAGE` (a full ref) or `benchmark.image` in the run
  config.

## Running jobs from the WAI repo

WAI evaluation jobs are launched from the **WAI checkout**, not from
XRLEnv — the integration is three scripts that import WAI's own
`evaluation/` modules (`agents`, `run_eval_parallel`, `tasks`). XRLEnv
keeps a versioned canonical copy at
`xrlenv_plugins/benchmarks/webarena_infinity/` (see its `README.md` for
the full reference); the files that actually run are the ones you place
into the WAI repo.

| File | Runs | Role |
|---|---|---|
| `run_eval_parallel_xrlenv.py` | host | Orchestrator — same CLI + output layout as WAI's `run_eval_parallel.py`, but each worker drives an xrlenv container instead of a local app-server port. |
| `xrlenv_config.py` | host | Cluster coordinates + credentials, read once from the WAI repo-root `.env`; holds the substrate `IMAGE_REF`. |
| `xrlenv_runner.py` | inside the container | Injected per container by the orchestrator and invoked there; you never run it directly. |

**1. Place the scripts into the WAI repo** — copy all three into the WAI
checkout's `evaluation/`, where they can import `agents` /
`run_eval_parallel` / `tasks`:

```bash
cp xrlenv_plugins/benchmarks/webarena_infinity/copy_to_call_site/* <wai-checkout>/evaluation/
```

**2. Prerequisites**

- The substrate image built + pushed (see *Rebuild and deploy workflow*
  above).
- `xrlenv` importable in the WAI venv — `pip install` it, or set
  `XRLENV_REPO=/path/to/xrlenv` (defaults to
  `/path/to/xrlenv-dev`).
- A `.env` at the WAI repo root, read once (with `override=True`) by
  `xrlenv_config.py`:

```bash
XRLENV_GRPC_HOST=<control-plane-host>    # control-plane host
XRLENV_GRPC_PORT=50051
XRLENV_CONSUMER_TOKEN=<consumer-token>   # xrlenv tokens issue consumer
XRLENV_PRIVATE_REGISTRY_HOST=<private-registry-host>
XRLENV_PRIVATE_REGISTRY_PORT=5011
OPENAI_API_KEY=...                       # plus GOOGLE_API_KEY / ANTHROPIC_API_KEY —
                                         # forwarded into each container
```

**3. Launch from the WAI checkout root**

```bash
cd <wai-checkout>

# 8 containers, real-tasks suite for one app, against the dev control plane
python evaluation/run_eval_parallel_xrlenv.py --model gemini-pro --workers 8 \
    --web-app apps/gmail

# one task, explicit image + control plane (overrides .env / defaults)
python evaluation/run_eval_parallel_xrlenv.py --model gpt --task-id task_e1 \
    --workers 1 --web-app apps/gmail \
    --image <registry-host>:5011/xrlenv-webarena-infinity/substrate:dev \
    --xrlenv-host <control-plane-host> --xrlenv-port 50051
```

`--workers N` is N containers in flight cluster-wide; the output
(`results.json` + `report.html`, multi-run merge, resume) is identical
to WAI's local runner.

Per run, the orchestrator acquires one container per worker (reused
across tasks), injects `evaluation/` + `xrlenv_runner.py`, starts the app
server inside the container, then for each task runs the agent → injects
the verifier (the answer) only *after* the agent exits → runs the
verifier → deletes the answer → pulls artifacts back. The answer-free
substrate stays answer-free at runtime: the agent is never co-resident
with an answer file.

## Auditing which digest ran

Because the control plane records the resolved digest on each raw
session, you can always determine exactly which image content ran for
a given rollout. In the admin panel, open `/rollouts/<id>` and check
the session metadata. The `image_ref` field shows the digest-pinned
form (`host/repo@sha256:…`) that was dispatched to the node, not the
channel tag.

## See also

- {doc}`../technical_details/images/registry_freshness` — how the
  control plane resolves tags to digests at acquire time.
- {doc}`../technical_details/images/cache_eviction` — operator-driven
  `xrlenv images evict` for proactive cleanup after a rebuild.
- {doc}`../technical_details/images/build_plan` — full build-plan
  schema, `xrlenv build calibrate`, and the `deploy/registry/build_and_push_images.py`
  rebuild workflow.
- {doc}`../deploy/multi_node_deployment/private_registry` — setting up
  the private registry that hosts the substrate image.
