# Bring-your-own-Dockerfile — build-on-demand

A template can supply a **Dockerfile + build context** instead of a prebuilt
image reference. When the first rollout that needs the image runs, the platform
builds it on the node that picks up the work, pushes the result to the scratch
registry, and every other node pulls it from there over the LAN. No operator
pre-build step is required.

This is the `image_build:` block in a template manifest.

## Why use `image_build:`

Use `image_build:` when:

- You are iterating on a custom environment Dockerfile and want to roll it out
  without waiting for an operator to build and push it.
- The benchmark ships a Dockerfile rather than a prebuilt image and you own the
  build (for operator-managed bulk builds, see {doc}`/deploy/multi_node_deployment/private_registry`).
- You want content-addressed build dedup across the fleet: the same
  Dockerfile+context pair is built exactly once, no matter how many concurrent
  rollouts request it.

## The `image_build:` block

Add an `image_build:` block to your template manifest in place of the `image:`
key. Exactly one of `context:` (a local directory) or `git:` (a repo-hosted
context) must be present.

```yaml
# template.yaml — bring-your-own-Dockerfile (local context)
name: my-env
version: "1.0"

image_build:
  context: ./environment        # local directory holding a Dockerfile + build files
  dockerfile: Dockerfile        # optional; defaults to "Dockerfile"
  build_args:
    PYTHON_VERSION: "3.11"      # optional; folded into the content-addressed tag
  durable_to: "reg.mycorp.internal:5000/team/my-env"  # optional; omit → scratch-only
```

```yaml
# template.yaml — bring-your-own-Dockerfile (git-hosted context)
name: my-env-git
version: "1.0"

image_build:
  git:
    repo: "https://github.com/yourorg/yourrepo"
    ref: "abc1234"          # commit sha preferred over a branch for reproducibility
    subdir: "docker/env"    # path within the repo that is the docker build context
    dockerfile: "Dockerfile"
  build_args:
    FOO: bar
  durable_to: "reg.mycorp.internal:5000/team/my-env-git"
```

Setting `image_build` implies the build-on-demand path (`image_pin_mode:
scratch_build`). You do not hand-edit `image_pin_mode`.

### Field reference

| Field | Required | Default | Description |
|---|---|---|---|
| `context` | one of `context`/`git` | — | Local directory holding a Dockerfile and build context. The platform tars it, content-addresses it, and ships the bytes to the build node. |
| `git` | one of `context`/`git` | — | Repo-hosted build context. The node clones the repo instead of receiving a tarball. See **`git:` sub-fields** below. |
| `dockerfile` | no | `"Dockerfile"` | Dockerfile filename within `context`. Ignored when `git:` is set; use `git.dockerfile` there. |
| `build_args` | no | `{}` | `--build-arg` key/value pairs. A changed build-arg forces a distinct image (it is folded into the content-addressed tag). |
| `durable_to` | no | unset | User-owned registry endpoint (`host:port/repo`) reachable over the LAN. When set, the built image is copied there digest-preserved and survives scratch GC. When omitted, the image lives only in the scratch registry. |
| `tag` | no | content-addressed | Override the scratch tag. Omit this in almost all cases — the content-addressed default is what guarantees build-once and no drift. |

**`git:` sub-fields:**

| Field | Default | Description |
|---|---|---|
| `repo` | (required) | Git URL — `https://...` or `git@...`. |
| `ref` | `"main"` | Branch, tag, or commit sha. **Pin a commit sha** for true reproducibility — a moving branch does not change the content-addressed tag when the branch advances. |
| `subdir` | `"."` | Path within the repo that is the docker build context. |
| `dockerfile` | `"Dockerfile"` | Dockerfile filename within `subdir`. |

## Content-addressing — built once, drift-free

The platform computes a deterministic **input digest** over the build inputs
before any build happens:

- For `context:` — a recursive sha256 over every file in the directory tree
  (path, executable bit, and content), sorted for stability.
- For `git:` — a sha256 over the tuple `(repo, ref, subdir, dockerfile)`. No
  clone is needed to compute this.
- Plus the effective Dockerfile name and any `build_args`.

The result is used as the scratch tag:

```
<scratch-host>:5012/scratch/<input_digest>
```

**Same Dockerfile+context ⇒ same `input_digest` ⇒ built exactly once for the
whole fleet, reused across runs** until GC reclaims it. Two nodes needing the
same image do not trigger two builds — the second node simply pulls from the
scratch registry after the first node pushes.

Unlike `per_node_local` mode — where each node builds independently and images
can diverge (apt mirror skew, `latest` base drift, timestamps) — `image_build:`
builds exactly one set of bytes and distributes it by pull.

## Durability — the `durable_to:` option

By default the built image lives only in the scratch registry (`:5012`) and may
be GC'd at any time by the operator's GC schedule. The platform emits a warning
at rollout submit when `durable_to:` is not set:

> `image_build` for `<template>` has no `durable_to`; the built image lives in
> the scratch registry and may be reclaimed by GC at any time (quota / TTL).
> Supply `durable_to: <your-registry>` to keep it.

To keep the image across GC passes, supply a registry endpoint you own and the
fleet can reach over the LAN:

```yaml
image_build:
  context: ./environment
  durable_to: "reg.mycorp.internal:5000/team/my-env"
```

After the build, the platform copies the image from scratch to your registry
**digest-preserved** (using `crane cp` or `skopeo copy`). The platform never
GCs your durable registry — you own its lifetime and quota.

The image is also available at `reg.mycorp.internal:5000/team/my-env` for
direct pulls, separate from the scratch ref.

:::{tip}
On a shared-FSx cluster you can stand up a small personal registry (a
`registry:3` container with a personal FSx subdirectory) and use that as your
`durable_to:` target. See {doc}`/deploy/multi_node_deployment/private_registry`
for the registry setup pattern — the same approach applies.
:::

## What happens at rollout time

1. **Register.** The control plane computes `input_digest` from the template
   manifest when you register the template. No build happens yet.
2. **First acquire.** The scheduler picks a node. That node checks whether the
   scratch tag already exists in `:5012` (a registry HEAD probe). If it does —
   because a previous run already built it — the node pulls it.
3. **Build.** If the image is absent, the node takes a build lease (singleflight:
   200 concurrent first-acquires of the same task trigger **one** build, not 200)
   and runs `docker build` against the context. The result is pushed to
   `<scratch-host>:5012/scratch/<input_digest>`.
4. **Manifest digest pin.** The control plane freezes the resolved manifest
   digest (`@sha256:M`) for the duration of the run. All subsequent nodes pull
   `@sha256:M` — not the tag — so no node can get a different image mid-run.
5. **`durable_to:` copy.** If `durable_to:` is set, the platform copies the
   image to your registry immediately after the build, before any rollout starts.

## Known limitations

- **Large local contexts (>128 MiB) take longer.** A large `context:` is
  shipped inline over gRPC using chunked transfer. For big contexts, prefer
  `git:` (the node clones rather than receiving bytes) or an FSx-staged context.
  This is an operational consideration, not a correctness issue.
- **Moving `git.ref` values weaken content-addressing.** A branch name like
  `main` produces a stable `input_digest` across branch advances — advancing the
  branch does **not** trigger a rebuild. Pin a commit sha for true
  reproducibility.
- **No SDK-time build overrides yet.** The `image_build:` block is template-level
  only. Passing a build context dynamically at SDK call time is a planned
  follow-on.

## Minimal working examples

### Local Dockerfile directory

```
my-project/
  template.yaml
  environment/
    Dockerfile
    requirements.txt
```

```yaml
# template.yaml
name: my-py-env
version: "1.0"

image_build:
  context: ./environment
  build_args:
    PY: "3.11"
```

```dockerfile
# environment/Dockerfile
FROM python:${PY}-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
```

Register and run — no operator build step needed:

```python
import asyncio
from xrlenv import Client

async def main():
    async with Client.in_process() as client:
        ref = await client.register_template("template.yaml")
        async with client.acquire_container(ref) as ctr:
            result = await ctr.exec(["python", "-c", "import numpy; print('ok')"])
            print(result.stdout)

asyncio.run(main())
```

### Git-hosted context with durability

```yaml
# template.yaml
name: my-git-env
version: "1.0"

image_build:
  git:
    repo: "https://github.com/yourorg/ml-envs"
    ref: "d3adb33f"          # pin a commit sha
    subdir: "envs/cuda"
    dockerfile: "Dockerfile.cu121"
  build_args:
    CUDA_VERSION: "12.1"
  durable_to: "<registry-host>:5099/team/cuda-env"
```

## See also

- {doc}`/deploy/multi_node_deployment/scratch_registry` — operator guide for
  standing up the scratch registry (`:5012`), pointing the control plane at it,
  and running the GC.
- {doc}`on_demand` — the general acquire flow that `image_build:` plugs into.
- {doc}`build_plan` — the proactive image placement path for known image sets
  (complement to `image_build:` for one-off or iterative Dockerfiles).
- {doc}`/deploy/multi_node_deployment/private_registry` — the durable,
  operator-managed alternative to `durable_to:` for images that need to be
  shared across many users or long-lived campaigns.
