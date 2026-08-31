# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is **stable** — invariants, architecture, principles, slow-changing project shape. For everything that moves with the code (current phase, what shipped when, what's next), read the live-state pointers below.

## Live state — read these (not this file)

If a fact in CLAUDE.md ever conflicts with one of these, **the live source wins**. CLAUDE.md describes the project's slow-changing shape, not point-in-time state.

- **What shipped recently**: `git log --oneline -20` and `git diff main...HEAD` on the current branch.
- **Internal sub-package layout**: walk the filesystem (`tree -L 2 xrlenv xrlenv_plugins`); only the top level is in this file.
- **Test suite**: `.venv/bin/python -m pytest -q` (the `pyproject.toml` `testpaths` includes `xrlenv_plugins`); `.venv/bin/python -m mypy` for strict type-check; `.venv/bin/python -m ruff check` for lint.
- **Sphinx docs site**: `uv pip install -e '.[docs]' && .venv/bin/sphinx-build -W -b html docs docs/_build/html`. Audience: end users + external developers.

## What XRLEnv is

Infrastructure for **agentic RL training**. Two cores:

1. **Sandboxing** — Docker (phase 0, universal) and CubeSandbox microVM (phase 2, Linux+KVM only) backends, plus a Function-Call execution mode (phase 1).
2. **Orchestration** — control plane + per-node agents managing thousands of concurrent long-horizon rollouts across cloud VMs (GCP + AWS, manual provisioning) and a local laptop.

It is **not** a trainer, not a model server, not a generic code interpreter. The policy lives trainer-side; the SDK calls whatever `policy.act(obs)` callable the user supplies.

## Architecture (three-plane split)

```
Trainer plane   (GPU host, runs policy, consumes trajectories)
       │   gRPC bidi-stream (rollout RPC)
Control plane   (single Python process in phase 0: scheduler, registry,
       ▲          capacity estimator, template catalog, state store, admin UI)
       │   outbound bidi gRPC stream from each node (spec 21)
       │   no inbound listener on the node side (invariant 7)
Data plane      (xrlenv-node daemons running BackendAdapters)
       │   in-sandbox stub protocol (uds / vsock)
Sandboxes       (Docker containers, CubeSandbox microVMs)
```

Trainer plane knows RL but nothing about sandboxes. Data plane knows sandboxes but nothing about RL. Control plane is the only thing that knows both, only in the narrow shape "schedule this template, return a rollout session." Don't violate this split.

## Top-level layout

Only the slow-changing top level is here. Sub-package detail (per-module roles inside `xrlenv/control/`, `xrlenv/node/`, etc.) belongs in the filesystem — read it directly when you need it.

```
xrlenv/              core platform (control plane, node agent, SDK, backends, observability, admin)
xrlenv_plugins/      benchmark + EnvAdapter plug-ins (PEP-420 namespace package; never inside xrlenv/)
specs/               00–21 design specs (the design source of truth)
deploy/              bootstrap/refresh/bring-up + systemd units; registry/ (3 registry servers + ops scripts), node/ (provisioning scripts)
nodes.yaml           operator inventory (loader at xrlenv/control/nodes_yaml.py)
docs/                Sphinx site (user + external-developer docs)
notes/               internal phase-gate docs + audit/rebuttal cycle (audit.md/rebuttal.md gitignored)
tests/unit/          unit tests (plus xrlenv_plugins/**/tests/ for plug-ins)
examples/            build-plans/, deployment_run_book/, nodes.yaml.example
README.md            developer entry point (dev setup + workflow; points users to the Sphinx docs)
```


## Critical design rules to never violate

These are the load-bearing design invariants. Most past mistakes in this design came from forgetting one of them:

1. **Sandbox identity ≠ rollout identity.** Separate columns, separate lifecycles. Phase 0 destroys the sandbox at rollout finish, but no code may assume `sandbox_id == rollout_id` — spec 18 sessions break that equation.
2. **Capacity is released only on node-confirmed destroy.** "Destroy enqueued" is not "slot free."
3. **Trajectories are immutable after seal.** Late reward updates write a new record set, never mutate.
4. **Template manifests are immutable for the duration of a training run** — pinned by `(name, version, digest)` at run start.
5. **Outbound-only node transport.** Never add an inbound listener on the node-agent without an explicit phase note + spec 04/07/09 update.
6. **State store holds metadata; blobs live on disk or object store.** Trajectory bodies, snapshot artifacts, image layers never go into SQLite/Redis.
7. **`task_key` is fairness; `instance_id` is identity.** Don't conflate them.

Plus these cross-cutting principles:

- **Mechanism not policy.** XRLEnv core ships primitives (`task_key`, `group_id`, `cancel_rollout`, `cancel_group`, anti-affinity, `max_runs_per_task`). Engine-specific over-request / filter / cancel loops live in the trainer adapters (`xrlenv/adapters/{slime,verl}.py`), never in core. Slime and verl have fundamentally different patterns; forcing one shape warps the others.
- **Design-first workflow.** Discuss design space and iterate before coding. When a fork in the road appears, surface it (e.g. via `AskUserQuestion`) rather than committing.
- **Don't reinvent benchmark-side wheels.** Benchmark plug-ins MUST delegate to upstream's published API (Python or filesystem-contract) for parsing, grading, and report generation. If upstream publishes `get_eval_report` / `MAP_REPO_TO_PARSER` (swebench), call them — don't carry your own resolution rule. If upstream publishes only a filesystem contract (harbor's `reward.txt` / `reward.json` at `/logs/verifier/`), honor it verbatim — don't make up your own format. The per-rollout verifier dir should be byte-compatible with what upstream's harness emits, so upstream report aggregators consume it without translation. If you find you genuinely *can't* delegate, surface it before working around — it likely means xrlenv core is missing the right hook (or is poorly designed for this case), not that the plug-in needs its own parser. Past failure: an earlier swebench-verified iteration carried `compute_resolved_status` + `build_report` helpers; they drifted silently when swebench v4 changed the per-repo parser signature, surfacing as a multi-layer operator-reported regression chain.
- **Concurrency is a trigger, not a cause.** When a failure appears only under high concurrency, it is *surfacing* a latent problem — an xrlenv bug (race, oversubscription, capacity-accounting error, admission-queue hang) OR a benchmark non-hermeticity (an external dependency). The fix is ALWAYS the underlying problem, NEVER lowering concurrency: lowering it hides the problem, it doesn't fix it. The admission queue is designed to gate load regardless of *requested* concurrency, so any level (4, 100, or arbitrarily high) must be safe. Do not touch the gate profile's concurrency or a benchmark's `workers` to make a red run go green. The discipline is: **reproduce the failing task in isolation, then identify WHICH underlying problem the concurrency surfaced.** Real xrlenv bug (2026-08): at high concurrency, over-cap acquires HUNG — fair-share `effective_cap` + a 24 h default queue timeout instead of a 240 s fail-fast; the fix was the admission path, not fewer rollouts. Counter-example, SAME triage but NOT an xrlenv bug: `psf__requests` swebench oracles returned an nginx `503` only under overlap yet passed in isolation — that traced to a NON-hermetic external `httpbin.org` rate-limit (HTTPBIN_URL unset, no local httpbin, no egress proxy), so the fix is hermeticity (a local httpbin), still never lowering concurrency. Isolation-testing to tell an xrlenv bug from a non-hermetic dependency (or a deterministic upstream/content failure) is the required first step before proposing any fix.

## Phases (capability progression, not feature gating)

These are *capability tiers*, not commit-tied feature flags. Don't pull a higher-phase feature into a lower-phase slice unless asked. The current phase + active slice are tracked outside this file.

- **Phase 0** — Docker, gym/step API, EnvAdapter layer, sqlite, GCP+AWS reserved VMs, read-only admin panel, static + online-refined capacity estimator, trainer-agnostic SDK with group/batch primitives. First benchmark to onboard end-to-end: **terminal-bench-2** (smaller per-task images than SWE-bench-Lite; simpler shell-driven harness exercises Pattern A + the in-sandbox stub without a complex VM). SWE-bench-Lite + OSWorld onboard after terminal-bench-2 has demonstrated the platform can carry a real benchmark.
- **Phase 1** — Redis StateStore (500-1k concurrent target), image-distribution architecture (D18/D19/D20 — three-mode `image_pin_mode`), **Slime + verl adapters**, SWE-bench Verified as the canonical phase-1 benchmark (oracle-driven gate), warm pools, allowlist egress, lazy image loading (eStargz Docker-only), external plug-in package mechanism (B11), Function-Call mode, mTLS hardening.
- **Phase 2** — CubeSandbox / microVM backend (image story is genuinely different from Docker), mixed-backend capacity accounting, overlaybd lazy-load image format, k8s/MIG/ASG autoscale, sandbox checkpoint/branch, durable trajectory store, predictive capacity, multi-tenant isolation. Example workloads: hours-long research agents at scale.
- **Phase 3** — sandbox sessions with preemption-safe resume (spec 18); aspirational extreme density (KSM, overcommit, 3FS).

The agent benchmarks listed per phase are *example workloads we'll exercise the platform with*, not features that gate the phase from shipping.

## Trainer integration

Slime is the **primary** trainer target; verl is **secondary**. Both ship in phase 1 as adapters in `xrlenv/adapters/`. Both are optional imports — neither is a runtime dependency of the core SDK. The phase-0 SDK shape (hard-deadline, partial-trajectory return, batch helper, group primitives) is designed so that the phase-1 adapters are thin shims.

## Cloud constraints

Deployment assumes **VM-only access** on GCP and AWS — no admin, no Terraform, no managed instance groups, no ASG autoscale until phase 2. Deployment is shell scripts run by hand on freshly provisioned VMs (`deploy/bootstrap-{gcp,aws}.sh`), plus a static `nodes.yaml`. AWS supports both Amazon Linux 2023 and Ubuntu 22.04 (bootstrap branches on `/etc/os-release`).

## Platform quirks worth knowing

- **macOS / Docker Desktop**: the host↔VM bridge does not route uds connections through bind-mounts. The Docker backend auto-detects platform and uses TCP transport (published port on `127.0.0.1`) on Darwin; uds on Linux per spec 01. Configured via `DockerBackendConfig.stub_transport`.


## Specialized agents available

`.claude/agents/` defines two project-scoped subagents that should be used proactively when their description matches:

- **qa-test-engineer** — invoke after a logical chunk of new code lands or an existing module's behavior is modified (new functions/classes, refactored public methods, bug fixes, new branches added to control-flow), and at PR-prep time before requesting review. Python/pytest-focused; writes unit tests for happy paths, edge cases, and error paths, runs the relevant suite to check for regressions, and reports a coverage summary. Do **not** invoke for pure-documentation/comment edits, formatting-only changes, spec edits under `specs/`, throwaway scripts under `examples/` that aren't part of the test surface, or before any implementation code exists in the repo (i.e. while still in design-only state).
- **sphinx-docs-writer** — invoke when new features, modules, or public APIs are added; when an existing public API's signature/behavior changes; or when the user explicitly asks for documentation. Maintains the Sphinx site under `docs/`, keeps API references in sync with code, adds runnable sample use cases for new features, and authors agent-readable summaries. Do **not** invoke for internal refactors with no public-surface change, for spec-only edits under `specs/` (those are design docs, not Sphinx output), or for changes that only touch `notes/` (internal-only).
