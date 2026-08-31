# beagle — conventions for contributors (human & agent)

beagle is a PyTorch-like framework for agent-harness evolution, built on a vendored
`xrlenv` (`vendor/xrlenv`) that provides the containerized rollout infrastructure.
These are the hard conventions; follow them.

## 1. AVOID environment variables. Prefer flags/config. (load-bearing)

**Do not introduce new environment variables.** Anything a run needs should be an
explicit **flag or config field**, not a hidden env var — env vars sneakily hide
what a run actually uses. If a value can be passed as a flag, pass it as a flag.

**Allowed env vars (do not add to this list without permission):**

- The xrlenv-introduced ones already in use:
  `XRLENV_NODE_TOKEN`, `XRLENV_CONSUMER_TOKEN`, `XRLENV_OPERATOR_TOKEN`,
  `XRLENV_GRPC_HOST`, `XRLENV_GRPC_PORT`, `XRLENV_REGISTRY_STORAGE`,
  `XRLENV_MIRROR_REGISTRY_HOST`, `XRLENV_MIRROR_REGISTRY_PORT`,
  `XRLENV_PRIVATE_REGISTRY_STORAGE`, `XRLENV_PRIVATE_REGISTRY_HOST`,
  `XRLENV_PRIVATE_REGISTRY_PORT`, `XRLENV_PRIVATE_REGISTRY_HTTP_SECRET`,
  `DOCKERHUB_USER`, `DOCKERHUB_TOKEN`, `XRLENV_HOME`, `XRLENV_BENCHMARK_CACHE`
  (formerly `XRLENV_HARBOR_CACHE`, retired by xrlenv 2026-07-31).
- `XRLENV_GROUP_ID` — set by DarwinX from the typed `xrlenv_group_id` config field to
  scope a run's xrlenv registrations to a per-run group; consumed by the xrlenv runtime.
- **Credentials / secrets** (API keys, tokens) — these legitimately come from the
  environment.

**To introduce any other new env var, ask the maintainer first, with justification.**

## 2. Other established rules

- **Respect the original harness — honor its filesystem contract verbatim.** Rollouts
  flow through each benchmark's *native* harness via the xrlenv drop-in; the per-rollout
  artifacts (harbor's `<job>/<trial>/{agent,verifier,artifacts,...}`, verifier
  `reward.txt`) must be **byte-compatible** with upstream's output — never reshape into a
  house format. For harbor: use `harbor.Job.create(JobConfig(...))` + `job.run()` (the
  Job driver), **not** the low-level `SingleStepTrial` + a custom `run.json` — a known
  past failure. A trial must also carry
  `agent/trajectory.json` like a native harbor trial: **ATIF (harbor's Agent Trajectory
  Interchange Format) is beagle's canonical trajectory format** — every agent's native
  trajectory is converted to ATIF by a per-format converter (`beagle/benchmarks/trajectory.py`,
  one converter per format, M+N) and written **by the harness POST-JOB** (`HarborHarness.
  _emit_trajectories`), NOT in the in-trial shim: on the xrlenv cluster the native stream is
  synced from the container to the host only *after* the agent step, so the shim runs too early.
  Don't invent a house trajectory format; align to ATIF.
- **Agents are benchmark-agnostic — no per-benchmark prompt templates.** Do NOT couple an
  agent's system prompt to the benchmark. Keying a `.j2` per benchmark on `benchmark_name`
  is an N×M trap (a past failure): a new benchmark then has to touch every agent, and a new
  agent has to ship a template per benchmark. The benchmark supplies the task instruction; the
  agent runs it. Benchmark-specific framing, if any, lives on the benchmark side.
- **Experiment-copy rule.** Every evolvable agent's `AgentSource.repo` points at an
  experiment copy we own; the evolver pushes candidate branches there, never the
  canonical upstream repo.
- **Token usage: normalize into `Usage`, never fold cache away.** When onboarding a new agent
  harness, its usage parser MUST map the native stream into `beagle/agents/core/usage.py:Usage`
  (four disjoint buckets: `input_uncached`, `cache_read`, `cache_write`, `output`) and set
  `TaskResult.tokens = usage.to_token_counts()`. This keeps the cache split in `result.json` /
  `run.json` so a downstream cost estimate can price each bucket (beagle stays pricing-agnostic —
  no price sheet in-repo). Watch the **per-provider cache semantics**, which differ: OpenAI-shaped
  streams report `cached_tokens` as a **subset** of `prompt_tokens` (fresh = `prompt − cached`);
  Anthropic-shaped ones report cache read/write **in addition** to input. A naive uniform
  `+= cached` double-counts the OpenAI case. Invariant: `prompt = input_uncached + cache_read +
  cache_write`. (Legacy `prompt`/`completion`/`total` stay for back-compat.)
- **Green-before-next:** keep `pytest tests/` passing; add a test per change.
