# Onboarding an agent harness

End-to-end runbook for adding a new agent to beagle. [docs/advanced.md](advanced.md#onboard-your-own-agent)
covers the *concept* (capabilities, `@register`, auto-discovery) in a page; this is the operational
checklist — the contracts an adapter must honor, and the traps that have actually cost us runs.

beagle **never vendors an agent's code.** You add a thin adapter; the agent itself lives in a git
repo you own (the *experiment copy*) and is cloned into the task container per trial. The adapter is
typically 200-400 lines.

## 0. Answer five questions first

Do this before writing code. If you can't answer #1 or #2, the agent cannot be onboarded as a
`Runnable` yet, and you'll discover that 300 lines in.

| # | Question | Why it decides the design |
| --- | --- | --- |
| 1 | **How does it run headless?** A single non-interactive command, one task, exits on its own? | No headless mode → no `run_in`. A TUI-only agent needs an upstream flag first. |
| 2 | **How does it reach an LLM?** Env var, config file, CLI flag? Can the base URL be overridden? | Decides whether gateway/proxy routing is expressible. An agent that hardcodes a vendor endpoint can't use `provider: gateway`. |
| 3 | **What does it install from?** pip/npm/bun/cargo, which registries, which language runtime? | Becomes `install()` and `install_hosts()`. A container's system Python is often too old. |
| 4 | **What does it emit?** A trajectory/log file, in what format, with token usage in it? | Becomes the ATIF converter and the `Usage` parse. No usage in the stream → no cost accounting. |
| 5 | **What's the evolvable surface?** Prompt YAML, a source tree, a config file? | Becomes `AgentSource.entrypoint` — what an evolver edits. |

Two answers are worth writing down verbatim: the exact headless command line, and the exact shape of
the usage numbers (are cached tokens a *subset* of input, or reported *in addition*? — see §4.2).

## 1. Onboard the source

Stand up the experiment copy — a repo **you own**, seeded with the upstream tree at a pinned commit.
The evolver pushes candidate branches there; upstream is never written to.

```bash
python -m beagle.tools.onboard \
    --upstream https://github.com/<upstream>/<agent> --ref <40-char-sha> \
    --repo "$YOUR_ORG/<agent>_<version>" --private --version <version> --branch-name baseline \
    --dir ../beagle-experiments/<agent>_<version> --profile-name <agent>_<version>
```

Pin a **full 40-char SHA**, not a branch — a branch drifts, so a re-onboard silently snapshots a
different baseline. This writes `.beagle/agents/<profile>.json`, the pointer everything downstream
reads. Add the invocation to [`scripts/onboard_all_agents.sh`](../scripts/onboard_all_agents.sh).

If the clone is large, consider a `--prune` profile (see [opencode-prune.md](opencode-prune.md)) —
it only ever removes whole paths, so kept blobs stay byte-identical and evolution diffs still apply
to upstream.

## 2. Write the adapter

One package: `beagle/agents/<your_agent>/__init__.py`. Start from
[`beagle/agents/core/_template.py`](../beagle/agents/core/_template.py). Auto-discovered on
`import beagle` — no registration list to edit.

Compose the capabilities the agent actually has:

| Mixin | Implement | Gives you |
| --- | --- | --- |
| `Runnable` | `install` + `run_in` | can attempt benchmark tasks (evolvee) |
| `Evolvable` | `_default_source` | versioned `repo@ref` source = θ |
| `Editor` | `edit` | can drive a workspace (evolver) |

Evolvee = `Runnable` + `Evolvable`. A closed-source CLI is `Editor` only, with
`transparency = Transparency.BLACK_BOX`.

### The two-phase lifecycle — implement `install` + `run_in`, not `run`

This is the part the conceptual docs gloss. A **network-phased** harness (pier/harbor) opens the
network for a trusted install, then locks it down to the LLM endpoint for the run. So split:

```python
def install(self, handle, task_ctx, *, runtime: ContainerRuntime) -> None:
    """INSTALL phase — network open. Clone repo@ref, build. Raise AgentInstallError to fail loud."""

def run_in(self, handle, task, task_ctx, *, runtime) -> TaskResult:
    """RUN phase — egress restricted to network_hosts(). Caller owns the container; no acquire/destroy."""
```

Inherit the default `run()` ([base.py:190](../beagle/agents/core/base.py)) — it composes the two for
always-open harnesses (docker drop-in) *and* records the harbor-shaped per-phase timing
(`environment_setup` / `agent_setup` / `agent_execution`). Overriding `run()` yourself throws that
away. Need a tweak to the container acquire (e.g. clearing an image `ENTRYPOINT` so `sleep infinity`
runs)? Override `_acquire_run_args()`, not `run()`.

**`runtime.exec` is `check=False`.** Every step you don't check is a silent failure that reads
downstream as a benign "scored 0". Check each one and raise `AgentInstallError` with a tail of
stderr.

### Cloning a private experiment copy

Use the shared helper — it handles the token URL rewrite and fetch-by-SHA:

```python
from beagle.rollout.runtime.transport import GitClone, clone_with_retry

clone = GitClone(repo_url=src.repo, ref=src.ref or "", container_path="/agent", token_env=token_env)
cloned = clone_with_retry(runtime, handle, clone, env=clone_env or None, timeout=600)
```

`clone_with_retry` exists because concurrent trials cloning one private repo trip GitHub's auth
throttle as a spurious 401; it backs off with jitter while definitive errors still fail fast.

### Language runtimes

Don't assume the container's interpreter. Task images pin their own (SWE-bench images ship Python
3.9). Stand up your own under `/agent` — `mini_swe` uses `uv` to create a managed 3.11 venv so the
*orchestrator* runs modern while the agent's tool-calls still execute in the task's own environment
([mini_swe/\_\_init\_\_.py:64](../beagle/agents/mini_swe/__init__.py)). Keep everything you install
out of the task workspace so it can never land in the agent's diff.

## 3. Honor the six contracts

These are not optional, and five of the six exist because of a specific past failure.

### 3.1 Timeout — never hardcode a wall clock

```python
from beagle.agents.core.base import resolve_agent_timeout
timeout = resolve_agent_timeout(cfg, task_ctx)
```

The benchmark's declared budget wins; `agent.timeout` in the run config applies only when nothing
was declared. There is deliberately **no house default** — an unstated budget raises
`AgentBudgetUndeclared` rather than inventing a number. Four adapters once each hardcoded 1800 s
while SWE-rebench tasks declared 3000 s and 6000 s, silently truncating every long task.

### 3.2 Token usage — normalize into `Usage`, never fold cache away

Parse the native stream into [`Usage`](../beagle/agents/core/usage.py) — four **disjoint** buckets —
and set `TaskResult.tokens = usage.to_token_counts()`.

**Per-provider cache semantics differ, and getting this wrong double-counts:**

| Native shape | Cached tokens are… | Fresh input = |
| --- | --- | --- |
| OpenAI-shaped (`prompt_tokens`, `prompt_tokens_details.cached_tokens`) | a **subset** of `prompt_tokens` | `prompt_tokens - cached_tokens` |
| Anthropic-shaped (`cache_read_input_tokens`, `cache_creation_input_tokens`) | **in addition to** `input_tokens` | `input_tokens` as-is |

Invariant: `prompt == input_uncached + cache_read + cache_write`. beagle stays pricing-agnostic — it
keeps the split so a downstream estimate can price each bucket. A parse miss must never fail the
run: return `({}, 0)` and let the patch stand.

### 3.3 Trajectory — convert to ATIF, post-job

**ATIF is beagle's canonical trajectory format** (harbor's Agent Trajectory Interchange Format).
Don't invent a house format. Register one converter per *format*, not per agent×harness — that's
what keeps this M+N:

```python
# beagle/benchmarks/trajectory.py
@register_converter("my-agent-json")
def _my_agent_to_atif(logs_dir: Path, *, instruction: str, agent_name: str,
                      agent_version: str, model_name: str | None): ...
```

Point `TaskResult.trajectory` at it: `TrajectoryRef(path=Path("my.traj.json"), format="my-agent-json")`.

Have the agent write its native stream into `/logs/agent` — pier/harbor sync that to the trial's
`agent/` directory. **Conversion happens POST-JOB** in `HarborHarness._emit_trajectories`, never in
the in-trial shim: on the cluster the native stream is synced out of the container only *after* the
agent step, so a shim-time converter runs too early and finds nothing.

### 3.4 Patch capture — record the base commit first

Capture `git rev-parse HEAD` **before** the agent runs, then commit whatever it left and diff
`base..HEAD`:

```python
base = runtime.exec(handle, ["bash", "-lc", f"{pre}cd {repo} && git rev-parse HEAD 2>/dev/null || true"]).stdout.strip()
# ... agent runs ...
# git add -A && git commit; then: git diff <base>..HEAD
```

This covers both grader styles: a working-tree grader (swe-bench reads `TaskResult.patch`) and a
`base..HEAD` grader (deep-swe/pier's `verifier.collect`). Agents that commit their own work produce
an *empty* post-run working-tree diff — that's the failure this prevents.

Prepend `task_ctx.shell_preamble` before every `cd` so the benchmark's own environment setup
(activating the task's Python env, etc.) is in scope.

### 3.5 Egress — declare both host sets

```python
def install_hosts(self) -> list[str]:   # git host + package registries, for the INSTALL phase
def network_hosts(self) -> list[str]:   # the LLM endpoint, for the RUN phase
```

A filtered-egress benchmark allowlists exactly these. Scheme-qualify entries
(`https://api.example.com`): pier urlparses each one, and a bare hostname has no `.hostname`, so it
drops out of the allowlist silently.

### 3.6 Provider routing — use the typed route

Read the route with `provider_config(cfg)` and declare what your adapter can honor via
`supported_provider_types` (default: `{"direct", "gateway", "internal"}`); preflight and runtime
enforce the same set. See [provider-routing.md](provider-routing.md).

Credentials must **not** ride the command line — argv is visible to every process in the container
via `ps`, and beagle echoes commands into runtime records, error tails, and test output. Pass secrets
through the exec's environment and, if the agent needs them in a file, materialize it with a
`umask 077` heredoc.

## 4. Keep the agent benchmark-agnostic

**No per-benchmark prompt templates.** Do not key a template on `benchmark_name`. That's an N×M trap:
a new benchmark then has to touch every agent, and a new agent has to ship a template per benchmark.
The benchmark supplies the task instruction; the agent runs it. Benchmark-specific framing, if any,
lives on the benchmark side.

One `run` works on every harness — you never write harness-specific code in an adapter.

## 5. Wire it up

1. **Onboard entry** — add the invocation to `scripts/onboard_all_agents.sh` (§1).
2. **Eval configs** — add an entry to `AGENTS` in
   [`scripts/generate_eval_configs.py`](../scripts/generate_eval_configs.py), keyed by your
   `harness.name`, listing the `versions` to generate for. **The join key is `version`** — it must
   match the `--version` you passed to `onboard`, which is how a generated config finds its manifest.

   ```python
   "my-agent": dict(
       versions=["v1.2.3"],
       extra_args={"my_agent_args": ["--headless"]},
       note="what a reader needs to know about these args",
   ),
   ```
3. **Run the smoke** for one benchmark before anything wider.

## 6. Test it

Add `tests/unit/test_<agent>.py`. The existing agent tests
([`test_mini_swe.py`](../tests/unit/test_mini_swe.py), [`test_opencode.py`](../tests/unit/test_opencode.py))
show the pattern: a fake runtime records `exec` calls, and assertions run against the composed command
string and the parsed `TaskResult` — no container needed. Cover at minimum:

- the headless command is composed correctly (model, task, config, output path);
- usage parsing, **including the cache split** — assert the invariant, and assert an OpenAI-shaped
  fixture isn't double-counted;
- a failed run that produced no patch surfaces as `FAILED` with a diagnostic tail;
- a trajectory-parse miss degrades to `({}, 0)` instead of raising;
- the ATIF converter round-trips a native fixture (see [`test_trajectory.py`](../tests/unit/test_trajectory.py)).

## Checklist

- [ ] Experiment copy onboarded at a pinned full SHA; `.beagle/agents/<profile>.json` exists
- [ ] `scripts/onboard_all_agents.sh` entry added
- [ ] Adapter registered with `@register`, capabilities composed, `transparency` set
- [ ] `install` + `run_in` split (not a monolithic `run`); every `exec` return checked
- [ ] `resolve_agent_timeout` used — no hardcoded wall clock
- [ ] Usage normalized into `Usage`; cache semantics verified against the native format
- [ ] ATIF converter registered; native stream written to `/logs/agent`
- [ ] Patch captured as `base..HEAD`; `shell_preamble` honored
- [ ] `install_hosts()` + `network_hosts()` declared, scheme-qualified
- [ ] No secrets in argv; no per-benchmark prompt templates
- [ ] `generate_eval_configs.py` `AGENTS` entry, version matching the manifest
- [ ] Unit tests added; `pytest tests/` green
- [ ] Smoke run passes on one benchmark

## Traps

| Trap | Symptom |
| --- | --- |
| Hardcoded timeout | Long tasks truncated; looks like the agent gave up |
| Unchecked `runtime.exec` | Silent install failure reads as "scored 0" |
| Post-run working-tree diff | Empty patch for any agent that commits its own work |
| Converting the trajectory in the shim | `trajectory.json` missing on the cluster, present locally |
| `+= cached` on an OpenAI-shaped stream | Input tokens double-counted |
| Bare hostname in an allowlist | Egress silently blocked under filtered egress |
| Template keyed on `benchmark_name` | N×M coupling; every new benchmark touches every agent |
| Secrets in argv | Credential visible via `ps` and echoed into logs/test output |
