# Quick start — evolve a harness on terminal-bench 2.1

Evolve one harness (**opencode**, θ) with **cursor-agent** as the proposer, on **1 task** of
terminal-bench 2.1, for **1 DarwinX iteration**. It spends (cursor + cluster + your model's API
key), so gate with a dry-run first.

```bash
beagle evolve --config examples/quick-start/config.yaml --dry-run  # preview: print the plan, no spend
beagle evolve --config examples/quick-start/config.yaml            # launch the evolution loop (spends)
```

That's the **canonical** way — a single `config.yaml` driven by the generic CLI (the sibling
`examples/evaluation/` runs *pure eval* the same way: `beagle evaluate --config …`).

> **Prefer editing Python?** `quick_start_inline.py` builds the same run from Python constants at the
> top of the file (reading the evolvee's repo/ref from the onboard manifest at runtime) —
> `python examples/quick-start/quick_start_inline.py` (`--dry-run` to preview). Same run, different surface.

## The `config.yaml`

```yaml
run:      {dir, name, runtime, parallelism}
evolvee:                                    # θ — the harness under evolution
  harness: {name, version, source}          # harness/adapter type + version + INLINE source (repo/ref/…)
  model:  {name}
  provider / forward_env                    # this agent's LLM routing (provider id + key to forward)
  effort / max_turns / timeout / extra_args # agent-level knobs (extra_args = the agent's CLI)
evolver:  {harness: {name, version}, model} # the proposer (cursor-agent)
algorithm: {name: darwinx, hparams: {…}}    # the optimizer + its typed knobs
data:     [{benchmark, tasks}]              # the benchmark + tasks to evolve on
```

### Where `evolvee.harness.source` comes from — the onboard artifact

Don't hand-write it: it's the onboard artifact `python -m beagle.tools.onboard` wrote at
`.beagle/agents/<profile>.json`. For opencode — `.beagle/agents/opencode_v1.18.16.json`:

```json
{ "profile": "opencode_v1.18.16", "version": "1.18.16",
  "repo": "https://github.com/<your-org>/opencode_v1.18.16",
  "ref": "<your copy's baseline commit>", "branch": "baseline", "token_env": "GH_TOKEN",
  "upstream": "https://github.com/anomalyco/opencode", "upstream_ref": "a3647eb…",
  "dir": "../beagle-experiments/opencode_v1.18.16" }
```

`ref` is your copy's baseline commit (the single orphan seed a run clones); `upstream_ref` is the
upstream commit it's a tree of; `branch` is where it lives (the `--branch-name`, default `baseline`). Its `repo`/`ref`/`token_env`/`dir` paste
**1:1** into `evolvee.harness.source.*`. You add
`harness.name` (the CLI/adapter that runs it — `opencode`) and `harness.version` (a label).
`harness.name` resolves to the registered adapter (exact, else longest-registered-prefix, so
`cursor-agent` → `cursor`).

### Version gate

For black-box agents (cursor-agent, …), `agent.version` is checked against the **installed** version
before any spend, and the run **fails loud** on a mismatch (`cursor-agent --version` vs your pin).
opencode is exempt — versioned by `source.ref`, not an installed binary.

### Task text vs. prompt framing

beagle ships **no** prompt framing. It hands the agent a **data payload** — the raw task plus any
benchmark **data hooks** (`additional_info_pre` / `additional_info_post`; SWE-bench uses a post-hook
for `hints_text`) — and the agent supplies its own system prompt + generic instruction (in its
source, where the evolver can see them).

## Prerequisites (verify once)

- **`.env`** at the repo root with the cluster facts + your model's API key. Export your provider
  key (e.g. `OPENAI_API_KEY`) and list it in the agent's `forward_env`; `GH_TOKEN` exported (fetch θ,
  clone the experiment copy, push candidate branches). `echo ${GH_TOKEN:+set}` → `set`.
- **cursor-agent installed + logged in** (`cursor-agent --version` works; the version gate checks it).
- **The evolvee checkout exists** at `source.dir` — DarwinX links it in under `<run_dir>/`.

## Where things land

`<run.dir>/<run.name>/` — the genealogy DB, the emitted campaign config, per-iteration logs,
the evolvee clone, and `_evals/` (per-eval scratch, kept out of the vendored tree).
The candidate branch is pushed to your experiment copy's `origin` as `evolve/<parent>__<pipeline>`
(`git -C <checkout> ls-remote origin 'refs/heads/evolve/*'`).

> A **0-token** task means the agent never actually ran (clone/install/auth failure) — check the
> per-task `error` + the trial's `agent/` tree, not the score.
