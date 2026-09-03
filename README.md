<h1 align="center">
  <img src="docs/assets/beagle-logo.png" alt="beagle" width="72" valign="middle">
  beagle
</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-%3E%3D3.12-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/license-Apache%202.0-10B981" alt="License"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/installer-uv-DE5FE9?logo=uv&logoColor=white" alt="uv"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/status-0.0.1-6366F1" alt="status"></a>
</p>


<p align="center">
  <a href="#setup">Setup</a> ·
  <a href="#evaluate">Evaluate</a> ·
  <a href="#evolve">Evolve</a> ·
  <a href="#more">More</a>
</p>

<table width="100%">
<tr>
<td width="50%" valign="top">
<h3>Evaluate</h3>
<p>Score <strong>your favorite agent-harness</strong> on a chosen benchmark. No evolution.</p>
<p>Each trial is driven by that benchmark's <strong>own</strong> runner (harbor, pier, upstream SWE-bench, …). beagle does not reimplement scoring. 
</td>
<td width="50%" valign="top">
<h3>Evolve</h3>
<p>Optimize the your choice of the agent harness (evolvee) <strong>source</strong> code and substrates by an <strong>agent</strong> (evolver).</p>
<p> Various evolution <strong>algorithms</strong> (e.g. DarwinX) are supported.</p>
</td>
</tr>
</table>


## Setup

### Install

```bash
git clone <beagle-url> # e.g. https://github.com/SalesforceAIResearch/Beagle
cd beagle
uv sync --extra all
```

In `vendor/xrlenv` is a rollout infrastructure for managing the containerized runtime environment at scale.
It requires a cluster of CPU-nodes to be set up. The user can also use the local docker daemon for development purposes.

Details about the rollout infrastructure `xrlenv` setup can be found in [xrlenv sphinx documentation](vendor/xrlenv/docs).
Use `uv run sphinx-autobuild docs docs/_build/html --open-browser --port 0` to self host the documentation.

### Prerequisites

```bash
source .venv/bin/activate
cp .env.example .env         # fill in credentials
set -a; source .env; set +a
```

### Bring your own API keys

Agents reach their model through **LiteLLM**, so bring the API key for whichever provider
you'll run — nothing else to stand up. Put it in `.env` (sourced above) and list the same
variable in the agent's `forward_env` so it reaches the run container:

```bash
# .env — set the key(s) for the model you'll use:
OPENAI_API_KEY=sk-...           # gpt-*, o-series
ANTHROPIC_API_KEY=sk-ant-...    # claude-*
# GEMINI_API_KEY / MISTRAL_API_KEY / GROQ_API_KEY / XAI_API_KEY — as your model needs
```

```yaml
# ...in the agent block of your config:
  model: {name: gpt-5.5}          # the model name selects the provider
  forward_env: [OPENAI_API_KEY]   # forwarded into the container; LiteLLM reads it
```

The model name picks the provider and LiteLLM authenticates with the forwarded key. On a
network-restricted benchmark, beagle allowlists that provider's API host automatically.


### Onboard agent experiment copies

Each evolvable agent needs an **experiment copy you own**. Onboarding pins an
upstream **commit SHA** (not a branch), seeds a parentless commit on `baseline`,
and writes `.beagle/agents/<profile>.json`.

```bash
export YOUR_ORG=<value>
# --version is the join key for generate_eval_configs.py (matches agent.harness.version)

# mini-swe — SWE-agent/mini-swe-agent
python -m beagle.tools.onboard \
    --upstream https://github.com/SWE-agent/mini-swe-agent --ref a83fcae82d2a08f0ee0c688f9d137b3566c097f8 \
    --repo $YOUR_ORG/mini_swe_agent_v2.4.6 --private --version v2.4.6 --branch-name baseline \
    --dir ../beagle-experiments/mini_swe_agent_v2.4.6 --profile-name mini_swe_agent_v2.4.6

# opencode — anomalyco/opencode  (--prune shrinks clones ~79 MB → ~12 MB)
python -m beagle.tools.onboard \
    --upstream https://github.com/anomalyco/opencode --ref a3647eb025c7615159d417dcc49fc39fdaeba65b \
    --repo $YOUR_ORG/opencode_v1.18.16 --private --version 1.18.16 --branch-name baseline \
    --prune opencode \
    --dir ../beagle-experiments/opencode_v1.18.16 --profile-name opencode_v1.18.16
```

- `--repo` — GitHub `org/name` created + pushed to
- `--dir` — local checkout (`origin` = your repo, `upstream` = source).
  Always pass it; omit and the copy lands under `.beagle/agents/` inside this repo
- `--profile-name` — manifest at `.beagle/agents/<profile>.json` (not the generator join key)
- `--branch-name` — branch for the baseline commit (default `baseline`)
- `--prune <profile>` — drop dead-weight paths (opencode only); patch-safe —
  see [docs/opencode-prune.md](docs/opencode-prune.md)

**Local vs containers.** `--dir` is for you; eval/evolve clone the **remote** `repo@ref`.

**Re-seed** with `--reseed` (destructive — don't use once candidate branches exist).
Updates remote + manifest `ref`, leaves `--dir` stale; `rm -rf <dir>` and re-run
without `--reseed` to refresh.

Batch option: `scripts/onboard_all_agents.sh`.

Later: `git -C <your-checkout> fetch upstream` to sync from upstream.


---

## Evaluate

Score an agent harness on a benchmark. Each task rolls through the benchmark's **own** runner
(harbor, pier, raw container path, ect) → graded → `run.json`; beagle does not reimplement scoring.

### Supported benchmarks

| Registry name | Tasks | Harness | Grading | Needs |
|---|---|---|---|---|
| `terminal_bench_2_1` | 89 (88 green) | harbor trial driver | in-band verifier reward | `beagle[terminal-bench]`, benchmark cache |
| `swe-rebench` | 860 (856 green) | harbor trial driver | in-band verifier reward (0/1) | `beagle[terminal-bench]`, benchmark cache |
| `deep-swe` | 113 (113 green) | pier trial driver (harbor fork) | in-band verifier reward | `beagle[deep-swe]`, benchmark cache |
| `swe-bench-verified` | 500 | docker drop-in | upstream swebench evaluator on the patch | `beagle[swe-bench]`, HuggingFace |

**More benchmarks are to be added soon.**

`beagle` runs **whatever the benchmark's corpus contains**. It does not filter tasks for you: a task's oracle
can give you zero reward if upstream dependencies are missing, changed or broken [**docs/benchmark-remarks.md**](docs/benchmark-remarks.md) lists them per benchmark
with the measured evidence; exclude them per run with `exclude_task_ids`. See the documentation for more details.

### Generate eval configs

Three config trees, three purposes:

| Tree | What it is | Written by |
|---|---|---|
| `examples/evaluation/*.yaml` | one file per **use case** (subset, mixture, pass@k, timeouts, retries), committed, `<your-org>` placeholders | hand-written |
| `experiments/configs/eval_baseline/…` | full-benchmark **sweeps** at the baseline knobs | `experiments/scripts/generate_eval_configs.py` |
| `tests/smoke/<bench>/<harness>-<version>_<variant>.yaml` | the **gate**: every copy × benchmark on 2 seeded-sample tasks | `scripts/generate_eval_configs.py` |

We only ship the `examples/evaluation/*.yaml` for user reference. The remaining two ties to your own agents's mainfest generated 
after the onboarding in `.beagle/agents/<profile>.json`. You are encourged to generate these configs yourself or handwrite them.

```bash
# both read your onboarded manifests in .beagle/agents/<profile>.json
python scripts/generate_eval_configs.py              # → tests/smoke/<bench>/<harness>-<version>_smoke2.yaml
python experiments/scripts/generate_eval_configs.py  # → experiments/configs/eval_baseline/<harness>-<version>_<bench>_<model>_<effort>_<turns>.yaml

# advanced — generate for one agent/benchmark, or at different knobs (each combination gets its own file)
python experiments/scripts/generate_eval_configs.py --check                                   # list, write nothing
python experiments/scripts/generate_eval_configs.py --agents opencode-1.18.16 --benches swe-rebench
python experiments/scripts/generate_eval_configs.py --model gpt-5.6 --effort high --max-turns 150
```

Both scripts fill in the part that is yours — the agent's repo and commit, read from the
`.beagle/agents/<profile>.json` that onboarding wrote:

```yaml
agent:
  harness:
    name: opencode          # which adapter runs it
    version: 1.18.16        # WHICH COPY — the `--version` you onboarded with
    source:                 # filled in for you, from that file
      repo: https://github.com/<you>/opencode_v1.18.16
      ref: a3647eb0…
```

`version` is the link between the two. Each script lists the agents it generates for, with the
version it expects; if you onboarded that agent under a different `--version`, there is nothing to
link and the agent is skipped. `--check` prints what would be written, and what was skipped, without
writing anything.

To add an agent or a benchmark — or a second copy of an agent you already have — edit the tables at
the top of `scripts/generate_eval_configs.py`; the sweep script reads the same ones. Files are named
`<agent>-<version>`, so two copies of one agent (say, before and after a change) never overwrite each
other.

Two knobs decide how long a task may run. `--timeout-multiplier 1.5` gives every task 1.5x the time
its benchmark allows; `--timeout` applies only to benchmarks that state no limit of their own — see
[05-timeouts.yaml](examples/evaluation/05-timeouts.yaml).

Whatever you generate or hand-write, `beagle evaluate --config <file> --dry-run` resolves it and
prints the plan without spending anything.

### Run an evaluation

```bash
beagle evaluate --config examples/evaluation/01-minimal.yaml --dry-run   # plan only, no spend
beagle evaluate --config examples/evaluation/01-minimal.yaml
```

`clone repo@sha → native harness → score + per-task rewards`. Writes the native `agent/` +
`verifier/` trees and `run.json` under `<run.dir>/<run.name>/`.

### Resume & retry

`beagle evaluate` reads each harness's **native** tree, so runs are resumable.
Categories split on one signal: did the trial record an error?

| Category | Meaning | Re-run with |
| --- | --- | --- |
| **missing** | no `result.json` (interrupted) | `--resume` |
| **error** | any recorded error (500, timeout, clone fail, no-attempt, …) | `--retry-errors` |
| **genuine-fail** | unresolved, no error (real attempt that didn't pass) | `--retry-unresolved` |
| **resolved** | passed | *never* |

Flags are **independent** — combine to union. Resolved tasks are never re-run.

| You pass | Re-runs |
| --- | --- |
| *(nothing)* | everything (fresh) |
| `--resume` | missing |
| `--retry-errors` | error |
| `--retry-unresolved` | error + genuine-fail |
| `--resume --retry-errors` | missing + error |
| `--resume --retry-unresolved` | missing + error + genuine-fail |

- Add `--dry-run` to any combo to print the plan without rolling out.
- `--retry-errors` re-runs every errored task — your call; never re-runs genuine fails.
- `--retry-unresolved` is the blunt superset (includes genuine fails). Use for
  deliberate re-sampling (pass@k / harness fix), not “the agent failed a task.”
  Ungraded trials (neither reward nor error) fail loud — grade first.
- Infra/setup failures and empty patches are stamped `NoAttempt` (errored), so
  `--retry-errors` catches them without hand-deleting `result.json`.
- `--task-ids t1,t2,…` restricts *which* tasks re-run — not a dataset filter.
  Full `run.json` aggregate stays whole; only named tasks re-run.
- `--force-resume` allows resume across a config change (records both hashes).
  In-run retry: `run.retry.infra` / `run.retry.content`.

---

## Evolve

Optimize the harness itself: an **evolver** agent edits the **evolvee**'s source, candidates are
scored on the same benchmark surface as above, and what survives the gate lands on your experiment
copy. Everything in **Evaluate** applies — evolution scores candidates the same way.

### Config shape

```yaml
run:      {dir, name, runtime, parallelism}
evolvee:                                    # θ — harness under evolution
  agent:  {name, version, source}           # type + version + INLINE source
  model:  {name}
  provider / forward_env                    # LLM routing (provider + key(s) to forward)
  effort / max_turns / timeout / extra_args # agent knobs (extra_args = CLI)
evolver:  {agent: {name, version}, model}   # proposer (e.g. cursor-agent)
algorithm: {name: darwinx, hparams: {…}}    # optimizer + typed knobs
data:     [{benchmark, tasks}]              # benchmark + tasks
```

**Derive** `evolvee.agent.source` from the onboard manifest (`.beagle/agents/<profile>.json`,
written by onboarding) — don't hand-write it. For the mini-swe copy from
[Onboard agent experiment copies](#onboard-agent-experiment-copies):

```json
{
  "profile": "mini_swe_agent_v2.4.6",
  "version": "v2.4.6",
  "repo": "https://github.com/<your-org>/mini_swe_agent_v2.4.6",
  "ref": "<baseline commit>",
  "branch": "baseline",
  "token_env": "GH_TOKEN",
  "upstream": "https://github.com/SWE-agent/mini-swe-agent",
  "upstream_ref": "a83fcae…",
  "dir": "../beagle-experiments/mini_swe_agent_v2.4.6"
}
```

Paste `repo` / `ref` / `token_env` / `dir` 1:1 into `evolvee.agent.source.*`.
Add `agent.name` (adapter — `mini-swe`) and `agent.version` (label). LLM routing
(`forward_env`) and CLI flags live on the **agent** block, not the model block.

Validate with `beagle evolve --config <file> --dry-run` (resolve + plan + version gate, no spend).

### Run a campaign

`beagle evolve` runs the loop named by the `algorithm` block — DarwinX unless you swap it.
One config is one **campaign**: every node it scores is recorded in a genealogy DB, so
re-launching the same config continues the tree instead of starting over.

```bash
beagle evolve --config examples/quick-start/config.yaml --dry-run   # resolve + plan, no spend
beagle evolve --config examples/quick-start/config.yaml
```

Each pipeline (one proposal attempt against one parent) walks these phases:

| Phase | What happens |
| --- | --- |
| **seed** | θ (the `evolvee`) is cloned once under `repo_root`; each pipeline gets its own worktree |
| **baseline** | the parent is scored on the subset — skipped when that parent already has a score |
| **propose** | the `evolver` edits the worktree; its diff is the candidate |
| **score** | the candidate runs the same tasks at the same budget as the baseline |
| **gate** | keep / reject — verification gates, guard (canary) tasks, equivalence, anti-cheat |
| **land** | a kept node is committed and pushed as `evolve/<parent-sha>__<pipeline-id>` |

Nodes end as `completed`, `no_change`, `rejected`, or `failed`. **`no_change` is a normal
outcome** — the proposer found nothing that survived the gate — not an error.

**Which tasks.** DarwinX is single-benchmark, so it evolves on `data[0]`. `tasks` is the
subset it optimizes against; `options.priority_tasks` / `variance_tasks` / `fullset_tasks`
stratify it further. All optional — omit them and the driver's own defaults stand.

**Knobs** go under `algorithm.hparams` and are typed (`extra='forbid'`, so a typo fails at
load instead of being silently ignored). The ones you reach for first:

| Knob | Controls |
| --- | --- |
| `max_loop_iters` | proposal attempts per pipeline before it gives up |
| `subset_eval_n_attempts` / `fullset_eval_n_attempts` | avg@k on the subset / on the scoring set |
| `mini_eval_k_samples` | cheap screen before committing to a full re-score |
| `parent_strategy` | which scored node the next pipeline branches from |
| `node_score` | `panel` (fixed task panel) or `mixture` (multi-benchmark) |
| `gate_enabled` · `anti_cheat` | the accept/reject machinery |
| `guard_enabled` · `guard_strict` | canary tasks a candidate is not allowed to regress |
| `absorb_timeouts` · `infra_retries` | keep infra flakiness out of the score |

That is the short list; the full surface is `DarwinXConfig` in
[beagle/algorithms/darwinx/config.py](beagle/algorithms/darwinx/config.py), which is also
where each knob's meaning is documented.

**What you get.** Under `<run.dir>/<run.name>/`, per campaign: `state.db` (the genealogy),
`nodes/<node-id>/` and `pipelines/<pipeline-id>/` for per-node and per-attempt logs, the
worktrees, and `_evals/` holding the raw native harness trees. Kept branches land on your
experiment copy; the CLI prints the best node when the campaign finishes.

---

## More

| Doc | What's in it |
|---|---|
| [docs/advanced.md](docs/advanced.md) | module map, onboarding your own agent adapter, the Python API |
| [docs/benchmark-remarks.md](docs/benchmark-remarks.md) | per-benchmark tasks we suggest excluding, with the measured evidence |
| [docs/opencode-prune.md](docs/opencode-prune.md) | what `--prune opencode` drops from a clone, and why it's patch-safe |
