# Pure evaluation — run an agent on a benchmark (no evolution)

Score **one agent** on **one benchmark** from a single `config.yaml`, via the generic CLI:

```bash
beagle evaluate --config <config.yaml> --dry-run   # preview the plan, no spend
beagle evaluate --config <config.yaml>             # evaluate (spends)
```

Under the hood it's `bgl.evaluate`: each task rolls through the benchmark's *native* harness → graded
→ `run.json`. Same canonical config shape as the evolution example (`../quick-start`), just **without**
the `evolver` + `algorithm` blocks — an `agent` to test + the `data` to score it on.

> **The eval matrix configs are GENERATED, not committed.** The config *shape* lives once in
> `scripts/generate_eval_configs.py` (edit its `AGENTS` / `BENCHMARKS` / `DEFAULTS` tables) — no M×N of
> near-duplicate YAML to maintain, no schema drift, and no private repo committed. **Onboard** each
> agent (with `--version`), then generate configs pointing at *your* copies:
> ```bash
> python scripts/generate_eval_configs.py     # → examples/evaluation/<bench>/<agent>.yaml (gitignored)
> beagle evaluate --config examples/evaluation/swe-bench-verified/opencode.yaml --dry-run
> ```
> The generator joins each agent to your manifest on `version` (the matrix `version` ↔ the `--version`
> you onboarded). See the top-level README §2 → §2b.

## Layout

The runnable matrix (agent × benchmark) is **generated** by `scripts/generate_eval_configs.py` and
gitignored. The config shape lives in that script (`build_config`); field meanings are in **Config
shape** below.

```
evaluation/
├── README.md
└── <benchmark>/<agent>.yaml        # GENERATED (gitignored) — the script's matrix + your manifests
```

| benchmark | harness | mini-swe | opencode |
|---|---|---|---|
| terminal_bench_2_1 | harbor | ✅ | ✅ |
| swe-bench-verified | harbor / docker | ✅ | ✅ |
| deep-swe | pier (filtered egress) | ✅ | ✅ ¹ |

¹ opencode (Bun) on DeepSWE installs behind pier's Squid allowlist; its `install_hosts` cover the
runtime + npm indexes across the common base-image package managers, but it's **best-effort** — a
task image whose apt/apk mirror isn't listed will 403 the runtime bootstrap (add the host to that
agent's `install_hosts`).

Each config runs the **whole suite** (the benchmark configs omit `tasks:`). Add `tasks: [id, …]`
under `data` to subset while iterating.

```bash
# generate first (§2b), then run:
beagle evaluate --config examples/evaluation/swe-bench-verified/mini-swe.yaml --dry-run
beagle evaluate --config examples/evaluation/deep-swe/mini-swe.yaml   # needs .[deep-swe]
```

## Config shape

```yaml
run:   {dir, name, runtime, parallelism}   # evaluate stamps <dir>/<name>-<timestamp>/
agent:
  harness: {name, version, source}         # agent type + version + source (repo/ref/token_env)
  model: {name}
  provider / effort / max_turns / forward_env / timeout   # first-level vocabulary (every agent)
  extra_args: {<agent>_args: …}            # the one agent-specific block (mini_swe_args / opencode_args)
data:  [{benchmark, tasks?, …}]            # omit `tasks` → the whole suite
```

- **`harness.source`** is filled by the generator from `.beagle/agents/<name>.json` — the manifest
  whose `version` matches the matrix entry — pulling its `repo`/`ref`/`token_env` (a public copy with
  no `token_env` omits the line). `harness.name` is the registered agent (`mini-swe`/`opencode`).
  Advanced: hand-write a config with a literal `source` and skip the generator.
- **`effort`** drives reasoning: mini-swe selects the Responses-API model class; opencode passes
  `--variant`. Each reaches the LLM via `provider` + the key you forward in `forward_env` (bring your
  own API key). (opencode accepts `max_turns` for a uniform vocabulary but has no turn-cap flag — it
  is a no-op there.)
- **`extra_args`** is keyed by `<agent>_args` so a config names which knobs belong to which agent —
  `mini_swe_args: [{config_path: …}]` (its `-c` preset) vs `opencode_args: [--…]` (its raw CLI).

## deep-swe (filtered egress)

DeepSWE is pier-driven with `allow_internet=false`, so the harness drives each agent across pier's
**phased network**: `install()` (INSTALL, open — clone + build, `install_hosts` allowlisted) then
`run_in()` (RUN, restricted to `network_hosts` — the model provider). mini-swe and opencode both
implement that split; the submission is `git diff base..HEAD`, so `run_in` commits the agent's edits.
Needs the `.[deep-swe]` extra (`uv pip install -e '.[deep-swe]'`).

opencode's `install_hosts` are **best-effort** across base-image package managers (it fetches Bun
from bun.sh + deps from npm). If a DeepSWE image uses an apt/apk mirror not listed, the runtime
bootstrap will 403 through the proxy — add that host to the agent's `install_hosts`. opencode's LLM
call rides pier's proxy with no extra shim (Bun's `fetch` honors `HTTPS_PROXY` natively); mini-swe
(uv, a bounded host set) doesn't hit the mirror issue at all.

## Reading results

`--dry-run` prints the resolved plan + the version gate (fails loud if a black-box agent's pinned
`agent.version` ≠ its installed version; opencode is exempt — versioned by `source.ref`).
A live run
prints `score` + per-task `resolved`/`reward`/`error`; artifacts land under `<run.dir>/<run.name>-<ts>/`.
Prerequisites (bring your own API key) are the top-level README §0–1.

> **A 0-token task means the agent never actually ran** (e.g. a clone/install failure) — check the
> per-task `error` and the trial's `agent/` tree, not the score.
