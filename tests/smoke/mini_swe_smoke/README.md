# mini-swe smoke — pure eval

Baseline **mini-swe-agent** on a few tasks, via `beagle evaluate`.

```bash
set -a; source .env; set +a                 # gateway creds; gateway up (scripts/gateway/)
cd /path/to/beagle

# terminal-bench-2.1 (2 tasks, harbor)
beagle evaluate --config tests/smoke/mini_swe_smoke/terminal_bench_2_1_smoke2.yaml --dry-run   # preview
beagle evaluate --config tests/smoke/mini_swe_smoke/terminal_bench_2_1_smoke2.yaml             # run

# swe-bench-verified (2 tasks)
beagle evaluate --config tests/smoke/mini_swe_smoke/swebench-verified_smoke2.yaml

# deep-swe (1 task, pier) — requires the extra: uv pip install -e '.[deep-swe]'
beagle evaluate --config tests/smoke/mini_swe_smoke/deep-swe_smoke2.yaml
```

Results → `<run.dir>/<run.name>/run.json`. Point `agent.harness.source.{repo,ref}` at your fork to evolve later.

**Config shape.** First-level (top of every agent's block, uniform across agents): `model`,
`provider`, `effort`, `max_turns`, `forward_env`, `timeout`. An agent-harness's *own* args live under
`extra_args:`, keyed by `<agent>_args` — `mini_swe_args:` (its `config_path` preset) / `monet_args:`
(monet's raw CLI) — so a config names which args belong to which agent.

**First run may need tuning** (mini-swe installs in-container from `repo@ref`, then runs its own `mini` CLI):
- gateway↔LiteLLM routing — litellm's base-url env is provider-specific (OpenAI vs Anthropic),
  so this must be model-agnostic (litellm `api_base` via model_kwargs, one URL for any model),
- `config_path` (the real preset path in the repo),
- swebench `cwd: /testbed` vs the task's repo path.
