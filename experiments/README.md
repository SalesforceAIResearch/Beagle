# experiments/ — baseline runbook

Parameterized baseline eval configs and their raw results, kept out of the canonical
`scripts/` matrix. Layout:

```
experiments/
  scripts/
    generate_eval_configs.py       # the generator (CLI flags; borrows scripts/generate_eval_configs.py)
    generate_all_eval_configs.sh   # wrapper: the baseline matrix, one command per config
  configs/eval_baseline/           # generated .yaml (name = the run identity)
  results/                         # raw run output (baked into each config's run.dir)
```

**Config name = the experiment identity:** `{harness}_{bench}_{model}_{effort}_{max_turns}.yaml`
(bench short tags: `terminal_bench_2_1→tb21`, `deep-swe→deepswe`, `swe-bench-verified→swebench`).

## 1. Generate

The current baseline = **monet + opencode × tb2.1 / deep-swe / swe-verified**, `gpt-5.6-sol` /
`medium` / `200` turns.

```bash
# everything (the 6 baseline configs)
bash experiments/scripts/generate_all_eval_configs.sh
CHECK=1 bash experiments/scripts/generate_all_eval_configs.sh    # dry-run, write nothing
```

Or drive the generator directly — every knob is a flag:

```bash
# one config
.venv/bin/python experiments/scripts/generate_eval_configs.py --agents monet --benches deep-swe

# a different sweep (new model/effort/turns → new filenames, no collision)
.venv/bin/python experiments/scripts/generate_eval_configs.py \
    --agents monet opencode --benches terminal_bench_2_1 deep-swe swe-bench-verified \
    --model gpt-5.6-sol --effort medium --max-turns 200

.venv/bin/python experiments/scripts/generate_eval_configs.py --help   # all flags
```

Flags: `--agents --benches --model --effort --max-turns --parallelism --timeout --retry-infra
--runtime --provider --out --results --manifest-dir --check`. Agent `extra_args`, benchmark
`dataset`/`split`/`parallelism`, and the manifest join are inherited from `scripts/generate_eval_configs.py`,
so onboard agents there once (`.beagle/agents/`) and this picks them up.

## 2. Run

```bash
source .venv/bin/activate
beagle evaluate --config experiments/configs/eval_baseline/monet_tb21_gpt-5.6-sol_medium_200.yaml
```

Raw results land in `experiments/results/<config-stem>/`.

## Prereqs
- **Gateway relay up** — `LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL` points at a live relay
  (`scripts/gateway/{laptop,login-node}.sh`). Models/efforts: see `notes/gateway-models.md`
  (`gpt-5.6-sol` + `medium` is confirmed live).
- **deep-swe** needs the extra: `uv pip install -e '.[deep-swe]'` (pier / filtered egress).
- **Agents onboarded** at the pinned versions (monet `20260816`, opencode `1.18.16`) — the
  generator skips any not present in `.beagle/agents/`.
