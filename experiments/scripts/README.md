# experiments/scripts — experiment utilities

Run these commands from the repository root. The scripts operate on the configs and
timestamped run directories described in [`experiments/README.md`](../README.md).

## Generate configs

- `generate_eval_configs.py` generates any requested agent × benchmark baseline sweep.
  It reuses the canonical agent, benchmark, credential-forwarding, and manifest metadata
  from `scripts/generate_eval_configs.py`.
- `generate_all_eval_configs.sh` materializes the current six-config baseline matrix with
  a uniform parallelism of 16.

```bash
# current baseline
bash experiments/scripts/generate_all_eval_configs.sh

# inspect without writing
CHECK=1 bash experiments/scripts/generate_all_eval_configs.sh

# custom cross-product
.venv/bin/python experiments/scripts/generate_eval_configs.py \
  --agents monet opencode \
  --benches terminal_bench_2_1 deep-swe swe-bench-verified \
  --model gpt-5.6-sol --effort medium --max-turns 200
```

Use `.venv/bin/python experiments/scripts/generate_eval_configs.py --help` for the full
flag list.

## Browse results

`dashboard.py` discovers completed and in-progress runs directly under
`experiments/results/`.

```bash
uv pip install -e '.[analysis]'                 # one-time; not a runtime dependency
streamlit run experiments/scripts/dashboard.py  # http://localhost:8501
```

The Overview page shows one row per run and benchmark: score, resolved/tasks, token and
cache usage, estimated cost, and median per-task agent latency and cost. It supports
benchmark, harness, model, and run filters plus editable model prices.

The Trajectory viewer drills into a run, benchmark, and trial to show reward, token
usage, errors, and the canonical ATIF `agent/trajectory.json`.

### Summary cache and token accounting

`results_data.py` performs the expensive artifact scan. For each trial, it prefers the
cache-split token counts in `result.json`; when those are absent, it runs the same
harness-specific usage parser used by the rollout pipeline. Unknown harnesses fall back
to ATIF `final_metrics`.

It writes a compact `<run-directory>/summary.json`; the dashboard reads these caches and only
opens a raw trajectory when requested. The cache is generated and git-ignored. Use
**Refresh (Rebuild Data)** after runs change.

Beagle remains pricing-agnostic. Dashboard costs are estimates using `DEFAULT_PRICES` in
`results_data.py`, expressed as dollars per million fresh-input, cached-input, and output
tokens. Prices can also be edited in the dashboard.

## Recover interrupted grading

`grade_run.py` is for a two-phase benchmark such as SWE-bench when agent trials produced
`patch.diff` files but batch grading did not finish. It reconstructs task results, invokes
the configured grader without rerunning agents, and writes trial results plus the run
record.

Inspect the recovery first:

```bash
.venv/bin/python experiments/scripts/grade_run.py \
  --config experiments/configs/eval_baseline/monet_swebench_verified_gpt-5.6-sol_medium_200.yaml \
  --run-dir experiments/results/<config-stem>-<timestamp> \
  --dry-run
```

Remove `--dry-run` to grade. Then resume the same directory so beagle runs only tasks
that are still missing:

```bash
beagle evaluate \
  --config experiments/configs/eval_baseline/monet_swebench_verified_gpt-5.6-sol_medium_200.yaml \
  --resume --run-dir experiments/results/<config-stem>-<timestamp>
```

## Extend the dashboard

For a new agent stream parser, add its harness mapping to `_PARSERS` in
`results_data.py`; otherwise the ATIF fallback is used. Add dashboard pages in
`dashboard.py` and register them with `st.navigation`.
