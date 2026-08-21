# monet smoke — pure eval

Baseline **monet** on 2 tasks, via `beagle evaluate`.

```bash
set -a; source .env; set +a                 # gateway creds; gateway up (scripts/gateway/)
cd /fsx/home/yutong/Github/beagle

# terminal-bench-2.1 (2 tasks)
beagle evaluate --config tests/smoke/monet_code/terminal_bench_2_1_smoke2.yaml --dry-run   # preview
beagle evaluate --config tests/smoke/monet_code/terminal_bench_2_1_smoke2.yaml             # run

# swe-bench-verified (2 tasks)
beagle evaluate --config tests/smoke/monet_code/swebench-verified_smoke2.yaml
```

Results → `<run.dir>/<run.name>/run.json`. `agent.harness.source` is monet's experiment copy (from
`python -m beagle.tools.onboard`); swap `repo`/`ref` to your own.
