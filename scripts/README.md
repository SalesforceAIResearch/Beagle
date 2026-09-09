# scripts/

Installation, onboarding, config generation, and operator utilities:

| path | purpose |
|---|---|
| [`install.sh`](install.sh) | install `uv` when needed and sync all Beagle dependencies |
| [`onboard_all_agents.sh`](onboard_all_agents.sh) | onboard the bundled reference agent harnesses |
| [`onboard/`](onboard/README.md) | create and seed an evolvable agent experiment copy |
| [`generate_eval_configs.py`](generate_eval_configs.py) | generate smoke evaluation configs from onboarded manifests |
| [`generate_evolution_config.py`](generate_evolution_config.py) | fill the DarwinX smoke config from an onboarded OpenCode manifest |
| [`populate_benchmarks_cache.py`](populate_benchmarks_cache.py) | populate registered benchmark task caches through their xrlenv builders |
| [`smoke_tasks.json`](smoke_tasks.json) | fixed task samples used by generated smoke configs |


## Evolution smoke config

Generate one two-task DarwinX smoke per onboarded reference harness × benchmark. The script reuses
the evaluation generator's matrix, provider/runtime profiles, and seeded tasks:

```bash
.venv/bin/python scripts/generate_evolution_config.py
# → tests/smoke/<bench>/<harness>-<version>_darwinx_evolution_smoke2.yaml

.venv/bin/beagle evolve \
  --config tests/smoke/terminal_bench_2_1/opencode-1.18.16_darwinx_evolution_smoke2.yaml \
  --dry-run
```

Use `--internal` to add internal harnesses and select the internal provider/runtime profile,
`--agents <harness>-<version>` or `--benches <name>` to narrow the matrix, and `--check` to list
outputs and missing manifests without writing. Generated smokes are gitignored and may be safely
regenerated.

For a real campaign, copy the selected smoke under `experiments/configs/`, change its `run.name`,
task pool, and DarwinX settings, then commit it with the experiment artifacts when appropriate.
DarwinX parameter guidance lives in
[`docs/darwinx-configuration.md`](../docs/darwinx-configuration.md).

## Benchmark cache

The project `.env.example` defaults `XRLENV_BENCHMARK_CACHE` to
`.cache/xrlenv_benchmark_cache` under the project root. Populate all cache-capable
benchmarks currently registered with Beagle:

```bash
.venv/bin/python scripts/populate_benchmarks_cache.py
```

Use `--list` to inspect the registry-driven set, `--benchmark NAME` (repeatable) to
populate a subset, or `--dest PATH` to override the cache root explicitly. Population
is sequential and idempotent; each benchmark's vendored xrlenv builder remains
responsible for downloading, normalizing, and patching its native corpus.

Transient `httpx`/`httpcore` transport failures, HTTP 429 responses, and HTTP 5xx
responses are retried three times with exponential backoff; completed task downloads
remain cached between attempts. Override with `--retries N`. Per-request HTTP INFO logs
are hidden by default; pass `--show-http-logs` when diagnosing connectivity.
