# DarwinX implementation hosted by Beagle

Beagle is the infrastructure layer: it provides the agent, benchmark, rollout, training, and
algorithm interfaces. This directory contains the **DarwinX algorithm implementation** that Beagle
hosts behind those interfaces.

Users should launch DarwinX through:

```bash
beagle evolve --config examples/evolution/config.yaml
```

See the [evolution quick start](../../../../examples/evolution/README.md) for a one-task smoke.
The modules below are implementation packages, not a separate supported CLI.

## Layout

| Package | Responsibility |
| --- | --- |
| [`evolve/`](evolve/) | genealogy, parent selection, proposal pipelines, candidate evaluation, preservation, compaction, and recombination |
| [`gate/`](gate/) | structural, anti-cheat, verifier, canary, transfer, and other candidate-acceptance checks |
| [`dx_trace/`](dx_trace/) | trajectory normalization, quality checks, failure-mode extraction, and evidence summaries |

The Beagle-facing adapter lives one directory above:

| File | Boundary |
| --- | --- |
| [`../algorithm.py`](../algorithm.py) | registered `darwinx` algorithm and `Trainer` entrypoint |
| [`../config.py`](../config.py) | typed public configuration (`DarwinXConfig`) |
| [`../_launch.py`](../_launch.py) | translates Beagle source/runtime/task settings and launches the algorithm |
| [`../eval.py`](../eval.py) | routes candidate scoring through Beagle's benchmark-native rollout infrastructure |
| [`../meta_agent.py`](../meta_agent.py) | injects the configured Beagle `Editor` as DarwinX's evolver |

## Launch flow

```text
beagle evolve
    └── Beagle Trainer
        └── DarwinX.evolve(...)
            ├── materialize the evolvee experiment copy
            ├── inject the configured evolver
            ├── translate typed config to the algorithm runtime
            ├── run evolve/ + gate/ + dx_trace/
            └── return the best candidate to Beagle
```

Candidate rollouts still use Beagle's benchmark integrations and native benchmark graders. DarwinX
owns the search and selection policy; Beagle owns the reusable infrastructure that supplies agents,
tasks, runtimes, and evaluation.

## Configuration

Configure the algorithm under `algorithm.hparams` in the Beagle run config. The fields are validated
by [`DarwinXConfig`](../config.py); unknown names fail during config loading. Do not set the
implementation's environment variables directly in new user-facing examples—the Beagle adapter
derives them from the typed run config.

The user-facing mapping from paper concepts to these fields lives in
[`docs/darwinx-configuration.md`](../../../../docs/darwinx-configuration.md).

```yaml
algorithm:
  name: darwinx
  hparams:
    max_loop_iters: 1
    fullset_eval_n_attempts: 1
```

## Import and maintenance boundary

The three implementation packages use top-level imports such as `from evolve...`, `from gate...`,
and `from dx_trace...`. [`prepare_import_path()`](../_launch.py) makes those packages available only
for a DarwinX launch and also exposes the evaluation shim to worker subprocesses.

Changes inside this directory change the DarwinX algorithm. Changes one level above should stay
limited to Beagle's hosting boundary—configuration translation, dependency injection, rollout
routing, and result conversion. Keeping that boundary explicit lets Beagle host DarwinX without
making Beagle itself synonymous with DarwinX.
