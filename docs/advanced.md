# Details & advanced

How beagle is put together, how to add your own agent adapter, and the Python API — the material
behind the [README](../README.md)'s Setup / Evaluate / Evolve walkthrough.

## Modules

```mermaid
flowchart TB
  cfg["config.yaml / Python API"] --> cli["beagle.config + beagle.cli"]
  cli --> agents["beagle.agents"]
  cli --> algos["beagle.algorithms"]
  cli --> benches["beagle.benchmarks"]
  benches --> data["beagle.data<br/>TaskDataset / DataMixture"]
  cli --> trainer["beagle.trainer.Trainer"]
  trainer --> runner["beagle.rollout.Runner"]
  runner --> xrlenv["vendor/xrlenv → native harnesses"]
```

Pure eval uses the same lower half: `beagle evaluate` → agent + dataset →
`beagle.eval` / `beagle.rollout` (no evolver / algorithm).

| Area | Path |
| --- | --- |
| Public facade | `beagle/__init__.py` |
| Config + CLIs | `beagle.config`, `beagle.cli` |
| Agents | `beagle.agents` |
| Algorithms | `beagle.algorithms` |
| Training loop | `beagle.trainer` |
| Data | `beagle.data` |
| Benchmarks | `beagle.benchmarks` |
| Rollout | `beagle.rollout` |
| Evaluation | `beagle.eval` |
| Tools | `beagle.tools` |
| Substrate | `vendor/xrlenv` |
| Examples / tests / notes | `examples/`, `tests/`, `notes/` |

## Onboard your own agent

Role (evolvee vs evolver) is chosen at **run time**, not baked into the agent.
Declare capabilities via mixins; `Trainer` checks the role you assign is supported:

| Capability | Implement | Meaning |
| --- | --- | --- |
| `Runnable` | `run` | attempt tasks (be scored) |
| `Evolvable` | `_default_source` | versioned git source (`repo@ref`) |
| `Editor` | `edit` | run one coding instruction (be an evolver) |

Evolvee = `Runnable` + `Evolvable`; evolver = `Editor`. The evolver is a thin
primitive — the algorithm owns prompts and the analyze/implement/review recipe.
Drop one package under `beagle/agents/` — no other edits:

```python
# beagle/agents/my_agent/__init__.py
from beagle.agents.core import Agent, AgentSource, Runnable, Evolvable, register
from beagle.rollout.runtime import ContainerRuntime
from beagle.types import Task, TaskContext, TaskResult

@register("my-agent")
class MyAgent(Agent, Runnable, Evolvable):   # white-box — usable as evolvee
    def _default_source(self):
        # source comes from run config (your experiment copy). entrypoint is intrinsic.
        return self.spec.source or AgentSource(entrypoint="bin/my-agent")

    def run(self, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime) -> TaskResult:
        src = self.source()                   # baseline ref, or evolved candidate
        handle = runtime.acquire(image=task_ctx.image or "", command=["sleep", "infinity"])
        try:
            ...  # clone src.repo@src.ref, build, run, collect patch
            return TaskResult(task_id=task.task_id)
        finally:
            runtime.destroy(handle)
    # add Editor + edit(instruction, workspace, ...) to also serve as evolver
```

Auto-discovered on `import beagle`. One `run` works on every harness — you never
write harness-specific code. Closed-source CLI you can't evolve:
`class MyAgent(Agent, Editor)`. Start from `beagle/agents/core/_template.py`.
Benchmarks and algorithms onboard the same way — one file, `@register`, done.

The sketch above is the shape; a real adapter also has to split `install`/`run_in`
for network-phased harnesses, resolve timeouts from the benchmark, normalize token
usage, and emit ATIF. See **[Onboarding an agent harness](onboarding-an-agent.md)**
for the full runbook, contracts, and checklist.

## Onboard your own benchmark

A benchmark is three pluggables behind the `Benchmark` ABC, each independently
overridable. Most harbor-family benchmarks need **zero** custom code:

```python
# beagle/benchmarks/my_bench/__init__.py
from beagle.benchmarks.harness import HarborBenchmark
from beagle.benchmarks.registry import register

@register("my-bench")
class MyBench(HarborBenchmark):
    cache_name = "my-bench"   # subdirectory under XRLENV_BENCHMARK_CACHE
    cache_builder_module = "xrlenv_plugins.benchmarks.my_bench.build_cache"
```

When the defaults don't fit, override the pluggables individually:

| Pluggable | Default (`HarborBenchmark`) | Override when |
| --- | --- | --- |
| `source()` → `TaskSource` | `HarborCache` (reads the benchmark cache directly) | Tasks come from Hugging Face, a local file, or a custom API |
| `harness()` → `BenchmarkHarness` | `HarborHarness` (harbor trial driver) | Using a docker drop-in, pier, or a benchmark's own vendored orchestrator |
| `grader()` → `Grader` | `InBandGrader` (reads verifier reward from the trial) | Patch-eval grading (`PatchEvalGrader`) or a custom judge |

### `cache_builder_module` — registry-driven cache population

Set `Benchmark.cache_builder_module` to the import path of the benchmark's xrlenv
`build_cache` module to make it discoverable by the cache bootstrap script:

```python
cache_builder_module = "xrlenv_plugins.benchmarks.my_bench.build_cache"
```

`None` (the default) means this benchmark has no local cache population step —
tasks are fetched at run time (e.g. from Hugging Face). When set, the module must
expose a `main(argv: list[str]) -> int` that accepts
`["--stage", "all", "--dest", "<path>"]`.

`scripts/populate_benchmarks_cache.py` discovers cache-capable benchmarks from
this attribute at run time — no edit to the script is needed when you add a new
benchmark. Use `--list` to verify your registration is visible:

```bash
.venv/bin/python scripts/populate_benchmarks_cache.py --list
# my-bench    xrlenv_plugins.benchmarks.my_bench.build_cache
```

See [`scripts/README.md`](../scripts/README.md) for the full operator workflow
(setting `XRLENV_BENCHMARK_CACHE` in `.env`, populating subsets with
`--benchmark NAME`, and idempotent reruns).

## Provider routing

How an agent reaches its model — `provider` type (`direct` / `gateway` / `internal`),
`forward_env` forwarding, and the agent support matrix — is covered in a dedicated page:

👉 [Provider routing](provider-routing.md)

## Prefer Python?

The CLI is a thin wrapper. Compose the pieces yourself (PyTorch-shaped
model / optimizer / dataloader → `fit`):

For the canonical YAML workflow and guidance on DarwinX's typed parameters, start with
[`examples/evolution/README.md`](../examples/evolution/README.md) and
[`docs/darwinx-configuration.md`](darwinx-configuration.md).

```python
import beagle as bgl

trainer = bgl.Trainer(
    evolvee=bgl.agents.build(evolvee_config),
    evolver=bgl.agents.build(evolver_config),
    algorithm=bgl.algorithms.build(darwinx_config),
    trainer_config={"runtime": {"kind": "xrlenv-cluster"}},
)
best = trainer.fit(train_dataset=bgl.TaskDataset.from_benchmark(benchmark_config))

bgl.evaluate(run_config)   # pure eval — no evolver/algorithm
```

Full example:
[examples/evolution/quick_start_inline.py](../examples/evolution/quick_start_inline.py).
Build by name: `bgl.agents.build(...)`, `bgl.algorithms.build(...)`,
`bgl.benchmarks.get(...)`.

> **Status.** Both paths run today from one `config.yaml` — pure evaluation and the
> DarwinX loop (baseline → edit → candidate eval → keep/reject → best node + branch).
> `Trainer.fit`, DarwinX, and the version gate are wired end-to-end. Beagle can evaluate a
> `DataMixture`; DarwinX currently trains on the first data group and can apply configured
> cross-benchmark or mixture gates during candidate selection.
