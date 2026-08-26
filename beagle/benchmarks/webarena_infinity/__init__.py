"""WebArena-Infinity (native-runner shape).

WAI is self-contained: it ships its *own* xrlenv-aware orchestrator
(``evaluation/run_eval_parallel_xrlenv.py``) that acquires containers, provisions
app servers, runs the agent, injects+runs+deletes the verifier (answer-free), and
produces the reward. So beagle **vendors the WAI repo** (a git submodule under
``vendor/benchmarks/webarena-infinity``) and *invokes that runner* — never
reimplementing the lifecycle.

* **source** — ``WaiSource`` enumerates the vendored ``apps/<app>/<suite>.json`` into
  ``Task``s (so WAI tasks mix in a ``DataMixture``), carrying the ``wai_*`` extras.
* **harness** — ``WaiHarness`` (a :class:`NativeRunnerHarness`) invokes WAI's runner
  for the selected tasks. **Agent injection is generic**: the runner runs the beagle
  agent via ``agent.run(...)`` in the provisioned container, so any browser-capable
  agent works; evolving monet flows the candidate ref into WAI's monet install.
* **grader** — ``InBandGrader``: the verifier's reward is already in ``result.json``.

(The agent<->environment compatibility check — does an agent operate a browser vs a
shell — is deferred until this benchmark is built for real; see notes/roadmap.md M3.)
"""

from __future__ import annotations

from collections.abc import Iterator

from beagle.benchmarks.base import Benchmark, BenchmarkHarness, BenchmarkSpec, Grader, TaskSource
from beagle.benchmarks.grader import InBandGrader
from beagle.benchmarks.harness import NativeRunnerHarness
from beagle.benchmarks.registry import register
from beagle.benchmarks.source import TaskItem

_BENCH = "webarena-infinity"
#: Substrate image (single image, apps as server.py processes) — overridable via spec.image.
_SUBSTRATE = "xrlenv-webarena-infinity/substrate:dev"


class WaiSource(TaskSource):
    """Enumerate WAI tasks from the vendored ``apps/<app>/<suite>.json`` manifests."""

    def tasks(self, spec: BenchmarkSpec) -> Iterator[TaskItem]:
        # TODO(M3): walk <wai_repo>/apps/<app>/*-tasks.json (wai_repo = spec.dataset or
        # the vendored submodule) → Task(extras={wai_web_app, wai_suite, wai_verify_rel,
        # wai_host_verify, wai_task_json, ...}), TaskContext(image=spec.image or _SUBSTRATE,
        # benchmark_name=_BENCH); then select_and_sample(items, spec).
        raise NotImplementedError("WaiSource.tasks not yet implemented (roadmap M3)")


class WaiHarness(NativeRunnerHarness):
    """Invoke WAI's native ``run_eval_parallel_xrlenv.py`` for a task batch.

    Overrides ``rollout`` (inherited stub): build the runner invocation for ``items``
    + the agent, run it (WAI owns provisioning/verifier/reuse), inject the beagle
    agent generically at the agent step, then read WAI's native results into
    ``TaskResult``s.
    """

    #: Path to the vendored WAI orchestrator.
    RUNNER = "vendor/benchmarks/webarena-infinity/evaluation/run_eval_parallel_xrlenv.py"


@register("webarena-infinity")
class WebArenaInfinity(Benchmark):
    """WebArena-Infinity: browser benchmark driven by its own vendored xrlenv runner."""

    name = _BENCH
    # No prompt framing here: beagle hands the agent only the task goal (the raw
    # problem_statement). Browser agents bring their own WAI-tuned browser system prompt, so any
    # framing on our side would fight it. A benchmark data hook (additional_info_pre/post) could
    # surface WAI dataset facts if needed, but the goal alone is the payload today.

    def source(self) -> TaskSource:
        return WaiSource()

    def harness(self, env_import_path: str | None = None) -> BenchmarkHarness:
        if env_import_path:      # native WAI harness — no harbor cluster Environment; warn, don't drop silently
            import warnings
            warnings.warn(f"{self.name}: env_import_path is ignored — this benchmark's harness has no "
                          "harbor cluster Environment.", stacklevel=2)
        return WaiHarness()

    def grader(self) -> Grader:
        return InBandGrader()


__all__ = ["WebArenaInfinity", "WaiSource", "WaiHarness"]
