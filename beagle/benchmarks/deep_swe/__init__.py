"""DeepSWE (pier family) — the thin beagle adapter.

Onboarded once in xrlenv (build_cache + oracle, 113/113 green as of 2026-07-18). Here it is the
whole integration: tasks come from the benchmark cache (pier ``task.toml`` + ``instruction.md``,
read by :class:`~beagle.benchmarks.source.HarborCache` — the pier task dir uses the same
``[environment].docker_image`` + ``instruction.md`` shape), rollouts run through pier's native Job
driver (:class:`~beagle.benchmarks.harness.PierHarness`), and the separate verifier's in-band
reward is read directly (:class:`~beagle.benchmarks.grader.InBandGrader`). No loader, grader, or
runner code — and no prompt framing: pier hands the agent its native ``instruction.md``.

DeepSWE = 113 SWE tasks across 5 languages (`datacurve-ai/deep-swe`), each a prebuilt public-ECR
image + a separate verifier (`environment_mode="separate"`) that grades with ``reward.json``. Needs
the ``beagle[deep-swe]`` extra (``datacurve-pier``). See ``vendor/xrlenv`` docs (deep_swe /
pier_framework) for how the corpus is materialized and driven.
"""

from __future__ import annotations

from typing import ClassVar

from beagle.benchmarks.base import Benchmark, BenchmarkHarness, Grader, TaskSource
from beagle.benchmarks.grader import InBandGrader
from beagle.benchmarks.harness import PierHarness
from beagle.benchmarks.registry import register
from beagle.benchmarks.source import HarborCache


@register("deep-swe")
class DeepSwe(Benchmark):
    """DeepSWE: benchmark-cache source + pier Job harness + in-band verifier reward.

    ``reward.json`` carries a headline ``reward`` (1.0 on a full solve). The xrlenv oracle gate
    keys on ``reward > 0``; here we keep beagle's standard resolved threshold (``reward >= 1.0``,
    a full solve), with the reward value itself the graded/fitness signal.
    """

    name = "deep-swe"
    #: The benchmark-cache shard dir xrlenv materialized (matches the registry name here).
    cache_name: ClassVar[str] = "deep-swe"

    def source(self) -> TaskSource:
        return HarborCache(self.name, cache_name=self.cache_name)

    def harness(self, env_import_path: str | None = None) -> BenchmarkHarness:
        return PierHarness(env_import_path=env_import_path)

    def grader(self) -> Grader:
        return InBandGrader()


__all__ = ["DeepSwe"]
