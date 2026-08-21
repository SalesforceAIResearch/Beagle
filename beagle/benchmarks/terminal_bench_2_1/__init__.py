"""Terminal-Bench 2.1 (harbor family) — the thin beagle adapter.

Onboarded once in xrlenv (build_cache + oracle). Here it is the whole integration:
tasks come from the harbor cache, rollouts run through harbor's native trial driver,
and the verifier's in-band reward is read directly. No loader, grader, or runner code —
and no prompt framing: on the harbor path the agent is handed harbor's own native
``instruction.md`` (the task goal), and the container/verifier context ("shell access,
the verifier grades the final container state, don't touch ``/tests/``") is the agent's
own to say (its system prompt + generic instruction), not beagle's. See
``notes/task-prompt-injection.md``.
"""

from __future__ import annotations

from beagle.benchmarks.harness import HarborBenchmark
from beagle.benchmarks.registry import register


@register("terminal_bench_2_1")
class TerminalBench21(HarborBenchmark):
    """Terminal-Bench 2.1 — ``terminal_bench_2_1`` is the canonical beagle name.

    xrlenv materialized the cache under the hyphenated dir, so ``cache_name`` maps the
    canonical (underscored) name to that on-disk location.
    """

    #: xrlenv's on-disk harbor-cache dir (hyphenated); the registry name is underscored.
    cache_name = "terminal-bench-2-1"


__all__ = ["TerminalBench21"]
