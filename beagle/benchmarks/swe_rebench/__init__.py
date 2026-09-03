"""SWE-rebench (harbor family) — the thin beagle adapter.

860 curated Python SWE tasks (Nebius AI R&D), onboarded once in xrlenv (build_cache + oracle).
Tasks come from the benchmark cache, rollouts run through harbor's native trial driver, and the
verifier's in-band 0/1 reward is read directly — the zero-code
:class:`~beagle.benchmarks.harness.HarborBenchmark` defaults.

The one thing it adds is :meth:`task_env`. Unlike terminal-bench, these images do **not** reliably
declare a ``WORKDIR`` (measured: 10 of 24 sampled images sit at ``/testbed``, the rest at ``/`` or
``""``) and do **not** put the task's conda env on ``PATH`` — the corpus's own verifier resolves
both for itself before running the tests (``tests/test.sh``: ``resolve_repo_dir`` /
``activate_testbed_env``). Placing the agent is therefore the harness's job here, exactly as it is
in upstream's evaluation, and beagle is the harness. The snippets below mirror the verifier's two
helpers so the agent phase gets the same workspace the verifier grades.

Because the oracle gate runs those same self-resolving scripts, a green oracle sweep says nothing
about whether the agent phase is set up correctly — see ``notes/swe-rebench-onboarding.md``.
"""

from __future__ import annotations

from typing import ClassVar

from beagle.benchmarks.harness import HarborBenchmark
from beagle.benchmarks.registry import register

#: Print the task's repo dir. Mirrors ``tests/test.sh:resolve_repo_dir()`` — ``/testbed`` is what
#: upstream's own harness assumes for all 860, with the ``find`` fallback the verifier also
#: carries — widened to ``-e`` / no ``-type d`` because a git worktree's ``.git`` is a file.
_RESOLVE_REPO_DIR = r"""
if [ -e /testbed/.git ]; then printf '/testbed'
else
  d=$(find / -maxdepth 3 -name .git 2>/dev/null | head -1 | sed 's|/\.git$||')
  if [ -n "$d" ]; then printf '%s' "$d"
  elif [ -d /testbed ]; then printf '/testbed'
  else pwd
  fi
fi
"""

#: Put the task's interpreter on PATH. Mirrors ``tests/test.sh:activate_testbed_env()``; upstream's
#: own setup hardcodes the env name ``testbed`` for the whole corpus, so that is tried first and the
#: single-non-base-env sweep is only a fallback. The Python version varies per task (3.11 / 3.12) but
#: is never chosen — it comes with the env.
_ACTIVATE_TASK_ENV = r"""
for r in /opt/conda /opt/miniconda3; do
  for n in testbed $(ls -1 "$r/envs" 2>/dev/null | grep -vx base); do
    [ -d "$r/envs/$n/bin" ] && { export PATH="$r/envs/$n/bin:$PATH"; break 2; }
  done
done
"""


@register("swe-rebench")
class SweRebench(HarborBenchmark):
    """SWE-rebench: benchmark-cache source + harbor Job harness + in-band verifier reward."""

    #: The cache shard xrlenv materialized (matches the registry name here).
    cache_name: ClassVar[str] = "swe-rebench"

    def task_env(self) -> dict[str, str]:
        return {"repo_path_cmd": _RESOLVE_REPO_DIR, "shell_preamble": _ACTIVATE_TASK_ENV}


__all__ = ["SweRebench"]
