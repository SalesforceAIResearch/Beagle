"""``runner.run`` shim (seam A) — the vendored DarwinX eval shells ``python -m runner.run
<config> --results-root <dir> …``; this routes that to beagle's eval adapter
(:func:`beagle.algorithms.darwinx.eval.run_eval` → the Runner → the ``run.json`` DarwinX reads).

Only reachable when ``_shims/`` is on ``PYTHONPATH`` (set by ``DarwinX.evolve`` at launch), so a
normal ``python -m runner.run`` elsewhere never picks it up. Unknown flags the vendored caller
passes are tolerated (``parse_known_args``) — extend the mapping if one turns out to matter."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from beagle.algorithms.darwinx.eval import run_eval

    p = argparse.ArgumentParser(prog="runner.run")
    p.add_argument("config")
    p.add_argument("--results-root", required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--campaign-id", default=None)
    p.add_argument("--include-task-name", action="append", default=None)
    args, _unknown = p.parse_known_args(argv)   # tolerate extra flags the vendored caller passes

    run_dir = run_eval(
        args.config, results_root=args.results_root, run_id=args.run_id,
        include_task_name=args.include_task_name, campaign_id=args.campaign_id,
    )
    print(str(run_dir))
    return 0


if __name__ == "__main__":  # `python -m runner.run`
    sys.exit(main())
