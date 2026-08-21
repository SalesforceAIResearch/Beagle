"""The ``beagle`` command-line entrypoint.

Thin wrapper over the library — parses a canonical ``config.yaml`` and dispatches to a subcommand.
Two verbs, one config shape:

* ``beagle evaluate --config config.yaml`` — roll one agent over a benchmark (no evolution).
* ``beagle evolve   --config config.yaml`` — evolve an agent (``Trainer.fit``).

Both take ``--dry-run`` (resolve + print the plan, no spend). ``evaluate`` also carries the run
ops flags (resume / retry / force-resume / campaign / run-id / run-dir). The parser + dispatch live
here; each subcommand's logic is its own module (:mod:`.evaluate`, :mod:`.evolve`). For a raw schema
lint of a config file, use ``python -m beagle.config <file>``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

# Re-exported so `beagle.cli._dry_run` stays importable (and monkeypatchable) for tests.
from beagle.cli.evaluate import _dry_run

__all__ = ["_dry_run", "main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="beagle", description="Agent-harness evolution.")
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", required=True, help="Path to a canonical config.yaml.")
        sp.add_argument("--dry-run", action="store_true",
                        help="resolve + print the plan and exit — no spend")
        sp.add_argument("--env-file", default=None,
                        help="path to a .env of facts/secrets (default: nearest .env); host env wins")

    evaluate = sub.add_parser("evaluate", help="Score one agent on a benchmark (no evolution) from a config.yaml.")
    _common(evaluate)
    # Three independent resume/retry flags — each re-runs only its own category; combine to union.
    evaluate.add_argument("--resume", action="store_true",
                          help="re-run tasks with NO result on disk (interrupted); keep completed tasks")
    evaluate.add_argument("--retry-errors", action="store_true",
                          help="re-run tasks that recorded an error (a 500, a timeout, a clone fail, a "
                               "no-attempt) — your call whether that helps. NOT genuine capability failures "
                               "(which carry no error). Preview with --dry-run")
    evaluate.add_argument("--retry-unresolved", action="store_true",
                          help="re-run EVERY unresolved task (error + genuine-fail) — a superset of "
                               "--retry-errors that ALSO re-runs genuine capability failures; for deliberate "
                               "re-sampling (pass@k / after a harness fix). Fails loud if a task has no "
                               "resolved signal")
    evaluate.add_argument("--task-ids", default=None,
                          help="comma-separated task ids to restrict the run to (subset of the benchmark). "
                               "Fresh run: run only these. With --retry-errors/--retry-unresolved: scope the "
                               "retry to these (only the ones in that category re-run). Unknown id → error")
    evaluate.add_argument("--force-resume", action="store_true",
                          help="resume even if the config changed (bypass the drift guard; records both hashes)")
    evaluate.add_argument("--campaign-id", default=None, help="logical grouping label (recorded, CLI-only)")
    evaluate.add_argument("--run-id", default=None, help="reuse/override the run id (default: derived from config)")
    evaluate.add_argument("--run-dir", default=None,
                          help="explicit output dir (verbatim); overrides the config's run.dir/run.name")

    evolve = sub.add_parser("evolve", help="Evolve an agent on a benchmark (Trainer.fit) from a config.yaml.")
    _common(evolve)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Both commands execute → load bucket-1 facts/secrets from .env first (host env wins);
    # run knobs come from the config, not here.
    from beagle.dotenv import load_project_dotenv

    load_project_dotenv(getattr(args, "env_file", None))

    try:
        if args.command == "evaluate":
            from beagle.cli.evaluate import _cmd_evaluate

            return _cmd_evaluate(args)

        if args.command == "evolve":
            from beagle.cli.evolve import _cmd_evolve

            return _cmd_evolve(args)
    except KeyboardInterrupt:
        # Ctrl-C already tore this run's containers down on the cluster (see
        # beagle.rollout.interrupt.stop_run_on_sigint). Exit cleanly with the conventional 130
        # instead of dumping a KeyboardInterrupt traceback at the user.
        print("\n[beagle] aborted (SIGINT).", file=sys.stderr, flush=True)
        return 130

    raise NotImplementedError(f"beagle {args.command!r} is not yet implemented")
