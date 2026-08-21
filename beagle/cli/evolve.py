"""``beagle evolve`` — run a full evolution loop (``Trainer.fit``) from a canonical config.yaml.

Loads the self-contained config (:mod:`beagle.cli._canonical`), runs the version gate, assembles
the Trainer (evolvee/evolver/algorithm), and hands off to ``Trainer.fit`` → ``algorithm.evolve``.
DarwinX's ``evolve`` launches the vendored driver, which needs a ``benchmark`` to score on and the
launch paths in ``algorithm.hparams`` (injected by the loader).
"""

from __future__ import annotations

import argparse


def _cmd_evolve(args: argparse.Namespace) -> int:
    import beagle as bgl
    from beagle.cli._canonical import build_evolution, check_versions, load

    raw = load(args.config)
    check_versions(raw)                             # fail loud on a pinned-vs-installed version mismatch
    cfg, run_dir, run_name = build_evolution(raw)
    print(f"[beagle evolve] campaign={run_name!r}  run dir: {run_dir}")

    trainer = bgl.Trainer.from_config(cfg)
    train_ds = bgl.TaskDataset.from_benchmark(cfg.benchmark)

    if args.dry_run:
        trainer.dry_run(train_dataset=train_ds)     # resolve + print the plan, no spend
        return 0
    best = trainer.fit(train_dataset=train_ds)
    print(f"[beagle evolve] best: {best}")
    return 0
