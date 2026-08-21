"""beagle — a PyTorch-like framework for agent-harness evolution.

Top-level surface (assembled like ``torch``)::

    import beagle as bgl

    trainer = bgl.Trainer(
        evolvee=bgl.agents.build("monet"),          # the harness being evolved
        evolver=bgl.agents.build("cursor"),         # the coding agent that edits it
        algorithm=bgl.algorithms.build("darwinx"),  # the evolution "optimizer"
    )
    train_ds, val_ds = bgl.DataMixture.from_config(cfg).split()
    best = trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

Subpackages map one-to-one onto the design plot:

* :mod:`beagle.agents`     — Model / Agent Factory
* :mod:`beagle.algorithms` — Optimizer / Evolve Algorithm Factory
* :mod:`beagle.data`       — dataloader (DataMixture)
* :mod:`beagle.benchmarks` — benchmark integration
* :mod:`beagle.rollout`    — rollout infra (xrlenv-backed runner)
* :mod:`beagle.trainer`    — Trainer entrypoint
"""

from __future__ import annotations

from beagle import agents, algorithms, benchmarks, data, eval, rollout
from beagle.config import RunConfig, BeagleConfig, load_config, load_evolve_config
from beagle.dotenv import load_project_dotenv as load_dotenv
from beagle.eval import evaluate
from beagle.data import DataMixture, TaskDataset
from beagle.trainer import Trainer
from beagle.types import (
    AgentRole,
    RolloutStatus,
    Task,
    TaskContext,
    TaskResult,
    Transparency,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    # subpackages
    "agents",
    "algorithms",
    "benchmarks",
    "data",
    "eval",
    "rollout",
    # top-level convenience
    "Trainer",
    "evaluate",
    "DataMixture",
    "TaskDataset",
    "RunConfig",
    "BeagleConfig",
    "load_config",
    "load_evolve_config",
    "load_dotenv",
    # core types
    "Task",
    "TaskContext",
    "TaskResult",
    "RolloutStatus",
    "AgentRole",
    "Transparency",
]
