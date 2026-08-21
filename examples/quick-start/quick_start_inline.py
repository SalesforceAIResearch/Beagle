"""Quick start — evolve a coding harness with the PyTorch-like ``Trainer`` API (inline variant).

The mental model mirrors ``torch``'s *model / optimizer / dataloader → ``trainer.fit(...)``*:

    evolvee    the harness being optimized        (θ — like the ``nn.Module``)
    evolver    the coding agent that edits it      (the mutation operator)
    algorithm  the evolution strategy              (the optimizer, e.g. DarwinX)
    dataset    the tasks candidates are scored on  (like the ``DataLoader``)

You compose those four as plain Python objects and hand them to a ``Trainer`` — no YAML. (The
identical run driven from a single ``config.yaml`` is ``beagle evolve --config config.yaml``.)

The evolvee (repo / ref / local checkout) is read from an **onboarded-agent manifest**
(``.beagle/agents/<name>.json``, produced by ``python -m beagle.tools.onboard``).

Usage
-----
    python examples/quick-start/quick_start_inline.py --dry-run       # preview the plan, NO spend
    python examples/quick-start/quick_start_inline.py --agent my_agent  # pick an onboarded agent
    python examples/quick-start/quick_start_inline.py                 # launch the evolution loop (spends)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import beagle as bgl
from beagle.algorithms import DarwinXConfig
from beagle.config import (
    AgentConfig,
    AgentSourceConfig,
    BenchmarkConfig,
    ClaudeModelConfig,
    ModelConfig,
)
from beagle.tools.onboard import latest_manifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]   # the beagle repo (examples/quick-start/<this>)

# Portable run knobs — edit to taste (these are benchmark-side, not machine-specific).
BENCHMARK = "terminal_bench_2_1"
TASKS = ["bn-fit-modify"]                    # keep it to 1 task for a cheap smoke
EVOLVEE_TYPE = "monet"                       # the agent adapter the onboarded repo is built with
EVOLVEE_MODEL = "claude-opus-4-8"
EVOLVEE_EFFORT = "high"                      # monet reasoning effort (→ --effort; default `none` is weak)
EVOLVER = "cursor"
#: The cursor proposer model — a bare family slug passed verbatim to `cursor-agent --model`.
EVOLVER_MODEL = "gpt-5.5-high"


def _evolvee_agent_config(m: dict) -> AgentConfig:
    """θ (the harness under evolution) as a declarative :class:`AgentConfig`, pinned to the
    manifest's experiment copy @ its baseline ref. The DarwinX driver supplies monet's gateway
    provider itself, so only the reasoning effort + clone credential live here."""
    config: dict = {"effort": EVOLVEE_EFFORT}
    if m.get("token_env"):
        config["token_env"] = m["token_env"]      # clone credential for the private experiment copy
    return AgentConfig(
        name=EVOLVEE_TYPE,
        model=ClaudeModelConfig(name=EVOLVEE_MODEL),
        source=AgentSourceConfig(repo=m["repo"], ref=m["ref"]),
        config=config,
    )


def build_trainer(agent: str, *, runtime: str, run_dir: Path, runname: str) -> bgl.Trainer:
    """Compose the four pieces from the declarative **Config** classes (what a YAML would hold —
    here inline). The evolvee is read from the onboarded-agent manifest."""
    m = load_manifest(agent, root=REPO_ROOT)          # {repo, ref, token_env?, dir}

    evolvee = bgl.agents.build(_evolvee_agent_config(m))
    # the mutation operator — the cursor CLI, driven as a black-box Editor. cursor bakes reasoning
    # effort into the slug (EVOLVER_MODEL, e.g. "gpt-5.5-high"), so a plain ModelConfig is right.
    evolver = bgl.agents.build(AgentConfig(name=EVOLVER, model=ModelConfig(name=EVOLVER_MODEL)))

    # the optimizer — DarwinX, configured by its typed DarwinXConfig (every field validated).
    #   repo_root         the run home (<dir>/<runname>): worktrees + the genealogy DB + config
    #   evolvee_checkout  the manifest's local clone, linked in as <repo_root>/monet_code
    #   campaign          the run's id (= runname), namespacing the genealogy DB
    #   evolvee_effort    monet --effort on the DarwinX eval path (else the driver's default `none`)
    algorithm = bgl.algorithms.build(DarwinXConfig(
        repo_root=str(run_dir),
        evolvee_checkout=str((REPO_ROOT / m["dir"]).resolve()),
        campaign=runname,
        max_loop_iters=1,
        evolvee_effort=EVOLVEE_EFFORT,
    ))
    return bgl.Trainer(
        evolvee=evolvee, evolver=evolver, algorithm=algorithm,
        trainer_config={"runtime": {"kind": runtime}},
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default=None,
                   help="onboarded agent manifest under .beagle/agents/ (default: the newest)")
    p.add_argument("--dir", default=None,
                   help="base directory to host run results (default: <repo>/.beagle/runs)")
    p.add_argument("--runname", default=None,
                   help="run name; results land in <dir>/<runname>/ (default: the agent name)")
    p.add_argument("--runtime", default="xrlenv-cluster", choices=["local", "xrlenv-cluster"])
    p.add_argument("--dry-run", action="store_true",
                   help="resolve + print the plan and exit — no spend (default: launch the loop)")
    args = p.parse_args()

    # Bucket-1 facts/secrets (xrlenv topology + gateway creds + benchmark cache) from .env.
    bgl.load_dotenv()

    agent = args.agent or latest_manifest(root=REPO_ROOT)
    base_dir = Path(args.dir) if args.dir else REPO_ROOT / ".beagle" / "runs"
    runname = args.runname or agent
    run_dir = base_dir / runname                       # results land here: <dir>/<runname>/
    print(f"[quick-start] run dir: {run_dir}")

    trainer = build_trainer(agent, runtime=args.runtime, run_dir=run_dir, runname=runname)

    # the data — the dataset carries its benchmark spec, so the Trainer derives the eval config
    # from it (the native harness loads the tasks in-trial). Loads from the .env benchmark cache.
    train_ds = bgl.TaskDataset.from_benchmark(BenchmarkConfig(name=BENCHMARK, task_ids=TASKS))

    if args.dry_run:
        trainer.dry_run(train_dataset=train_ds)       # resolve + print, no spend
        return 0

    best = trainer.fit(train_dataset=train_ds)         # launch the loop → the best evolved harness
    print(f"\nbest candidate: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
