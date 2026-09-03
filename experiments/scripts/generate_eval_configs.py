#!/usr/bin/env python3
"""Generate BASELINE eval configs (a parameterized sweep) for the experiments/ area.

This BORROWS the canonical matrix from ``scripts/generate_eval_configs.py`` — each agent's
``extra_args`` (e.g. monet's REQUIRED stream flags), each benchmark's ``dataset``/``split`` and
per-benchmark ``parallelism``, the ``forward_env`` list, and the version→manifest join — so the
agent/benchmark facts stay in ONE place. What lives HERE is the *experiment* surface: the model,
reasoning effort, max turns, and the output naming/layout, all exposed as CLI flags.

Config name (one per agent × benchmark)::

    {harness}_{bench-short}_{model}_{effort}_{max_turns}.yaml
    e.g.  monet-20260826_tb21_gpt-5.6-sol_medium_200.yaml

(bench-short comes from the canonical BENCHMARKS table: terminal_bench_2_1→tb21,
deep-swe→deepswe, swe-bench-verified→swebench_verified, swe-rebench→swerebench.)

Examples::

    # ONE command = every onboarded agent × every registered benchmark, at the baseline knobs
    # (gpt-5.6-sol / medium / 200 turns). Both matrices come from the canonical tables, so a newly
    # onboarded agent or benchmark is picked up here with no edit.
    python experiments/scripts/generate_eval_configs.py

    # one config
    python experiments/scripts/generate_eval_configs.py --agents monet-20260826 --benches deep-swe

    # a variant sweep (different model/effort/turns → different filenames, no collision)
    python experiments/scripts/generate_eval_configs.py --model gpt-5.6 --effort high --max-turns 150

    python experiments/scripts/generate_eval_configs.py --check      # dry-run: list, write nothing
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = REPO_ROOT / "scripts" / "generate_eval_configs.py"
OUT_DIR = REPO_ROOT / "experiments" / "configs" / "eval_baseline"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
MANIFEST_DIR = REPO_ROOT / ".beagle" / "agents"


def _load_canonical():
    """Import ``scripts/generate_eval_configs.py`` as a module (it's a script, not a package),
    to reuse AGENTS / BENCHMARKS / _FORWARD_ENV / _manifests_by_version. No side effects on import."""
    spec = importlib.util.spec_from_file_location("canonical_gen", CANONICAL)
    assert spec and spec.loader, f"cannot load {CANONICAL}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_canonical()


# --- baseline defaults (overridable by flags) --------------------------------
# DERIVED from the canonical tables, not restated: the whole point of borrowing AGENTS /
# BENCHMARKS is that ONE command regenerates every config, and a hand-kept copy silently drops any
# newly-onboarded benchmark from the default sweep (swe-rebench was missing for exactly that
# reason). Narrow a run with --agents / --benches; never by editing a list here.
DEF_AGENTS = [label for _n, _v, label in gen.agent_cells()]   # one per experiment COPY
DEF_BENCHES = list(gen.BENCHMARKS)
DEF_MODEL = "gpt-5.6-sol"
DEF_EFFORT = "medium"
DEF_MAX_TURNS = 200
DEF_PROVIDER = "llm-gateway-express-local-proxy"
DEF_RUNTIME = "xrlenv-cluster"
# Last-resort agent wall clock, emitted explicitly into each config; it applies ONLY to a benchmark
# that ships no agent budget of its own. A harbor task's own task.toml budget always wins, and is
# scaled (not replaced) by the multiplier.
DEF_TIMEOUT = 1800
DEF_TIMEOUT_MULTIPLIER = 1.0
DEF_RETRY_INFRA = 2
DEF_PARALLELISM = 32  # deep-swe overrides lower (from the canonical BENCHMARKS table)


#: label -> (harness name, version). A label is the bare harness name while it has one copy, and
#: ``<name>-<version>`` once it has more, so two copies never collide on a filename or a run.name.
_CELLS = {label: (name, version) for name, version, label in gen.agent_cells()}


def _agent_label(value: str) -> str:
    """Accept a copy LABEL (``<harness>-<version>``), not a bare harness name.

    A harness can have several experiment copies, and the version is what says which one — so a
    bare name would be ambiguous the moment a second copy is added, and silently mean "the first
    one" until then. Names the valid labels, since the bare form is what older commands pass.
    """
    if value in _CELLS:
        return value
    same = [label for label in _CELLS if label.rsplit("-", 1)[0] == value]
    hint = f"did you mean {', '.join(same)}?" if same else f"choose from: {', '.join(_CELLS)}"
    raise argparse.ArgumentTypeError(
        f"{value!r} is not an experiment copy — pass <harness>-<version>; {hint}")


def config_stem(agent: str, bench: str, *, model: str, effort: str, max_turns: int) -> str:
    """``{label}_{bench-short}_{model}_{effort}_{max_turns}`` — the file name and the run.name."""
    return f"{agent}_{gen.BENCHMARKS[bench]['short']}_{model}_{effort}_{max_turns}"


def build_config(agent: str, bench: str, manifest: dict, args: argparse.Namespace) -> dict:
    """The eval-config shape, with the experiment knobs from ``args`` and the agent/benchmark facts
    reused from the canonical tables. ``source`` is filled from the onboarded ``manifest``."""
    harness, version = _CELLS[agent]
    a, b = gen.AGENTS[harness], gen.BENCHMARKS[bench]
    source = {"repo": manifest["repo"], "ref": manifest["ref"]}
    if manifest.get("token_env"):
        source["token_env"] = manifest["token_env"]
    data_entry: dict = {"benchmark": bench}
    for k in ("dataset", "split"):
        if b.get(k):
            data_entry[k] = b[k]
    parallelism = args.parallelism if args.parallelism is not None else b.get("parallelism", DEF_PARALLELISM)
    return {
        "run": {
            "dir": str(args.results),
            "name": config_stem(agent, bench, model=args.model, effort=args.effort, max_turns=args.max_turns),
            "runtime": args.runtime,
            "parallelism": parallelism,
            # A two-phase benchmark (SWE-bench) fans patch EVAL out wider than patch generation.
            **({"parallelism_eval_patches": b["parallelism_eval_patches"]}
               if b.get("parallelism_eval_patches") else {}),
            # Scales the TASK's own declared phase budgets. RUN-level, not under `retry`.
            "timeout_multiplier": args.timeout_multiplier,
            # Task-level infra retry: re-run a trial on an infra-transient in a fresh container;
            # content outcomes are never re-rolled. See notes/retry-coverage.md.
            "retry": {"infra": args.retry_infra},
        },
        "agent": {
            "harness": {"name": harness, "version": version, "source": source},
            "model": {"name": args.model},
            "provider": args.provider,
            "effort": args.effort,
            "max_turns": args.max_turns,
            "forward_env": list(gen._FORWARD_ENV),
            "timeout": args.timeout,
            "extra_args": a["extra_args"],
        },
        "data": [data_entry],
    }


def _dump(agent: str, bench: str, cfg: dict, args: argparse.Namespace) -> str:
    header = [
        f"# BASELINE eval config — GENERATED by experiments/scripts/generate_eval_configs.py "
        f"(do not edit; regenerate).",
        f"# baseline: model={args.model} effort={args.effort} max_turns={args.max_turns} "
        f"| agent={agent} benchmark={bench}",
    ]
    for src in (gen.AGENTS[_CELLS[agent][0]].get("note"), gen.BENCHMARKS[bench].get("note")):
        if src:
            header.append(f"# {src}")
    body = gen._annotate_timeouts(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100))
    return "\n".join(header) + "\n" + body


def generate(args: argparse.Namespace) -> tuple[int, list[str]]:
    """Write one config per (agent × benchmark) in the flags' cross-product. Returns
    (written, missing-agent names)."""
    manifests = gen._manifests_by_version(Path(args.manifest_dir))
    missing = [f"{ag} (v{_CELLS[ag][1]})"
               for ag in args.agents if manifests.get(_CELLS[ag][1]) is None]
    out = Path(args.out)
    print(f"[gen] baseline configs → {out}  (model={args.model} effort={args.effort} turns={args.max_turns})")
    written = 0
    for agent in args.agents:
        manifest = manifests.get(_CELLS[agent][1])
        if manifest is None:
            continue
        for bench in args.benches:
            cfg = build_config(agent, bench, manifest, args)
            dest = out / f"{config_stem(agent, bench, model=args.model, effort=args.effort, max_turns=args.max_turns)}.yaml"
            if not args.check:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(_dump(agent, bench, cfg, args), encoding="utf-8")
            print(f"  {'would write' if args.check else '✓'} {dest.relative_to(REPO_ROOT)}")
            written += 1
    return written, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate baseline eval configs (experiments/).")
    ap.add_argument("--agents", nargs="+", default=DEF_AGENTS, type=_agent_label,
                    metavar="AGENT",
                    help=f"experiment copies, as <harness>-<version> (default: all — "
                         f"{', '.join(DEF_AGENTS)})")
    ap.add_argument("--benches", nargs="+", default=DEF_BENCHES, choices=list(gen.BENCHMARKS),
                    metavar="BENCH", help=f"benchmarks (default: {' '.join(DEF_BENCHES)})")
    ap.add_argument("--model", default=DEF_MODEL, help=f"model name (default: {DEF_MODEL})")
    ap.add_argument("--effort", default=DEF_EFFORT, help=f"reasoning effort (default: {DEF_EFFORT})")
    ap.add_argument("--max-turns", type=int, default=DEF_MAX_TURNS, dest="max_turns",
                    help=f"agent turn cap (default: {DEF_MAX_TURNS})")
    ap.add_argument("--parallelism", type=int, default=None,
                    help="override trial parallelism (default: per-benchmark, else 32)")
    ap.add_argument("--timeout", type=int, default=DEF_TIMEOUT,
                    help="last-resort wall clock s, used only by benchmarks that declare none "
                         f"(default: {DEF_TIMEOUT})")
    ap.add_argument("--timeout-multiplier", type=float, default=DEF_TIMEOUT_MULTIPLIER,
                    dest="timeout_multiplier",
                    help="scale the TASK's declared agent/verifier budgets, e.g. 1.5 "
                         f"(default: {DEF_TIMEOUT_MULTIPLIER} = the task's own value)")
    ap.add_argument("--retry-infra", type=int, default=DEF_RETRY_INFRA, dest="retry_infra",
                    help=f"infra-transient retries (default: {DEF_RETRY_INFRA})")
    ap.add_argument("--runtime", default=DEF_RUNTIME, help=f"runtime (default: {DEF_RUNTIME})")
    ap.add_argument("--provider", default=DEF_PROVIDER, help=f"model provider (default: {DEF_PROVIDER})")
    ap.add_argument("--out", default=str(OUT_DIR), metavar="DIR", help="output dir for the .yaml configs")
    ap.add_argument("--results", default=str(RESULTS_DIR), metavar="DIR",
                    help="run.dir baked into each config (where raw results land)")
    ap.add_argument("--manifest-dir", default=str(MANIFEST_DIR), metavar="DIR",
                    help="onboarded manifests (default: .beagle/agents)")
    ap.add_argument("--check", action="store_true", help="dry-run: list what would be written, write nothing")
    args = ap.parse_args(argv)

    written, missing = generate(args)
    print(f"[gen] {'would generate' if args.check else 'generated'} {written} config(s)")
    if missing:
        print(f"[gen] not onboarded (skipped): {', '.join(missing)} — onboard with the matching --version")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
