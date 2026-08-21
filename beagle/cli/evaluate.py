"""``beagle evaluate`` — score ONE agent on a benchmark (no evolution) from a canonical config.yaml.

Loads the self-contained config (:mod:`beagle.cli._canonical`), runs the version gate, builds a
:class:`~beagle.config.RunConfig` from the ``agent`` + ``data`` blocks, and hands it to
``bgl.evaluate`` — which rolls every task through the benchmark's native harness and writes run.json.

``--dry-run`` resolves the plan (tasks, source, gateway) without spending. The ops flags
(``--resume`` / ``--retry-errors`` / ``--force-resume`` / ``--campaign-id`` / ``--run-id`` /
``--run-dir``) drive resume + retry; ``--run-dir``/``--run-id`` override the config's ``run:`` block.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import beagle as bgl
    from beagle.cli._canonical import build_evaluation, check_versions, load

    raw = load(args.config)
    check_versions(raw, roles=("agent",))           # fail loud on a pinned-vs-installed version mismatch
    run_cfg, cfg_run_dir = build_evaluation(raw)
    run_dir = args.run_dir or str(cfg_run_dir)       # --run-dir overrides the config's run.dir/run.name
    spec = run_cfg.agent_spec()
    dataset = bgl.TaskDataset.from_benchmark(run_cfg.benchmark_spec())
    # --task-ids RESTRICTS the re-run set (not a dataset filter): the full dataset still drives
    # run.json so its aggregate covers the whole benchmark; only the named tasks actually re-run.
    only_task_ids = None
    if args.task_ids:
        only_task_ids = {t.strip() for t in args.task_ids.split(",") if t.strip()}
        unknown = only_task_ids - {t.task_id for t, _ in dataset}
        if unknown:
            raise SystemExit(f"--task-ids: unknown task id(s) {sorted(unknown)} not in "
                             f"benchmark {run_cfg.benchmark.name!r}")
    print(f"[beagle evaluate] agent={run_cfg.agent.name!r}  benchmark={run_cfg.benchmark.name!r}  "
          f"run dir: {run_dir}")

    if args.dry_run:  # resolve the plan (tasks, source, gateway) — roll out nothing, spend nothing
        return _dry_run(run_cfg, spec, list(dataset), run_dir=Path(run_dir),
                        resume=args.resume, retry_errors=args.retry_errors,
                        retry_unresolved=args.retry_unresolved, only_task_ids=only_task_ids)

    from beagle.rollout.interrupt import stop_run_on_sigint
    from beagle.rollout.run_id import build_run_id, compute_config_hash
    from beagle.rollout.runtime import RuntimeConfig as RtCfg
    from beagle.rollout.runtime import build_runtime

    # Resolve the run_id up front (same as the Runner would) so the runtime can stamp every
    # container it acquires with ``xrlenv.group_id = run_id`` — that tag is what a Ctrl-C teardown
    # targets. Passing the SAME run_id into evaluate keeps the Runner from recomputing a different one.
    run_id = args.run_id or build_run_id(run_cfg, compute_config_hash(run_cfg.model_dump(mode="json")))

    # The container substrate the harness hands each rollout (None on the harbor path — harbor owns
    # its trial container; a real runtime for docker-harness benchmarks like SWE-bench).
    rt = build_runtime(RtCfg(kind=run_cfg.runtime.kind, run_id=run_id))
    # Ctrl-C → actively stop THIS run's containers on the cluster (node-confirmed destroy frees
    # capacity now) instead of leaving them for xrlenv's ~120 s raw-liveness reaper.
    with stop_run_on_sigint(rt, run_id):
        result = bgl.evaluate(
            run_cfg, run_dir=run_dir, run_id=run_id, resume=args.resume,
            retry_errors=args.retry_errors, retry_unresolved=args.retry_unresolved,
            only_task_ids=only_task_ids, force_resume=args.force_resume,
            campaign_id=args.campaign_id, config_path=args.config,
            agent=bgl.agents.build(spec), dataset=dataset, runtime=rt,
        )
    print(f"[beagle evaluate] run dir: {result.artifact_dir}   score: {result.score:.3f}")
    for r in result.results:
        line = f"    {r.task_id:40} resolved={r.resolved!s:5} reward={r.reward} status={r.status.name}"
        if r.error:
            line += f"\n        error: {r.error}"
        print(line)
    return 0


def _resume_plan_lines(items, *, run_dir: Path, resume: bool, retry_errors: bool,
                       retry_unresolved: bool,
                       only_task_ids: set[str] | None = None) -> tuple[list[str], int]:
    """Render the resume/retry plan (what re-runs vs is kept) as printable lines for ``--dry-run``,
    plus the count of tasks that would actually re-run (for the cost estimate).

    Groups ``items`` by benchmark (the Runner's unit), reads each harness's ``completed(run_dir)``,
    and applies :func:`~beagle.rollout.resume.plan_resume` — the SAME decision the Runner makes.
    Every re-run task is listed in full (task id · category · signal); kept tasks are summarized by
    category (a 500-task run shouldn't dump 500 rows)."""
    from beagle import benchmarks
    from beagle.rollout.resume import plan_resume

    groups: dict[str, list] = {}
    for t, c in items:
        groups.setdefault(t.benchmark, []).append((t, c))

    flags = f"resume={resume}  retry-errors={retry_errors}  retry-unresolved={retry_unresolved}"
    lines = ["", f"  resume plan  ·  {flags}"]
    tot_retry = tot_keep = 0
    for name, grp in groups.items():
        harness = benchmarks.get(name).harness()
        prior = harness.completed(grp, run_dir=run_dir)
        if len(groups) > 1:
            lines.append(f"    [{name}]")
        try:
            plan = plan_resume(grp, prior, resume=resume, retry_errors=retry_errors,
                               retry_unresolved=retry_unresolved, only_task_ids=only_task_ids,
                               label=name)
        except RuntimeError as e:
            lines.append(f"      ⚠ {e}")
            continue
        retry_rows = [d for d in plan.decisions if d.retry]
        lines.append(f"      ↻ RETRY ({len(retry_rows)}):" if retry_rows else "      ↻ RETRY (0)")
        for d in retry_rows:
            lines.append(f"        {d.task_id:44.44}  {d.category:18}  {d.signal:.60}")
        kept_by_cat: dict[str, int] = {}   # count the KEPT tasks per category (a category can be
        for d in plan.decisions:           # split — e.g. some errored re-run in scope, others kept)
            if not d.retry and d.task_id in plan.keep:
                kept_by_cat[d.category] = kept_by_cat.get(d.category, 0) + 1
        kept = "  ·  ".join(f"{c} {n}" for c, n in kept_by_cat.items())
        lines.append(f"      · KEEP ({len(plan.keep)}): {kept}" if plan.keep else "      · KEEP (0)")
        tot_retry += len(plan.rerun_ids)
        tot_keep += len(plan.keep)
    skipped = len(items) - tot_retry - tot_keep          # out of --task-ids scope + no result to keep
    tail = f", {skipped} out-of-scope" if skipped else ""
    lines.append(f"    → {tot_retry} re-run, {tot_keep} kept{tail}  ({len(items)} tasks)"
                 f"   [DRY RUN — nothing spent]")
    return lines, tot_retry


def _dry_run(cfg, spec, items, *, run_dir: Path, resume: bool = False,
             retry_errors: bool = False, retry_unresolved: bool = False,
             only_task_ids: set[str] | None = None) -> int:
    """Print what ``beagle evaluate`` *would* do — no rollout, no cost. The pre-flight for a gate
    run: confirms task selection, the resolved agent source, that every task's benchmark actually
    resolves the way the Runner will look it up, and that the gateway env the agent needs is present —
    all before you spend on containers + LLM calls.

    When ``resume`` is set, also prints the **resume plan** (:func:`_resume_plan_lines`): per task,
    the signal it keys off and whether it re-runs — the same decision the Runner makes, so you glimpse
    exactly what a ``--resume [--retry-errors|--retry-unresolved]`` invocation would touch.

    ``items`` is the materialized ``[(Task, TaskContext), …]`` (same order the Runner sees);
    ``run_dir`` is the resolved output dir (from the config's ``run:`` block, or ``--run-dir``)."""
    import os

    from beagle import benchmarks
    from beagle.agents.core.forward_env import normalize_forward_env
    from beagle.rollout.run_id import build_run_id, compute_config_hash

    chash = compute_config_hash(cfg.model_dump(mode="json"))
    run_id = build_run_id(cfg, chash)

    src = spec.source
    cfg_dict = spec.config or {}
    # `provider` is a first-level knob (config["provider"]); fall back to a `--provider` spelled into
    # an agent's raw argv, else none.
    monet_args = list(cfg_dict.get("monet_args") or [])
    provider = (cfg_dict.get("provider")
                or (monet_args[monet_args.index("--provider") + 1] if "--provider" in monet_args else None)
                or "(none)")
    forward = [host for _container, host in normalize_forward_env((spec.config or {}).get("forward_env"))]
    fwd_set = [v for v in forward if os.environ.get(v)]
    fwd_unset = [v for v in forward if not os.environ.get(v)]

    # The Runner groups by Task.benchmark and calls benchmarks.get(that) — resolve it HERE so a
    # name mismatch (e.g. a cache dir name leaking into task identity) fails the pre-flight, not the
    # live run after money's been spent.
    task_ids = [t.task_id for t, _ in items]
    bench_names: list[str] = []
    for t, _ in items:
        if t.benchmark not in bench_names:
            bench_names.append(t.benchmark)
    unresolved: list[tuple[str, str]] = []
    for b in bench_names:
        try:
            benchmarks.get(b)
        except Exception as e:  # noqa: BLE001 — surface any resolution failure as a pre-flight ⚠
            unresolved.append((b, str(e).splitlines()[0]))

    print("[beagle evaluate] DRY RUN — resolves the plan, rolls out NOTHING (no cost).\n")
    print(f"  run_id      : {run_id}")
    print(f"  run_dir     : {run_dir}/")
    print(f"  config_hash : {chash[:12]}…")
    print(f"  model       : {cfg.model.name}   (agent --provider {provider})")
    print(f"  agent       : {spec.name} @ {src.repo}@{src.ref}" if src
          else f"  agent       : {spec.name}   ⚠ NO SOURCE resolved")
    print(f"  benchmark   : {cfg.benchmark.name}")
    if unresolved:
        for b, err in unresolved:
            print(f"                  ⚠ Task.benchmark {b!r} does NOT resolve — {err}")
    else:
        print(f"                  ✓ resolves via benchmarks.get({', '.join(repr(b) for b in bench_names)})")
    print(f"  runtime     : {cfg.runtime.kind}   (parallelism {cfg.parallelism})")
    print(f"  tasks       : {len(task_ids)}")
    n_live = len(task_ids)                            # fresh run → every task rolls out
    if resume or retry_errors or retry_unresolved or only_task_ids:   # resume/retry/scope → show plan
        plan_lines, n_live = _resume_plan_lines(items, run_dir=run_dir, resume=resume,
                                                retry_errors=retry_errors,
                                                retry_unresolved=retry_unresolved,
                                                only_task_ids=only_task_ids)
        for line in plan_lines:
            print(line)
    else:
        for tid in task_ids:
            print(f"                  - {tid}")
    # forward_env is best-effort: an unset host var is skipped, not fatal (the agent adapter warns).
    # The only real failure is ZERO creds forwarded (agent gets nothing). A single unset alternative
    # is fine: the gateway needs ONE of API_KEY / API_KEY_LIST, not both, plus the proxy URL — so we
    # report, we don't judge which subset is present.
    if not forward:
        print("  forward_env : (none)")
    elif not fwd_set:
        print(f"  forward_env : ⚠ 0/{len(forward)} host vars set — nothing forwarded, agent gets no creds")
        print(f"                  not set : {', '.join(fwd_unset)}")
    else:
        print(f"  forward_env : {len(fwd_set)}/{len(forward)} host vars set → forwarded to the agent")
        print(f"                  set     : {', '.join(fwd_set)}")
        if fwd_unset:
            print(f"                  not set : {', '.join(fwd_unset)}  (skipped)")
    print("\n  each trial → agent/{trajectory.json (ATIF), the agent's native stream}, "
          "verifier/reward.txt, artifacts/…")
    print(f"  aggregate  → {run_dir}/run.json  (config + env + per-task reward/tokens "
          "+ totals + fitness score)")
    scope = "tasks" if n_live == len(task_ids) else f"re-run tasks (of {len(task_ids)})"
    print("\n  cost when live: real "
          f"{cfg.model.name} calls × {n_live} {scope} through the gateway + "
          f"{n_live} trial containers on {cfg.runtime.kind}.")
    return 0
