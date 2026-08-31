"""Skeleton smoke tests: the public API imports and the factories/registries wire up.

These assert the *shape* of beagle, not behavior — every execution path is still
a stub. They exist so the interface layer can't silently break while the internals
are ported. No xrlenv / harbor / model access required.
"""

from __future__ import annotations

import pytest

import beagle as bgl
from beagle.types import AgentRole, Task, TaskContext, Transparency


def test_top_level_surface() -> None:
    assert bgl.__version__
    for attr in ("Trainer", "DataMixture", "TaskDataset", "load_config"):
        assert hasattr(bgl, attr), attr
    for pkg in ("agents", "algorithms", "benchmarks", "data", "rollout"):
        assert hasattr(bgl, pkg), pkg


def test_registries_populated() -> None:
    assert {"monet", "mini-swe", "cursor", "claude-code", "codex"} <= set(bgl.agents.available())
    assert "darwinx" in bgl.algorithms.available()
    assert {"terminal_bench_2_1", "swe-bench-verified"} <= set(bgl.benchmarks.available())


def test_capabilities_not_roles() -> None:
    from beagle.agents import Capability

    monet = bgl.agents.build("monet")
    cursor = bgl.agents.build("cursor")
    # transparency is a declared class attribute.
    assert monet.transparency is Transparency.WHITE_BOX
    assert cursor.transparency is Transparency.BLACK_BOX
    # white-box coding agents can serve EITHER role — the user's choice.
    assert monet.can_be_evolvee() and monet.can_be_evolver()
    assert monet.capabilities == {
        Capability.ROLLOUT,
        Capability.EVOLVABLE,
        Capability.EDIT,
    }
    # closed-source CLIs can only edit — they cannot be evolved.
    assert cursor.can_be_evolver() and not cursor.can_be_evolvee()
    assert cursor.capabilities == {Capability.EDIT}


def test_evolvable_rebind() -> None:
    # with_source pins an evolvable agent to a candidate ref; run() then
    # installs/invokes *that* version. The original is untouched.
    from beagle.agents.core.base import AgentSource, Evolvable

    monet = bgl.agents.build("monet")
    assert isinstance(monet, Evolvable)
    variant = AgentSource(repo="https://example/fork", ref="cand-1")
    rebound = monet.with_source(variant)
    assert rebound is not monet
    assert rebound.source().ref == "cand-1"


def test_trainer_validates_role_against_capabilities() -> None:
    # cursor lacks Runnable+Evolvable, so it cannot be assigned the evolvee role.
    with pytest.raises(TypeError):
        bgl.Trainer(
            evolvee=bgl.agents.build("cursor"),
            evolver=bgl.agents.build("cursor"),
            algorithm=bgl.algorithms.build("darwinx"),
        )
    # monet can play both roles in the same run.
    t = bgl.Trainer(
        evolvee=bgl.agents.build("monet"),
        evolver=bgl.agents.build("monet"),
        algorithm=bgl.algorithms.build("darwinx"),
    )
    assert t.evolvee.role is AgentRole.EVOLVEE
    assert t.evolver.role is AgentRole.EVOLVER


def test_benchmark_pluggables() -> None:
    # Harbor benchmark = zero code: default source/harness/grader.
    from beagle.benchmarks import HarborBenchmark, InBandGrader, PatchEvalGrader
    from beagle.types import TaskResult

    tb = bgl.benchmarks.get("terminal_bench_2_1")
    assert isinstance(tb, HarborBenchmark)
    assert type(tb.source()).__name__ == "HarborCache"
    assert isinstance(tb.grader(), InBandGrader)

    # Override case: SWE-bench brings its own source + a patch-eval grader.
    sv = bgl.benchmarks.get("swe-bench-verified")
    assert isinstance(sv.grader(), PatchEvalGrader)
    assert type(sv.harness()).__name__ == "DockerHarness"

    # InBandGrader is a real reduction over rewards the rollout already produced.
    rep = InBandGrader().grade(
        [TaskResult(task_id="a", reward=1.0), TaskResult(task_id="b", reward=0.0)],
        runtime=None,  # type: ignore[arg-type]
        run_dir=None,  # type: ignore[arg-type]
    )
    assert (rep.num_tasks, rep.num_resolved, rep.score) == (2, 1, 0.5)


def test_harbor_cache_stamps_canonical_name_not_cache_dir(tmp_path) -> None:
    """Regression: a benchmark whose cache dir differs from its registry name (e.g.
    ``terminal_bench_2_1`` cached under ``terminal-bench-2-1``) must stamp the *canonical*
    name on ``Task.benchmark`` — otherwise the Runner's ``benchmarks.get(task.benchmark)``
    raises KeyError on the hyphenated cache name."""
    from beagle.benchmarks.base import BenchmarkSpec
    from beagle.benchmarks.source import HarborCache

    # A fake cache under the HYPHENATED dir, holding one task.
    task_dir = tmp_path / "terminal-bench-2-1" / "some-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[environment]\ndocker_image = "img:1"\nworkdir = "/w"\n')
    (task_dir / "instruction.md").write_text("do the thing")

    src = HarborCache("terminal_bench_2_1", cache_name="terminal-bench-2-1", cache_root=tmp_path)
    (task, ctx), = list(src.tasks(BenchmarkSpec(name="terminal_bench_2_1")))

    assert task.task_id == "some-task"                       # read from the hyphenated dir
    assert task.benchmark == "terminal_bench_2_1"            # CANONICAL, not "terminal-bench-2-1"
    assert ctx.benchmark_name == "terminal_bench_2_1"
    bgl.benchmarks.get(task.benchmark)                        # the Runner path — must NOT KeyError


def test_harbor_benchmark_source_keeps_canonical_and_cache_names_split() -> None:
    """The benchmark wires the two names correctly: identity=canonical, cache dir=cache_name."""
    from beagle.benchmarks.source import HarborCache

    src = bgl.benchmarks.get("terminal_bench_2_1").source()
    assert isinstance(src, HarborCache)
    assert src.name == "terminal_bench_2_1"                  # identity
    assert src.cache_name == "terminal-bench-2-1"            # filesystem


def test_container_runtime_ported() -> None:
    # The runtime is the real ported impl (LocalDocker + Xrlenv), not a stub.
    from beagle.rollout import (
        ContainerRuntime,
        GitClone,
        LocalDockerRuntime,
        RuntimeConfig,
        XrlenvDockerRuntime,
        build_runtime,
        git_clone_argv,
    )

    assert isinstance(LocalDockerRuntime(), ContainerRuntime)
    assert isinstance(build_runtime(RuntimeConfig(kind="local")), LocalDockerRuntime)
    assert XrlenvDockerRuntime.__name__ == "XrlenvDockerRuntime"  # importable w/o xrlenv (lazy)
    # GitClone transport: token injected via shell var (not interpolated), ref passed
    # as a positional arg (not baked into the script).
    argv = git_clone_argv(GitClone("https://github.com/o/r", "ref", "/agent", token_env="GH_TOKEN"))
    assert argv[0] == "sh" and "$GH_TOKEN" in argv[2]
    assert "ref" in argv[3:] and "ref" not in argv[2]  # positional, not interpolated


def test_native_runner_shape() -> None:
    # WAI is driven by its own vendored xrlenv runner (the native-runner shape).
    from beagle.benchmarks import NativeRunnerHarness

    wai = bgl.benchmarks.get("webarena-infinity")
    assert isinstance(wai.harness(), NativeRunnerHarness)


def test_unknown_names_raise() -> None:
    with pytest.raises(KeyError):
        bgl.agents.build("nope")
    with pytest.raises(KeyError):
        bgl.algorithms.build("nope")
    with pytest.raises(KeyError):
        bgl.benchmarks.get("nope")


def test_dataset_container_ops() -> None:
    items = [(Task(task_id=f"t{i}", benchmark="demo"), TaskContext(image=None)) for i in range(10)]
    ds = bgl.TaskDataset(items, name="demo")
    assert len(ds) == 10
    train, val = ds.split(0.2)
    assert (len(train), len(val)) == (8, 2)
    assert ds.select(["t3", "t1"]).task_ids == ["t3", "t1"]
    assert ds.filter(lambda t: t.task_id in {"t0", "t9"}).task_ids == ["t0", "t9"]
    with pytest.raises(KeyError):
        ds.select(["missing"])


def test_build_parity() -> None:
    # agents and algorithms are both built by name via build(...).
    algo = bgl.algorithms.build("darwinx", max_loop_iters=8)
    assert type(algo).__name__ == "DarwinX"
    assert algo.hparams["max_loop_iters"] == 8


def test_agent_onboarding_contract() -> None:
    # Onboarding = compose the capability mixins you support, @register, done.
    from beagle.agents import Agent, Editor, register

    @register("test-onboard-editor")
    class _TmpEditor(Agent, Editor):
        transparency = Transparency.BLACK_BOX

        def edit(self, instruction, workspace, **kw):  # noqa: ANN001
            raise NotImplementedError

    assert "test-onboard-editor" in bgl.agents.available()
    built = bgl.agents.build("test-onboard-editor")
    assert isinstance(built, Editor)
    assert built.can_be_evolver() and not built.can_be_evolvee()


def test_trainer_fit_delegates_to_algorithm_evolve() -> None:
    # Trainer.fit is thin: it builds evaluate + hands off to algorithm.evolve(). Built directly
    # (no BeagleConfig), it names no benchmark → DarwinX's launch fails loud, surfaced by fit.
    trainer = bgl.Trainer(
        evolvee=bgl.agents.build("monet"),
        evolver=bgl.agents.build("cursor"),
        algorithm=bgl.algorithms.build("darwinx"),
    )
    with pytest.raises(ValueError, match="needs a benchmark"):
        trainer.fit(train_dataset=bgl.TaskDataset([]))


def test_config_from_dict() -> None:
    cfg = bgl.BeagleConfig.from_dict(
        {
            "evolvee": {"name": "monet", "source": {"repo": "git@…/monet_code", "ref": "main"}},
            "evolver": {"name": "cursor", "model": {"name": "cursor-fast", "provider": "cursor"}},
            "algorithm": {"name": "darwinx", "hparams": {"max_loop_iters": 8}},
            "data": {"components": [{"benchmark": {"name": "terminal_bench_2_1"}}]},
            "runtime": {"kind": "xrlenv-cluster"},
        }
    )
    assert cfg.evolvee.role is AgentRole.EVOLVEE
    assert cfg.evolvee.source is not None and cfg.evolvee.source.ref == "main"
    assert cfg.evolver.role is AgentRole.EVOLVER
    assert cfg.algorithm.hparams["max_loop_iters"] == 8
    assert cfg.runtime.kind == "xrlenv-cluster"
