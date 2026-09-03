"""Unit tests for the SWE-rebench cache builder (pure logic; no network, no cluster)."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.swe_rebench.build_cache import (
    apply_all_patches,
    normalize_task_toml_text,
    repin_all,
    set_environment_docker_image,
    set_environment_env_marker,
    set_environment_memory,
    upstream_image_ref,
)

# The task.toml shape the SWE-rebench corpus actually ships (deprecated string
# memory/storage only, no canonical _mb duplicate, no docker_image).
_UPSTREAM_TOML = """\
[task]
name = "swe-rebench/demo__repo-1"

[verifier]
timeout_sec = 3000

[agent]
timeout_sec = 3000

[environment]
build_timeout_sec = 1800.0
cpus = 1
memory = '8G'
storage = '16G'
"""


def _make_task(
    shard: Path,
    name: str,
    *,
    image: str = "swerebench/sweb.eval.x86_64.demo-1:latest",
    toml_text: str = _UPSTREAM_TOML,
    dockerfile_from: str | None = None,
    test_sh: str = "#!/bin/bash\npytest -q\n",
    with_config: bool = True,
) -> Path:
    """Build a synthetic SWE-rebench task dir. ``dockerfile_from`` defaults to
    ``image`` (the consistent case)."""
    d = shard / name
    (d / "tests").mkdir(parents=True)
    (d / "solution").mkdir(parents=True)
    (d / "environment").mkdir(parents=True)
    (d / "task.toml").write_text(toml_text)
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\ngit apply p.diff\n")
    (d / "tests" / "test.sh").write_text(test_sh)
    if with_config:
        (d / "tests" / "config.json").write_text(
            json.dumps({"instance_id": name, "docker_image": image}),
        )
    base = image if dockerfile_from is None else dockerfile_from
    (d / "environment" / "Dockerfile").write_text(
        f'FROM {base}\n\nENV _JAVA_OPTIONS=""\nRUN mkdir -p /logs\n',
    )
    return d


# ── task.toml normalization ───────────────────────────────────────────────────


def test_normalize_is_noop_on_the_real_corpus_shape() -> None:
    """SWE-rebench sets only the deprecated string form, which harbor coerces
    cleanly — so there is no conflict to strip and the text is byte-preserved."""
    out, changed = normalize_task_toml_text(_UPSTREAM_TOML)
    assert changed is False
    assert out == _UPSTREAM_TOML


def test_normalize_strips_deprecated_key_only_when_canonical_present() -> None:
    text = (
        "[environment]\n"
        "cpus = 1\n"
        "memory = '8G'\n"
        "memory_mb = 8192\n"
        "storage = '16G'\n"   # no storage_mb -> must be KEPT
    )
    out, changed = normalize_task_toml_text(text)
    assert changed is True
    parsed = tomllib.loads(out)["environment"]
    assert "memory" not in parsed
    assert parsed["memory_mb"] == 8192
    assert parsed["storage"] == "16G"
    assert parsed["cpus"] == 1


def test_normalize_scopes_the_strip_to_the_right_section() -> None:
    """A `memory` key in another section must survive untouched."""
    text = (
        "[metadata]\n"
        "memory = 'irrelevant'\n"
        "\n"
        "[environment]\n"
        "memory = '8G'\n"
        "memory_mb = 8192\n"
    )
    out, _ = normalize_task_toml_text(text)
    parsed = tomllib.loads(out)
    assert parsed["metadata"]["memory"] == "irrelevant"
    assert "memory" not in parsed["environment"]


# ── set_environment_docker_image (pure) ───────────────────────────────────────


def test_set_docker_image_appends_to_existing_section() -> None:
    out, changed = normalize_task_toml_text(_UPSTREAM_TOML)
    out, changed = set_environment_docker_image(_UPSTREAM_TOML, "repo/img:tag")
    assert changed is True
    parsed = tomllib.loads(out)
    assert parsed["environment"]["docker_image"] == "repo/img:tag"
    # everything else byte-preserved
    assert parsed["environment"]["memory"] == "8G"
    assert parsed["task"]["name"] == "swe-rebench/demo__repo-1"
    assert _UPSTREAM_TOML.rstrip("\n") in out.replace(
        '\ndocker_image = "repo/img:tag"', "",
    )


def test_set_docker_image_is_idempotent() -> None:
    once, _ = set_environment_docker_image(_UPSTREAM_TOML, "repo/img:tag")
    twice, changed = set_environment_docker_image(once, "repo/img:tag")
    assert changed is False
    assert twice == once


def test_set_docker_image_rewrites_in_place() -> None:
    first, _ = set_environment_docker_image(_UPSTREAM_TOML, "repo/old:1")
    second, changed = set_environment_docker_image(first, "repo/new:2")
    assert changed is True
    assert tomllib.loads(second)["environment"]["docker_image"] == "repo/new:2"
    assert "repo/old:1" not in second
    # exactly one docker_image line survives
    assert sum(1 for ln in second.splitlines() if "docker_image" in ln) == 1


def test_set_docker_image_inserts_before_a_following_section() -> None:
    text = "[environment]\ncpus = 1\n\n[verifier]\ntimeout_sec = 10\n"
    out, _ = set_environment_docker_image(text, "repo/img:tag")
    parsed = tomllib.loads(out)
    assert parsed["environment"]["docker_image"] == "repo/img:tag"
    assert parsed["verifier"]["timeout_sec"] == 10
    # the key landed inside [environment], not after [verifier]
    lines = out.splitlines()
    assert lines.index("docker_image = \"repo/img:tag\"") < lines.index("[verifier]")


def test_set_docker_image_appends_a_missing_section() -> None:
    out, changed = set_environment_docker_image("[task]\nname = 'x'\n", "repo/i:t")
    assert changed is True
    assert tomllib.loads(out)["environment"]["docker_image"] == "repo/i:t"


# ── upstream_image_ref (the authoritative pin + its cross-check) ───────────────


def test_upstream_image_ref_reads_config_json(tmp_path: Path) -> None:
    d = _make_task(tmp_path / "swe-rebench", "t", image="swerebench/x:latest")
    assert upstream_image_ref(d) == "swerebench/x:latest"


def test_upstream_image_ref_fails_loud_on_dockerfile_mismatch(tmp_path: Path) -> None:
    d = _make_task(
        tmp_path / "swe-rebench", "t",
        image="swerebench/declared:latest",
        dockerfile_from="swerebench/actually-different:latest",
    )
    with pytest.raises(SystemExit, match="image ref disagreement"):
        upstream_image_ref(d)


def test_upstream_image_ref_fails_loud_without_config(tmp_path: Path) -> None:
    d = _make_task(tmp_path / "swe-rebench", "t", with_config=False)
    with pytest.raises(SystemExit, match=r"no .*config\.json"):
        upstream_image_ref(d)


def test_upstream_image_ref_fails_loud_without_docker_image(tmp_path: Path) -> None:
    d = _make_task(tmp_path / "swe-rebench", "t")
    (d / "tests" / "config.json").write_text(json.dumps({"instance_id": "t"}))
    with pytest.raises(SystemExit, match="declares no 'docker_image'"):
        upstream_image_ref(d)


# ── repin ─────────────────────────────────────────────────────────────────────


def test_repin_all_pins_every_task_and_is_idempotent(tmp_path: Path) -> None:
    shard = tmp_path / "swe-rebench"
    _make_task(shard, "a", image="swerebench/a:latest")
    _make_task(shard, "b", image="swerebench/b:latest")

    assert repin_all(shard) == 2
    for name, ref in (("a", "swerebench/a:latest"), ("b", "swerebench/b:latest")):
        toml = tomllib.loads((shard / name / "task.toml").read_text())
        assert toml["environment"]["docker_image"] == ref

    assert repin_all(shard) == 0  # second run is a no-op


# ── patches ───────────────────────────────────────────────────────────────────


def test_apply_all_patches_overlays_present_tasks_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    patches = tmp_path / "patches"
    (patches / "present" / "tests").mkdir(parents=True)
    (patches / "present" / "tests" / "test.sh").write_text("#!/bin/bash\nfixed\n")
    (patches / "absent").mkdir(parents=True)
    (patches / "absent" / "task.toml").write_text("[task]\n")
    monkeypatch.setattr(bc, "PATCHES_DIR", patches)

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "present")

    assert apply_all_patches(shard) == 1
    assert (shard / "present" / "tests" / "test.sh").read_text() == "#!/bin/bash\nfixed\n"
    assert not (shard / "absent").exists()


def test_apply_all_patches_is_a_noop_when_empty(tmp_path: Path) -> None:
    """The shipped patches/ dir starts empty — the hook must be inert."""
    shard = tmp_path / "swe-rebench"
    _make_task(shard, "a")
    before = (shard / "a" / "tests" / "test.sh").read_text()
    apply_all_patches(shard)
    assert (shard / "a" / "tests" / "test.sh").read_text() == before


# ── resources: cpu-pinning markers + FAIR memory overrides ───────────────────


def test_set_environment_env_marker_appends_the_subtable() -> None:
    out, changed = set_environment_env_marker(_UPSTREAM_TOML, "XRLENV_CPU_PINNING", "1")
    assert changed is True
    parsed = tomllib.loads(out)
    assert parsed["environment"]["env"]["XRLENV_CPU_PINNING"] == "1"
    # the parent table's own keys survive untouched
    assert parsed["environment"]["memory"] == "8G"
    assert parsed["environment"]["cpus"] == 1


def test_set_environment_env_marker_is_idempotent() -> None:
    once, _ = set_environment_env_marker(_UPSTREAM_TOML, "XRLENV_CPU_PINNING", "1")
    twice, changed = set_environment_env_marker(once, "XRLENV_CPU_PINNING", "1")
    assert changed is False
    assert twice == once


def test_set_environment_key_lands_before_the_env_subtable() -> None:
    """A bare key written after [environment.env] would silently belong to the
    SUB-table, not to [environment] — harbor would then ignore it."""
    marked, _ = set_environment_env_marker(_UPSTREAM_TOML, "XRLENV_CPU_PINNING", "1")
    out, _ = set_environment_docker_image(marked, "repo/img:tag")
    parsed = tomllib.loads(out)
    assert parsed["environment"]["docker_image"] == "repo/img:tag"
    assert "docker_image" not in parsed["environment"]["env"]
    assert parsed["environment"]["env"]["XRLENV_CPU_PINNING"] == "1"


def test_set_environment_memory_rewrites_in_place() -> None:
    out, changed = set_environment_memory(_UPSTREAM_TOML, "16G")
    assert changed is True
    assert tomllib.loads(out)["environment"]["memory"] == "16G"
    assert sum(1 for ln in out.splitlines() if ln.startswith("memory")) == 1
    assert set_environment_memory(out, "16G")[1] is False


def test_apply_resource_routing_writes_markers_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "pin-only")
    _make_task(shard, "pin-and-mem")
    _make_task(shard, "untouched")
    monkeypatch.setattr(bc, "CPU_PINNING_TASKS", frozenset({"pin-only", "pin-and-mem"}))
    monkeypatch.setattr(bc, "MEMORY_OVERRIDES", {"pin-and-mem": "16G"})

    pinned, bumped, missing = bc.apply_resource_routing(shard)
    assert (pinned, bumped, missing) == (2, 1, [])

    a = tomllib.loads((shard / "pin-only" / "task.toml").read_text())
    assert a["environment"]["env"]["XRLENV_CPU_PINNING"] == "1"
    assert a["environment"]["memory"] == "8G"          # untouched
    b = tomllib.loads((shard / "pin-and-mem" / "task.toml").read_text())
    assert b["environment"]["env"]["XRLENV_CPU_PINNING"] == "1"
    assert b["environment"]["memory"] == "16G"
    c = tomllib.loads((shard / "untouched" / "task.toml").read_text())
    assert "env" not in c["environment"]

    assert bc.apply_resource_routing(shard) == (0, 0, [])   # idempotent


def test_apply_resource_routing_reports_absent_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "present")
    monkeypatch.setattr(bc, "CPU_PINNING_TASKS", frozenset({"present", "gone"}))
    monkeypatch.setattr(bc, "MEMORY_OVERRIDES", {})
    pinned, _bumped, missing = bc.apply_resource_routing(shard)
    assert pinned == 1
    assert missing == ["gone"]


def test_memory_override_refused_when_upstream_declares_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE FAIRNESS RULE. Raising a converter default is fine; overriding a
    resource upstream chose for itself would change the benchmark's envelope."""
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    d = _make_task(shard, "upstream-sized")
    cfg = json.loads((d / "tests" / "config.json").read_text())
    cfg["harbor_memory"] = "16G"          # upstream DID declare
    (d / "tests" / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(bc, "CPU_PINNING_TASKS", frozenset())
    monkeypatch.setattr(bc, "MEMORY_OVERRIDES", {"upstream-sized": "32G"})

    with pytest.raises(SystemExit, match="upstream DECLARES harbor_memory"):
        bc.apply_resource_routing(shard)


def test_memory_override_allowed_when_upstream_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    d = _make_task(shard, "converter-default")
    cfg = json.loads((d / "tests" / "config.json").read_text())
    cfg["harbor_memory"] = None           # upstream declared nothing
    (d / "tests" / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(bc, "CPU_PINNING_TASKS", frozenset())
    monkeypatch.setattr(bc, "MEMORY_OVERRIDES", {"converter-default": "16G"})

    _pinned, bumped, _missing = bc.apply_resource_routing(shard)
    assert bumped == 1
    assert tomllib.loads((d / "task.toml").read_text())["environment"]["memory"] == "16G"


def test_every_shipped_memory_override_targets_a_pinned_task() -> None:
    """A memory bump is a last resort AFTER pinning — the evidence trail in
    STATUS.md only justifies it for tasks that pinning alone did not fix."""
    from xrlenv_plugins.benchmarks.swe_rebench.build_cache import (
        CPU_PINNING_TASKS,
        MEMORY_OVERRIDES,
    )
    assert set(MEMORY_OVERRIDES) <= set(CPU_PINNING_TASKS)


def test_shipped_resource_routing_matches_the_documented_evidence() -> None:
    """The sets are what a 3h44m sweep + three follow-up experiments cost to
    learn; pin them so a careless edit is caught, and so the counts in STATUS.md
    and run_full_sweep.sh's EXCLUDE note cannot silently drift from the code."""
    from xrlenv_plugins.benchmarks.swe_rebench.build_cache import (
        CPU_PINNING_TASKS,
        MEMORY_OVERRIDES,
    )
    assert len(CPU_PINNING_TASKS) == 16
    assert MEMORY_OVERRIDES == {
        "ImperialCollegeLondon__virtual_ecosystem-1232": "16G",
        "calliope-project__calliope-854": "16G",
        "pybamm-team__PyBaMM-4871": "16G",
        "owkin__PyDESeq2-356": "32G",
    }
    # the 3 tasks upstream sizes itself (2 cpu / 16G) were fixed by PINNING
    # alone — no upstream-declared resource may ever be overridden
    for upstream_sized in ("sktime__sktime-8723", "sktime__sktime-8921",
                           "sktime__sktime-8937"):
        assert upstream_sized in CPU_PINNING_TASKS
        assert upstream_sized not in MEMORY_OVERRIDES


@pytest.mark.parametrize("stage", ["all", "patch"])
def test_patch_stage_applies_the_resource_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    """Routing is a curated content fix, so it rides the `patch` stage next to
    the patches/ overlays — the same two-kind split terminal_bench_2_1 uses
    (PATCHES + ENV_PATCHES). `--stage all` must apply it too: run_full_sweep.sh's
    step 1 is `--stage all`, which is what makes the findings survive a wipe."""
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "pinned-task", image="swerebench/x:latest")
    _make_task(shard, "hermetic-task", image="swerebench/y:latest")
    monkeypatch.setattr(bc, "CPU_PINNING_TASKS", frozenset({"pinned-task"}))
    monkeypatch.setattr(bc, "MEMORY_OVERRIDES", {"pinned-task": "16G"})
    monkeypatch.setattr(bc, "HERMETICITY_ENV", {"hermetic-task": {"UV_NO_SYNC": "1"}})
    monkeypatch.setattr(bc, "PATCHES_DIR", tmp_path / "no-patches")

    if stage == "patch":                 # patch alone must not need repin first
        bc.repin_all(shard)
    assert bc.main(["--stage", stage, "--dest", str(tmp_path)]) == 0

    env = tomllib.loads((shard / "pinned-task" / "task.toml").read_text())["environment"]
    assert env["docker_image"] == "swerebench/x:latest"
    assert env["env"]["XRLENV_CPU_PINNING"] == "1"
    assert env["memory"] == "16G"

    # the hermeticity env rides the SAME stage — a task can need one without the
    # other, so neither routing may leak onto the other's tasks
    other = tomllib.loads((shard / "hermetic-task" / "task.toml").read_text())["environment"]
    assert other["env"] == {"UV_NO_SYNC": "1"}
    assert other["memory"] == "8G"


def test_patch_stage_fails_loud_when_a_routed_task_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corpus drift must not silently drop a measured fix."""
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "present")
    monkeypatch.setattr(bc, "CPU_PINNING_TASKS", frozenset({"present", "vanished"}))
    monkeypatch.setattr(bc, "MEMORY_OVERRIDES", {})
    monkeypatch.setattr(bc, "HERMETICITY_ENV", {})
    monkeypatch.setattr(bc, "PATCHES_DIR", tmp_path / "no-patches")

    with pytest.raises(SystemExit, match="absent from the shard"):
        bc.main(["--stage", "patch", "--dest", str(tmp_path)])


@pytest.mark.parametrize("absent_in", ["CPU_PINNING_TASKS", "HERMETICITY_ENV"])
def test_patch_stage_drift_guard_covers_both_routings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absent_in: str,
) -> None:
    """A hermeticity fix is as measured as a resource one, so corpus drift must
    fail loud for EITHER table — not just the resource one."""
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "present")
    monkeypatch.setattr(bc, "CPU_PINNING_TASKS", frozenset())
    monkeypatch.setattr(bc, "MEMORY_OVERRIDES", {})
    monkeypatch.setattr(bc, "HERMETICITY_ENV", {})
    monkeypatch.setattr(bc, "PATCHES_DIR", tmp_path / "no-patches")
    if absent_in == "CPU_PINNING_TASKS":
        monkeypatch.setattr(bc, absent_in, frozenset({"vanished"}))
    else:
        monkeypatch.setattr(bc, absent_in, {"vanished": {"UV_NO_SYNC": "1"}})

    with pytest.raises(SystemExit, match="vanished"):
        bc.main(["--stage", "patch", "--dest", str(tmp_path)])


def test_apply_hermeticity_env_writes_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "needs-env")
    _make_task(shard, "untouched")
    monkeypatch.setattr(
        bc, "HERMETICITY_ENV", {"needs-env": {"UV_NO_SYNC": "1", "PIP_NO_INDEX": "1"}},
    )

    written, missing = bc.apply_hermeticity_env(shard)
    assert (written, missing) == (1, [])
    env = tomllib.loads((shard / "needs-env" / "task.toml").read_text())["environment"]
    assert env["env"] == {"UV_NO_SYNC": "1", "PIP_NO_INDEX": "1"}
    assert "env" not in tomllib.loads(
        (shard / "untouched" / "task.toml").read_text())["environment"]

    assert bc.apply_hermeticity_env(shard) == (0, [])          # idempotent


def test_apply_hermeticity_env_reports_absent_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "present")
    monkeypatch.setattr(
        bc, "HERMETICITY_ENV",
        {"present": {"UV_NO_SYNC": "1"}, "gone": {"UV_NO_SYNC": "1"}},
    )
    written, missing = bc.apply_hermeticity_env(shard)
    assert (written, missing) == (1, ["gone"])


def test_hermeticity_env_coexists_with_cpu_pinning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both routings write into the same [environment.env] sub-table; whichever
    runs second must ADD its key rather than replace the table."""
    import xrlenv_plugins.benchmarks.swe_rebench.build_cache as bc

    shard = tmp_path / "swe-rebench"
    _make_task(shard, "both")
    monkeypatch.setattr(bc, "CPU_PINNING_TASKS", frozenset({"both"}))
    monkeypatch.setattr(bc, "MEMORY_OVERRIDES", {})
    monkeypatch.setattr(bc, "HERMETICITY_ENV", {"both": {"UV_NO_SYNC": "1"}})

    bc.apply_resource_routing(shard)
    bc.apply_hermeticity_env(shard)
    env = tomllib.loads((shard / "both" / "task.toml").read_text())["environment"]["env"]
    assert env == {"XRLENV_CPU_PINNING": "1", "UV_NO_SYNC": "1"}


def test_shipped_hermeticity_env_matches_the_documented_evidence() -> None:
    """guppylang-1259's `uv run pytest` re-resolves the workspace, so an
    UNPINNED PEP-517 build requirement (hatchling >= 1.32) breaks a task authored
    against an older one. UV_NO_SYNC=1 was measured to restore reward 1 AND to
    remove PyPI from the verify phase. Pin it so the reason cannot be lost."""
    from xrlenv_plugins.benchmarks.swe_rebench.build_cache import HERMETICITY_ENV

    assert HERMETICITY_ENV == {"CQCL__guppylang-1259": {"UV_NO_SYNC": "1"}}


def test_status_md_names_every_hermeticity_env_task() -> None:
    """Same doc-drift guard the memory overrides get."""
    from xrlenv_plugins.benchmarks.swe_rebench.build_cache import HERMETICITY_ENV

    status = (Path(__file__).resolve().parents[1] / "STATUS.md").read_text()
    for task, env in HERMETICITY_ENV.items():
        assert f"`{task}`" in status, f"STATUS.md does not name {task}"
        for key in env:
            assert f"`{key}`" in status, f"STATUS.md does not name {key}"


def test_status_md_names_every_memory_override() -> None:
    """Guard against hand-maintained doc drift — the exact failure that left
    terminal_bench_2_1's README listing 5 ENV_PATCHES tasks when the code had 6.
    STATUS.md quotes the override table, so it must name every task and value."""
    from xrlenv_plugins.benchmarks.swe_rebench.build_cache import MEMORY_OVERRIDES

    status = (Path(__file__).resolve().parents[1] / "STATUS.md").read_text()
    for task, memory in MEMORY_OVERRIDES.items():
        assert f"`{task}`" in status, f"STATUS.md does not name {task}"
        assert memory in status, f"STATUS.md does not state {memory} for {task}"
    assert str(len(MEMORY_OVERRIDES)) in status
