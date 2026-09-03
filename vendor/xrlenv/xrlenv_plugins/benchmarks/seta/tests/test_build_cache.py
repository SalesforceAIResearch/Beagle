"""Unit tests for the seta build_cache pipeline.

The clone itself is network + git (covered operationally), but the deterministic
pieces are pure: shard resolution, the "already populated" idempotency signal,
and the task-selection logic that decides which cloned dirs to move (task.toml
present, not already in the shard). Offline: synthetic dirs in tmp.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.seta import build_cache as bc
from xrlenv_plugins.benchmarks.seta.build_cache import (
    BASE_IMAGE_FIX_TASKS,
    DROPPED_COMMAND_TASKS,
    PATCHES_DIR,
    SYSBOX_TASKS,
    TBENCH_BASE,
    VERIFIER_ROOT_TASKS,
    apply_all_patches,
    apply_all_sysbox_markers,
    apply_all_verifier_root_markers,
    apply_sysbox_marker,
    apply_verifier_root_marker,
    restore_base_images,
    restore_dropped_commands,
)

# ── shard resolution ──────────────────────────────────────────────────────────


def test_shard_dir_from_explicit_dest() -> None:
    assert bc._shard_dir("/tmp/cache") == Path("/tmp/cache/seta-env")


def test_shard_dir_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", "/tmp/envcache")
    assert bc._shard_dir(None) == Path("/tmp/envcache/seta-env")


def test_shard_dir_fails_loud_without_a_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset root used to resolve to ``~/.cache/harbor/tasks`` — a plausible-but-wrong
    directory that reads as an empty shard instead of reporting the operator error. Now
    it raises, the same as every other kit and as this kit's own --dest path."""
    monkeypatch.delenv("XRLENV_BENCHMARK_CACHE", raising=False)
    with pytest.raises(SystemExit, match="XRLENV_BENCHMARK_CACHE"):
        bc._shard_dir(None)


def test_shard_dir_prefers_an_explicit_dest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", "/tmp/envcache")
    assert bc._shard_dir("/tmp/explicit") == Path("/tmp/explicit/seta-env")


# ── idempotency signal ────────────────────────────────────────────────────────


def _make_task(shard: Path, task_id: str, *, with_toml: bool = True) -> None:
    (shard / task_id).mkdir(parents=True)
    if with_toml:
        (shard / task_id / "task.toml").write_text("[environment]\n")


def test_is_populated(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"
    assert not bc.is_populated(shard)            # missing dir
    shard.mkdir()
    assert not bc.is_populated(shard)            # empty
    _make_task(shard, "0")
    assert bc.is_populated(shard)                # a task with task.toml


# ── task selection (what gets moved out of the clone) ─────────────────────────


def test_tasks_to_move_selects_new_tasks_with_toml(tmp_path: Path) -> None:
    hd = tmp_path / "Harbor-Dataset"
    _make_task(hd, "0")
    _make_task(hd, "1")
    _make_task(hd, "2", with_toml=False)  # no task.toml → skipped
    shard = tmp_path / "seta-env"
    _make_task(shard, "0")                # already present → skipped

    picked = [d.name for d in bc._tasks_to_move(hd, shard)]
    assert picked == ["1"]  # 0 already there, 2 has no task.toml


def test_tasks_to_move_is_sorted(tmp_path: Path) -> None:
    hd = tmp_path / "Harbor-Dataset"
    for tid in ("10", "2", "1"):
        _make_task(hd, tid)
    shard = tmp_path / "seta-env"
    # Lexicographic sort of the dir names (the git ids are strings on disk).
    assert [d.name for d in bc._tasks_to_move(hd, shard)] == ["1", "10", "2"]


# ── populate idempotency (no clone when already populated) ─────────────────────


def test_populate_skips_when_already_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "seta-env"
    _make_task(shard, "0")

    # A populated shard must short-circuit BEFORE any git clone is attempted.
    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("populate must not clone when already populated")

    monkeypatch.setattr(
        "xrlenv_plugins.benchmarks.seta.build_cache.subprocess.run", _boom,
    )
    assert bc.populate(str(tmp_path)) == 0


# ── sysbox DinD routing markers ───────────────────────────────────────────────

# The real seta task.toml shape: metadata + verifier/agent + a scalar-only
# [environment] table (no [environment.env] yet — the marker must create it).
_SETA_TOML = (
    'version = "1.0"\n\n'
    "[metadata]\n"
    'category = "software-engineering"\n'
    "custom_docker_compose = true\n\n"
    "[verifier]\ntimeout_sec = 360.0\n\n"
    "[agent]\ntimeout_sec = 360.0\n\n"
    "[environment]\nbuild_timeout_sec = 600.0\ncpus = 1\n"
)

_TASK8 = next(s for s in SYSBOX_TASKS if s.task == "8")
_TASK1004 = next(s for s in SYSBOX_TASKS if s.task == "1004")


def test_sysbox_set_is_the_validated_set() -> None:
    # Exactly the tasks validated end-to-end (reward 1.0) — grown one proven task
    # at a time (guards against an unvetted row sneaking in).
    assert {s.task for s in SYSBOX_TASKS} == {
        "8", "1004", "1117", "1347", "311", "119", "1225", "830", "1059", "484", "345",
        "846",
    }
    # every id routes to sysbox-runc.
    assert all(s.runtime == "sysbox-runc" for s in SYSBOX_TASKS)
    by = {s.task: s for s in SYSBOX_TASKS}
    # DinD tasks nest a dockerd; 1004 (docker-ce-cli, no daemon) also installs it.
    assert _TASK8.inner_dockerd and not _TASK8.install_dockerd
    assert _TASK1004.inner_dockerd and _TASK1004.install_dockerd
    # privilege-only tasks (iptables/mount/netns) don't nest a dockerd.
    assert not by["1117"].inner_dockerd and not by["311"].inner_dockerd
    assert not by["1059"].inner_dockerd
    # the systemd task boots PID 1.
    assert by["345"].systemd_init and not by["345"].inner_dockerd


def test_marker_creates_env_table_with_runtime_and_inner_dockerd() -> None:
    out, status = apply_sysbox_marker(_SETA_TOML, _TASK8)
    assert status == "patched"
    env = tomllib.loads(out)["environment"]["env"]
    assert env["XRLENV_CONTAINER_RUNTIME"] == "sysbox-runc"
    assert env["XRLENV_INNER_DOCKERD"] == "1"
    assert "XRLENV_INSTALL_DOCKERD" not in env  # task 8 ships a full daemon


def test_marker_install_dockerd_companion_for_cli_only_image() -> None:
    out, _ = apply_sysbox_marker(_SETA_TOML, _TASK1004)
    env = tomllib.loads(out)["environment"]["env"]
    assert env["XRLENV_INSTALL_DOCKERD"] == "1"
    assert env["XRLENV_INNER_DOCKERD"] == "1"


def test_marker_is_idempotent() -> None:
    once, _ = apply_sysbox_marker(_SETA_TOML, _TASK8)
    twice, status = apply_sysbox_marker(once, _TASK8)
    assert status == "already"
    assert twice == once


def test_marker_output_is_valid_toml_and_preserves_existing_keys() -> None:
    out, _ = apply_sysbox_marker(_SETA_TOML, _TASK1004)
    parsed = tomllib.loads(out)  # must not raise
    # existing content survives the surgical insert
    assert parsed["environment"]["build_timeout_sec"] == 600.0
    assert parsed["metadata"]["custom_docker_compose"] is True


def test_apply_all_marks_set_and_skips_unlisted(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"
    listed = {s.task for s in SYSBOX_TASKS}
    for tid in listed | {"101"}:  # 101 = an unlisted control task
        (shard / tid).mkdir(parents=True)
        (shard / tid / "task.toml").write_text(_SETA_TOML)
    results = dict(apply_all_sysbox_markers(shard, only=None))
    assert set(results) == listed  # every listed task marked; 101 untouched
    assert "XRLENV_CONTAINER_RUNTIME" not in \
        tomllib.loads((shard / "101" / "task.toml").read_text()).get(
            "environment", {}).get("env", {})
    # the marked ones carry the runtime
    for tid in ("8", "1004"):
        env = tomllib.loads((shard / tid / "task.toml").read_text())["environment"]["env"]
        assert env["XRLENV_CONTAINER_RUNTIME"] == "sysbox-runc"


def test_apply_all_fails_loud_on_unpopulated_target(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"  # nothing cloned
    with pytest.raises(SystemExit, match="sysbox-marker target missing"):
        apply_all_sysbox_markers(shard, only=None)


def test_apply_all_rejects_unknown_only_task(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"
    (shard / "8").mkdir(parents=True)
    (shard / "8" / "task.toml").write_text(_SETA_TOML)
    with pytest.raises(SystemExit, match="not in SYSBOX_TASKS"):
        apply_all_sysbox_markers(shard, only=["8", "does-not-exist"])


# ── migration-repair overlays (patches/) ──────────────────────────────────────


def test_patch_309_overlay_is_the_path_repair() -> None:
    # The committed 309 overlay repairs the /oracle -> /solution run-path guard
    # (harbor 0.20 runs the oracle from /solution, not the pre-Harbor /oracle).
    p = PATCHES_DIR / "309" / "solution" / "solve.sh"
    assert p.is_file(), "309 solve.sh overlay missing"
    body = p.read_text()
    assert '"$SCRIPT_DIR" == "/solution"' in body
    assert "/oracle" not in body


def test_apply_all_patches_overlays_present_tasks(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"
    # a synthetic 309 with the BROKEN (/oracle) guard + an unrelated task
    (shard / "309" / "solution").mkdir(parents=True)
    (shard / "309" / "task.toml").write_text(_SETA_TOML)
    (shard / "309" / "solution" / "solve.sh").write_text(
        'if [[ "$SCRIPT_DIR" == "/oracle" ]]; then cp "$0" /app/x; fi\n',
    )
    (shard / "999" / "solution").mkdir(parents=True)
    (shard / "999" / "task.toml").write_text(_SETA_TOML)

    n = apply_all_patches(shard)
    assert n == 1  # only 309 has an overlay
    fixed = (shard / "309" / "solution" / "solve.sh").read_text()
    assert '"$SCRIPT_DIR" == "/solution"' in fixed and "/oracle" not in fixed


def test_apply_all_patches_skips_absent_task(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"
    shard.mkdir(parents=True)  # empty shard — 309 not populated
    assert apply_all_patches(shard) == 0  # skipped, no crash


# ── base-image restore (Harbor swapped the t-bench base for bare ubuntu:24.04) ──


def test_restore_base_images_rewrites_from_for_fix_tasks(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"
    fix = next(iter(BASE_IMAGE_FIX_TASKS))
    d = shard / fix / "environment"
    d.mkdir(parents=True)
    (d / "Dockerfile").write_text("# canary\nFROM ubuntu:24.04\nRUN echo hi\n")
    # an UNLISTED task must be left on ubuntu:24.04
    other = shard / "999999" / "environment"
    other.mkdir(parents=True)
    (other / "Dockerfile").write_text("FROM ubuntu:24.04\n")

    assert restore_base_images(shard) == 1
    fixed = (d / "Dockerfile").read_text()
    assert f"FROM {TBENCH_BASE}" in fixed
    assert "FROM ubuntu:24.04" not in fixed
    assert (other / "Dockerfile").read_text() == "FROM ubuntu:24.04\n"  # untouched
    # idempotent — a second pass rewrites nothing
    assert restore_base_images(shard) == 0


def test_base_image_fix_set_is_the_validated_ten() -> None:
    # Exactly the 10 base-image tasks validated green after the t-bench-base rebuild
    # (2026-08-04, 10/15). The 5 that failed even rebuilt (15/304/729/1092 solve
    # breaks the curl->uv bootstrap; 172 pins an unavailable wget) are blacklisted,
    # NOT here.
    assert {
        "240", "367", "390", "617", "906", "953",  # python3
        "60", "827",                                # curl
        "1203",                                     # uv
        "723",                                      # tmux
    } == BASE_IMAGE_FIX_TASKS
    # A task is fixed exactly one way; the two fix sets must not overlap.
    assert not (BASE_IMAGE_FIX_TASKS & {s.task for s in SYSBOX_TASKS})


# ── dropped-command restore (Harbor dropped the compose command) ───────────────


def test_restore_dropped_commands_bakes_entrypoint_wrapper(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"
    tid = next(iter(DROPPED_COMMAND_TASKS))
    cmd = DROPPED_COMMAND_TASKS[tid]
    env = shard / tid / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text("# canary\nFROM ubuntu:24.04\nRUN echo hi\n")

    assert restore_dropped_commands(shard) == 1
    df = (env / "Dockerfile").read_text()
    assert "COPY .xrlenv-boot.sh /.xrlenv-boot.sh" in df
    assert 'ENTRYPOINT ["/.xrlenv-boot.sh"]' in df
    boot = (env / ".xrlenv-boot.sh").read_text()
    # runs the recovered command in the background, then execs harbor's CMD (PID 1)
    assert f"( {cmd} ) &" in boot
    assert 'exec "$@"' in boot
    # idempotent — a second pass appends nothing
    assert restore_dropped_commands(shard) == 0


def test_dropped_command_set_disjoint_from_other_fixes() -> None:
    # A task is fixed exactly one way (no double-fix).
    assert not (set(DROPPED_COMMAND_TASKS) & BASE_IMAGE_FIX_TASKS)
    assert not (set(DROPPED_COMMAND_TASKS) & {s.task for s in SYSBOX_TASKS})
    assert "227" in DROPPED_COMMAND_TASKS  # the first proven task


# ── verifier-as-root markers ([verifier] user = "root" for custom-user tasks) ──


def test_verifier_root_marker_adds_user_root_under_existing_table() -> None:
    # A custom-user task's task.toml already has a [verifier] table (timeout only);
    # the marker inserts user = "root" under it (surgical, valid TOML).
    toml = (
        'version = "1.0"\n\n[metadata]\nsets_custom_user = true\n\n'
        "[verifier]\ntimeout_sec = 360.0\n\n[agent]\ntimeout_sec = 360.0\n"
    )
    out, status = apply_verifier_root_marker(toml)
    assert status == "patched"
    parsed = tomllib.loads(out)
    assert parsed["verifier"]["user"] == "root"
    assert parsed["verifier"]["timeout_sec"] == 360.0   # existing key preserved
    assert parsed["agent"]["timeout_sec"] == 360.0
    # idempotent
    assert apply_verifier_root_marker(out)[1] == "already"


def test_verifier_root_marker_never_overwrites_existing_user() -> None:
    # If a task deliberately pins a different verifier user, leave it alone.
    toml = '[verifier]\nuser = "someone"\ntimeout_sec = 5.0\n'
    out, status = apply_verifier_root_marker(toml)
    assert status == "already"
    assert tomllib.loads(out)["verifier"]["user"] == "someone"


def test_apply_all_verifier_root_markers_marks_the_set(tmp_path: Path) -> None:
    shard = tmp_path / "seta-env"
    for tid in VERIFIER_ROOT_TASKS | {"999"}:  # 999 = an unlisted control task
        (shard / tid).mkdir(parents=True)
        (shard / tid / "task.toml").write_text("[verifier]\ntimeout_sec = 1.0\n")
    results = dict(apply_all_verifier_root_markers(shard, only=None))
    assert set(results) == VERIFIER_ROOT_TASKS
    for tid in VERIFIER_ROOT_TASKS:
        got = tomllib.loads((shard / tid / "task.toml").read_text())["verifier"]["user"]
        assert got == "root"
    # unlisted task untouched
    assert "user" not in tomllib.loads((shard / "999" / "task.toml").read_text())["verifier"]


def test_verifier_root_set_is_expected_and_disjoint() -> None:
    assert {"15", "304", "729", "1092"} == VERIFIER_ROOT_TASKS
    # fixed exactly one way — no overlap with the other fix mechanisms.
    assert not (VERIFIER_ROOT_TASKS & BASE_IMAGE_FIX_TASKS)
    assert not (VERIFIER_ROOT_TASKS & set(DROPPED_COMMAND_TASKS))
    assert not (VERIFIER_ROOT_TASKS & {s.task for s in SYSBOX_TASKS})
