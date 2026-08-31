"""Unit tests for the node-side compose runner (``xrlenv.node.raw_compose``).

The runner's orchestration is exercised through an injected fake shell runner —
no docker required. We pin the command sequence (``up`` → ``ps`` → ``inspect``),
image-ensure ordering, container-ID resolution, main-service resolution, and the
teardown-on-failure guarantee (a failed acquire never leaks a project).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from xrlenv.node.raw_compose import (
    ComposeError,
    ComposeProjectRunner,
    ShellResult,
    _parse_ps_json,
)


class FakeRunner:
    """Records every argv and replies based on the compose subcommand. Defaults
    to success for every command; override per-subcommand via ``responses``."""

    def __init__(self, responses: dict[str, ShellResult] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.responses = responses or {}

    def _subcommand(self, argv: list[str]) -> str:
        # the compose verb is the first token after the -f <file> block, but it's
        # simplest to scan for a known verb.
        for verb in ("up", "ps", "down", "logs", "inspect"):
            if verb in argv:
                return verb
        return argv[-1]

    async def __call__(self, argv, *, timeout_s=None) -> ShellResult:
        self.calls.append(list(argv))
        sub = self._subcommand(list(argv))
        if sub in self.responses:
            return self.responses[sub]
        if sub == "ps":
            # default: one main container (short id, as real `compose ps` emits)
            return ShellResult(
                rc=0,
                stdout=json.dumps({"Service": "main", "ID": "cid_main"}),
                stderr="",
            )
        if sub == "inspect":
            # default: batch inspect upgrades the short id -> full id | name | svc
            return ShellResult(
                rc=0, stdout="cidmainfull|/proj-main|main\n", stderr="",
            )
        return ShellResult(rc=0, stdout="", stderr="")

    def verbs(self) -> list[str]:
        return [self._subcommand(c) for c in self.calls]


def _two_service_ps() -> ShellResult:
    return ShellResult(
        rc=0,
        stdout="\n".join([
            json.dumps({"Service": "main", "ID": "cid_main", "State": "running"}),
            json.dumps({"Service": "postgres", "ID": "cid_pg", "State": "running"}),
        ]),
        stderr="",
    )


def _two_service_inspect() -> ShellResult:
    # full ids (64-char-ish) + /name + compose service, one line per container
    return ShellResult(
        rc=0,
        stdout="\n".join([
            "cidmainfull|/proj-main|main",
            "cidpgfull|/proj-postgres|postgres",
        ]),
        stderr="",
    )


@pytest.mark.asyncio
async def test_up_happy_path_resolves_main_and_members(tmp_path: Path) -> None:
    runner = FakeRunner({"ps": _two_service_ps(), "inspect": _two_service_inspect()})
    ensured: list[str] = []

    async def ensure(ref: str) -> None:
        ensured.append(ref)

    cpr = ComposeProjectRunner(run=runner, ensure_image=ensure, root_dir=tmp_path)
    rec = await cpr.up(
        project_name="proj",
        compose_yaml="services:\n  main: {}\n  postgres: {image: postgres:14}\n",
        images=["ns/app:main", "postgres:14"],
        main_service="main",
    )
    # images ensured before up, in order
    assert ensured == ["ns/app:main", "postgres:14"]
    # command sequence: up then ps then inspect (ps discovers, inspect upgrades)
    assert runner.verbs()[:3] == ["up", "ps", "inspect"]
    # the compose file was written into the project dir
    assert (Path(rec.project_dir) / "docker-compose.yaml").is_file()
    # main + members resolved to FULL ids (matches list_raw_containers)
    assert rec.main_container_id == "cidmainfull"
    assert rec.main_container_name == "proj-main"
    assert set(rec.member_container_ids) == {"cidmainfull", "cidpgfull"}
    assert rec.service_container_ids == {
        "main": "cidmainfull", "postgres": "cidpgfull",
    }


@pytest.mark.asyncio
async def test_resolve_upgrades_short_ps_ids_to_full_via_inspect(tmp_path: Path) -> None:
    # the node-truth diff needs FULL ids; `compose ps` gives short ids, so the
    # runner must inspect the short ids and return the full ones.
    runner = FakeRunner({"ps": _two_service_ps(), "inspect": _two_service_inspect()})
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    rec = await cpr.up(project_name="proj", compose_yaml="services:\n  main: {}\n")
    # inspect was called with the SHORT ids that ps reported
    inspect_call = next(c for c in runner.calls if "inspect" in c)
    assert "cid_main" in inspect_call and "cid_pg" in inspect_call
    # ...and the record carries the FULL ids
    assert rec.main_container_id == "cidmainfull"
    assert rec.service_container_ids["postgres"] == "cidpgfull"


@pytest.mark.asyncio
async def test_inspect_failure_raises_and_tears_down(tmp_path: Path) -> None:
    runner = FakeRunner({
        "ps": _two_service_ps(),
        "inspect": ShellResult(rc=1, stdout="", stderr="no such object"),
    })
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    with pytest.raises(ComposeError, match=r"inspect.*failed"):
        await cpr.up(project_name="proj", compose_yaml="services:\n  main: {}\n")
    assert "down" in runner.verbs()  # partial project torn down


@pytest.mark.asyncio
async def test_up_includes_wait_flag_by_default(tmp_path: Path) -> None:
    runner = FakeRunner()
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    await cpr.up(project_name="p", compose_yaml="services:\n  main: {}\n")
    up_call = next(c for c in runner.calls if "up" in c)
    assert "--wait" in up_call and "-d" in up_call


@pytest.mark.asyncio
async def test_up_wait_false_omits_flag(tmp_path: Path) -> None:
    runner = FakeRunner()
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    await cpr.up(
        project_name="p", compose_yaml="services:\n  main: {}\n", wait=False,
    )
    up_call = next(c for c in runner.calls if "up" in c)
    assert "--wait" not in up_call


@pytest.mark.asyncio
async def test_up_failure_captures_logs_and_tears_down(tmp_path: Path) -> None:
    runner = FakeRunner({
        "up": ShellResult(rc=1, stdout="", stderr="dependency failed to start"),
        "logs": ShellResult(rc=0, stdout="postgres | FATAL: boom", stderr=""),
    })
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    with pytest.raises(ComposeError, match=r"dependency failed to start"):
        await cpr.up(project_name="p", compose_yaml="services:\n  main: {}\n")
    verbs = runner.verbs()
    assert "logs" in verbs  # captured a diagnostic
    assert "down" in verbs  # tore the partial project down


@pytest.mark.asyncio
async def test_up_missing_main_service_raises_and_tears_down(tmp_path: Path) -> None:
    runner = FakeRunner({
        "ps": ShellResult(
            rc=0, stdout=json.dumps({"Service": "sidecar", "ID": "x"}), stderr="",
        ),
        "inspect": ShellResult(rc=0, stdout="xfull|/p-sidecar|sidecar\n", stderr=""),
    })
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    with pytest.raises(ComposeError, match=r"no 'main' service"):
        await cpr.up(project_name="p", compose_yaml="services:\n  sidecar: {}\n")
    assert "down" in runner.verbs()


@pytest.mark.asyncio
async def test_up_ps_empty_raises(tmp_path: Path) -> None:
    runner = FakeRunner({"ps": ShellResult(rc=0, stdout="[]", stderr="")})
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    with pytest.raises(ComposeError, match=r"no containers"):
        await cpr.up(project_name="p", compose_yaml="services:\n  main: {}\n")


@pytest.mark.asyncio
async def test_up_tears_down_when_ensure_image_raises(tmp_path: Path) -> None:
    runner = FakeRunner()

    async def bad_ensure(ref: str) -> None:
        raise RuntimeError("pull failed")

    cpr = ComposeProjectRunner(run=runner, ensure_image=bad_ensure, root_dir=tmp_path)
    with pytest.raises(RuntimeError, match="pull failed"):
        await cpr.up(
            project_name="p", compose_yaml="services:\n  main: {}\n",
            images=["x:1"],
        )
    # up never ran (ensure failed first) but teardown still fired
    assert "up" not in runner.verbs()
    assert "down" in runner.verbs()


@pytest.mark.asyncio
async def test_down_success_removes_dir(tmp_path: Path) -> None:
    runner = FakeRunner()  # down defaults to rc=0
    proj = tmp_path / "proj"
    proj.mkdir()
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    await cpr.down(project_name="p", project_dir=str(proj))
    assert "down" in runner.verbs()
    assert not proj.exists()  # temp dir removed on confirmed teardown


@pytest.mark.asyncio
async def test_down_strict_raises_and_keeps_dir_on_failure(tmp_path: Path) -> None:
    # P1 audit: explicit down is STRICT — a non-zero `docker compose down` raises
    # (capacity released only on confirmed teardown, invariant 2) and keeps the
    # project dir so a retry can re-issue down with the same compose file.
    runner = FakeRunner({
        "down": ShellResult(rc=1, stdout="", stderr="daemon wedged"),
    })
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "docker-compose.yaml").write_text("services:\n  main: {}\n")
    cpr = ComposeProjectRunner(run=runner, root_dir=tmp_path)
    with pytest.raises(ComposeError, match=r"down.*failed"):
        await cpr.down(project_name="p", project_dir=str(proj))
    assert proj.exists()  # kept for retry — NOT cleaned on failure


@pytest.mark.asyncio
async def test_default_run_timeout_kills_process_group(tmp_path: Path) -> None:
    # P2 audit: on timeout the WHOLE process group is killed, so no compose helper
    # child outlives the deadline. A shell that would `touch` a marker after a
    # sleep must never create it once we time out before the sleep completes.
    import asyncio

    from xrlenv.node.raw_compose import _default_run

    marker = tmp_path / "marker"
    argv = ["bash", "-c", f"sleep 3; touch {marker}"]
    with pytest.raises(ComposeError, match="timed out"):
        await _default_run(argv, timeout_s=0.3)
    # wait well past when the child would have created the marker
    await asyncio.sleep(3.2)
    assert not marker.exists()  # group killed → no orphaned command past deadline


@pytest.mark.asyncio
async def test_default_run_kills_process_group_on_cancellation(tmp_path: Path) -> None:
    # audit H10: a CANCELLED up/down must kill the whole `docker compose` process group too — not
    # only on timeout. Otherwise the subprocess keeps running and can create containers AFTER the
    # caller's rollback, leaking an unowned stack. A shell that would `touch` a marker after a
    # sleep must never create it once the run is cancelled mid-sleep.
    import asyncio

    from xrlenv.node.raw_compose import _default_run

    marker = tmp_path / "marker"
    argv = ["bash", "-c", f"sleep 3; touch {marker}"]
    task = asyncio.create_task(_default_run(argv, timeout_s=None))
    await asyncio.sleep(0.3)     # let the subprocess start + enter its sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(3.2)     # wait past when the child would have touched the marker
    assert not marker.exists()   # group killed on cancel → no orphaned command


@pytest.mark.asyncio
async def test_empty_project_name_rejected(tmp_path: Path) -> None:
    cpr = ComposeProjectRunner(run=FakeRunner(), root_dir=tmp_path)
    with pytest.raises(ComposeError, match="empty project_name"):
        await cpr.up(project_name="", compose_yaml="services:\n  main: {}\n")


def test_parse_ps_json_array_and_jsonl() -> None:
    array = json.dumps([{"Service": "a", "ID": "1"}, {"Service": "b", "ID": "2"}])
    assert _parse_ps_json(array) == [
        {"Service": "a", "ID": "1"}, {"Service": "b", "ID": "2"},
    ]
    jsonl = '{"Service": "a", "ID": "1"}\n{"Service": "b", "ID": "2"}\n'
    assert _parse_ps_json(jsonl) == [
        {"Service": "a", "ID": "1"}, {"Service": "b", "ID": "2"},
    ]
    assert _parse_ps_json("") == []
    assert _parse_ps_json("  \n ") == []
    # a single object (older compose)
    assert _parse_ps_json('{"Service": "a", "ID": "1"}') == [{"Service": "a", "ID": "1"}]
