"""Unit tests for the seta-env build-plan generator.

seta-env publishes only Dockerfiles (no prebuilt registry image), so every plan
entry must be ``context_source: type: git`` pointing at the task's
``Harbor-Dataset/<id>/environment`` subdir — and the range/blacklist parsing that
feeds task selection must be exact. Offline: no network, no cluster.
"""
from __future__ import annotations

from pathlib import Path

from xrlenv_plugins.benchmarks.seta import build_plan_gen as gen
from xrlenv_plugins.benchmarks.seta.build_cache import (
    BASE_IMAGE_FIX_TASKS,
    DROPPED_COMMAND_TASKS,
)


def test_parse_range_mixes_singletons_and_ranges() -> None:
    assert gen._parse_range("0-2,5,10-11") == ["0", "1", "2", "5", "10", "11"]
    assert gen._parse_range("42") == ["42"]
    assert gen._parse_range(" 0-1 , 3 ") == ["0", "1", "3"]


def test_generate_plan_emits_git_source() -> None:
    plan = gen.generate_plan(["0", "1"], ref="main")
    assert plan["version"] == 1
    assert len(plan["entries"]) == 2
    e0 = plan["entries"][0]
    assert e0["image_ref"] == "seta-env/0:main"
    cs = e0["context_source"]
    assert cs["type"] == "git"
    assert cs["repo"].endswith("camel-ai/seta-env")
    assert cs["ref"] == "main"
    assert cs["subdir"] == "Harbor-Dataset/0/environment"
    assert cs["dockerfile"] == "Dockerfile"
    assert e0["placement"]["size_hint_source"] == "heuristic"
    assert e0["labels"] == {"xrlenv.benchmark": "seta-env", "xrlenv.task_id": "0"}


def test_generate_plan_name_reflects_count() -> None:
    assert gen.generate_plan(["0", "1", "2"], ref="main")["name"] == "seta-env-3-task"


def test_image_ref_sanitizes_ref() -> None:
    # A ref with tag-illegal chars (slash/colon) is normalized for the tag slot.
    assert gen._image_ref("7", "feature/x") == "seta-env/7:feature-x"


def test_load_blacklist_parses_first_token(tmp_path: Path) -> None:
    bl = tmp_path / "black_list.txt"
    bl.write_text(
        "# comment line\n"
        "\n"
        "25      # COPY of an uncommitted file\n"
        "387     # python3 not installed\n",
        encoding="utf-8",
    )
    assert gen._load_blacklist(bl) == {"25", "387"}
    assert gen._load_blacklist(tmp_path / "missing.txt") == set()


def test_committed_blacklist_build_and_runtime_exclusions() -> None:
    # The committed black_list.txt carries BOTH axes (see STATUS.md): the 5
    # build-unbuildable ids AND the harbor-0.20 runtime-excluded set. Assert the
    # build ids are present, and that every FIXED task is NOT excluded (regression
    # guard: never blacklist a task we fixed).
    committed = Path(gen.__file__).resolve().parent / gen.DEFAULT_BLACKLIST_NAME
    bl = gen._load_blacklist(committed)
    assert {"25", "305", "387", "683", "999"} <= bl          # build-unbuildable
    assert len(bl) >= 60                                      # + the runtime set
    # 13 pure-cache fixes (sysbox markers + 309 overlay) ...
    fixed = {"8", "1004", "1117", "1347", "311", "119", "1225",
             "830", "1059", "484", "345", "846", "309"}
    # ... + the base-image-restore tasks (built type: local, not excluded) ...
    fixed |= set(BASE_IMAGE_FIX_TASKS)
    # ... + the dropped-command tasks (ENTRYPOINT-restored, built type: local).
    fixed |= set(DROPPED_COMMAND_TASKS)
    assert not (fixed & bl), f"fixed tasks must not be blacklisted: {fixed & bl}"


def test_generate_plan_base_image_fix_emits_local_with_cache_root() -> None:
    # A base-image-restore task builds type: local from the cache Dockerfile that
    # build_cache rewrote to the t-bench base; a normal task stays type: git.
    fix = next(iter(BASE_IMAGE_FIX_TASKS))
    plan = gen.generate_plan([fix, "0"], ref="main", cache_root="/cache")
    by = {e["labels"]["xrlenv.task_id"]: e["context_source"] for e in plan["entries"]}
    assert by[fix]["type"] == "local"
    assert by[fix]["path"] == f"/cache/seta-env/{fix}/environment"
    assert by[fix]["dockerfile"] == "Dockerfile"
    assert by[fix]["shared_fs"] == gen.DEFAULT_SHARED_FS
    assert by["0"]["type"] == "git"  # non-fix task unchanged


def test_generate_plan_base_image_fix_warns_without_cache_root(capsys) -> None:  # type: ignore[no-untyped-def]
    # Without --cache-root a base-restore task can't build type: local; it falls
    # back to type: git (unrestored → still fails) and warns loudly.
    fix = next(iter(BASE_IMAGE_FIX_TASKS))
    plan = gen.generate_plan([fix], ref="main", cache_root=None)
    assert plan["entries"][0]["context_source"]["type"] == "git"
    assert "unpatched" in capsys.readouterr().err.lower()
