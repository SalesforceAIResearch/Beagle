#!/usr/bin/env python3
"""build_cache.py — materialize SWE-bench Pro into the shared cache as harbor tasks.

SWE-bench Pro (ScaleAI/SWE-bench_Pro, 731 public instances; Go/Python/JS/TS across 11 repos)
is distributed as a dataset + a per-instance evaluation kit in the upstream harness repo
(``run_scripts/<id>/{run_script.sh,parser.py}``, ``dockerfiles/{base,instance}_dockerfile/<id>``)
+ a prebuilt image per instance on Docker Hub (``jefzda/sweap-images:<dockerhub_tag>``). Its
xrlenv shape is the **harbor golden path**: this builder turns every selected instance into a
self-contained harbor task dir under ``<cache>/swebench-pro/<instance_id>/`` and the sweep
reuses ``xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`` with zero adapter code.

    <cache>/swebench-pro/<instance_id>/
    ├── task.toml              [environment] docker_image = jefzda/sweap-images:<tag>, cpus/memory, timeouts
    ├── instruction.md         problem statement + requirements + interface (what an agent reads)
    ├── instance.json          the full dataset row (anchor, written last)
    ├── environment/Dockerfile FROM <image>            (harbor bookkeeping; the cluster pulls the image)
    ├── solution/gold.patch + solve.sh                 the oracle (dataset ``patch``)
    └── tests/test.sh, run_script.sh, parser.py, env.sh, f2p.json, p2p.json, grade.py
        test.sh reproduces upstream's entry script (swe_bench_pro_eval.create_entryscript): export the
        Dockerfiles' ENV lines, reset /app to the base commit, apply the submission, check out the
        solution's test files (before_repo_set_cmd), run the instance run_script on the selected test
        files, parse with the instance parser, grade with upstream's rule
        resolved <=> FAIL_TO_PASS | PASS_TO_PASS ⊆ {tests PASSED}  ->  /logs/verifier/reward.txt (+ reward.json)

Selection (exactly one flag): ``--all`` (the full corpus, 731), ``--filtered`` (the quality-filtered
set, ``filtered_instance_ids.txt``, 478), ``--subset-100`` (the 100-task sample of the filtered set,
``subset_100_instance_ids.txt``), plus ``--smoke`` (the first 8 rows), ``--ids-file``, ``--instances``.

Inputs (no network, no default locations): ``$SWEBENCH_PRO_PARQUET`` — the dataset parquet, or the
directory of a ``ScaleAI/SWE-bench_Pro`` snapshot (``huggingface-cli download ScaleAI/SWE-bench_Pro
--repo-type dataset --local-dir <dir>``) — and ``$SWEBENCH_PRO_HARNESS`` — a checkout of
https://github.com/scaleapi/SWE-bench_Pro-os (``run_scripts/`` + ``dockerfiles/``). Both may live in
this repo's ``.env`` next to ``XRLENV_BENCHMARK_CACHE`` (loaded automatically).

    .venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/build_cache.py --subset-100
    .venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/build_cache.py --filtered
    .venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/build_cache.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SHARD = "swebench-pro"
GOLDEN_SUBDIR = "golden_patches"     # second supported shard layout (see shard_dir)


def has_task_dirs(d: Path) -> bool:
    """True when ``d`` holds materialized task dirs (``<d>/<instance_id>/task.toml``)."""
    return d.is_dir() and next(d.glob("*/task.toml"), None) is not None


def shard_dir(root: str | Path) -> Path:
    """The task-dir shard under a cache ROOT — the single rule every entry point uses.

    Two layouts are supported: ``<root>/swebench-pro/<instance_id>/`` (canonical) and
    ``<root>/swebench-pro/golden_patches/<instance_id>/``. Whichever holds task dirs is used;
    the canonical one wins when both do, and an empty root resolves to it so a populate writes
    the canonical layout.
    """
    base = Path(root).expanduser() / SHARD
    if has_task_dirs(base):
        return base
    nested = base / GOLDEN_SUBDIR
    return nested if has_task_dirs(nested) else base
FILTERED_IDS = HERE / "scripts" / "filtered_instance_ids.txt"          # the quality-filtered set (478)
SUBSET_100_IDS = HERE / "scripts" / "subset_100_instance_ids.txt"      # 100 of the filtered set, spread over the 11 repos (sample_subset.py)
DATASET_ID = "ScaleAI/SWE-bench_Pro"
HARNESS_URL = "https://github.com/scaleapi/SWE-bench_Pro-os"
PARQUET_ENV = "SWEBENCH_PRO_PARQUET"                       # the parquet file, or the snapshot directory holding data/test-*.parquet
HARNESS_ENV = "SWEBENCH_PRO_HARNESS"                       # the upstream checkout (run_scripts/ + dockerfiles/)
PARQUET_GLOBS = ("data/test-*.parquet", "test-*.parquet", "*.parquet")
IMAGE_REPO = "jefzda/sweap-images"
SMOKE_COUNT = 8
EXPECTED_FILES = ("task.toml", "instruction.md", "instance.json", "environment/Dockerfile", "solution/gold.patch",
                  "solution/solve.sh", "tests/test.sh", "tests/run_script.sh", "tests/parser.py", "tests/env.sh",
                  "tests/f2p.json", "tests/p2p.json", "tests/grade.py")

# per-language container sizing (harbor task.toml); the JS/TS and Go suites parallelise on nproc
RESOURCES = {
    "python": {"cpus": 2, "memory_mb": 8192, "storage_mb": 20480, "verifier_timeout_sec": 1800.0},
    "js": {"cpus": 4, "memory_mb": 12288, "storage_mb": 30720, "verifier_timeout_sec": 2400.0},
    "ts": {"cpus": 4, "memory_mb": 12288, "storage_mb": 30720, "verifier_timeout_sec": 2400.0},
    "go": {"cpus": 4, "memory_mb": 8192, "storage_mb": 30720, "verifier_timeout_sec": 2400.0},
}
# repos with very large images / slow suites (protonmail 15-20 GB; tutao "take a long time to eval" per upstream)
# element-web: 32 GiB, not 16 — its ``npx jest`` runs without ``--maxWorkers`` and the full-sweep oracle of
# instance 53a9b644 had jest workers SIGKILLed (OOM) at 16 GiB on all 3 attempts (2026-08-27, 729/731).
HEAVY_REPOS = {"protonmail/webclients": (16384, 61440, 3600.0), "tutao/tutanota": (16384, 40960, 3600.0),
               "element-hq/element-web": (32768, 40960, 3600.0), "gravitational/teleport": (16384, 40960, 3600.0),
               "NodeBB/NodeBB": (12288, 30720, 2400.0)}
AGENT_TIMEOUT_SEC = 5400.0


# ── dataset / harness inputs ───────────────────────────────────────────────────

def parquet_path(explicit: str | None = None) -> Path:
    """The dataset parquet: ``explicit`` or ``$SWEBENCH_PRO_PARQUET`` — the file itself, or the directory of a
    HF snapshot (``data/test-*.parquet`` inside it). No default location and no download: fail loud instead."""
    raw = explicit or os.environ.get(PARQUET_ENV)
    if not raw:
        raise SystemExit(f"set {PARQUET_ENV} (or pass --parquet) to the {DATASET_ID} parquet or snapshot directory: "
                         f"huggingface-cli download {DATASET_ID} --repo-type dataset --local-dir <dir>")
    p = Path(raw).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        for pattern in PARQUET_GLOBS:
            hits = sorted(p.glob(pattern))
            if hits:
                return hits[0]
    raise SystemExit(f"{PARQUET_ENV}={raw}: not a parquet file or a snapshot directory holding data/test-*.parquet")


def harness_dir(explicit: str | None = None) -> Path:
    """The upstream evaluation kit: ``explicit`` or ``$SWEBENCH_PRO_HARNESS`` (a clone of SWE-bench_Pro-os)."""
    raw = explicit or os.environ.get(HARNESS_ENV)
    if not raw:
        raise SystemExit(f"set {HARNESS_ENV} (or pass --harness) to a checkout of {HARNESS_URL} (run_scripts/ + dockerfiles/)")
    d = Path(raw).expanduser()
    if not (d / "run_scripts").is_dir() or not (d / "dockerfiles").is_dir():
        raise SystemExit(f"{HARNESS_ENV}={raw} is not the upstream SWE-bench_Pro-os checkout (needs run_scripts/ and dockerfiles/): "
                         f"git clone {HARNESS_URL}")
    return d


def load_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    for r in rows:
        safe_instance_id(r["instance_id"])
    return rows


def safe_instance_id(iid: str) -> str:
    """The id is interpolated into cache paths: it must be a bare path component."""
    if not iid or iid in (".", "..") or iid != Path(iid).name or "/" in iid:
        raise SystemExit(f"unsafe instance id {iid!r}: must be a bare name")
    return iid


def read_ids_file(path: Path) -> list[str]:
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if not ids:
        raise SystemExit(f"{path} selected no instances")
    return ids


def select_rows(rows: list[dict[str, Any]], *, all_: bool, smoke: bool, ids_file: Path | None, instances: str | None,
                filtered: bool = False, subset_100: bool = False) -> list[dict[str, Any]]:
    """The three shipped configurations (``all_`` = full 731, ``filtered`` = 478, ``subset_100`` = 100) plus the
    ad-hoc selections; manifests keep their own order."""
    by_id = {r["instance_id"]: r for r in rows}
    if smoke:
        return rows[:SMOKE_COUNT]
    if instances:
        want = [t.strip() for t in instances.split(",") if t.strip()]
    elif all_:
        return list(rows)
    else:
        f = ids_file or (SUBSET_100_IDS if subset_100 else FILTERED_IDS if filtered else None)
        if f is None:
            raise SystemExit("no selection: pass --all, --filtered, --subset-100, --smoke, --ids-file or --instances")
        if not f.is_file():
            raise SystemExit(f"no id manifest at {f}" + (" — regenerate it with scripts/sample_subset.py" if subset_100 else ""))
        want = read_ids_file(f)
    missing = [w for w in want if w not in by_id]
    if missing:
        raise SystemExit(f"unknown instance id(s): {missing[:5]}{' …' if len(missing) > 5 else ''}")
    return [by_id[w] for w in want]


# ── pure renderers (unit-tested) ───────────────────────────────────────────────

def _toml_str(s: str) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def _json_list(value: Any) -> list[str]:
    """fail_to_pass / pass_to_pass / selected_test_files_to_run are JSON-encoded lists (strings) in the parquet."""
    if isinstance(value, list):
        return [str(x) for x in value]
    if not value:
        return []
    try:
        v = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        import ast
        v = ast.literal_eval(value)          # upstream uses eval(); some rows are Python-literal quoted
    return [str(x) for x in v]


def image_ref(row: dict[str, Any]) -> str:
    tag = (row.get("dockerhub_tag") or "").strip()
    if not tag:
        raise SystemExit(f"{row['instance_id']}: dataset row has no dockerhub_tag")
    return f"{IMAGE_REPO}:{tag}"


def resources_for(row: dict[str, Any]) -> dict[str, Any]:
    lang = (row.get("repo_language") or "python").lower()
    res = dict(RESOURCES.get(lang, RESOURCES["python"]))
    heavy = HEAVY_REPOS.get(row.get("repo") or "")
    if heavy:
        res["memory_mb"], res["storage_mb"], res["verifier_timeout_sec"] = max(res["memory_mb"], heavy[0]), max(res["storage_mb"], heavy[1]), max(res["verifier_timeout_sec"], heavy[2])
    return res


def render_task_toml(row: dict[str, Any], *, kept: bool | None = None) -> str:
    res = resources_for(row)
    iid = row["instance_id"]
    lines = [
        'schema_version = "1.1"', "artifacts = []", "",
        "[task]", f"name = {_toml_str('swebench-pro/' + iid)}",
        f"description = {_toml_str(str(row.get('repo')) + ' — SWE-bench Pro instance ' + iid)}", "",
        "[metadata]", 'benchmark = "swebench-pro"', f"instance_id = {_toml_str(iid)}", f"repo = {_toml_str(str(row.get('repo') or ''))}",
        f"language = {_toml_str(str(row.get('repo_language') or ''))}", f"base_commit = {_toml_str(str(row.get('base_commit') or ''))}",
        f"dockerhub_tag = {_toml_str(str(row.get('dockerhub_tag') or ''))}",
        f"issue_categories = {_toml_str(str(row.get('issue_categories') or ''))}",
        f"filter_kept = {'true' if kept else 'false'}" if kept is not None else "filter_kept = true", "",
        "[verifier]", f"timeout_sec = {float(res['verifier_timeout_sec'])}", "",
        "[agent]", f"timeout_sec = {AGENT_TIMEOUT_SEC}", "",
        "[environment]", "build_timeout_sec = 1800.0", f"docker_image = {_toml_str(image_ref(row))}", f"cpus = {res['cpus']}",
        f"memory_mb = {res['memory_mb']}", f"storage_mb = {res['storage_mb']}", "gpus = 0", "allow_internet = true", "mcp_servers = []", "",
        "[verifier.env]", "",
        # the images' ENTRYPOINT is /bin/bash (upstream: "bash runs by default"); the cluster env must exec the
        # keep-alive directly or the container exits at once (xrlenv_plugins.harbor.environment._keepalive_argv)
        "[environment.env]", 'XRLENV_KEEPALIVE_ENTRYPOINT = "1"', "",
    ]
    return "\n".join(lines)


def render_instruction(row: dict[str, Any]) -> str:
    parts = [f"# {row.get('repo')} — {row['instance_id']}", "",
             f"The repository is checked out at `/app` (base commit `{row.get('base_commit')}`). Resolve the issue below by "
             "editing the code in place. Hidden tests will be run against your working tree; do not edit or delete existing test files.", "",
             "## Problem statement", "", str(row.get("problem_statement") or "").strip(), ""]
    if row.get("requirements"):
        parts += ["## Requirements", "", str(row["requirements"]).strip(), ""]
    if row.get("interface"):
        parts += ["## Interface", "", str(row["interface"]).strip(), ""]
    return "\n".join(parts)


def dockerfile_env_exports(base_dockerfile: str, instance_dockerfile: str) -> str:
    """Upstream (create_entryscript) turns every ``ENV …`` line of both Dockerfiles into an ``export …``
    line, verbatim (``ENV`` -> ``export``). Mirrored exactly, quirks included, so grading matches."""
    out = ["#!/bin/bash", "# ENV lines of the instance's base + instance Dockerfiles, as upstream exports them"]
    for content in (base_dockerfile, instance_dockerfile):
        for line in (content or "").split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                out.append(line.replace("ENV", "export", 1))
    return "\n".join(out) + "\n"


def render_test_sh(row: dict[str, Any]) -> str:
    base = str(row.get("base_commit") or "").strip()
    before = (row.get("before_repo_set_cmd") or "").strip().split("\n")[-1]      # upstream: only the last line
    selected = ",".join(_json_list(row.get("selected_test_files_to_run")))       # upstream: ONE comma-joined argument
    return f"""#!/bin/bash
# swebench-pro verifier (harbor tests/test.sh) — upstream swe_bench_pro_eval.create_entryscript, adapted to grade the
# working tree the agent (or the oracle's solve.sh) left in /app. Writes /logs/verifier/reward.txt (+ reward.json).
set -uo pipefail
mkdir -p /logs/verifier
source /tests/env.sh
cd /app || {{ echo 0 > /logs/verifier/reward.txt; echo '{{"reward": 0, "error": "no /app"}}' > /logs/verifier/reward.json; exit 0; }}
git config --global --add safe.directory /app >/dev/null 2>&1 || true
# 1. the submission = everything changed vs the base commit (committed or not, new files included)
git add -A >/dev/null 2>&1 || true
git diff --cached --binary {shlex.quote(base)} > /logs/verifier/model.patch 2>/dev/null || true
echo "[verifier] submission: $(wc -c < /logs/verifier/model.patch) bytes"
# 2. pristine base (upstream: git reset --hard <base>; git checkout <base>)
git reset --hard {shlex.quote(base)} >/dev/null 2>&1
git checkout {shlex.quote(base)} >/dev/null 2>&1 || true
# 3. re-apply the submission (upstream: git apply -v /workspace/patch.diff)
if [ -s /logs/verifier/model.patch ]; then
  git apply -v /logs/verifier/model.patch > /logs/verifier/apply.log 2>&1 || echo "[verifier] patch apply failed rc=$?" >> /logs/verifier/apply.log
fi
# 4. the solution commit's test files (dataset before_repo_set_cmd, last line)
{before}
# 5. run the instance's test script on the selected test files, then parse
bash /tests/run_script.sh {shlex.quote(selected)} > /logs/verifier/stdout.log 2> /logs/verifier/stderr.log
PY=$(command -v python3 || command -v python)
"$PY" /tests/parser.py /logs/verifier/stdout.log /logs/verifier/stderr.log /logs/verifier/output.json
# 6. upstream's rule: resolved <=> FAIL_TO_PASS | PASS_TO_PASS ⊆ {{tests PASSED}}
"$PY" /tests/grade.py /logs/verifier/output.json /tests/f2p.json /tests/p2p.json /logs/verifier/reward.json > /logs/verifier/reward.txt
echo "[verifier] reward=$(cat /logs/verifier/reward.txt)"
"""


GRADE_PY = '''#!/usr/bin/env python3
"""grade.py — upstream SWE-bench Pro rule: resolved <=> (FAIL_TO_PASS | PASS_TO_PASS) ⊆ {tests PASSED}.
Prints the reward (1/0) and writes reward.json with the breakdown. stdlib only (runs inside the task image)."""
import json, sys
out_path, f2p_path, p2p_path, reward_path = sys.argv[1:5]
try:
    tests = json.load(open(out_path)).get("tests", [])
except Exception as exc:
    tests, err = [], f"{type(exc).__name__}: {exc}"
else:
    err = None
passed = {t.get("name") for t in tests if t.get("status") == "PASSED"}
f2p = set(json.load(open(f2p_path))); p2p = set(json.load(open(p2p_path)))
# Upstream's rule is exact name equality. The dataset lists carry two mangling artifacts that no run can ever
# satisfy exactly (observed 2026-08-26 on NodeBB-00c70ce7: 4/681 names): a name cut at an embedded double quote
# (`… ACP default "day` for `… ACP default "day"`) and trailing-whitespace differences. Curated, auditable
# loosening: a listed name also matches a parsed name that equals it after rstrip(), or — when the listed name
# has an unbalanced quote — a parsed name that starts with it. Everything else stays exact.
passed_stripped = {n.rstrip() for n in passed}
def hit(name):
    if name in passed or name.rstrip() in passed_stripped:
        return True
    if name.count('"') % 2 == 1:
        return any(pn.startswith(name) for pn in passed)
    return False
f2p_hit = {n for n in f2p if hit(n)}; p2p_hit = {n for n in p2p if hit(n)}
f2p_ok = len(f2p_hit); p2p_ok = len(p2p_hit)
resolved = f2p_hit == f2p and p2p_hit == p2p and not err
# harbor parses reward.json as ``rewards: dict[str, float | int]`` — numbers ONLY here; the rest goes to grade_details.json
rec = {"reward": 1 if resolved else 0, "resolved": 1 if resolved else 0, "f2p_total": len(f2p), "f2p_passed": f2p_ok,
       "p2p_total": len(p2p), "p2p_passed": p2p_ok, "f2p": (f2p_ok / len(f2p)) if f2p else 0.0, "p2p": (p2p_ok / len(p2p)) if p2p else 1.0,
       "n_parsed": len(tests), "n_passed": len(passed)}
json.dump(rec, open(reward_path, "w"), indent=1)
details = {"error": err, "missing_f2p": sorted(f2p - f2p_hit)[:50], "missing_p2p": sorted(p2p - p2p_hit)[:50],
           "lenient_matches": sorted((f2p_hit | p2p_hit) - passed)[:50],
           "statuses": dict(__import__("collections").Counter(t.get("status") for t in tests))}
json.dump(details, open(reward_path.replace("reward.json", "grade_details.json"), "w"), indent=1)
print(rec["reward"])
'''

SOLVE_SH = """#!/bin/bash
# oracle: apply the dataset gold patch to /app (upstream applies it with `git apply -v`)
set -uo pipefail
cd /app || exit 1
git config --global --add safe.directory /app >/dev/null 2>&1 || true
for cmd in "git apply --verbose" "git apply --verbose --3way --recount --ignore-space-change --whitespace=nowarn" "patch --batch --fuzz=5 -p1 -i"; do
  if $cmd /solution/gold.patch; then echo ">>>>> Applied Patch ($cmd)"; exit 0; fi
  git reset --hard HEAD >/dev/null 2>&1; git clean -fd >/dev/null 2>&1
done
echo ">>>>> Patch Apply Failed"; exit 1
"""


# ── writer ─────────────────────────────────────────────────────────────────────

def _write(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:      # keep dataset bytes (CRLF issue text) verbatim
        fh.write(text)
    if mode is not None:
        path.chmod(mode)


def is_complete(task_dir: Path, iid: str) -> bool:
    if not all((task_dir / f).is_file() for f in EXPECTED_FILES):
        return False
    try:
        return json.loads((task_dir / "instance.json").read_text(encoding="utf-8")).get("instance_id") == iid
    except Exception:
        return False


def refresh_kit_files(row: dict[str, Any], task_dir: Path, *, kept: bool | None = None) -> bool:
    """A complete task dir whose kit-rendered files predate the current renderers gets just those files
    rewritten: ``tests/grade.py`` (the grading rule), ``tests/test.sh`` (the verifier), ``solution/solve.sh``,
    and ``task.toml`` (container sizing / timeouts — a ``RESOURCES`` / ``HEAVY_REPOS`` change must reach the
    cache too). A grading fix must reach every cached task on the next ``build_cache`` run, without
    ``--overwrite`` (2026-08-27: 481 cached dirs still carried the pre-lenient grade.py and failed a passing
    oracle). ``kept`` is the same filter flag ``write_task`` renders into ``task.toml``.
    Returns True when something was rewritten."""
    want = {"tests/grade.py": (GRADE_PY, None), "tests/test.sh": (render_test_sh(row), 0o755), "solution/solve.sh": (SOLVE_SH, 0o755),
            "task.toml": (render_task_toml(row, kept=kept), None)}
    changed = False
    for rel, (text, mode) in want.items():
        p = task_dir / rel
        if not p.is_file() or p.read_text(encoding="utf-8") != text:
            _write(p, text, mode)
            changed = True
    return changed


def write_task(row: dict[str, Any], task_dir: Path, harness: Path, *, kept: bool | None = None) -> None:
    iid = row["instance_id"]
    rs = harness / "run_scripts" / iid
    base_df = harness / "dockerfiles" / "base_dockerfile" / iid / "Dockerfile"
    inst_df = harness / "dockerfiles" / "instance_dockerfile" / iid / "Dockerfile"
    for p in (rs / "run_script.sh", rs / "parser.py"):
        if not p.is_file():
            raise SystemExit(f"{iid}: upstream kit missing {p}")
    _write(task_dir / "task.toml", render_task_toml(row, kept=kept))
    _write(task_dir / "instruction.md", render_instruction(row))
    _write(task_dir / "environment" / "Dockerfile", f"FROM {image_ref(row)}\n")
    _write(task_dir / "solution" / "gold.patch", str(row.get("patch") or ""))
    _write(task_dir / "solution" / "solve.sh", SOLVE_SH, 0o755)
    _write(task_dir / "tests" / "run_script.sh", (rs / "run_script.sh").read_text(encoding="utf-8"), 0o755)
    _write(task_dir / "tests" / "parser.py", (rs / "parser.py").read_text(encoding="utf-8"))
    _write(task_dir / "tests" / "env.sh", dockerfile_env_exports(base_df.read_text(encoding="utf-8") if base_df.is_file() else "",
                                                                 inst_df.read_text(encoding="utf-8") if inst_df.is_file() else ""))
    _write(task_dir / "tests" / "f2p.json", json.dumps(_json_list(row.get("fail_to_pass")), ensure_ascii=False, indent=0))
    _write(task_dir / "tests" / "p2p.json", json.dumps(_json_list(row.get("pass_to_pass")), ensure_ascii=False, indent=0))
    _write(task_dir / "tests" / "grade.py", GRADE_PY)
    _write(task_dir / "tests" / "test.sh", render_test_sh(row), 0o755)
    _write(task_dir / "instance.json", json.dumps(row, ensure_ascii=False, indent=1))     # anchor, written last


def load_dotenv() -> None:
    """This repo's ``.env`` (XRLENV_BENCHMARK_CACHE, SWEBENCH_PRO_PARQUET, SWEBENCH_PRO_HARNESS, XRLENV_GRPC_*) for
    the standalone entrypoints — the same loader ``xrlenv`` runs at import; silent if xrlenv is not importable."""
    try:
        from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
    except Exception:
        return
    _maybe_auto_load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swebench-pro build_cache", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true", help="full: all 731 public instances")
    sel.add_argument("--filtered", action="store_true", help=f"filtered: the quality-filtered set, {FILTERED_IDS.name} (478)")
    sel.add_argument("--subset-100", action="store_true", help=f"subset-100: the 100-task sample of the filtered set across the 11 repos, {SUBSET_100_IDS.name}")
    sel.add_argument("--smoke", action="store_true", help=f"the first {SMOKE_COUNT} rows")
    sel.add_argument("--ids-file", default=None, help="an explicit id manifest (one id per line, '#' comments)")
    sel.add_argument("--instances", default=None, help="comma list of instance ids")
    p.add_argument("--dest", default=None, help="cache ROOT (default $XRLENV_BENCHMARK_CACHE); the shard is <root>/swebench-pro (or <root>/swebench-pro/golden_patches when only that level is populated)")
    p.add_argument("--parquet", default=None, help=f"dataset parquet or snapshot directory (default ${PARQUET_ENV})")
    p.add_argument("--harness", default=None, help=f"upstream SWE-bench_Pro-os checkout (default ${HARNESS_ENV})")
    p.add_argument("--overwrite", action="store_true", help="re-materialize complete task dirs too")
    p.add_argument("--list", action="store_true", help="print the selected instance ids and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    load_dotenv()
    from xrlenv_plugins.benchmarks._benchmark_cache import (
        benchmark_cache_root,
        guard_legacy_cache_env,
    )
    guard_legacy_cache_env(a.dest)
    rows = load_rows(parquet_path(a.parquet))
    selected = select_rows(rows, all_=a.all, smoke=a.smoke, ids_file=Path(a.ids_file).expanduser() if a.ids_file else None, instances=a.instances,
                           filtered=a.filtered, subset_100=a.subset_100)
    if a.list:
        print("\n".join(r["instance_id"] for r in selected))
        return 0
    harness = harness_dir(a.harness)
    shard = shard_dir(benchmark_cache_root(a.dest))
    shard.mkdir(parents=True, exist_ok=True)
    kept_ids = set(read_ids_file(FILTERED_IDS)) if FILTERED_IDS.is_file() else None
    written = skipped = refreshed = 0
    for row in selected:
        iid = row["instance_id"]
        task_dir = shard / iid
        kept = (iid in kept_ids) if kept_ids is not None else None
        if not a.overwrite and is_complete(task_dir, iid):
            if refresh_kit_files(row, task_dir, kept=kept):      # renderer changed since this dir was written
                refreshed += 1
            else:
                skipped += 1
            continue
        write_task(row, task_dir, harness, kept=kept)
        written += 1
    print(f"swebench-pro: {written} task(s) written, {refreshed} refreshed (kit files), {skipped} already current -> {shard} "
          f"({len(selected)} selected of {len(rows)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
