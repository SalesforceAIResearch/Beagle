"""SWE-bench Verified end-to-end smoke against ``xrlenv.from_env()``.

This is a **smoke test**, not a unit / integration test:

- Pulls a real Docker image from the upstream SWE-bench registry
  (~1-3 GiB on first run; cached after).
- Runs the upstream ``swebench.harness.run_evaluation.main()``
  unmodified against ``xrlenv.from_env()`` via a ``docker.from_env``
  monkey-patch.
- Wall-clock: ~5-10 min on first run (image pull dominant), ~30-90 s
  per subsequent run.

Excluded from the default ``pytest -q`` suite via ``addopts =
"--ignore=tests/smoke"`` in ``pyproject.toml``. Run explicitly:

pytest single-instance (default ``sphinx-doc__sphinx-10323``)::

    .venv/bin/python -m pytest tests/smoke/test_swebench_drop_in.py -v

script — single instance, no artifact archiving (default)::

    .venv/bin/python tests/smoke/test_swebench_drop_in.py

script — single instance, archive to default ``<repo>/tmp/``
(gitignored)::

    .venv/bin/python tests/smoke/test_swebench_drop_in.py \\
        --save-artifacts

script — broader infrastructure soak (the 8-instance ``SMOKE_8``
reference set), 2-way parallel, archive to a custom out-of-repo
path (substitute ``$XRLENV_SMOKE_ARCHIVE_ROOT`` with whichever
durable directory you want — typically a long-lived eval-results
tree outside this repo)::

    .venv/bin/python tests/smoke/test_swebench_drop_in.py \\
        --instance-ids \\
        astropy__astropy-7166,django__django-11099,sympy__sympy-18189,astropy__astropy-12907,astropy__astropy-14182,sympy__sympy-13615,django__django-11138,sympy__sympy-12489 \\
        --max-workers 2 \\
        --save-artifacts "$XRLENV_SMOKE_ARCHIVE_ROOT" \\
        --job-id claude-opus-4-7-50-v1.12.0

Output layout under ``<save-artifacts>/<job-id>/`` (only when the
flag is passed)::

    summary-<utc-ts>.json                                               # per-run snapshot (sort by name for chronology; latest = `tail -1`)
    logs/run_evaluation/<run_id>/<model>/<instance>/                    # swebench's per-instance logs
        report.json   run_instance.log   test_output.txt   eval.sh   patch.diff

The architectural validation: SWE-bench's harness drives our
docker-py drop-in unmodified end-to-end with a real container,
real patch application, real test execution, and an
``resolved=True`` signal in the report. Pinning that here so a
future change to ``xrlenv/compat/docker_client.py`` can't silently
break the upstream-harness contract.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_DEFAULT_INSTANCE_ID = "sphinx-doc__sphinx-10323"
_DEFAULT_TIMEOUT_S = 900

# 8-instance reference set the swebench-verified plug-in pinned in
# ``examples/swebench_smoke_tasks.py``. Copied here so the smoke
# survives the slim deletion of that plug-in. Operators can pass
# this whole set to ``--instance-ids`` for a longer batch run; the
# default is one small instance for fast infrastructure validation.
SMOKE_8: tuple[str, ...] = (
    "astropy__astropy-7166",
    "django__django-11099",
    "sympy__sympy-18189",
    "astropy__astropy-12907",
    "astropy__astropy-14182",
    "sympy__sympy-13615",
    "django__django-11138",
    "sympy__sympy-12489",
)


# ──────────────────────────────────────────────────────────────────────────────
# Skip gates
# ──────────────────────────────────────────────────────────────────────────────


def _docker_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False
    return True


def _swebench_available() -> bool:
    try:
        import swebench.harness.run_evaluation  # noqa: F401
    except ImportError:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Smoke implementation
# ──────────────────────────────────────────────────────────────────────────────


def _install_drop_in() -> None:
    """Replace ``docker.from_env`` with our drop-in BEFORE importing
    any swebench module. Must run before the harness loads —
    swebench's ``run_evaluation.main()`` calls ``docker.from_env()``
    directly at line 535."""
    import docker
    from xrlenv.compat.docker_client import from_env as xrlenv_from_env

    docker.from_env = xrlenv_from_env  # type: ignore[assignment]


@contextlib.contextmanager
def _hf_offline_mode() -> Iterator[None]:
    """Force HF offline mode for the duration of the block.

    By default, ``datasets.load_dataset`` does revision-check HEAD
    requests (plus legacy-loader probes — ``SWE-bench_Verified.py``,
    ``dataset_infos.json``, ``.huggingface.yaml`` — all 404 by
    design) on every single call, even when the parquet is fully
    cached locally. In offline mode, ``load_dataset`` reads straight
    from the HF cache with zero network and zero httpx INFO chatter.

    The offline flags are read at *import* time from
    ``HF_DATASETS_OFFLINE`` / ``HF_HUB_OFFLINE`` env vars; once those
    libraries are imported (swebench imports ``datasets`` /
    ``huggingface_hub`` when its harness module loads), env-var flips
    have no effect. We mutate the in-memory module globals instead —
    restored on exit so we don't leak the offline flag.

    Offline mode only affects HF SDK calls — Docker registry pulls,
    GitHub fetches, etc. inside the swebench harness are unaffected.
    """
    import datasets.config as _ds_cfg
    import huggingface_hub.constants as _hf_const

    prev_ds_offline = _ds_cfg.HF_DATASETS_OFFLINE
    prev_hub_offline = _hf_const.HF_HUB_OFFLINE
    _ds_cfg.HF_DATASETS_OFFLINE = True
    _hf_const.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        _ds_cfg.HF_DATASETS_OFFLINE = prev_ds_offline
        _hf_const.HF_HUB_OFFLINE = prev_hub_offline


def _gold_patch_for(instance_id: str) -> str:
    """Fetch the dataset's gold patch for ``instance_id`` from the
    HuggingFace cache at ``~/.cache/huggingface/datasets/``.

    Tries offline mode first (zero network). On a true cache miss
    (first ever run on this host, or the operator nuked the HF
    cache) we transparently fall back to an online fetch, which
    populates the standard HF cache so subsequent runs are
    offline-quiet. The noisy 404-probe pattern on first run is
    unavoidable — it's a one-time price.
    """
    from datasets import load_dataset

    cache_hit = False
    ds = None
    with _hf_offline_mode():
        try:
            ds = load_dataset(
                "SWE-bench/SWE-bench_Verified", split="test",
            )
            cache_hit = True
        except (FileNotFoundError, ConnectionError, OSError):
            cache_hit = False
    if not cache_hit:
        ds = load_dataset(
            "SWE-bench/SWE-bench_Verified", split="test",
        )

    assert ds is not None  # narrows type after either branch
    rows = [r for r in ds if r["instance_id"] == instance_id]
    if not rows:
        raise RuntimeError(
            f"instance {instance_id!r} not found in "
            f"SWE-bench/SWE-bench_Verified",
        )
    return str(rows[0]["patch"])


def _make_predictions_file(
    tmp_dir: Path, instance_ids: list[str],
) -> Path:
    """SWE-bench predictions.jsonl using the dataset's gold patches.

    Empty patches get filtered out by the harness before any
    container work happens, so they don't actually exercise the
    docker-py drop-in. The gold patch (the upstream-recorded
    correct fix) makes the harness pull the instance image, create
    a container, apply the patch, run tests — all the docker-py
    surface we need to validate. ``resolved=True`` for every
    instance is the green-light signal.
    """
    pred_path = tmp_dir / "predictions.jsonl"
    lines: list[str] = []
    n = len(instance_ids)
    for i, instance_id in enumerate(instance_ids, start=1):
        # Progress on stderr so operators see forward motion during
        # the gold-patch fetch loop. Cache-hit reads are fast (<1s
        # each) but a fresh ``datasets`` cache miss can take ~30s
        # while the parquet downloads, and a long ``--instance-ids``
        # list amplifies that into apparent silence. The HF offline
        # flip in ``_gold_patch_for`` keeps this fast on warm caches.
        print(
            f"[xrlenv-smoke] [{i}/{n}] resolving gold patch for "
            f"{instance_id}",
            file=sys.stderr, flush=True,
        )
        gold = _gold_patch_for(instance_id)
        lines.append(json.dumps({
            "instance_id": instance_id,
            "model_patch": gold,
            "model_name_or_path": "xrlenv-drop-in-smoke-gold",
        }))
    pred_path.write_text("\n".join(lines) + "\n")
    return pred_path


def _run_smoke(
    *, instance_ids: list[str], timeout_s: int, keep_images: bool,
    save_report_to: Path | None = None,
    save_artifacts: Path | None = None,
    job_id: str | None = None,
    max_workers: int = 1,
) -> dict:
    """Run the smoke and return the parsed swebench report dict.

    The harness writes its top-level summary to ``cwd`` with a
    ``<model_name>.<run_id>.json`` filename; ``report_dir`` doesn't
    cover this top-level file. We chdir into a tempdir so the leak
    lands somewhere we can clean up, parse the summary into a dict,
    and (if ``save_report_to`` is given) copy it to a persistent
    path before the tempdir is reaped.
    """
    _install_drop_in()
    # Now safe to import swebench — its module-level docker.from_env
    # references will resolve at call time to the patched version.
    from swebench.harness.run_evaluation import main as run_eval

    with tempfile.TemporaryDirectory(prefix="xrlenv-swebench-smoke-") as td:
        tmp = Path(td)
        preds_path = _make_predictions_file(tmp, instance_ids)
        report_dir = tmp / "reports"
        report_dir.mkdir()

        # ``cwd`` matters: swebench's ``run_evaluation`` writes its
        # top-level summary file to ``cwd`` with a run_id-derived
        # filename, ignoring the ``report_dir`` arg. Run in tmp so
        # the artifact lands in tmp, not in the project root.
        prev_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp)
            # Wrap in offline mode so swebench's *own* internal
            # ``load_dataset`` call (it filters the test split by
            # instance_id before scheduling containers) reads from
            # the HF cache without revision-check HEAD requests +
            # legacy-loader 404 probes. Same trick as
            # ``_gold_patch_for``; only affects HF SDK calls so
            # Docker registry pulls are unaffected.
            with _hf_offline_mode():
                run_eval(
                    # Canonical home is ``SWE-bench/SWE-bench_Verified``;
                    # ``SWE-bench/SWE-bench_Verified`` is a legacy mirror
                    # that also lives on HF. Pinning the canonical name here
                    # so swebench's harness and our ``_gold_patch_for`` share
                    # one cache dir under
                    # ``~/.cache/huggingface/datasets/SWE-bench___swe-bench_verified/``
                    # — otherwise the host accumulates two copies of the
                    # same dataset and ``rm``-ing one keeps reappearing as
                    # the other code path repopulates it.
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    split="test",
                    instance_ids=list(instance_ids),
                    predictions_path=str(preds_path),
                    max_workers=max_workers,
                    force_rebuild=False,
                    # ``env`` cleans the instance image post-eval to
                    # avoid disk creep across smoke runs; ``instance``
                    # keeps it for fast subsequent iteration.
                    cache_level="env" if not keep_images else "instance",
                    clean=False,
                    open_file_limit=4096,
                    run_id="xrlenv-drop-in-smoke",
                    timeout=timeout_s,
                    # pull-and-retag from Docker Hub instead of building
                    # from source — faster + matches our existing
                    # swebench-verified plug-in's pull-and-retag path.
                    namespace="swebench",
                    rewrite_reports=False,
                    modal=False,
                    report_dir=str(report_dir),
                )
        finally:
            import os
            os.chdir(prev_cwd)

        # The summary file is named
        # ``<model_name>.<run_id>.json`` and lives in the eval cwd.
        candidates = list(tmp.glob("*xrlenv-drop-in-smoke*.json"))
        if not candidates:
            raise RuntimeError(
                f"no swebench summary file under {tmp}",
            )
        summary_path = candidates[0]
        if save_report_to is not None:
            save_report_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(summary_path, save_report_to)
        summary = json.loads(summary_path.read_text())

        # Optional: archive the harness's native per-instance
        # artifacts (logs/run_evaluation/...) plus the summary to a
        # persistent location for trajectory reference. Done before
        # the tempdir reaper runs.
        if save_artifacts is not None:
            from tests.smoke._artifacts import (
                archive_artifacts,
                default_job_id,
            )
            archive_artifacts(
                src_dir=tmp,
                save_root=save_artifacts,
                job_id=job_id or default_job_id(),
                summary=summary,
                # SWE-bench's run_evaluation writes per-instance
                # logs under ``logs/run_evaluation/<run_id>/<model>/
                # <instance>/`` in cwd — that tree carries everything
                # operator-relevant (report.json, run_instance.log,
                # test_output.txt, eval.sh, patch.diff). The
                # ``reports/`` arg we pass to ``run_eval`` below is
                # not populated by upstream — don't archive it.
                subtrees=["logs"],
            )
        return summary


# ──────────────────────────────────────────────────────────────────────────────
# pytest entry point
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _docker_reachable(), reason="docker daemon not reachable")
@pytest.mark.skipif(not _datasets_available(), reason="`datasets` not installed")
@pytest.mark.skipif(not _swebench_available(), reason="`swebench` not installed")
def test_swebench_verified_drop_in_resolves_one_instance() -> None:
    """SWE-bench's stock ``run_evaluation.main()`` resolves one
    Verified instance through ``xrlenv.from_env()`` end-to-end."""
    summary = _run_smoke(
        instance_ids=[_DEFAULT_INSTANCE_ID],
        timeout_s=_DEFAULT_TIMEOUT_S,
        keep_images=False,
    )

    assert summary["total_instances"] == 1
    assert summary["completed_instances"] == 1, summary
    assert summary["error_instances"] == 0, summary
    assert summary["empty_patch_instances"] == 0, summary
    assert _DEFAULT_INSTANCE_ID in summary["resolved_ids"], summary
    assert summary["resolved_instances"] == 1, summary


# ──────────────────────────────────────────────────────────────────────────────
# Standalone-script entry point — same flow, parses argv for ad-hoc use
# ──────────────────────────────────────────────────────────────────────────────


def _main_script() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--instance-id", default=None,
        help="Single Verified instance id to smoke. Default: "
             "sphinx-doc__sphinx-10323 (small + <15 min fix).",
    )
    group.add_argument(
        "--instance-ids", default=None,
        help="Comma-separated list of Verified instance ids. Pass "
             "the whole SMOKE_8 reference list when you want the "
             "broader infrastructure soak.",
    )
    parser.add_argument(
        "--keep-images", action="store_true",
        help="Keep the pulled instance image after the run "
             "(default: drops images above ``cache_level=env``).",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=_DEFAULT_TIMEOUT_S,
        help="Per-instance test timeout (default 900s).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Concurrent instances per run (default 1; raise when "
             "there's spare CPU/disk on the host).",
    )
    parser.add_argument(
        "--save-report", type=Path, default=None,
        help="Copy the swebench summary JSON to this path. By default "
             "the report lives briefly inside a tempdir that gets "
             "reaped at exit, so the contents print inline either way "
             "but the JSON file is gone.",
    )
    from tests.smoke._artifacts import default_save_artifacts_root
    parser.add_argument(
        "--save-artifacts", nargs="?", type=Path,
        # ``default=None`` (omitted) → archiving OFF.
        # ``--save-artifacts`` (no value) → archive to ``const`` (the
        # default path).
        # ``--save-artifacts /custom/path`` → archive to /custom/path.
        default=None, const=default_save_artifacts_root(),
        help="Persist the harness's per-instance artifacts (logs, "
             "test outputs) under <PATH>/<job-id>/ for trajectory "
             "reference. Layout mirrors swebench's native tree: "
             "logs/run_evaluation/<run_id>/<model>/<instance>/. "
             "Pass ``--save-artifacts`` (no value) to archive under "
             f"``{default_save_artifacts_root()}`` (gitignored), or "
             "``--save-artifacts /your/path`` to override (e.g. "
             "``~/.../monet_code_eval/jobs``). Omit the flag "
             "entirely to skip archiving.",
    )
    parser.add_argument(
        "--job-id", default=None,
        help="Subdirectory under --save-artifacts to group this run's "
             "artifacts (e.g. ``claude-opus-4-7-50-v1.12.0``). "
             "Defaults to a UTC timestamp.",
    )
    args = parser.parse_args()

    if args.instance_ids is not None:
        instance_ids = [s.strip() for s in args.instance_ids.split(",") if s.strip()]
    elif args.instance_id is not None:
        instance_ids = [args.instance_id]
    else:
        instance_ids = [_DEFAULT_INSTANCE_ID]

    # Resolve job_id once so the print below + the archive layout
    # see the same timestamp.
    if args.save_artifacts is not None:
        from tests.smoke._artifacts import default_job_id
        resolved_job_id = args.job_id or default_job_id()
    else:
        resolved_job_id = args.job_id

    print(f"[xrlenv-smoke] instances={instance_ids}")
    try:
        summary = _run_smoke(
            instance_ids=instance_ids,
            timeout_s=args.timeout_s,
            keep_images=args.keep_images,
            save_report_to=args.save_report,
            save_artifacts=args.save_artifacts,
            job_id=resolved_job_id,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        print(
            f"[xrlenv-smoke] FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        return 1

    # Print the report inline so the green-light signal is visible
    # without hunting for files. The report lives in a tempdir that
    # gets reaped at exit unless the operator passed --save-report.
    print()
    print("[xrlenv-smoke] swebench summary:")
    print(json.dumps(summary, indent=2))
    if args.save_report is not None:
        print(f"[xrlenv-smoke] report saved to {args.save_report.resolve()}")
    if args.save_artifacts is not None:
        archive_dest = (args.save_artifacts / resolved_job_id).resolve()
        print(f"[xrlenv-smoke] artifacts saved under {archive_dest}/")

    resolved_ids = set(summary.get("resolved_ids") or [])
    expected = set(instance_ids)
    missing = expected - resolved_ids
    if not missing:
        print(
            f"\n[xrlenv-smoke] SUCCESS: {len(expected)}/{len(expected)} "
            f"resolved=True in "
            f"{summary.get('completed_instances', 0)} completion(s)",
        )
        return 0
    print(
        f"\n[xrlenv-smoke] FAIL: {len(missing)}/{len(expected)} "
        f"not resolved: {sorted(missing)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(_main_script())
