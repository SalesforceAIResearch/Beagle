"""Build-on-acquire after eviction smoke (sub-slice 2, Plan 4).

Verifies the load-bearing property of the per-image_ref source-spec
registry: an evicted source-built image rebuilds automatically on
the next ``acquire_container``, with no operator re-apply needed.

This test is **operator-driven** — it cannot delete a remote node's
docker image on its own without coupling to your SSH setup. The
operator performs the eviction step manually, then runs this
smoke. The smoke issues a ``Client.acquire_container`` for the
target image and times how long it took:

- **Rebuild fired** (typical 10-60s for a Harbor-style image):
  the source-spec registry kicked in, build-on-acquire works.
- **Sub-second acquire**: the scheduler routed to a node that
  STILL had the image. Either the operator didn't delete it on
  the right node, OR a replicated entry has copies elsewhere.
  The test does NOT fail this case — the property under test
  (image-comes-back-from-source-spec) isn't actually exercised.
  Re-run with a wider eviction, or pick an image_ref whose
  ``preferred_home_count`` is 1.

See ``tests/smoke/build_plan/README.md`` § "Build-on-acquire
manual recipe" for the full step-by-step operator setup before
running this smoke.

Remote-only (the consumer SDK dials the live cluster); skips
upfront when ``XRLENV_GRPC_HOST`` isn't set.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from tests.smoke._build_plan_dispatch_helpers import (
    smoke_artifact_dir,
    write_summary,
)

# How long to give the acquire to complete. A real rebuild of a
# seta-env-style image takes 10-60s; the worst-case is a registry
# pull of a base layer + docker build, ~3-5min. Be generous.
ACQUIRE_TIMEOUT_S = 5 * 60.0

# Below this, the acquire was almost certainly a docker cache hit
# rather than a real rebuild. We don't fail the test on this
# condition — the property under test isn't exercised — but we
# log it loudly in the JSON artifact so the operator notices.
REBUILD_LIKELY_THRESHOLD_S = 2.0


_OUT_DIR: Path | None = None


@pytest.fixture
def out_dir() -> Path:
    if not (os.environ.get("XRLENV_ADMIN_HOST")
            or os.environ.get("XRLENV_GRPC_HOST")):
        pytest.skip(
            "build-on-acquire smoke is remote-only; set "
            "XRLENV_GRPC_HOST to point at the consumer-facing "
            "control plane",
        )
    global _OUT_DIR
    if _OUT_DIR is None:
        _OUT_DIR = smoke_artifact_dir("build-on-acquire-remote")
    return _OUT_DIR


def _resolve_target_image() -> str:
    target = os.environ.get("SMOKE_TARGET_IMAGE")
    if not target:
        pytest.fail(
            "SMOKE_TARGET_IMAGE not set. Set it to a source-built "
            "image_ref the cluster has built (e.g. "
            "xrlenv-seta-env/0:main) and that you've evicted on "
            "its preferred-home node via ``docker rmi``. See the "
            "operator setup steps in "
            "tests/smoke/build_plan/README.md § Build-on-acquire "
            "manual recipe.",
        )
    return target


def _read_token_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def test_acquire_after_eviction_rebuilds_image(out_dir: Path) -> None:
    """Drive an ``acquire_container`` for the operator-evicted
    image. The cluster's scheduler routes to a node, ensure_present
    on that node fires the source-spec registry's lookup_producer,
    and the image rebuilds.

    Asserts the acquire succeeded. Reports the timing so the
    operator can confirm the precondition (image actually evicted
    on the chosen node) was met.
    """
    from xrlenv.client import Client

    target = _resolve_target_image()
    host = os.environ["XRLENV_GRPC_HOST"]
    port = int(os.environ.get("XRLENV_GRPC_PORT", "50051"))
    token = (
        os.environ.get("XRLENV_CONSUMER_TOKEN")
        or _read_token_file(Path.home() / ".xrlenv" / "secrets" / "consumer.token")
    )

    summary: dict[str, object] = {
        "target_image": target,
        "host": host,
        "port": port,
        "have_token": bool(token),
        "acquire_timeout_s": ACQUIRE_TIMEOUT_S,
    }

    async def _drive() -> tuple[float, str]:
        client = Client.grpc(host=host, port=port, token=token)
        t0 = time.monotonic()
        session = await asyncio.wait_for(
            client.acquire_container(
                image=target,
                command=["sleep", "5"],
            ),
            timeout=ACQUIRE_TIMEOUT_S,
        )
        elapsed = time.monotonic() - t0
        container_id = session.container_id
        # Destroy immediately — we just wanted to prove the
        # acquire works.
        try:
            await session.destroy()
        except Exception as exc:
            # Destroy errors are non-fatal for this test's
            # intent (we proved the rebuild + acquire path);
            # report but don't fail.
            summary["destroy_error"] = f"{type(exc).__name__}: {exc}"
        return elapsed, container_id

    try:
        elapsed, container_id = asyncio.run(_drive())
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        write_summary(out_dir, "test_acquire_after_eviction.json", summary)
        raise

    summary["status"] = "ok"
    summary["acquire_duration_s"] = round(elapsed, 3)
    summary["container_id"] = container_id
    summary["rebuild_likely_fired"] = elapsed >= REBUILD_LIKELY_THRESHOLD_S
    if elapsed < REBUILD_LIKELY_THRESHOLD_S:
        summary["interpretation"] = (
            f"acquire completed in {elapsed:.3f}s — under the "
            f"{REBUILD_LIKELY_THRESHOLD_S}s threshold for a real "
            "rebuild. Almost certainly a docker cache hit. "
            "Did the eviction precondition fail? Check that you "
            "deleted the image on the SAME node the scheduler "
            "would pick. See README § Build-on-acquire manual recipe."
        )
    else:
        summary["interpretation"] = (
            f"acquire took {elapsed:.1f}s — consistent with a real "
            "source-spec-registry rebuild (typical 10-60s for a "
            "Harbor-style task image). Build-on-acquire works."
        )
    write_summary(out_dir, "test_acquire_after_eviction.json", summary)

    # The test asserts the acquire SUCCEEDED. The timing-based
    # "did the rebuild actually fire?" interpretation is in the
    # JSON artifact + emitted to stdout for operator inspection.
    print(f"\n[build-on-acquire] {summary['interpretation']}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone-script entry point
# ──────────────────────────────────────────────────────────────────────────────


def _main_script() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-image", default=None,
        help="Override $SMOKE_TARGET_IMAGE for this run.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args, passthrough = parser.parse_known_args()

    if args.target_image:
        os.environ["SMOKE_TARGET_IMAGE"] = args.target_image

    pytest_args = [__file__, "-s"]
    pytest_args.append("-vv" if args.verbose else "-v")
    pytest_args += passthrough
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(_main_script())
