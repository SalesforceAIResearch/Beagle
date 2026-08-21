"""Tarball-source build dispatch smoke (sub-slice 1.b).

Three tests:

1. ``test_tarball_cap_rejects_oversized`` — operator-side cap fires
   BEFORE any wire traffic. Local-only; never touches the cluster.
   Pure ``resolve_tarball_sources`` invocation with an oversized
   tarball + a tight cap. Lowest-cost regression check that the
   `--build-tarball-max-bytes` knob is wired end-to-end.
2. ``test_tarball_happy_path`` — build a tiny FROM-busybox image
   from a tarball, assert the image exists locally and carries the
   reserved labels (``xrlenv.image.rebuild-cost=local-build-cheap``
   + ``xrlenv.cancel-key=<image_ref>``). Local-only because remote
   mode can't inspect docker labels via the admin API today.
3. ``test_tarball_source_registry_persists`` — after a successful
   tarball build, the per-image_ref source-spec registry has
   ``spec.json`` + ``content.bin`` on disk under the builder's
   cache root. Pins the build-on-acquire prerequisite (sub-slice
   2) so eviction recovery works without operator re-apply.

Excluded from default pytest; run with::

    .venv/bin/python -m pytest \\
        tests/smoke/build_plan/test_tarball_dispatch.py -v -s

Standalone script::

    .venv/bin/python tests/smoke/build_plan/test_tarball_dispatch.py
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import tarfile
from pathlib import Path

import pytest

from tests.smoke._build_plan_dispatch_helpers import (
    apply_plan,
    docker_available,
    image_present_locally,
    smoke_artifact_dir,
    write_summary,
)

# Every test in this file is local-only:
#
# - the cap check fires entirely client-side in
#   ``resolve_tarball_sources`` (no cluster reachability needed);
# - the happy-path build inspects docker labels via ``docker image
#   inspect`` on this host;
# - the source-spec registry assertion reads the local cache root.
#
# Skipping the parametrize avoids creating empty artifact dirs +
# guaranteed-SKIPPED pytest entries when the operator has
# ``XRLENV_GRPC_HOST`` exported.


_OUT_DIR: Path | None = None


@pytest.fixture
def out_dir() -> Path:
    if not docker_available():
        pytest.skip("docker daemon not reachable")
    global _OUT_DIR
    if _OUT_DIR is None:
        _OUT_DIR = smoke_artifact_dir("tarball-local")
    return _OUT_DIR


@pytest.fixture
def isolated_state(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "state.db", tmp_path / "runs"


def _make_tarball(payload: dict[str, bytes]) -> bytes:
    """Build an in-memory tar with ``{name: bytes}`` entries."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in payload.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — cap rejects oversized at apply time, before any wire traffic
# ──────────────────────────────────────────────────────────────────────────────


def test_tarball_cap_rejects_oversized(
    tmp_path: Path, out_dir: Path,
) -> None:
    """Operator-side cap fires inside ``resolve_tarball_sources``.
    No cluster needed — pure function exercise that confirms the
    ``--build-tarball-max-bytes`` plumbing is intact."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        BuildPlan,
        EntryPlacement,
        TarballSource,
        resolve_tarball_sources,
    )
    from xrlenv.errors import ManifestInvalid

    tarball_path = tmp_path / "oversized.tar"
    tarball_path.write_bytes(b"X" * (2 * 1024 * 1024))   # 2 MB

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="smoke/oversized:1",
            context_source=TarballSource(
                path=str(tarball_path), dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    payload: dict[str, str] = {}
    with pytest.raises(ManifestInvalid) as excinfo:
        resolve_tarball_sources(plan, max_bytes=1 * 1024 * 1024)   # 1 MB cap
    msg = str(excinfo.value)
    payload["message"] = msg
    write_summary(out_dir, "test_tarball_cap_rejects.json", payload)

    assert "smoke/oversized:1" in msg
    assert "over the" in msg
    # Name the right flag so the operator can act.
    assert "--build-tarball-max-bytes" in msg


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — happy path: tarball build produces image + labels
# ──────────────────────────────────────────────────────────────────────────────


def test_tarball_happy_path(
    isolated_state: tuple[Path, Path], out_dir: Path,
) -> None:
    """Drive a tiny tarball build through the dispatch pipeline and
    inspect the resulting image. Local-only because a node-side
    ``docker image inspect`` wire-call doesn't ship today; the
    file-level docstring covers the rationale."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        BuildPlan,
        EntryPlacement,
        TarballSource,
        resolve_tarball_sources,
    )
    from xrlenv.node.source_builder import CANCEL_KEY_LABEL, REBUILD_COST_LABEL

    image_ref = "xrlenv-smoke/tarball-hello:1"
    tar_bytes = _make_tarball({
        "Dockerfile": (
            b"FROM busybox:1.36\n"
            b"COPY hello.txt /hello.txt\n"
            b"CMD [\"cat\", \"/hello.txt\"]\n"
        ),
        "hello.txt": b"tarball dispatch smoke\n",
    })

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref=image_ref,
            context_source=TarballSource(
                path="<in-memory>",   # path is metadata-only when content_b64 set
                dockerfile="Dockerfile",
                content_b64=base64.b64encode(tar_bytes).decode("ascii"),
            ),
            placement=EntryPlacement(
                size_hint_bytes=10 * 1024 * 1024,
                preferred_home_count=1,
            ),
        ),
    ))
    plan = resolve_tarball_sources(plan, max_bytes=20 * 1024 * 1024)

    state_db, runs_root = isolated_state
    state_db.parent.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    result = apply_plan(
        plan, mode="local", state_db=state_db, runs_root=runs_root,
        dry_run=False, eager=True,
    )
    payload = {
        "image_ref": image_ref,
        "status": result.status,
        "successes": result.successes,
        "failures": result.failures,
    }
    write_summary(out_dir, "test_tarball_happy_path.json", payload)

    assert result.status == "completed", (
        f"apply did not complete; status={result.status!r}, "
        f"failures={result.failures}"
    )
    assert result.successes == 1
    assert result.failures == 0
    assert image_present_locally(image_ref), (
        f"docker images -q {image_ref!r} returned nothing post-apply"
    )

    # Label assertions: rebuild-cost = local-build-cheap (tarball
    # tier) AND cancel-key = image_ref (operator-cancel hook).
    import subprocess
    r = subprocess.run(
        ["docker", "image", "inspect", image_ref,
         "-f", "{{json .Config.Labels}}"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    import json
    labels = json.loads(r.stdout) or {}
    assert labels.get(REBUILD_COST_LABEL) == "local-build-cheap", (
        f"expected rebuild-cost label local-build-cheap; "
        f"got {labels.get(REBUILD_COST_LABEL)!r} in {labels!r}"
    )
    assert labels.get(CANCEL_KEY_LABEL) == image_ref, (
        f"expected cancel-key={image_ref!r}; "
        f"got {labels.get(CANCEL_KEY_LABEL)!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — source-spec registry persists for build-on-acquire
# ──────────────────────────────────────────────────────────────────────────────


def test_tarball_source_registry_persists(
    isolated_state: tuple[Path, Path], out_dir: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a successful tarball build, the per-image_ref
    source-spec registry has ``spec.json`` + ``content.bin`` on
    disk. This is the load-bearing precondition for build-on-
    acquire (sub-slice 2): without persistence, an evicted image
    forces operator re-apply instead of automatic rebuild.

    Local-only — the registry lives on the node-side filesystem;
    the file-level docstring covers the rationale. Uses a per-test
    ``XRLENV_BUILD_CONTEXT_CACHE`` so we don't poison the
    operator's real registry."""
    # Isolate the cache root so the registry created by this build
    # doesn't leak into the operator's persistent home directory.
    cache_root = tmp_path / "build-context-cache"
    cache_root.mkdir()
    monkeypatch.setenv("XRLENV_BUILD_CONTEXT_CACHE", str(cache_root))

    from xrlenv.control.build_plan import (
        BuildEntry,
        BuildPlan,
        EntryPlacement,
        TarballSource,
        resolve_tarball_sources,
    )

    image_ref = "xrlenv-smoke/registry-persists:1"
    tar_bytes = _make_tarball({
        "Dockerfile": b"FROM busybox:1.36\nCMD [\"echo\", \"registered\"]\n",
    })
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref=image_ref,
            context_source=TarballSource(
                path="<in-memory>", dockerfile="Dockerfile",
                content_b64=base64.b64encode(tar_bytes).decode("ascii"),
            ),
            placement=EntryPlacement(
                size_hint_bytes=5 * 1024 * 1024,
                preferred_home_count=1,
            ),
        ),
    ))
    plan = resolve_tarball_sources(plan, max_bytes=20 * 1024 * 1024)

    state_db, runs_root = isolated_state
    state_db.parent.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    result = apply_plan(
        plan, mode="local", state_db=state_db, runs_root=runs_root,
        dry_run=False, eager=True,
    )
    assert result.status == "completed", (
        f"prerequisite build failed; status={result.status!r}, "
        f"failures={result.failures}"
    )

    # The source-registry root is ``<cache_root>.parent /
    # source-registry`` per ``GitSourceBuilder.__init__``. The
    # cache_root we set above is the build-context cache (sibling).
    registry_root = cache_root.parent / "source-registry"
    assert registry_root.is_dir(), (
        f"source-registry root not created at {registry_root}"
    )

    # One subdir per image_ref (sha256 hash truncated to 32 chars).
    entries = list(registry_root.iterdir())
    assert len(entries) == 1, (
        f"expected exactly one source-registry entry; got {entries}"
    )
    entry_dir = entries[0]
    spec_path = entry_dir / "spec.json"
    content_path = entry_dir / "content.bin"
    assert spec_path.is_file(), f"spec.json missing under {entry_dir}"
    assert content_path.is_file(), f"content.bin missing under {entry_dir}"

    import json
    spec = json.loads(spec_path.read_text())
    assert spec["type"] == "tarball"
    assert spec["image_ref"] == image_ref
    assert spec["dockerfile"] == "Dockerfile"
    # Bytes survive byte-exact.
    assert content_path.read_bytes() == tar_bytes

    write_summary(out_dir, "test_tarball_source_registry_persists.json", {
        "image_ref": image_ref,
        "registry_root": str(registry_root),
        "spec_keys": sorted(spec.keys()),
        "content_size_bytes": content_path.stat().st_size,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Standalone-script entry point
# ──────────────────────────────────────────────────────────────────────────────


def _main_script() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("local", "remote", "all"), default="local",
    )
    parser.add_argument("-k", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args, passthrough = parser.parse_known_args()

    pytest_args: list[str] = [__file__, "-s"]
    pytest_args.append("-vv" if args.verbose else "-v")
    if args.mode == "local":
        pytest_args += ["-k", "local or not ["]
    elif args.mode == "remote":
        pytest_args += ["-k", "remote or not ["]
    if args.k:
        if pytest_args[-2] == "-k":
            pytest_args[-1] = f"({pytest_args[-1]}) and ({args.k})"
        else:
            pytest_args += ["-k", args.k]
    pytest_args += passthrough
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(_main_script())
