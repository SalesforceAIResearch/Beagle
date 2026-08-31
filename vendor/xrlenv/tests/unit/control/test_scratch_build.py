"""Unit tests for :mod:`xrlenv.control.scratch_build` — turning an
``image_build`` spec into a (scratch_ref, build_source) pair (slice 2c-ii)."""

from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

import pytest
from xrlenv.control.build_plan import GitSource, TarballSource
from xrlenv.control.scratch_build import resolve_scratch_image
from xrlenv.errors import XRLEnvError
from xrlenv.image_build import (
    GitContext,
    ImageBuildSpec,
    compute_context_digest_for_dir,
    compute_input_digest,
    git_context_digest,
    scratch_ref,
)

HOST = "cp-box:5012"


def _ctx(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "ctx"
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


def test_resolve_git_spec_returns_gitsource_and_expected_ref() -> None:
    git = GitContext(repo="https://x/y", ref="abc123", subdir="env")
    spec = ImageBuildSpec(git=git, build_args={"A": "1"})
    ref, source = resolve_scratch_image(spec, scratch_host=HOST)
    assert isinstance(source, GitSource)
    assert (source.repo, source.ref, source.subdir) == ("https://x/y", "abc123", "env")
    expected = scratch_ref(
        HOST, compute_input_digest(spec, context_digest=git_context_digest(git)),
    )
    assert ref == expected


def test_resolve_context_spec_returns_tarball_that_untars_to_context(tmp_path: Path) -> None:
    root = _ctx(tmp_path, {"Dockerfile": "FROM busybox\n", "app/main.py": "print(1)\n"})
    spec = ImageBuildSpec(context=str(root))
    ref, source = resolve_scratch_image(spec, scratch_host=HOST)
    assert isinstance(source, TarballSource)
    assert source.dockerfile == "Dockerfile"
    assert source.content_b64 is not None
    # ref matches a manual content-addressed computation
    expected = scratch_ref(
        HOST, compute_input_digest(spec, context_digest=compute_context_digest_for_dir(root)),
    )
    assert ref == expected
    # the tarball untars to the context contents, Dockerfile at root
    raw = base64.b64decode(source.content_b64)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        names = {m.name.lstrip("./") for m in tf.getmembers() if m.isfile()}
    assert "Dockerfile" in names
    assert "app/main.py" in names


def test_resolve_is_deterministic(tmp_path: Path) -> None:
    root = _ctx(tmp_path, {"Dockerfile": "FROM busybox\n"})
    spec = ImageBuildSpec(context=str(root))
    ref1, _ = resolve_scratch_image(spec, scratch_host=HOST)
    ref2, _ = resolve_scratch_image(spec, scratch_host=HOST)
    assert ref1 == ref2


def test_resolve_build_args_change_the_ref(tmp_path: Path) -> None:
    root = _ctx(tmp_path, {"Dockerfile": "FROM busybox\n"})
    ref_a, _ = resolve_scratch_image(
        ImageBuildSpec(context=str(root), build_args={"V": "1"}), scratch_host=HOST,
    )
    ref_b, _ = resolve_scratch_image(
        ImageBuildSpec(context=str(root), build_args={"V": "2"}), scratch_host=HOST,
    )
    assert ref_a != ref_b


def test_resolve_dockerfile_selection_changes_ref(tmp_path: Path) -> None:
    root = _ctx(tmp_path, {"Dockerfile": "FROM a\n", "Dockerfile.gpu": "FROM b\n"})
    ref_a, _ = resolve_scratch_image(
        ImageBuildSpec(context=str(root), dockerfile="Dockerfile"), scratch_host=HOST,
    )
    ref_b, _ = resolve_scratch_image(
        ImageBuildSpec(context=str(root), dockerfile="Dockerfile.gpu"), scratch_host=HOST,
    )
    assert ref_a != ref_b


def test_resolve_empty_host_raises(tmp_path: Path) -> None:
    root = _ctx(tmp_path, {"Dockerfile": "FROM busybox\n"})
    with pytest.raises(XRLEnvError, match="XRLENV_SCRATCH_REGISTRY_HOST"):
        resolve_scratch_image(ImageBuildSpec(context=str(root)), scratch_host="")


def test_resolve_missing_context_dir_raises(tmp_path: Path) -> None:
    spec = ImageBuildSpec(context=str(tmp_path / "does-not-exist"))
    with pytest.raises(XRLEnvError, match="not a directory"):
        resolve_scratch_image(spec, scratch_host=HOST)


def test_resolve_git_never_touches_disk(tmp_path: Path) -> None:
    # git spec resolves purely from the pinning tuple — no context dir needed.
    spec = ImageBuildSpec(git=GitContext(repo="https://x/y", ref="main"))
    ref, source = resolve_scratch_image(spec, scratch_host=HOST)
    assert ref.startswith(f"{HOST}/scratch/")
    assert isinstance(source, GitSource)


def test_scratch_registry_host_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from xrlenv.control.scratch_build import scratch_registry_host_from_env
    monkeypatch.delenv("XRLENV_SCRATCH_REGISTRY_HOST", raising=False)
    monkeypatch.delenv("XRLENV_SCRATCH_REGISTRY_PORT", raising=False)
    assert scratch_registry_host_from_env() is None
    monkeypatch.setenv("XRLENV_SCRATCH_REGISTRY_HOST", "cp-box")
    assert scratch_registry_host_from_env() == "cp-box:5012"  # default port
    monkeypatch.setenv("XRLENV_SCRATCH_REGISTRY_PORT", "6000")
    assert scratch_registry_host_from_env() == "cp-box:6000"


def test_durable_ref_for_none_when_no_durable() -> None:
    from xrlenv.control.scratch_build import durable_ref_for
    spec = ImageBuildSpec(context="/x")
    assert durable_ref_for(spec, f"{HOST}/scratch/deadbeef") is None


def test_durable_ref_for_appends_input_digest_when_untagged() -> None:
    from xrlenv.control.scratch_build import durable_ref_for
    spec = ImageBuildSpec(context="/x", durable_to="reg.internal:5000/team/env")
    ref = durable_ref_for(spec, f"{HOST}/scratch/deadbeef")
    assert ref == "reg.internal:5000/team/env:deadbeef"


def test_durable_ref_for_verbatim_when_tagged() -> None:
    from xrlenv.control.scratch_build import durable_ref_for
    spec = ImageBuildSpec(context="/x", durable_to="reg.internal:5000/team/env:v9")
    ref = durable_ref_for(spec, f"{HOST}/scratch/deadbeef")
    assert ref == "reg.internal:5000/team/env:v9"


def test_durable_ref_for_bare_untagged_appends_digest() -> None:
    """durable_to with no slash and no tag (bare repo name) → appends input_digest.
    rsplit("/", 1) on a no-slash string gives a single-element list; the tail
    is the whole string, no colon → append."""
    from xrlenv.control.scratch_build import durable_ref_for
    spec = ImageBuildSpec(context="/x", durable_to="myimage")
    ref = durable_ref_for(spec, f"{HOST}/scratch/deadbeef")
    assert ref == "myimage:deadbeef"


def test_durable_ref_for_bare_tagged_is_verbatim() -> None:
    """durable_to with no slash but with a tag (e.g. env:v1) → verbatim."""
    from xrlenv.control.scratch_build import durable_ref_for
    spec = ImageBuildSpec(context="/x", durable_to="myimage:v1")
    ref = durable_ref_for(spec, f"{HOST}/scratch/deadbeef")
    assert ref == "myimage:v1"
