"""Unit tests for :mod:`xrlenv.image_build` — the bring-your-own-Dockerfile
spec + content-addressing foundation (slice 1 of the scratch-registry design,
``notes/scratch-registry-build-on-demand.md``)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from xrlenv.image_build import (
    GitContext,
    ImageBuildSpec,
    compute_context_digest_for_dir,
    compute_input_digest,
    git_context_digest,
    scratch_ref,
)

# --------------------------------------------------------------------------
# ImageBuildSpec validation
# --------------------------------------------------------------------------


def test_context_only_spec_is_valid() -> None:
    spec = ImageBuildSpec(context="./environment")
    assert spec.context == "./environment"
    assert spec.git is None
    assert spec.effective_dockerfile() == "Dockerfile"


def test_git_only_spec_is_valid() -> None:
    spec = ImageBuildSpec(
        git=GitContext(repo="https://x/y", ref="abc123", subdir="env", dockerfile="Dockerfile.gpu"),
    )
    assert spec.context is None
    assert spec.effective_dockerfile() == "Dockerfile.gpu"


def test_context_and_git_both_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one of 'context'"):
        ImageBuildSpec(context="./environment", git=GitContext(repo="https://x/y"))


def test_neither_context_nor_git_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one of 'context'"):
        ImageBuildSpec()


def test_empty_context_string_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ImageBuildSpec(context="")


def test_unknown_field_is_rejected() -> None:
    # Via model_validate(dict) so the deliberately-invalid extra field is a
    # runtime concern (pydantic extra="forbid"), not a static-typing error.
    with pytest.raises(ValidationError):
        ImageBuildSpec.model_validate({"context": "./x", "bogus": True})


def test_git_context_requires_repo() -> None:
    with pytest.raises(ValidationError, match="repo must be non-empty"):
        GitContext(repo="")


def test_effective_dockerfile_context_uses_top_level() -> None:
    spec = ImageBuildSpec(context="./x", dockerfile="Dockerfile.custom")
    assert spec.effective_dockerfile() == "Dockerfile.custom"


# --------------------------------------------------------------------------
# compute_context_digest_for_dir — determinism + sensitivity
# --------------------------------------------------------------------------


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_context_digest_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    for root in (a, b):
        _write(root, "Dockerfile", "FROM python:3.11\n")
        _write(root, "src/app.py", "print('hi')\n")
    assert compute_context_digest_for_dir(a) == compute_context_digest_for_dir(b)


def test_context_digest_changes_when_a_file_changes(tmp_path: Path) -> None:
    a = tmp_path / "a"
    _write(a, "Dockerfile", "FROM python:3.11\n")
    before = compute_context_digest_for_dir(a)
    _write(a, "Dockerfile", "FROM python:3.12\n")
    assert compute_context_digest_for_dir(a) != before


def test_context_digest_changes_when_a_file_is_added(tmp_path: Path) -> None:
    a = tmp_path / "a"
    _write(a, "Dockerfile", "FROM python:3.11\n")
    before = compute_context_digest_for_dir(a)
    _write(a, "extra.txt", "data\n")
    assert compute_context_digest_for_dir(a) != before


def test_context_digest_is_independent_of_absolute_location(tmp_path: Path) -> None:
    a = tmp_path / "here" / "ctx"
    b = tmp_path / "somewhere" / "else" / "ctx"
    for root in (a, b):
        _write(root, "Dockerfile", "FROM busybox\n")
    assert compute_context_digest_for_dir(a) == compute_context_digest_for_dir(b)


def test_context_digest_sensitive_to_exec_bit(tmp_path: Path) -> None:
    a = tmp_path / "a"
    _write(a, "run.sh", "#!/bin/sh\necho hi\n")
    before = compute_context_digest_for_dir(a)
    (a / "run.sh").chmod(0o755)
    assert compute_context_digest_for_dir(a) != before


def test_context_digest_on_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        compute_context_digest_for_dir(tmp_path / "nope")


# --------------------------------------------------------------------------
# git_context_digest — sensitivity to each pinning field
# --------------------------------------------------------------------------


def test_git_context_digest_deterministic() -> None:
    g1 = GitContext(repo="https://x/y", ref="abc", subdir="env")
    g2 = GitContext(repo="https://x/y", ref="abc", subdir="env")
    assert git_context_digest(g1) == git_context_digest(g2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repo": "https://x/other"},
        {"ref": "def"},
        {"subdir": "other"},
        {"dockerfile": "Dockerfile.gpu"},
    ],
)
def test_git_context_digest_changes_per_field(kwargs: dict[str, str]) -> None:
    base = GitContext(repo="https://x/y", ref="abc", subdir="env", dockerfile="Dockerfile")
    other = base.model_copy(update=kwargs)
    assert git_context_digest(base) != git_context_digest(other)


# --------------------------------------------------------------------------
# compute_input_digest — the content-addressed tag body
# --------------------------------------------------------------------------


def test_input_digest_is_deterministic() -> None:
    spec = ImageBuildSpec(context="./x", build_args={"A": "1", "B": "2"})
    d1 = compute_input_digest(spec, context_digest="ctx", base_image_digest="sha256:base")
    d2 = compute_input_digest(spec, context_digest="ctx", base_image_digest="sha256:base")
    assert d1 == d2


def test_input_digest_changes_with_context_digest() -> None:
    spec = ImageBuildSpec(context="./x")
    a = compute_input_digest(spec, context_digest="ctxA")
    b = compute_input_digest(spec, context_digest="ctxB")
    assert a != b


def test_input_digest_changes_with_base_image_digest() -> None:
    spec = ImageBuildSpec(context="./x")
    a = compute_input_digest(spec, context_digest="ctx", base_image_digest="sha256:aaa")
    b = compute_input_digest(spec, context_digest="ctx", base_image_digest="sha256:bbb")
    assert a != b


def test_input_digest_changes_with_build_args() -> None:
    ctx = "ctx"
    a = compute_input_digest(ImageBuildSpec(context="./x", build_args={"A": "1"}), context_digest=ctx)
    b = compute_input_digest(ImageBuildSpec(context="./x", build_args={"A": "2"}), context_digest=ctx)
    assert a != b


def test_input_digest_changes_with_dockerfile_selection() -> None:
    ctx = "ctx"
    a = compute_input_digest(ImageBuildSpec(context="./x", dockerfile="Dockerfile"), context_digest=ctx)
    b = compute_input_digest(ImageBuildSpec(context="./x", dockerfile="Dockerfile.gpu"), context_digest=ctx)
    assert a != b


def test_input_digest_build_args_order_insensitive() -> None:
    ctx = "ctx"
    a = compute_input_digest(ImageBuildSpec(context="./x", build_args={"A": "1", "B": "2"}), context_digest=ctx)
    b = compute_input_digest(ImageBuildSpec(context="./x", build_args={"B": "2", "A": "1"}), context_digest=ctx)
    assert a == b


def test_input_digest_default_base_is_empty() -> None:
    spec = ImageBuildSpec(context="./x")
    explicit = compute_input_digest(spec, context_digest="ctx", base_image_digest="")
    default = compute_input_digest(spec, context_digest="ctx")
    assert explicit == default


# --------------------------------------------------------------------------
# scratch_ref
# --------------------------------------------------------------------------


def test_scratch_ref_format() -> None:
    assert scratch_ref("cp-box:5012", "deadbeef") == "cp-box:5012/scratch/deadbeef"


def test_scratch_ref_custom_namespace() -> None:
    assert scratch_ref("h:5012", "d", namespace="ns") == "h:5012/ns/d"


@pytest.mark.parametrize("host,digest", [("", "d"), ("h:5012", "")])
def test_scratch_ref_rejects_empty(host: str, digest: str) -> None:
    with pytest.raises(ValueError):
        scratch_ref(host, digest)


def test_scratch_ref_rejects_empty_namespace() -> None:
    with pytest.raises(ValueError, match="namespace"):
        scratch_ref("h:5012", "d", namespace="")


def test_input_digest_is_hex_sha256() -> None:
    d = compute_input_digest(ImageBuildSpec(context="./x"), context_digest="ctx")
    assert len(d) == 64
    int(d, 16)  # raises if not hex


# --------------------------------------------------------------------------
# compute_context_digest_for_dir — additional edge cases
# --------------------------------------------------------------------------


def test_context_digest_symlink_file_target_change_changes_digest(tmp_path: Path) -> None:
    """A symlink is hashed by its target path (via readlink), not by the
    target file's content.  Changing the symlink target must change the
    digest even when the target files have identical content."""
    root = tmp_path / "ctx"
    (root / "targets").mkdir(parents=True)
    (root / "targets" / "a.txt").write_text("same content")
    (root / "targets" / "b.txt").write_text("same content")
    link = root / "link.txt"
    link.symlink_to("targets/a.txt")
    before = compute_context_digest_for_dir(root)
    link.unlink()
    link.symlink_to("targets/b.txt")
    after = compute_context_digest_for_dir(root)
    assert before != after, (
        "changing symlink target must change context digest even when file contents are identical"
    )


def test_context_digest_symlink_differs_from_regular_file_with_same_content(tmp_path: Path) -> None:
    """A symlink entry (kind='L') and a regular-file entry (kind='F') that
    refer to the same bytes must produce different digests.  Swapping a
    regular file for a symlink is a structural change in the context tree."""
    a = tmp_path / "with_file"
    b = tmp_path / "with_symlink"
    for root in (a, b):
        root.mkdir()
    (a / "Dockerfile").write_text("FROM busybox\n")
    (b / "target").write_text("FROM busybox\n")
    (b / "Dockerfile").symlink_to("target")
    assert compute_context_digest_for_dir(a) != compute_context_digest_for_dir(b), (
        "a regular file and a symlink to identical content must hash differently"
    )


def test_context_digest_empty_subdir_not_captured(tmp_path: Path) -> None:
    """Empty directories are not included in the digest — only files matter.
    Two context trees that differ only by an extra empty subdirectory hash
    identically.  This is documented behavior (conservative = can only
    over-rebuild, never wrongly dedupe a file-content change)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write(a, "Dockerfile", "FROM busybox\n")
    _write(b, "Dockerfile", "FROM busybox\n")
    (b / "empty_dir").mkdir()
    assert compute_context_digest_for_dir(a) == compute_context_digest_for_dir(b), (
        "an empty directory must not change the context digest"
    )


def test_context_digest_root_and_subdir_file_order_is_deterministic(tmp_path: Path) -> None:
    """Files in the root directory mixed with files in a subdirectory that
    lexicographically sorts before the root files (e.g. root 'b.txt' and
    subdir 'a_sub/z.txt') must produce the same digest as a tree with
    identical content built in any other order.

    This pins the ``entries.sort()`` call at the end of
    ``compute_context_digest_for_dir``: the sorted walk-order and the global
    lexicographic sort of relative paths differ for this tree shape, so the
    sort is load-bearing."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for root in (a, b):
        _write(root, "b.txt", "root-file\n")
        _write(root, "a_sub/z.txt", "subdir-file\n")
    d_a = compute_context_digest_for_dir(a)
    d_b = compute_context_digest_for_dir(b)
    assert d_a == d_b, (
        "identical trees with root + subdir files must produce the same digest"
    )


# --------------------------------------------------------------------------
# compute_input_digest — git: source edge cases
# --------------------------------------------------------------------------


def test_input_digest_git_spec_dockerfile_changes_are_captured() -> None:
    """For a git: spec, compute_input_digest hashes git.dockerfile via
    effective_dockerfile() — so changing git.dockerfile changes the
    input_digest.  Two builds from the same git context but different
    Dockerfiles must never collide."""
    ctx = "same-ctx"
    spec_default = ImageBuildSpec(git=GitContext(repo="https://x/y", ref="abc"))
    spec_gpu = ImageBuildSpec(
        git=GitContext(repo="https://x/y", ref="abc", dockerfile="Dockerfile.gpu"),
    )
    assert compute_input_digest(spec_default, context_digest=ctx) != compute_input_digest(
        spec_gpu, context_digest=ctx
    ), "different git.dockerfile must produce a different input_digest"


def test_git_context_default_ref_equals_explicit_main() -> None:
    """GitContext.ref defaults to 'main'.  A GitContext with ref omitted and
    one with ref='main' must produce the same git_context_digest — so the
    default is stable and doesn't silently change meaning if the field is
    later renamed."""
    g_implicit = GitContext(repo="https://x/y")
    g_explicit = GitContext(repo="https://x/y", ref="main")
    assert git_context_digest(g_implicit) == git_context_digest(g_explicit)
