"""Unit tests for the agent-onboarding tool's pure helpers (no git / gh / network)."""

from __future__ import annotations

import json

from beagle.tools import onboard


def test_resolve_sha_disambiguates_bare_branch_name(monkeypatch) -> None:
    # Repro: `git ls-remote <url> dev` matches BOTH refs/heads/dev and refs/heads/K-Mistele/dev
    # (trailing path component), K-Mistele/dev listed first. The bare-name query must resolve to the
    # EXACT refs/heads/dev, not `out.split()[0]` (which grabbed the wrong fork branch).
    heads_dev = ("1f94d8a3c86b67f4f49a0e341de74e9188381b3a\trefs/heads/dev\n")
    ambiguous = ("5a7b678553e8ed9cdd0fc3ff26a2edcd07556001\trefs/heads/K-Mistele/dev\n"
                 "1f94d8a3c86b67f4f49a0e341de74e9188381b3a\trefs/heads/dev\n")

    def fake_run(argv, *, token=None, capture=False):
        pattern = argv[-1]
        if pattern == "refs/heads/dev":
            return heads_dev            # exact refspec → unambiguous
        if pattern == "refs/tags/dev":
            return ""
        if pattern == "dev":
            return ambiguous            # bare name → both refs
        return ""

    monkeypatch.setattr(onboard, "_run", fake_run)
    assert onboard.resolve_sha("https://github.com/o/r", "dev", "TOK") == (
        "1f94d8a3c86b67f4f49a0e341de74e9188381b3a")


def test_resolve_sha_passes_full_sha_through() -> None:
    sha = "1f94d8a3c86b67f4f49a0e341de74e9188381b3a"
    assert onboard.resolve_sha("https://github.com/o/r", sha, "TOK") == sha


def test_resolve_sha_dereferences_annotated_tag(monkeypatch) -> None:
    def fake_run(argv, *, token=None, capture=False):
        if argv[-1] == "refs/tags/v1":
            return ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/v1\n"
                    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\trefs/tags/v1^{}\n")
        return ""

    monkeypatch.setattr(onboard, "_run", fake_run)
    # the ^{} deref (the commit the annotated tag points at) wins over the tag object
    assert onboard.resolve_sha("https://github.com/o/r", "v1", "TOK") == "b" * 40


def test_authed_url_injects_token_for_https_only() -> None:
    assert onboard.authed_url("https://github.com/o/r.git", "TOK") == (
        "https://x-access-token:TOK@github.com/o/r.git"
    )
    # ssh + tokenless pass through unchanged.
    assert onboard.authed_url("git@github.com:o/r.git", "TOK") == "git@github.com:o/r.git"
    assert onboard.authed_url("https://github.com/o/r.git", "") == "https://github.com/o/r.git"


def test_redact_strips_token_and_url_form() -> None:
    msg = "cloning https://x-access-token:sekret@github.com/o/r using sekret"
    out = onboard.redact(msg, "sekret")
    assert "sekret" not in out and "x-access-token:***@" in out
    assert onboard.redact("no token here", None) == "no token here"


def test_is_full_sha() -> None:
    assert onboard.is_full_sha("b8264d2b8b8c5ddf6d5eb4ad8d48cc9fea89552b")
    assert not onboard.is_full_sha("b8264d2")  # abbreviated
    assert not onboard.is_full_sha("develop")
    assert not onboard.is_full_sha("")


def test_derive_name() -> None:
    assert onboard.derive_name("airesearch-emu/monet_code-beagle") == "monet_code-beagle"
    assert onboard.derive_name("org/name/") == "name"


def test_agent_source_config_is_pointer_only() -> None:
    # Onboarding emits ONLY the agent-source pointer — no eval knobs (model /
    # benchmark selection) and no agent-intrinsic entrypoint.
    private = onboard.agent_source_config("https://github.com/o/r", "deadbeef", "GH_TOKEN")
    assert private == {"repo": "https://github.com/o/r", "ref": "deadbeef", "token_env": "GH_TOKEN"}
    # public repos need no clone credential.
    public = onboard.agent_source_config("https://github.com/o/r", "deadbeef", None)
    assert public == {"repo": "https://github.com/o/r", "ref": "deadbeef"}


def test_manifest_path_is_canonical() -> None:
    from pathlib import Path

    assert onboard.manifest_path("monet", root="/x") == Path("/x/.beagle/agents/monet.json")


def test_resolve_onboarded_source_from_manifest(tmp_path) -> None:
    # Fill agent.source from the onboarded manifest named in agent.config.onboarded — no
    # hand-copying repo/ref. Manifest-only keys (name/upstream/dir) are ignored; the
    # `onboarded` directive is popped.
    from beagle.config import RunConfig
    from beagle.tools.onboard import resolve_onboarded_source

    (tmp_path / ".beagle" / "agents").mkdir(parents=True)
    (tmp_path / ".beagle" / "agents" / "monet_code.json").write_text(json.dumps(
        {"name": "monet_code", "repo": "https://h/r", "ref": "deadbeef",
         "token_env": "GH_TOKEN", "upstream": "https://up", "dir": "/d"}))

    def _cfg(agent: dict) -> RunConfig:
        return RunConfig.from_dict({"model": {"name": "gpt-5.5"}, "agent": agent,
                                    "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]}})

    resolved = resolve_onboarded_source(
        _cfg({"name": "monet", "config": {"onboarded": "monet_code", "monet_args": ["--x"]}}),
        root=tmp_path)
    spec = resolved.agent_spec()
    assert spec.source and spec.source.repo == "https://h/r" and spec.source.ref == "deadbeef"
    assert resolved.agent.config["token_env"] == "GH_TOKEN"  # manifest-only keys dropped
    assert "onboarded" not in resolved.agent.config          # the directive is popped

    # An explicit agent.source wins — the manifest is not consulted.
    explicit = resolve_onboarded_source(
        _cfg({"name": "monet", "source": {"repo": "b", "ref": "1"},
              "config": {"onboarded": "monet_code"}}), root=tmp_path)
    assert explicit.agent_spec().source.repo == "b"


def test_load_manifest_reads_and_raises(tmp_path) -> None:
    import pytest

    d = tmp_path / ".beagle" / "agents"
    d.mkdir(parents=True)
    (d / "monet_code_x.json").write_text(json.dumps(
        {"name": "monet_code_x", "repo": "https://h/r", "ref": "abc", "token_env": "GH_TOKEN",
         "dir": "../exp/monet_code_x"}))

    m = onboard.load_manifest("monet_code_x", root=tmp_path)
    assert m["repo"] == "https://h/r" and m["ref"] == "abc" and m["dir"] == "../exp/monet_code_x"

    with pytest.raises(FileNotFoundError):
        onboard.load_manifest("nope", root=tmp_path)


def test_latest_manifest_picks_newest_by_mtime(tmp_path) -> None:
    import os

    d = tmp_path / ".beagle" / "agents"
    d.mkdir(parents=True)
    (d / "old.json").write_text("{}")
    (d / "new.json").write_text("{}")
    os.utime(d / "old.json", (1000, 1000))     # explicit mtimes (no wall-clock dependence)
    os.utime(d / "new.json", (2000, 2000))
    assert onboard.latest_manifest(root=tmp_path) == "new"


def test_opencode_prune_profile_targets() -> None:
    # The opencode profile drops the apps/assets but KEEPS the vendored client tarball (session-ui
    # needs it for `bun install`) and README.md (only the translated READMEs go).
    p = onboard.PRUNE_PROFILES["opencode"]
    for dead in ("packages/console", "packages/web", "packages/desktop", "artifacts", "screenshot-uk.png"):
        assert dead in p
    assert ":(glob)README.*.md" in p                      # translated READMEs only
    assert "packages/app" in p and ":(exclude)packages/app/vendor" in p   # keep the vendored tarball


def _git(cwd, *args: str) -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def test_seed_from_upstream_prunes_tree(tmp_path) -> None:
    """End-to-end (real git, no network/gh): a pruned seed drops the profile's paths and keeps
    everything else byte-identical — the patch-safety invariant the evolver relies on."""
    import subprocess

    # A fake upstream mirroring opencode's shape: apps + assets to drop, runtime pkgs + vendor to keep.
    up = tmp_path / "up"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _git(up, "config", "user.email", "t@t"); _git(up, "config", "user.name", "t")
    (up / "README.md").write_text("keep")
    (up / "README.zh-CN.md").write_text("drop")
    (up / "screenshot-uk.png").write_bytes(b"img")
    for rel, body in [
        ("artifacts/demo.mp4", "vid"),
        ("packages/console/x.ts", "console"), ("packages/web/x.ts", "web"),
        ("packages/desktop/x.ts", "desktop"),
        ("packages/app/src/main.ts", "app-src"),                 # dropped
        ("packages/app/vendor/client.tgz", "keep-me"),           # KEPT
        ("packages/core/index.ts", "core"), ("packages/session-ui/index.ts", "ui"),
    ]:
        f = up / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    _git(up, "add", "-A"); _git(up, "commit", "-q", "-m", "init")
    sha = _git(up, "rev-parse", "HEAD")

    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)

    new_sha = onboard.seed_from_upstream(
        upstream_auth=str(up / ".git"), github_auth=str(bare), branch="baseline", sha=sha, token="",
        prune=onboard.PRUNE_PROFILES["opencode"])

    files = set(_git(bare, "ls-tree", "-r", "--name-only", new_sha).splitlines())
    # dropped
    assert "README.zh-CN.md" not in files and "screenshot-uk.png" not in files
    assert not any(f.startswith(("artifacts/", "packages/console/", "packages/web/",
                                 "packages/desktop/", "packages/app/src/")) for f in files)
    # kept — byte-identical to upstream (same blob names + a preserved vendor tarball)
    assert {"README.md", "packages/app/vendor/client.tgz", "packages/core/index.ts",
            "packages/session-ui/index.ts"} <= files
    # and the kept blobs are truly identical to upstream's (patch-safety)
    assert _git(bare, "rev-parse", f"{new_sha}:packages/core/index.ts") == \
           _git(up, "rev-parse", f"{sha}:packages/core/index.ts")


def test_seed_from_upstream_no_prune_keeps_full_tree(tmp_path) -> None:
    import subprocess

    up = tmp_path / "up"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _git(up, "config", "user.email", "t@t"); _git(up, "config", "user.name", "t")
    (up / "keep.txt").write_text("a")
    (up / "packages").mkdir(); (up / "packages" / "console").mkdir()
    (up / "packages" / "console" / "x.ts").write_text("c")
    _git(up, "add", "-A"); _git(up, "commit", "-q", "-m", "init")
    sha = _git(up, "rev-parse", "HEAD")
    bare = tmp_path / "o.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)

    new_sha = onboard.seed_from_upstream(
        upstream_auth=str(up / ".git"), github_auth=str(bare), branch="baseline", sha=sha, token="")
    files = set(_git(bare, "ls-tree", "-r", "--name-only", new_sha).splitlines())
    assert files == {"keep.txt", "packages/console/x.ts"}    # nothing dropped without a profile


def _mirror_checkout(tmp_path, branch: str = "baseline"):
    """A pristine local checkout of a bare 'origin' on `branch` (origin/<branch> == HEAD), plus the
    bare repo — the state a fresh onboard leaves, so `local_checkout_work` should call it clean."""
    import subprocess
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", branch, str(bare)], check=True)
    seed = tmp_path / "seed"
    _git(seed.parent, "clone", "-q", str(bare), str(seed))
    _git(seed, "config", "user.email", "t@t"); _git(seed, "config", "user.name", "t")
    (seed / "code.txt").write_text("v1")
    _git(seed, "add", "-A"); _git(seed, "commit", "-q", "-m", "baseline")
    _git(seed, "push", "-q", "origin", branch)
    co = tmp_path / "checkout"
    _git(seed.parent, "clone", "-q", str(bare), str(co))
    _git(co, "config", "user.email", "t@t"); _git(co, "config", "user.name", "t")
    return bare, co


def test_local_checkout_work_clean_mirror_is_none(tmp_path) -> None:
    _bare, co = _mirror_checkout(tmp_path)
    assert onboard.local_checkout_work(co, "baseline") is None


def test_local_checkout_work_flags_uncommitted_changes(tmp_path) -> None:
    _bare, co = _mirror_checkout(tmp_path)
    (co / "code.txt").write_text("dirty edit")     # unstaged change
    assert onboard.local_checkout_work(co, "baseline") == "1 uncommitted change(s)"


def test_local_checkout_work_flags_unpushed_commit(tmp_path) -> None:
    _bare, co = _mirror_checkout(tmp_path)
    (co / "mine.txt").write_text("local work")
    _git(co, "add", "-A"); _git(co, "commit", "-q", "-m", "my candidate")
    assert onboard.local_checkout_work(co, "baseline") == "1 commit(s) not on any remote"


def test_local_checkout_work_flags_wrong_branch(tmp_path) -> None:
    _bare, co = _mirror_checkout(tmp_path)
    _git(co, "checkout", "-q", "-b", "sidebar")     # not on the baseline branch
    assert onboard.local_checkout_work(co, "baseline") == "HEAD is 'sidebar', not 'baseline'"


def test_refresh_local_checkout_hard_resets_to_new_orphan_baseline(tmp_path) -> None:
    """A reseed replaces origin/baseline with an ORPHAN commit (unrelated history). A clean checkout
    must land exactly on it — proving refresh is a hard reset, not a (would-fail) fast-forward."""
    bare, co = _mirror_checkout(tmp_path)
    old = _git(co, "rev-parse", "HEAD")
    # Re-seed the bare 'origin' with a parentless commit (what onboard --reseed does).
    reseed = tmp_path / "reseed"
    _git(reseed.parent, "clone", "-q", str(bare), str(reseed))
    _git(reseed, "config", "user.email", "t@t"); _git(reseed, "config", "user.name", "t")
    _git(reseed, "checkout", "-q", "--orphan", "fresh")
    _git(reseed, "rm", "-rfq", "."); (reseed / "code.txt").write_text("v2")
    _git(reseed, "add", "-A"); _git(reseed, "commit", "-q", "-m", "new baseline")
    _git(reseed, "push", "-qf", "origin", "fresh:baseline")
    new = _git(reseed, "rev-parse", "HEAD")
    assert new != old

    assert onboard.local_checkout_work(co, "baseline") is None   # still clean → safe to refresh
    onboard.refresh_local_checkout(co, "baseline", str(bare), token="")
    assert _git(co, "rev-parse", "HEAD") == new                  # landed exactly on the orphan baseline
    assert (co / "code.txt").read_text() == "v2"
