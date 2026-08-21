"""Onboard an evolvable agent — stand up its experiment-copy repo.

Given an upstream agent source (git URL + ref), this makes a **copy you own** in two
places and tells you how to point an eval at it:

  1. resolves the baseline commit SHA (the exact ``refs/heads/<ref>`` — a bare name never
     matches a namespaced fork branch like ``refs/heads/<fork>/<ref>``),
  2. creates a GitHub repo you own (``--repo <org>/<name>``) — the *experiment copy*
     the evolver pushes candidate branches to, never upstream (the experiment-copy rule),
  3. seeds it with a **single parentless commit** on a ``baseline`` branch (override with
     ``--branch-name`` — e.g. to mirror the snapshot ``opencode_v1.18.16``) — the tree at ``--ref``,
     no history, no other branches, none of upstream's PR refs (an experiment copy only needs the
     baseline tree; a full-history mirror of a large upstream is gigabytes of waste). ``--prune
     <profile>`` optionally drops dead-weight paths from that tree (patch-safe — see
     :data:`PRUNE_PROFILES`) so every per-container clone is smaller,
  4. checks out a **local git-tracked working copy** at ``--dir`` (``origin`` = your
     GitHub repo, ``upstream`` = the source, so you can inspect it, sync later, and —
     Phase E — branch candidates from it). A re-onboard **auto-refreshes** this checkout
     to the new baseline **only when it's clean** (no uncommitted changes / un-pushed
     commits); if it holds local work it's left untouched with a warning,
  5. writes the **agent-source pointer** ``{repo, ref, token_env}`` to
     ``.beagle/agents/<profile>.json`` so downstream tasks discover it by profile — no
     hand-copying.

**No vendoring.** The agent's code is never committed into beagle. At eval the trial
container clones ``<experiment-repo>@<sha>`` (the M+N harbor path). The experiment
repo is the single durable home for the agent code (θ).

Run it::

    python -m beagle.tools.onboard \\
        --upstream https://github.com/<upstream-org>/<agent> --ref <commit-sha> \\
        --repo <your-org>/<agent>-beagle --private \\
        --dir ./experiments/<agent>-beagle --profile-name <agent> --version <v>

Pass a **commit SHA** for ``--ref`` (a branch like ``develop`` drifts — a later re-onboard would
snapshot a *different* baseline); a branch/tag is resolved to its current SHA if you must.

Downstream reads the pointer from the manifest by agent name (the baseline smoke does
this); you add only the eval knobs (model / gateway / benchmark) to your run config —
see the README's "Onboard an agent and run a baseline". Auth: ``gh`` + git over HTTPS using the token
in ``--token-env`` (default ``GH_TOKEN`` — a credential, from the environment); tokens
are passed only as command arguments (never persisted to ``.git/config``) and redacted
from output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beagle.config import RunConfig

_FULL_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")


# --- pure helpers (unit-tested) ----------------------------------------------


def is_full_sha(ref: str) -> bool:
    """True if ``ref`` is a full 40-char commit SHA (used directly, not resolved)."""
    return bool(_FULL_SHA.fullmatch(ref or ""))


def authed_url(url: str, token: str) -> str:
    """Inject a token into an HTTPS GitHub URL for a single command's auth.

    Returns the URL unchanged for non-HTTPS (ssh) URLs or when no token is given —
    those authenticate by other means (ssh keys). The result is passed as a command
    argument only, never stored in a remote.
    """
    if token and url.startswith("https://"):
        return url.replace("https://", f"https://x-access-token:{token}@", 1)
    return url


def redact(text: str, token: str | None) -> str:
    """Strip a token (and its ``x-access-token:<tok>@`` URL form) from text."""
    if not token:
        return text
    return text.replace(f"x-access-token:{token}@", "x-access-token:***@").replace(token, "***")


def derive_name(repo: str) -> str:
    """``org/name`` → ``name`` (the default ``--profile-name`` / ``--dir`` / manifest key)."""
    return repo.rstrip("/").split("/")[-1]


def manifest_path(profile: str, root: str | Path = ".") -> Path:
    """Canonical location onboarding records an onboarded agent's pointer at, so
    downstream tasks discover it by profile instead of hand-copying. Gitignored."""
    return Path(root) / ".beagle" / "agents" / f"{profile}.json"


def load_manifest(profile: str, *, root: str | Path = ".") -> dict:
    """The onboarded agent's pointer dict (``{profile, version, repo, ref, token_env?, …}``),
    read from ``.beagle/agents/<profile>.json``. Raises if it's not there."""
    path = manifest_path(profile, root=root)
    if not path.exists():
        raise FileNotFoundError(
            f"no onboarded agent manifest at {path} — run "
            f"`python -m beagle.tools.onboard --profile-name {profile} …` first"
        )
    return json.loads(path.read_text())


def latest_manifest(*, root: str | Path = ".") -> str:
    """Profile of the most-recently-written onboarded agent (handy default for tooling/examples)."""
    d = Path(root) / ".beagle" / "agents"
    metas = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []
    if not metas:
        raise FileNotFoundError(f"no onboarded agents under {d} — run `python -m beagle.tools.onboard …` first")
    return metas[0].stem


def agent_source_config(repo: str, ref: str, token_env: str | None) -> dict:
    """The **agent-source pointer** baseline eval reads: ``repo@ref`` (+ how to auth a
    private clone).

    This is *all* onboarding uniquely produces, and it's agent-agnostic. Model,
    gateway, and task selection are eval-run knobs the run config supplies — not
    onboarding's business. The agent's entrypoint is agent-intrinsic (its adapter's
    ``_default_source`` owns it). A valid (partial) run config, whose other fields
    default.
    """
    cfg = {"repo": repo, "ref": ref}
    if token_env:  # only a private clone needs a credential
        cfg["token_env"] = token_env
    return cfg


def resolve_onboarded_source(cfg: RunConfig, *, root: str | Path) -> RunConfig:
    """Fill the agent's source from the onboarded manifest named in
    ``agent.config['onboarded']`` — so repo/ref are read from where onboarding filed them,
    never hand-copied. Returns a **copy**; a no-op if an explicit ``agent.source`` (or an
    inline ``agent.config.agent_source``) is already present, or no ``onboarded`` is named.
    Raises if the named manifest is missing. Shared by ``beagle evaluate`` and the smoke.
    """
    cfg = cfg.model_copy(deep=True)
    profile = cfg.agent.config.pop("onboarded", None)  # smoke/run directive, not an agent knob
    if cfg.agent.source is not None or "agent_source" in cfg.agent.config or not profile:
        return cfg
    path = manifest_path(profile, root=root)
    if not path.exists():
        raise RuntimeError(
            f"agent.config.onboarded={profile!r} but no manifest at {path} — run "
            f"`python -m beagle.tools.onboard --profile-name {profile} …` first, or set agent.source"
        )
    manifest = json.loads(path.read_text())
    cfg.agent.config["agent_source"] = {"repo": manifest["repo"], "ref": manifest["ref"]}
    if manifest.get("token_env"):
        cfg.agent.config["token_env"] = manifest["token_env"]
    return cfg


# --- git / gh (impure) -------------------------------------------------------


def _run(cmd: list[str], *, token: str | None = None, capture: bool = False) -> str:
    """Run a command, redacting the token from any error output."""
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    except subprocess.CalledProcessError as e:
        out = redact((e.stderr or "") + (e.stdout or ""), token)
        raise SystemExit(f"command failed: {redact(' '.join(cmd), token)}\n{out}") from None
    return result.stdout if capture else ""


def resolve_sha(upstream_auth: str, ref: str, token: str) -> str:
    """Resolve ``ref`` (branch/tag/sha, or upstream default when empty) to a commit SHA.

    A **bare** branch/tag name is disambiguated to the EXACT ``refs/heads/<ref>`` (then
    ``refs/tags/<ref>``): ``git ls-remote <url> dev`` matches on the trailing path component, so it
    also returns ``refs/heads/<fork>/dev`` — taking the first line there silently seeds the WRONG
    branch. An annotated tag resolves to the commit it points at (the ``^{}`` deref line)."""
    if is_full_sha(ref):
        return ref
    if not ref or ref == "HEAD":
        out = _run(["git", "ls-remote", upstream_auth, "HEAD"], token=token, capture=True)
        if not out.strip():
            raise SystemExit("upstream has no HEAD to resolve")
        return out.split()[0]
    for pattern in (f"refs/heads/{ref}", f"refs/tags/{ref}", ref):
        rows = [ln.split() for ln in
                _run(["git", "ls-remote", upstream_auth, pattern], token=token, capture=True).splitlines()
                if ln.strip()]
        if not rows:
            continue
        deref = next((sha for sha, name in rows if name.endswith("^{}")), None)  # annotated tag → commit
        if deref:
            return deref
        if pattern.startswith("refs/") or len(rows) == 1:
            return rows[0][0]
        # Bare fallback matched multiple refs (ambiguous): accept only an exact full-ref equality.
        exact = next((sha for sha, name in rows
                      if name in (ref, f"refs/heads/{ref}", f"refs/tags/{ref}")), None)
        if exact:
            return exact
        raise SystemExit(
            f"ref {ref!r} is ambiguous on upstream ({', '.join(n for _, n in rows)}); "
            f"pass an exact ref (e.g. refs/heads/{ref}) or a full SHA")
    raise SystemExit(f"ref {ref!r} not found on upstream")


def repo_exists(repo: str) -> bool:
    return subprocess.run(
        ["gh", "repo", "view", repo], capture_output=True, text=True
    ).returncode == 0


def create_repo(repo: str, visibility: str) -> None:
    _run(["gh", "repo", "create", repo, f"--{visibility}"])


#: Default branch the single baseline commit lands on. Unified across agents — the ``--repo`` name
#: already carries the version (e.g. ``opencode_v1.18.16``). Override per-onboard with ``--branch-name``.
DEFAULT_SEED_BRANCH = "baseline"


#: Optional per-agent seed **prune profiles** — a list of git pathspecs whose blobs are dropped from the
#: baseline tree at seed time (``--prune <name>``), so the experiment copy (and every per-container
#: clone) ships without dead weight. **Patch-safe:** pruning only *removes* whole paths, so every KEPT
#: blob stays byte-identical to upstream — an evolution ``git diff base..HEAD`` touches only kept files
#: and applies cleanly to upstream, and the removed paths never appear in that diff. Pathspecs that match
#: nothing are ignored (``--ignore-unmatch``), so a profile degrades gracefully across upstream versions.
PRUNE_PROFILES: dict[str, list[str]] = {
    # opencode: drop the web/desktop/marketing apps + demo assets from the monorepo (per-container clone
    # 79 → ~12 MB) while keeping the whole headless-run closure. KEEP packages/app/vendor/*.tgz — the
    # kept `session-ui` package needs that vendored tarball for `bun install`. Derived + verified (bun
    # install + `run`) in docs/opencode-prune.md.
    "opencode": [
        "packages/console", "packages/web", "packages/desktop",
        "artifacts", "screenshot-uk.png",
        ":(glob)README.*.md",                        # translated READMEs (README.md itself is kept)
        "packages/app", ":(exclude)packages/app/vendor",   # all of app EXCEPT the vendored client tarball
    ],
}


def seed_from_upstream(*, upstream_auth: str, github_auth: str, branch: str, sha: str, token: str,
                       prune: list[str] | None = None) -> str:
    """Seed the (empty) GitHub repo with a SINGLE parentless commit on the ``branch`` ref (the
    caller's ``--branch-name``, default ``baseline`` — where the evolver branches candidates off) —
    the tree at ``sha``, no history, no other branches, none of upstream's ``refs/pull/*``.

    An experiment copy only needs the baseline *tree*: the runtime clones ``<repo>@<baseline-sha>``
    shallow (``git fetch --depth 1``). A ``git clone --mirror`` of a large upstream is **gigabytes** of
    pure waste (every branch + PR ref + full history), so we instead depth-1 fetch just ``sha`` and
    re-commit its tree as a fresh orphan. (We can't push the shallow commit as-is: its omitted parents
    would dangle and the server rejects it — hence the orphan.)

    ``prune`` (a :data:`PRUNE_PROFILES` list) drops dead-weight paths from that tree *in the index*
    (``read-tree`` → ``rm --cached`` → ``write-tree`` — no working-tree checkout), so the seeded tree
    carries only the kept blobs. Patch-safe: kept blobs are byte-identical to upstream (see
    :data:`PRUNE_PROFILES`).

    Returns the **new baseline commit SHA on the copy** (a fresh orphan, so it differs from the
    upstream ``sha``; the caller records it as the manifest ref). The throwaway clone (and the token it
    briefly holds) is deleted immediately after.
    """
    with tempfile.TemporaryDirectory() as tmp:
        seed = Path(tmp) / "seed"
        _run(["git", "init", "-q", "-b", branch, str(seed)])
        g = ["git", "-C", str(seed)]
        _run(g + ["remote", "add", "origin", upstream_auth], token=token)
        # Depth-1 fetch of just the baseline commit (its tree + blobs) — mirrors the runtime clone.
        _run(g + ["fetch", "-q", "--depth", "1", "origin", sha], token=token)
        # The tree the orphan will carry: the baseline's exact tree, or a pruned tree built in the index.
        tree_ref = f"{sha}^{{tree}}"
        if prune:
            _run(g + ["read-tree", sha], token=token)                       # index := baseline tree
            _run(g + ["rm", "-r", "--cached", "--quiet", "--ignore-unmatch", "--", *prune], token=token)
            tree_ref = _run(g + ["write-tree"], token=token, capture=True).strip()  # reduced tree object
        # A fresh orphan commit carrying that tree (no parents → a clean, minimal push).
        new_sha = _run(
            g + ["-c", "user.name=beagle-onboard", "-c", "user.email=onboard@beagle.local",
                 "commit-tree", tree_ref, "-m", f"{branch}: baseline @ upstream {sha}"],
            token=token, capture=True).strip()
        # --force so --reseed overwrites an existing baseline (the orphan shares no ancestry); a no-op
        # on the initial push into the empty repo.
        _run(g + ["push", "-q", "--force", github_auth, f"{new_sha}:refs/heads/{branch}"],
             token=token)
        return new_sha


def clone_working(
    *, github_auth: str, github_clean: str, upstream_clean: str, dir_path: Path, ref: str, token: str,
) -> None:
    """Check out a local working copy: ``origin`` = your GitHub repo, ``upstream`` =
    source, at the baseline ref. The token is scrubbed from the stored remote."""
    _run(["git", "clone", github_auth, str(dir_path)], token=token)
    g = ["git", "-C", str(dir_path)]
    _run(g + ["remote", "set-url", "origin", github_clean])  # scrub token from .git/config
    _run(g + ["remote", "add", "upstream", upstream_clean])
    if ref:
        _run(g + ["checkout", ref])  # branch/tag/sha (all present via the mirror seed)


def _git_out(dir_path: Path, *args: str) -> str:
    """`git -C dir <args>` stdout (stripped); '' on any non-zero exit (treated as 'unknown')."""
    r = subprocess.run(["git", "-C", str(dir_path), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def local_checkout_work(dir_path: Path, branch: str) -> str | None:
    """Reason a pre-existing ``--dir`` checkout holds local work a baseline refresh would destroy,
    or ``None`` if it's a pristine mirror safe to move.

    A reseed makes an ORPHAN baseline (no shared history), so moving the checkout is a hard reset,
    not a fast-forward — only safe when nothing local is lost. 'Clean' requires ALL of, checked
    BEFORE any fetch (so remote-tracking refs still point at the OLD baseline the checkout mirrors):

      * working tree + index clean (no uncommitted changes),
      * HEAD on ``branch`` (not detached / some other branch),
      * no commits on HEAD absent from every remote-tracking ref (nothing un-pushed).

    Any failure returns a short human reason; the caller then warns instead of resetting. Fail-safe:
    an unreadable git state (``''``) trips the un-pushed check and is reported, never silently reset."""
    if _git_out(dir_path, "status", "--porcelain"):
        n = len(_git_out(dir_path, "status", "--porcelain").splitlines())
        return f"{n} uncommitted change(s)"
    cur = _git_out(dir_path, "rev-parse", "--abbrev-ref", "HEAD")
    if cur != branch:
        return f"HEAD is '{cur or 'unknown'}', not '{branch}'"
    unpushed = _git_out(dir_path, "rev-list", "HEAD", "--not", "--remotes")
    if unpushed:
        return f"{len(unpushed.splitlines())} commit(s) not on any remote"
    return None


def refresh_local_checkout(dir_path: Path, branch: str, remote_auth: str, token: str) -> None:
    """Move a CLEAN checkout onto the remote ``branch`` baseline: fetch it (via the authed URL —
    ``origin`` stores a token-free URL, so a private fetch needs the token here) and hard-reset onto
    it. A reseed baseline is an orphan commit (unrelated history), so this is a reset, NOT a
    fast-forward — the caller MUST have confirmed :func:`local_checkout_work` is ``None`` first, since
    this discards anything local."""
    _run(["git", "-C", str(dir_path), "fetch", remote_auth,
          f"+{branch}:refs/remotes/origin/{branch}"], token=token)
    _run(["git", "-C", str(dir_path), "reset", "--hard", f"origin/{branch}"], token=token)


# --- CLI ---------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m beagle.tools.onboard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", required=True, help="source agent git URL (the repo to copy)")
    p.add_argument("--ref", default="",
                   help="baseline commit SHA (RECOMMENDED — a branch/tag drifts, so a re-onboard "
                        "would snapshot a different baseline); a branch/tag also works. "
                        "Default: upstream's default branch")
    p.add_argument("--repo", required=True, metavar="ORG/NAME",
                   help="the GitHub repo you own to create + push the copy to")
    p.add_argument("--dir", default="", metavar="PATH",
                   help="local dir for the git-tracked working copy (default: .beagle/agents/<profile>)")
    p.add_argument("--profile-name", default="", metavar="PROFILE",
                   help="profile name — the pointer is filed at .beagle/agents/<profile>.json "
                        "(default: the repo name). It labels the copy for you; the eval-config "
                        "generator joins on --version, not this.")
    p.add_argument("--version", default="", metavar="VERSION",
                   help="snapshot version recorded in the manifest (e.g. 1.18.16 / v2.4.6 / 20260816) "
                        "— the join key `scripts/generate_eval_configs.py` matches against a config's "
                        "`agent.harness.version`. Set it to the version your run configs pin.")
    p.add_argument("--branch-name", default="", metavar="BRANCH",
                   help="branch on the copy where the single baseline commit lands (and where the "
                        "evolver branches candidates off). Default: 'baseline' — the --repo name "
                        "already carries the version. Pass e.g. --branch-name opencode_v1.18.16 to "
                        "mirror the snapshot, or any custom name.")
    p.add_argument("--prune", default="", metavar="PROFILE", choices=["", *PRUNE_PROFILES],
                   help="apply a seed prune profile — drop dead-weight paths from the baseline tree so "
                        "every per-container clone is smaller (PATCH-SAFE: only removes whole paths, kept "
                        f"files stay byte-identical to upstream). Available: {', '.join(PRUNE_PROFILES)}. "
                        "Default: none (full tree).")
    vis = p.add_mutually_exclusive_group()
    vis.add_argument("--private", dest="visibility", action="store_const", const="private", default="private")
    vis.add_argument("--public", dest="visibility", action="store_const", const="public")
    p.add_argument("--token-env", default="GH_TOKEN", help="env var holding the GitHub token")
    p.add_argument("--reseed", action="store_true",
                   help="re-seed an existing repo from upstream (DESTRUCTIVE: overwrites candidate branches)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(
            f"{args.token_env} is not set in the environment (needed to create + push the "
            f"repo). If it's in .env, EXPORT it — plain `source .env` only sets shell vars:\n"
            f"    set -a; source .env; set +a"
        )

    upstream_auth = authed_url(args.upstream, token)
    github_clean = f"https://github.com/{args.repo}.git"
    github_auth = authed_url(github_clean, token)
    github_web = f"https://github.com/{args.repo}"
    profile = args.profile_name or derive_name(args.repo)
    # The branch on the copy where the single baseline commit lands (and where the evolver branches
    # candidates off). ``--profile-name`` is the local/manifest identifier, NOT the branch; the branch
    # defaults to a unified ``baseline`` (the --repo name already carries the version) and is set by
    # --branch-name (e.g. --branch-name opencode_v1.18.16 to mirror the snapshot).
    branch = args.branch_name or DEFAULT_SEED_BRANCH
    dir_path = Path(args.dir).expanduser() if args.dir else Path(".beagle/agents") / derive_name(args.repo)

    print(f"[onboard] resolving {args.ref or 'default branch'} on {args.upstream} …")
    upstream_sha = resolve_sha(upstream_auth, args.ref, token)
    print(f"[onboard] upstream baseline commit: {upstream_sha}")

    if repo_exists(args.repo):
        created = False
        print(f"[onboard] github repo {args.repo} already exists — not creating")
    else:
        create_repo(args.repo, args.visibility)
        created = True
        print(f"[onboard] created {args.visibility} github repo {args.repo}")

    # The copy's baseline SHA — a fresh orphan of the upstream tree (small seed), so it differs from
    # ``upstream_sha`` and is what the manifest/runtime pin. Recovered from the copy when we skip seeding.
    if created or args.reseed:
        if args.reseed and not created:
            print("[onboard] --reseed: re-seeding from upstream (OVERWRITES any candidate branches)")
        prune = PRUNE_PROFILES[args.prune] if args.prune else None
        sha = seed_from_upstream(upstream_auth=upstream_auth, github_auth=github_auth,
                                 branch=branch, sha=upstream_sha, token=token, prune=prune)
        pruned = f"; pruned '{args.prune}' profile ({len(prune)} pathspecs)" if prune else ""
        print(f"[onboard] seeded {args.repo} branch '{branch}' = single baseline commit {sha} "
              f"(tree of upstream {upstream_sha[:12]}; no history{pruned})")
    else:
        sha = resolve_sha(github_auth, branch, token)
        print(f"[onboard] {args.repo} not empty — skipping seed (use --reseed to force); "
              f"branch '{branch}' = {sha}")

    if dir_path.exists():
        # A pre-existing --dir checkout that predates this seed is STALE. It's the working tree the
        # evolver branches candidates off (`evolvee.source.dir`), so a stale one is an evolve landmine
        # (eval clones the REMOTE, so eval is unaffected either way). Auto-refresh it to the new baseline
        # WHEN CLEAN — a reseed is an orphan commit, so this is fetch + hard-reset, not a fast-forward —
        # but never clobber local work: if the checkout has any, warn and leave it for the human.
        head = _git_out(dir_path, "rev-parse", "HEAD")
        if not head or head == sha:
            print(f"[onboard] local copy {dir_path} already at baseline {sha[:12]} — leaving it")
        else:
            reason = local_checkout_work(dir_path, branch)  # checked BEFORE fetch (remote refs = old)
            if reason is None:
                refresh_local_checkout(dir_path, branch, github_auth, token)
                print(f"[onboard] refreshed clean local copy {dir_path}: {head[:12]} → {sha[:12]}")
            else:
                print(f"[onboard] WARNING: local copy {dir_path} is at {head[:12]} but the baseline is now "
                      f"{sha[:12]} and it holds local work ({reason}) — NOT touching it. The remote + "
                      f"manifest ARE updated (eval/evolve clone the remote), so runs use the new baseline. "
                      f"To move this checkout yourself: commit/stash, then  git -C {dir_path} fetch origin "
                      f"&& git -C {dir_path} reset --hard origin/{branch}")
    else:
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        clone_working(github_auth=github_auth, github_clean=github_web, upstream_clean=args.upstream,
                      dir_path=dir_path, ref=branch, token=token)
        print(f"[onboard] local working copy → {dir_path} (origin={args.repo}, upstream=source)")

    token_env = args.token_env if args.visibility == "private" else None
    cfg = agent_source_config(repo=github_web, ref=sha, token_env=token_env)

    # Write the pointer to the canonical manifest so downstream tasks discover it by
    # profile — no hand-copying. Provenance (upstream + the upstream sha the baseline is a tree of,
    # + the baseline branch the orphan sits on) rides along; consumers read whatever they need.
    manifest = {"profile": profile, "version": args.version, **cfg, "upstream": args.upstream,
                "upstream_ref": upstream_sha, "branch": branch, "dir": str(dir_path)}
    mpath = manifest_path(profile)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n[onboard] done. Wrote the agent-source pointer → {mpath}")
    print(json.dumps(cfg, indent=2))
    print("\nNext (needs a live cluster + your gateway config — see README):")
    print("  python -m pytest -v -s tests/smoke/monet_tb21_smoke/test_baseline.py")
    print(f"Sync from upstream later:  git -C {dir_path} fetch upstream")
    return 0


if __name__ == "__main__":
    sys.exit(main())
