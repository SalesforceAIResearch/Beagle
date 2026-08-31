"""The golden-solution ``oracle`` flow — host-side extraction + in-container apply.

Two tightly-coupled halves of one acceptance oracle (DESIGN.md §5.1):

* **Host side** (:func:`extract_selected` / :func:`extract_one`): EvoClaw's agent
  container is answer-free — ``ContainerSetup`` deletes every tag, clears the
  reflog, and ``git gc --prune``s, so the END commit objects are physically gone.
  The golden source survives only as the ``milestone-<mid>-end`` tag inside each
  **milestone image**. This module pulls that out host-side: acquire the milestone
  image, ``git archive`` the END tree for the repo's source dirs, write the tar.

* **In-container side** (:class:`OracleFramework`, ``@register_framework("oracle")``):
  behaves like a real agent (runs through ``ContainerSetup`` → ``AgentRunner`` →
  watcher → evaluator), but instead of an LLM it **applies the golden solution**:
  the host-extracted ``<mid>.tar`` is mounted into the agent container at
  ``/golden`` (the ``docker_shim`` turns the ``-v`` bind into a ``put_archive``).
  For each milestone it untars the END source over ``/testbed``, commits, and tags
  ``agent-impl-<mid>``. The watcher then ``git archive``s the agent container's tag
  and the evaluator runs the verifiers → ``resolved``.

This proves the whole pipeline is sound — only the real LLM-driven agent is left
untested (Harbor `solve.sh` / WAI run-the-solution model). The agent container is
answer-free by design, so the solution must come from the host mount.

``run_e2e_xrlenv.py`` calls :func:`extract_selected` *before* it hands off to
``run_e2e``, then exposes the tars via ``EVOCLAW_GOLDEN_DIR`` (host path, mounted
into the agent container). Importing this module registers the ``oracle`` framework.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import image_resolution
from harness.e2e.agents.base import (  # type: ignore[import-not-found]
    AgentFramework,
    register_framework,
)

LOGGER = logging.getLogger("xrlenv.evoclaw.oracle")


# ── Host-side golden extraction ───────────────────────────────────────────────


def _src_dirs(workspace_root: Path) -> list[str]:
    meta = json.loads((workspace_root / "metadata.json").read_text())
    dirs = meta.get("repo_src_dirs") or []
    if not dirs:
        raise ValueError(f"metadata.json has no repo_src_dirs: {workspace_root}")
    return list(dirs)


def extract_one(
    client: Any,
    *,
    repo_full: str,
    milestone_id: str,
    src_dirs: list[str],
    out_tar: Path,
    name_prefix: str = "",
) -> Path:
    """Acquire the milestone image and write the END source tree to ``out_tar``.

    ``git archive milestone-<mid>-end -- <src_dirs>`` at ``/testbed``, run as the
    image's default user (root — the milestone image has no ``fakeroot`` user).
    Returns ``out_tar``.
    """
    image_base = f"{repo_full.lower()}/{milestone_id.lower()}"
    image_ref = image_resolution.dockerhub_ref(image_base, _repo_map()) or image_base
    tag = milestone_id  # the END git tag is milestone-<mid>-end
    # `git archive -- A B C` fails HARD if ANY pathspec is absent at the tag. In
    # monorepos, or when a src dir is only introduced in a later milestone, some
    # of metadata's repo_src_dirs don't exist at *this* milestone's END tag. So
    # filter to the dirs that actually exist at the tag, then archive those.
    import shlex

    quoted = " ".join(shlex.quote(d.rstrip("/")) for d in src_dirs)
    script = (
        "cd /testbed && git config --global --add safe.directory /testbed >/dev/null 2>&1; "
        f'tag=milestone-{tag}-end; present=""; '
        f'for d in {quoted}; do '
        'if git ls-tree "$tag" -- "$d" 2>/dev/null | grep -q .; then present="$present $d"; fi; '
        "done; "
        '[ -n "$present" ] || { echo "none of the repo_src_dirs exist at $tag" >&2; exit 3; }; '
        'git archive --format=tar "$tag" -- $present'
    )
    LOGGER.info("extracting golden for %s/%s from %s", repo_full, milestone_id, image_ref)
    container = client.containers.run(
        image_ref, ["tail", "-f", "/dev/null"],
        name=f"{name_prefix}golden-{milestone_id.lower()}", detach=True,
    )
    try:
        # Run as the milestone image's default user (root). Unlike EvoClaw's agent
        # container (which ContainerSetup provisions with a `fakeroot` user), the
        # milestone/evaluator image has no `fakeroot` in /etc/passwd.
        res = container.exec_run(
            ["bash", "-c", script], workdir="/testbed", demux=True
        )
        exit_code = getattr(res, "exit_code", res[0] if isinstance(res, tuple) else 1)
        output = getattr(res, "output", res[1] if isinstance(res, tuple) else (b"", b""))
        stdout, stderr = output if isinstance(output, tuple) else (output, b"")
        if exit_code != 0 or not stdout:
            raise RuntimeError(
                f"git archive failed for {repo_full}/{milestone_id} "
                f"(rc={exit_code}): {(stderr or b'').decode(errors='replace')[:400]}"
            )
        out_tar.parent.mkdir(parents=True, exist_ok=True)
        out_tar.write_bytes(stdout)
        return out_tar
    finally:
        try:
            container.remove(force=True)
        except Exception as exc:
            LOGGER.debug("remove golden container: %s", exc)


def extract_selected(
    client: Any,
    *,
    workspace_root: Path,
    milestone_ids: list[str],
    golden_dir: Path,
    name_prefix: str = "",
    refresh: bool = False,
) -> Path:
    """Extract golden tars for ``milestone_ids`` into ``golden_dir`` (a content
    cache keyed by repo+tag). A cached ``<mid>.tar`` is **reused** — the milestone
    image is only acquired on a cache miss. Set ``refresh=True`` to force re-extract.
    Returns ``golden_dir``.
    """
    src_dirs = _src_dirs(workspace_root)
    repo_full = workspace_root.name
    golden_dir.mkdir(parents=True, exist_ok=True)
    hits, misses = 0, 0
    for mid in milestone_ids:
        out = golden_dir / f"{mid}.tar"
        if not refresh and out.is_file() and out.stat().st_size > 0:
            LOGGER.info("golden cache hit: %s", out)
            hits += 1
            continue
        extract_one(
            client,
            repo_full=repo_full,
            milestone_id=mid,
            src_dirs=src_dirs,
            out_tar=out,
            name_prefix=name_prefix,
        )
        misses += 1
    LOGGER.info("golden cache: %d hit(s), %d extracted -> %s", hits, misses, golden_dir)
    return golden_dir


_REPO_MAP_CACHE: dict[str, str] | None = None


def _repo_map() -> dict[str, str]:
    global _REPO_MAP_CACHE
    cached = _REPO_MAP_CACHE
    if cached is None:
        import harness.e2e.image_version as iv  # type: ignore[import-not-found]

        f = iv.__file__
        if not f:
            raise RuntimeError("cannot locate harness.e2e.image_version on disk")
        cached = image_resolution.load_repo_short_map(Path(f).resolve().parents[2])
        _REPO_MAP_CACHE = cached
    return cached


# ── In-container oracle agent framework ───────────────────────────────────────

# Per milestone: apply the golden END source mounted at /golden/<mid>.tar, then
# tag the completion marker the orchestrator's watcher polls for.
_ORACLE_SCRIPT = r"""
set -u
cd /testbed || { echo "oracle: FATAL /testbed missing" >&2; exit 1; }
# Fail loud if the base image isn't a git repo. EvoClaw's submission is git-tag
# based, so a base image without .git (a bad image build — seen on go-zero) makes
# git commit/tag silently no-op and the watcher stalls with no signal. We do NOT
# work around it (a real agent fails the same way); we surface it clearly.
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "oracle: FATAL /testbed is not a git repository (base image has no .git) -- cannot submit" >&2
  exit 1
fi
git config --global --add safe.directory /testbed >/dev/null 2>&1 || true
git config --global user.name oracle >/dev/null 2>&1 || true
git config --global user.email oracle@evoclaw.local >/dev/null 2>&1 || true
status=0
# The orchestrator stages each unlocked milestone's SRS in the container as a
# flat file /e2e_workspace/srs/<mid>_SRS.md (NOT a dir). Derive <mid> from it.
for f in /e2e_workspace/srs/*; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  mid=${base%_SRS.md}
  mid=${mid%.md}
  tarball="/golden/${mid}.tar"
  if [ -f "${tarball}" ]; then
    # Replace each source dir the golden tar carries (top-level entries) with the
    # END content — robust to the DAG start state and to files the milestone
    # deletes. Then commit + tag the completion marker the watcher polls for.
    tar -tf "${tarball}" | cut -d/ -f1 | sort -u | while read -r d; do
      [ -n "${d}" ] && rm -rf "/testbed/${d}"
    done
    tar -xf "${tarball}" -C /testbed
    git add -A >/dev/null 2>&1 || true
    git commit -q -m "oracle: golden solution for ${mid}" >/dev/null 2>&1 || true
    # Verify the tag actually got created — don't echo a lie if git failed.
    if git tag -f "agent-impl-${mid}" >/dev/null 2>&1 \
       && git rev-parse -q --verify "refs/tags/agent-impl-${mid}" >/dev/null; then
      echo "oracle: applied golden ${tarball} -> tagged agent-impl-${mid}"
    else
      echo "oracle: FATAL git tag agent-impl-${mid} failed (git state broken)" >&2
      status=1
    fi
  else
    echo "oracle: MISSING ${tarball} (run oracle extract first)" >&2
    status=1
  fi
done
# Race fix: EvoClaw submits a milestone's eval ASYNCHRONOUSLY — a background
# watcher polls ``git tag -l`` every ~2s (detection lags further under cluster
# load) and only then marks the milestone submitted. A no-LLM oracle would
# otherwise exit in milliseconds, before the watcher ever sees the tag, so the
# runner's no-progress loop abandons the milestone with Completed:0 (observed
# deterministically on fast-recovering milestones like ripgrep
# milestone_seed_a6e0be3_1_sub-01, while slower siblings passed). Linger here so
# the watcher reliably detects the tag while this agent is still "running" —
# exactly as a real (minutes-long) agent would. Only when we actually tagged
# (status==0); a missing-golden / git failure gains nothing from waiting.
if [ "${status}" -eq 0 ]; then
  sleep @@TAG_SETTLE_S@@
fi
exit ${status}
""".strip()

# Seconds the oracle lingers after tagging (set from the --oracle-tag-settle-s
# wrapper flag; default keeps the script standalone-runnable). Substituted into
# the script at command-build time — a placeholder, not a bash ${var}, so no env
# needs to reach the container.
_TAG_SETTLE_S = 12


@register_framework("oracle")
class OracleFramework(AgentFramework):
    """Applies the host-injected golden END source as the agent's submission."""

    def get_container_mounts(self) -> list[str]:
        golden = os.environ.get("EVOCLAW_GOLDEN_DIR")
        # The shim converts this host bind into a put_archive of the golden tars.
        return ["-v", f"{golden}:/golden:ro"] if golden else []

    def get_container_init_script(self, agent_name: str) -> str:
        return "pass"

    def build_run_command(self, model: str, session_id: str, prompt_path: str) -> str:
        script = _ORACLE_SCRIPT.replace("@@TAG_SETTLE_S@@", str(int(_TAG_SETTLE_S)))
        return f"sh -c {_sh_quote(script)}"

    def build_resume_command(self, model: str, session_id: str, message_path: str) -> str:
        return self.build_run_command(model, session_id, message_path)

    def get_effective_reasoning_effort(self) -> str | None:
        return None


def _sh_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)
