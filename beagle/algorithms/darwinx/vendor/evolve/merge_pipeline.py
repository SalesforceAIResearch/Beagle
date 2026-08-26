"""Node merger pipeline for self-evolve campaigns."""

from __future__ import annotations

import logging
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import codingbench_eval as eval_runner  # route merge evals through the terminal-bench harness (not legacy monet run_harbor)
from . import cursor_agent, cursor_failures, meta_agent, run_config, tree, worktree
from .pipeline import (
    PipelineConfig,
    _dir_size_bytes,
    _final_eval_archive_max_bytes,
    _final_eval_extra_args,
)

def _prompt_path(name: str) -> Path:
    """Prompt template lookup with optional overlay directories.

    DARWINX_GATE_PROMPT_DIR may hold variants for a non-monet agent; only the files
    that actually differ need to exist there, everything else falls through to
    the defaults. Accepts an ``os.pathsep``-separated list searched left to
    right, so a benchmark-specific variant can layer on an agent-specific one.
    Kept identical to ``pipeline._prompt_path`` on purpose: a merge worker that
    resolved prompts differently from an evolve worker would be very hard to
    diagnose.
    """
    override = os.environ.get("DARWINX_GATE_PROMPT_DIR", "").strip()
    for part in override.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        candidate = Path(part).expanduser() / name
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parent / "prompts" / name




DEFAULT_MERGE_PARENT_WIN_VALIDATION_MAX = 8
DEFAULT_MERGE_RISK_SAMPLE_MAX = 6


class MergeConflictResolutionFailed(RuntimeError):
    """A merge candidate could not be materialized by git or Cursor."""


@dataclass
class MergePipelineConfig:
    base: PipelineConfig
    primary_parent_id: str
    secondary_parent_id: str
    # N-way merge: additional complementary specialists to fold in after the first
    # secondary (empty => classic pairwise behavior). Resolved via the deterministic
    # skill-union for additive-skill conflicts.
    extra_secondary_parent_ids: list = field(default_factory=list)
    pipeline_id_override: str | None = None
    max_repair_iters: int = 4


@dataclass
class _RejectedMergeCandidate:
    label: str
    result: eval_runner.EvalResult
    commit_sha: str
    validation_metadata: dict | None


def _make_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


class NodeMergePipeline:
    """Merge two parent nodes into one first-class child node."""

    def __init__(self, cfg: MergePipelineConfig) -> None:
        self.cfg = cfg
        self.base = cfg.base
        self.pipeline_id = cfg.pipeline_id_override or tree.new_id()
        self.db_path = tree.db_path_for(self.base.reports_root, self.base.campaign)
        self.conn = tree.connect(self.db_path)
        self.pipeline_log_dir = tree.pipeline_dir(
            self.base.reports_root, self.base.campaign, self.pipeline_id,
        )
        self.pipeline_log_dir.mkdir(parents=True, exist_ok=True)
        self.cursor_log_dir = self.pipeline_log_dir / "cursor"
        self.cursor_log_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir_path = tree.prompts_dir(
            self.base.reports_root, self.base.campaign, self.pipeline_id,
        )
        self.prompts_dir_path.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.pipeline_log_dir / "run.log"
        self.log = _make_logger(f"selfmerge.{self.pipeline_id}", self.run_log_path)
        self.primary: tree.Node | None = None
        self.secondary: tree.Node | None = None
        self.extra_secondaries: list[tree.Node] = []
        self.child: tree.Node | None = None
        self.worktree: worktree.Worktree | None = None
        self.pr_url: str | None = None
        self.final_score: float | None = None
        self.repair_notes: list[str] = []
        self.best_rejected: _RejectedMergeCandidate | None = None
        self.merge_contract_json: str | None = None
        self._rejection_history: list[dict[str, object]] = []
        self._last_rejection: dict[str, object] | None = None

    def run(self) -> int:
        try:
            self._register_pipeline()
            self._load_parents()
            self._create_worktree_and_child()
            self._preflight_cursor_model()
            self._create_intermediate_pr()
            self._merge_secondary_into_child()
            self._merge_extra_secondaries()
            accepted = self._evaluate_and_repair()
            if accepted:
                self._merge_intermediate_pr()
                self._mark_done("done")
            else:
                self._mark_done("no_change")
            return 0
        except MergeConflictResolutionFailed as e:
            self.log.warning("merge pipeline skipped: %s", e)
            try:
                if self.child:
                    self._record_note({"merge_skipped": str(e)})
                    if self.best_rejected is not None:
                        self._persist_best_rejected_candidate()
                if self.child and self.child.status == "in_progress":
                    tree.update_node(self.conn, self.child.id, status="no_change")
                tree.update_pipeline(
                    self.conn, self.pipeline_id,
                    status="no_change", finished_at=tree.utcnow_iso(),
                )
            except Exception:
                self.log.warning("failed to mark skipped merge", exc_info=True)
                return 1
            return 0
        except Exception as e:
            self.log.exception("merge pipeline failed: %s", e)
            try:
                if self.child and self.child.status == "in_progress":
                    tree.update_node(self.conn, self.child.id, status="failed")
                tree.update_pipeline(
                    self.conn, self.pipeline_id,
                    status="failed", finished_at=tree.utcnow_iso(),
                )
            except Exception:
                pass
            return 1
        finally:
            try:
                if self.base.cleanup_worktree and self.worktree is not None:
                    removed = worktree.remove_eval_worktree(
                        self.worktree.eval_dir, repo_root=self.base.repo_root,
                    )
                    if not removed:
                        self.log.warning(
                            "worktree cleanup fell back or timed out for %s",
                            self.worktree.eval_dir,
                        )
                    deleted = worktree.delete_eval_branch(
                        self.worktree.eval_branch, repo_root=self.base.repo_root,
                    )
                    if not deleted:
                        self.log.warning(
                            "eval branch cleanup did not delete %s",
                            self.worktree.eval_branch,
                        )
            except Exception:
                self.log.warning("worktree cleanup failed", exc_info=True)
            self.conn.close()

    def _register_pipeline(self) -> None:
        tree.insert_pipeline(
            self.conn,
            id=self.pipeline_id,
            campaign=self.base.campaign,
            parent_node_id=self.cfg.primary_parent_id,
            log_path=str(self.run_log_path),
            worktree_path=None,
            pid=os.getpid(),
        )
        tree.update_pipeline(self.conn, self.pipeline_id, status="preparing")

    def _load_parents(self) -> None:
        primary = tree.get_node(self.conn, self.cfg.primary_parent_id)
        secondary = tree.get_node(self.conn, self.cfg.secondary_parent_id)
        if primary is None or secondary is None:
            raise RuntimeError("merge parent id not found")
        if primary.parent_id is None or secondary.parent_id is None:
            raise RuntimeError("root nodes cannot be merge parents")
        if primary.score is None or secondary.score is None:
            raise RuntimeError("merge parents must be scored")
        self.primary = primary
        self.secondary = secondary
        self.extra_secondaries = []
        for nid in (self.cfg.extra_secondary_parent_ids or []):
            n = tree.get_node(self.conn, nid)
            if n is None or n.parent_id is None or n.score is None:
                self.log.warning("skipping invalid extra merge parent %s", nid)
                continue
            self.extra_secondaries.append(n)
        self.log.info(
            "merge parents primary=%s score=%s secondary=%s score=%s extras=%s",
            primary.id, primary.score, secondary.id, secondary.score,
            [e.id for e in self.extra_secondaries],
        )

    def _create_worktree_and_child(self) -> None:
        assert self.primary and self.secondary and self.primary.commit_sha
        self.worktree = worktree.add_eval_worktree(
            pipeline_id=self.pipeline_id,
            parent_commit=self.primary.commit_sha,
            repo_root=self.base.repo_root,
        )
        tree.update_pipeline(
            self.conn, self.pipeline_id, worktree_path=str(self.worktree.eval_dir),
        )
        child_id = tree.new_id()
        tree.insert_node(
            self.conn,
            id=child_id,
            campaign=self.base.campaign,
            branch_name=self.worktree.monet_branch,
            commit_sha=self.primary.commit_sha,
            parent_id=self.primary.id,
            subset=self.base.subset_label,
            status="in_progress",
            pipeline_id=self.pipeline_id,
            commits=[],
        )
        tree.insert_node_edge(
            self.conn,
            campaign=self.base.campaign,
            parent_id=self.primary.id,
            child_id=child_id,
            edge_type="merge",
            parent_role="primary",
            pipeline_id=self.pipeline_id,
        )
        tree.insert_node_edge(
            self.conn,
            campaign=self.base.campaign,
            parent_id=self.secondary.id,
            child_id=child_id,
            edge_type="merge",
            parent_role="secondary",
            pipeline_id=self.pipeline_id,
        )
        tree.update_pipeline(self.conn, self.pipeline_id, child_node_id=child_id)
        node_dir = tree.node_dir(self.base.reports_root, self.base.campaign, child_id)
        node_dir.mkdir(parents=True, exist_ok=True)
        works_path = node_dir / "works.md"
        effort_path = node_dir / "effort.md"
        works_path.write_text(
            f"# Merge node `{child_id}`\n\n"
            f"- primary parent: `{self.primary.id}` score={self.primary.score}\n"
            f"- secondary parent: `{self.secondary.id}` score={self.secondary.score}\n"
            f"- branch: `{self.worktree.monet_branch}`\n"
            f"- pipeline: `{self.pipeline_id}`\n"
            f"- run log: `{self.run_log_path}`\n"
        )
        effort_path.write_text(
            f"# Merge effort: {self.worktree.monet_branch}\n\n"
            f"Pipeline `{self.pipeline_id}` started at {tree.utcnow_iso()}.\n"
        )
        tree.update_node(
            self.conn, child_id,
            works_md_path=str(works_path),
            effort_md_path=str(effort_path),
        )
        self.child = tree.get_node(self.conn, child_id)
        self._record_note({
            "merge_artifacts": {
                "pipeline_id": self.pipeline_id,
                "run_log_path": str(self.run_log_path),
                "cursor_log_dir": str(self.cursor_log_dir),
                "prompts_dir": str(self.prompts_dir_path),
            }
        })

    def _preflight_cursor_model(self) -> None:
        # Best-effort note on the proposer model the merge will use. Only the
        # cursor backend can be probed via `cursor-agent models`; for monet_code /
        # claude_code we just record the active backend + model (those backends
        # validate themselves per-call — claude_code preflights its cc_setup
        # translator on the first agent call, with actionable guidance on failure).
        backend = self.base.meta_backend
        if backend != "cursor":
            self._record_note({
                "merge_meta_preflight": {
                    "ok": True,
                    "backend": backend,
                    "model": self.base.meta_model,
                    "reasoning_effort": self.base.meta_effort,
                }
            })
            return
        # `cursor-agent models` lists fully-qualified reasoning/context variants
        # (e.g. `gpt-5.5-extra-high`) but accepts bare family slugs like
        # `gpt-5.5` via --model and resolves them against the user's
        # account-level preference. So a slug missing from the listing is
        # not a launch blocker; we just record what we see.
        try:
            models = run_config._available_cursor_models()
        except Exception:
            models = {}
        listed = self.base.cursor_model in models if models else None
        self._record_note({
            "merge_cursor_preflight": {
                "ok": True,
                "model": self.base.cursor_model,
                "listed_in_advertised_models": listed,
            }
        })

    def _create_intermediate_pr(self) -> None:
        assert self.worktree and self.primary and self.secondary and self.child
        # Best effort: PR creation is external state, but the local merge path
        # still creates the child node if GitHub is temporarily unavailable.
        self._ensure_remote_branch(
            branch=self.worktree.monet_branch,
            commit_sha=worktree.head_sha(self.worktree.monet_dir),
        )
        self._ensure_remote_branch(
            branch=self.secondary.branch_name,
            commit_sha=self.secondary.commit_sha,
        )
        body = (
            f"Intermediate self-evolve merge PR.\n\n"
            f"- pipeline: `{self.pipeline_id}`\n"
            f"- child node: `{self.child.id}`\n"
            f"- primary parent: `{self.primary.id}`\n"
            f"- secondary parent: `{self.secondary.id}`\n"
        )
        cmd = [
            "gh", "pr", "create",
            "--base", self.worktree.monet_branch,
            "--head", self.secondary.branch_name,
            "--title", f"[self-evolve merge] {self.secondary.id} into {self.child.id}",
            "--body", body,
        ]
        if self.base.monet_repo_url:
            cmd[2:2] = ["--repo", self.base.monet_repo_url]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.worktree.monet_dir),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if proc.returncode == 0:
                self.pr_url = (proc.stdout or "").strip().splitlines()[-1]
                self.log.info("intermediate PR: %s", self.pr_url)
                self._record_note({"merge_pr_url": self.pr_url})
            else:
                self.log.warning("gh pr create failed: %s", proc.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.log.warning("gh pr create unavailable: %s", e)

    def _merge_secondary_into_child(self) -> None:
        assert self.worktree and self.secondary
        target = self.secondary.commit_sha or self.secondary.branch_name
        proc = _git(
            ["merge", "--no-edit", target],
            cwd=self.worktree.monet_dir,
            check=False,
        )
        self.merge_contract_json = self._merge_contract_json(
            git_stdout=proc.stdout or "",
            git_stderr=proc.stderr or "",
        )
        self._record_note({
            "merge_contract": {
                "path": str(self.prompts_dir_path / "merge_contract.json"),
                "primary_parent": self.primary.id if self.primary else None,
                "secondary_parent": self.secondary.id,
            }
        })
        (self.prompts_dir_path / "merge_contract.json").write_text(self.merge_contract_json)
        if proc.returncode == 0:
            self.log.info("git merge %s succeeded cleanly", target)
            self._record_merge_commit("clean merge")
            return
        # Deterministic additive-skill union: if the ONLY conflicts are in the skill
        # registry, resolve by UNIONING the skill objects (clean, valid JS) rather
        # than the LLM diff-merge (which can corrupt multi-line promptTemplate
        # literals). Falls through to Cursor for any code/other conflict.
        if self._try_skill_union_resolution():
            return
        self.log.warning("git merge had conflicts/regression risk; invoking Cursor")
        prompt = cursor_agent.render_prompt(
            self._template_path("merge_implement.md"),
            {
                "pipeline_id": self.pipeline_id,
                "wt_dir": str(self.worktree.eval_dir),
                "monet_branch": self.worktree.monet_branch,
                "primary_parent": self.primary.id if self.primary else "?",
                "secondary_parent": self.secondary.id,
                "secondary_commit": self.secondary.commit_sha,
                "git_stdout": proc.stdout or "",
                "git_stderr": proc.stderr or "",
                "merge_contract_json": self.merge_contract_json,
            },
        )
        try:
            from . import preserve_extend as _pe
            _targets = sorted(set(self.primary.failed_tasks) & set(self.secondary.failed_tasks))
            if os.environ.get("DARWINX_GATE_SPECIALIST_CONTRACT", "").strip().lower() in ("1", "true", "yes", "on"):
                # Tiered specialist contract: preserve CORE, allow net-positive
                # PERIPHERY trades. Tier from campaign-local per-task history.
                _stats = tree.task_outcome_stats(
                    self.conn, campaign=self.base.campaign, subset=self.base.subset_label,
                )
                _core, _peri, _ev = _pe.tier_tasks(sorted(self._lineage_invariant()), _stats)
                _contract = _pe.build_specialist_contract(
                    _core, _ev, _targets, kind="recombination",
                )
            else:
                _contract = _pe.build_contract(
                    sorted(self._lineage_invariant()),
                    _targets,
                    kind="recombination",
                )
            if _contract:
                prompt = _contract + "\n" + prompt
        except Exception:
            pass
        self._save_prompt("merge_implement", prompt)
        log_path = self.cursor_log_dir / "merge_implement.log"
        result = meta_agent.run(
            prompt,
            workspace=self.worktree.eval_dir,
            log_path=log_path,
            model=self.base.meta_model,
            timeout_s=self.base.implement_timeout_s,
            reasoning_effort=self.base.meta_effort,
        )
        if result.error:
            summary = self._record_cursor_failure("merge_implement", log_path, result)
            raise MergeConflictResolutionFailed(
                f"merge conflict resolution failed: {summary}",
            )
        self._record_merge_commit("cursor merge resolution")

    def _merge_extra_secondaries(self) -> None:
        """N-way: fold each additional complementary specialist into the child,
        reusing the same git-merge + deterministic skill-union path."""
        # The fold loop overwrites ``self.secondary``; capture the ORIGINAL so the
        # acceptance gate checks the union of ALL merged parents' wins (primary +
        # original secondary + every extra), not just primary+last-extra — else an
        # earlier specialist's wins can be silently dropped yet still accepted.
        if getattr(self, "_original_secondary", None) is None:
            self._original_secondary = self.secondary
        for extra in self.extra_secondaries:
            tree.insert_node_edge(
                self.conn, campaign=self.base.campaign,
                parent_id=extra.id, child_id=self.child.id if self.child else "",
                edge_type="merge", parent_role="secondary", pipeline_id=self.pipeline_id,
            )
            self.secondary = extra
            self.log.info("N-way: folding extra secondary %s", extra.id)
            self._merge_secondary_into_child()

    def _try_skill_union_resolution(self) -> bool:
        """Resolve a merge conflict deterministically IFF it is confined to the
        additive-skill registry, by unioning the skill objects. Returns True on a
        clean resolution (and commits it); False to defer to the LLM resolver."""
        SKILLS = "src/core/bundled-skills.js"
        TEST = "tests/bundled-skills.test.js"
        try:
            u = _git(["diff", "--name-only", "--diff-filter=U"],
                     cwd=self.worktree.monet_dir, check=False)
            conflicted = [f for f in (u.stdout or "").split() if f]
            if not conflicted or any(f not in (SKILLS, TEST) for f in conflicted):
                return False
            from . import skill_union
            base_txt = _git(["show", f"{self.primary.commit_sha}:{SKILLS}"],
                            cwd=self.worktree.monet_dir, check=False).stdout or ""
            sec_txt = _git(["show", f"{self.secondary.commit_sha}:{SKILLS}"],
                           cwd=self.worktree.monet_dir, check=False).stdout or ""
            merged, added = skill_union.union_bundled_skills(base_txt, sec_txt)
            (Path(self.worktree.monet_dir) / SKILLS).write_text(merged)
            _git(["add", "--", SKILLS], cwd=self.worktree.monet_dir, check=False)
            if TEST in conflicted:
                # keep primary's tests (additive skills do not need the secondary's)
                _git(["checkout", "--ours", "--", TEST], cwd=self.worktree.monet_dir, check=False)
                _git(["add", "--", TEST], cwd=self.worktree.monet_dir, check=False)
            self.merge_contract_json = self._merge_contract_json()
            (self.prompts_dir_path / "merge_contract.json").write_text(self.merge_contract_json)
            self.log.info("deterministic skill-union resolved conflict (+%d skills: %s)",
                          len(added), added)
            self._record_merge_commit("deterministic skill-union")
            return True
        except Exception as exc:
            self.log.warning("skill-union resolution failed (%s); deferring to LLM", exc)
            return False

    def _merge_contract_json(self, *, git_stdout: str = "", git_stderr: str = "") -> str:
        assert self.primary and self.secondary and self.worktree
        primary_solved = set(self.primary.solved_tasks)
        secondary_solved = set(self.secondary.solved_tasks)
        base = self._merge_base_sha()
        primary_changed = self._changed_files(base, self.primary.commit_sha)
        secondary_changed = self._changed_files(base, self.secondary.commit_sha)
        solved_union = primary_solved | secondary_solved
        payload = {
            # PRESERVE the full lineage invariant (both/all parents + extras + root
            # known-solved), not just the two parents' union -- this is the unified
            # preserve-extend discipline that stops "additive" bundles from regressing
            # tasks outside the parents.
            "must_preserve": sorted(self._lineage_invariant()),
            "primary_unique_wins": sorted(primary_solved - secondary_solved),
            "secondary_unique_wins": sorted(secondary_solved - primary_solved),
            "both_parent_wins": sorted(primary_solved & secondary_solved),
            "both_parent_failures": sorted(
                set(self.primary.failed_tasks) & set(self.secondary.failed_tasks)
            ),
            "focus_batches": {
                "primary": self._node_focus_batch(self.primary),
                "secondary": self._node_focus_batch(self.secondary),
            },
            "coverage_tags": {
                task: self._coverage_tags_for_task(task, primary_changed, secondary_changed)
                for task in sorted(solved_union)
            },
            "known_parent_regressions": {
                "primary": self.primary.regressed_tasks,
                "secondary": self.secondary.regressed_tasks,
            },
            "changed_files": {
                "primary": primary_changed,
                "secondary": secondary_changed,
                "overlap": sorted(set(primary_changed) & set(secondary_changed)),
            },
            "eval_logs": {
                "primary": self.primary.job_log_path,
                "secondary": self.secondary.job_log_path,
            },
            "git_merge_output": {
                "stdout": git_stdout.strip(),
                "stderr": git_stderr.strip(),
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _merge_base_sha(self) -> str | None:
        assert self.primary and self.secondary and self.worktree
        if not self.primary.commit_sha or not self.secondary.commit_sha:
            return None
        proc = _git(
            ["merge-base", self.primary.commit_sha, self.secondary.commit_sha],
            cwd=self.worktree.monet_dir,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return (proc.stdout or "").strip() or None

    def _changed_files(self, base: str | None, head: str | None) -> list[str]:
        assert self.worktree
        if not base or not head:
            return []
        proc = _git(
            ["diff", "--name-only", f"{base}..{head}"],
            cwd=self.worktree.monet_dir,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return sorted(line.strip() for line in (proc.stdout or "").splitlines() if line.strip())

    def _node_focus_batch(self, node: tree.Node) -> list[str]:
        tasks = list(node.claimed_task_scores.keys())
        for item in node.resolved_tasks:
            if isinstance(item, dict) and isinstance(item.get("task"), str):
                tasks.append(item["task"])
        tasks.extend(node.improved_tasks)
        return list(dict.fromkeys(tasks))[:4]

    def _coverage_tags_for_task(
        self,
        task: str,
        primary_changed: list[str],
        secondary_changed: list[str],
    ) -> dict[str, object]:
        stats = tree.task_outcome_stats(self.conn, campaign=self.base.campaign, subset=self.base.subset_label).get(task)
        changed = sorted(set(primary_changed) | set(secondary_changed))
        return {
            "family": task.split("-", 1)[0],
            "historical_passes": stats.passes if stats else 0,
            "historical_failures": stats.failures if stats else 0,
            "historical_regressions": stats.regressions if stats else 0,
            "changed_file_roots": sorted({p.split("/", 1)[0] for p in changed})[:6],
        }

    def _evaluate_and_repair(self) -> bool:
        if self._evaluate_current_candidate("merge_final"):
            return True
        for i in range(1, self.cfg.max_repair_iters + 1):
            self._restore_best_rejected_before_repair(i)
            before_repair = self._last_rejection
            if not self._analyze_regression(i):
                break
            self._repair_iteration(i)
            if self._evaluate_current_candidate(f"merge_repair_{i}"):
                return True
            if self._should_stop_repair_after_rejection(i, before_repair):
                break
        self._persist_best_rejected_candidate()
        self._write_notes()
        return False

    def _evaluate_current_candidate(self, label: str) -> bool:
        validation = self._run_merge_validation(label)
        validation_metadata = self._validation_metadata(validation) if validation else None
        if validation is not None and not self._validation_preserves_parent_wins(validation):
            snapshot = self._record_rejection_snapshot(
                label,
                validation,
                validation_only=True,
            )
            self.repair_notes.append(
                f"{label}: rejected before broad eval because parent-win validation failed "
                f"on {self._lost_validation_parent_wins(validation)}."
            )
            self._record_note({"merge_rejection": snapshot})
            self._persist_validation_reject(validation, validation_metadata=validation_metadata)
            return False

        final = self._run_final_eval(label)
        if self._acceptable_completed_merge(final):
            self._persist_eval(final, status="completed", validation_metadata=validation_metadata)
            return True

        self.repair_notes.append(
            f"{label}: rejected because score={final.score} did not beat both parents "
            f"or lost {self._parent_win_loss_count(final)} parent win(s)."
        )
        snapshot = self._record_rejection_snapshot(
            label,
            final,
            validation_only=False,
        )
        self._record_note({"merge_rejection": snapshot})
        self._remember_rejected_candidate(label, final, validation_metadata)
        self._persist_eval(final, status="rejected", validation_metadata=validation_metadata)
        return False

    def _run_merge_validation(self, label: str) -> eval_runner.EvalResult | None:
        assert self.worktree
        tasks = self._merge_validation_tasks()
        if not tasks:
            return None
        self.log.info("merge validation tasks for %s: %s", label, tasks)
        return eval_runner.run_subset(
            config_path=self.base.config_path,
            cwd=self.worktree.eval_dir,
            task_names=tasks,
            job_name=f"{label}_validation_{self.pipeline_id}",
        )

    def _merge_validation_tasks(self) -> list[str]:
        assert self.primary and self.secondary
        parent_wins = set(self.primary.solved_tasks) | set(self.secondary.solved_tasks)
        unique_wins = (
            (set(self.primary.solved_tasks) - set(self.secondary.solved_tasks)) |
            (set(self.secondary.solved_tasks) - set(self.primary.solved_tasks))
        )
        both_wins = set(self.primary.solved_tasks) & set(self.secondary.solved_tasks)
        stats = tree.task_outcome_stats(self.conn, campaign=self.base.campaign, subset=self.base.subset_label)
        max_parent_wins = _env_int("DARWINX_EVAL_MERGE_PARENT_WIN_VALIDATION_MAX", DEFAULT_MERGE_PARENT_WIN_VALIDATION_MAX)
        max_canaries = _env_int("DARWINX_EVAL_MERGE_RISK_SAMPLE_MAX", DEFAULT_MERGE_RISK_SAMPLE_MAX)

        selected: list[str] = []
        for task in sorted(unique_wins, key=lambda t: _task_validation_priority(t, stats), reverse=True):
            if len(selected) >= max_parent_wins:
                break
            selected.append(task)
        for task in sorted(both_wins, key=lambda t: _task_validation_priority(t, stats), reverse=True):
            if len(selected) >= max_parent_wins:
                break
            if task not in selected:
                selected.append(task)

        # Also evaluate a bounded sample of the broader LINEAGE INVARIANT (root +
        # N-way extras' solved) beyond the two parents' wins -- otherwise the
        # lineage-invariant preserve check never actually inspects those tasks and
        # an "additive" merge could regress them undetected. Capped to bound cost.
        invariant_extra = sorted(
            (self._lineage_invariant() - parent_wins - set(selected)),
            key=lambda t: _task_validation_priority(t, stats), reverse=True,
        )
        selected.extend(invariant_extra[:max_parent_wins])

        canary_pool = [
            task for task in (self.base.subset_tasks or [])
            if task not in parent_wins and task not in selected
        ]
        canary_pool.sort(key=lambda t: (_task_validation_priority(t, stats), t), reverse=True)
        selected.extend(canary_pool[:max_canaries])
        return selected

    def _validation_preserves_parent_wins(self, result: eval_runner.EvalResult) -> bool:
        return not self._lost_validation_parent_wins(result)

    def _lineage_invariant(self) -> set:
        """The known-solved INVARIANT the merged child must preserve: the union of
        all merge parents' (primary + secondary + N-way extras) solved tasks plus
        the root's solved set. Unifies the preservation surface with the mutation
        path (which preserves the parent's solved set)."""
        from . import preserve_extend
        inv = preserve_extend.union_solved(
            self.primary, self.secondary, *getattr(self, "extra_secondaries", []),
        )
        try:
            root_ev = tree.root_full_eval(self.conn, campaign=self.base.campaign)
            if root_ev is not None:
                inv |= set(getattr(root_ev, "solved_tasks", None) or [])
        except Exception:
            pass
        return inv

    def _lost_validation_parent_wins(self, result: eval_runner.EvalResult) -> list[str]:
        assert self.primary and self.secondary
        from . import preserve_extend
        # Check against the FULL lineage invariant, intersected with the tasks
        # actually evaluated (an unevaluated invariant task is not "lost").
        invariant = self._lineage_invariant() & set(result.task_names)
        return preserve_extend.preserved(result.solved_tasks, invariant)

    def _validation_metadata(self, result: eval_runner.EvalResult) -> dict:
        assert self.primary and self.secondary
        tags = sorted({task.split("-", 1)[0] for task in result.task_names})
        full_subset_task_count = len(self.base.subset_tasks) if self.base.subset_tasks else None
        parent_wins = set(self.primary.solved_tasks) | set(self.secondary.solved_tasks)
        validated_parent_wins = parent_wins & set(result.task_names)
        lost_validated_parent_wins = self._lost_validation_parent_wins(result)
        return {
            "validated_tasks": list(result.task_names),
            "validation_stage": "parent_win_and_risk_sample",
            "validation_score": result.score,
            "validation_n_trials": result.n_trials,
            "score_basis": "merge_validation",
            "full_subset_task_count": full_subset_task_count,
            "coverage_tags_checked": tags,
            "confidence_level": "sampled" if full_subset_task_count and len(result.task_names) < full_subset_task_count else "full_subset",
            "audit_required": bool(full_subset_task_count and len(result.task_names) < full_subset_task_count),
            "validated_parent_wins": sorted(validated_parent_wins),
            "lost_validated_parent_wins": lost_validated_parent_wins,
            "unvalidated_parent_wins": sorted(parent_wins - validated_parent_wins),
            # Backward-compatible key for existing visualizer/report consumers.
            "lost_parent_wins": lost_validated_parent_wins,
        }

    def _run_final_eval(self, label: str) -> eval_runner.EvalResult:
        assert self.worktree
        tree.update_pipeline(self.conn, self.pipeline_id, status="eval")
        return eval_runner.run_full(
            config_path=self.base.config_path,
            cwd=self.worktree.eval_dir,
            subset=self.base.subset_label,
            task_names=self.base.subset_tasks,
            job_name=f"{label}_{self.pipeline_id}",
            extra_args=(
                _final_eval_extra_args(self.base.config_path)
                if self.base.subset_label == "full" else []
            ),
        )

    def _persist_eval(
        self,
        result: eval_runner.EvalResult,
        *,
        status: str,
        validation_metadata: dict | None = None,
        best_rejected_label: str | None = None,
    ) -> None:
        assert self.child and self.worktree
        latest = worktree.head_sha(self.worktree.monet_dir)
        commits = (
            worktree.commits_since(self.worktree.monet_dir, self.primary.commit_sha)
            if self.primary else []
        )
        solved_tasks = list(result.solved_tasks)
        unsolved_tasks = list(result.unsolved_tasks)
        partially_solved_tasks = list(result.partially_solved_tasks)
        failed_tasks = list(result.failed_task_names)
        improved_tasks: list[str] = []
        regressed_tasks: list[str] = []
        if self.primary:
            improved_tasks, regressed_tasks = tree.task_deltas(
                parent_solved=self.primary.solved_tasks,
                parent_unsolved=self.primary.failed_tasks,
                child_solved=solved_tasks,
                child_unsolved=failed_tasks,
            )
        archived_job_dir = self._archive_final_eval_job_if_small(result.job_dir)
        result.job_dir = archived_job_dir
        resolved_tasks = self._merge_resolved_metadata(
            solved_tasks=solved_tasks,
            validation_metadata=validation_metadata,
            validation_only=False,
            best_rejected_label=best_rejected_label,
        )
        tree.update_node(
            self.conn,
            self.child.id,
            commit_sha=latest,
            commits_json=list(reversed(commits)),
            score=result.score,
            subset=self.base.subset_label,
            job_log_path=str(archived_job_dir),
            failed_tasks_json=failed_tasks,
            solved_tasks_json=solved_tasks,
            unsolved_tasks_json=unsolved_tasks,
            partially_solved_tasks_json=partially_solved_tasks,
            improved_tasks_json=improved_tasks,
            regressed_tasks_json=regressed_tasks,
            resolved_tasks_json=resolved_tasks,
            status=status,
        )
        parent_eval = (
            tree.node_search_eval(
                self.conn, campaign=self.base.campaign, node_id=self.primary.id,
            )
            if self.primary else None
        )
        node_eval = tree.upsert_node_eval(
            self.conn,
            campaign=self.base.campaign,
            node_id=self.child.id,
            eval_kind="subset_final",
            subset_label=self.base.subset_label,
            task_names=list(result.task_names),
            n_trials=result.n_trials,
            n_errors=result.n_errors,
            score=result.score,
            job_log_path=str(archived_job_dir),
            solved_tasks=solved_tasks,
            unsolved_tasks=unsolved_tasks,
            partially_solved_tasks=partially_solved_tasks,
            task_rewards=dict(result.task_rewards),
            improved_tasks=improved_tasks,
            regressed_tasks=regressed_tasks,
            source_pipeline_id=self.pipeline_id,
            metadata={"basis": "merge_subset_final"},
        )
        self._record_experiences_from_eval(node_eval, parent_eval)
        self.child = tree.get_node(self.conn, self.child.id)
        self.final_score = result.score
        self._write_notes()

    def _persist_validation_reject(
        self,
        result: eval_runner.EvalResult,
        *,
        validation_metadata: dict | None,
    ) -> None:
        """Persist sampled validation metadata without making it canonical task state."""
        assert self.child and self.worktree
        latest = worktree.head_sha(self.worktree.monet_dir)
        commits = (
            worktree.commits_since(self.worktree.monet_dir, self.primary.commit_sha)
            if self.primary else []
        )
        archived_job_dir = self._archive_final_eval_job_if_small(result.job_dir)
        validation_solved = list(result.solved_tasks)
        tree.upsert_node_eval(
            self.conn,
            campaign=self.base.campaign,
            node_id=self.child.id,
            eval_kind="merge_validation",
            subset_label=self.base.subset_label,
            task_names=list(result.task_names),
            n_trials=result.n_trials,
            n_errors=result.n_errors,
            score=result.score,
            job_log_path=str(archived_job_dir),
            solved_tasks=list(result.solved_tasks),
            unsolved_tasks=list(result.unsolved_tasks),
            partially_solved_tasks=list(result.partially_solved_tasks),
            task_rewards=dict(result.task_rewards),
            source_pipeline_id=self.pipeline_id,
            metadata=validation_metadata or {},
        )
        resolved_tasks = self._merge_resolved_metadata(
            solved_tasks=validation_solved,
            validation_metadata=validation_metadata,
            validation_only=True,
            best_rejected_label=None,
        )
        tree.update_node(
            self.conn,
            self.child.id,
            commit_sha=latest,
            commits_json=list(reversed(commits)),
            score=result.score,
            subset=self.base.subset_label,
            job_log_path=str(archived_job_dir),
            resolved_tasks_json=resolved_tasks,
            status="rejected",
        )
        self.child = tree.get_node(self.conn, self.child.id)
        self.final_score = result.score
        self._write_notes()

    def _archive_final_eval_job_if_small(self, job_dir: Path) -> Path:
        if not self.child:
            return job_dir
        job_dir = Path(job_dir)
        if not job_dir.is_dir():
            return job_dir
        max_bytes = _final_eval_archive_max_bytes()
        size = _dir_size_bytes(job_dir, max_bytes=max_bytes if max_bytes > 0 else None)
        if max_bytes > 0 and size > max_bytes:
            self.log.info(
                "not archiving merge eval job %s: %.1f MiB exceeds cap %.1f MiB",
                job_dir,
                size / (1024 * 1024),
                max_bytes / (1024 * 1024),
            )
            return job_dir
        node_dir = tree.node_dir(self.base.reports_root, self.base.campaign, self.child.id)
        dest = node_dir / "evals" / job_dir.name
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(job_dir, dest, symlinks=True)
            self.log.info(
                "archived merge eval job logs: %s -> %s (%.1f MiB)",
                job_dir,
                dest,
                size / (1024 * 1024),
            )
            return dest
        except Exception:
            self.log.warning("could not archive merge eval job %s", job_dir, exc_info=True)
            return job_dir

    def _beats_both(self, score: float) -> bool:
        assert self.primary and self.secondary
        return score > (self.primary.score or 0.0) and score > (self.secondary.score or 0.0)

    def _acceptable_completed_merge(self, result: eval_runner.EvalResult) -> bool:
        # A merge is accepted iff it PRESERVES every task either parent solved
        # (no regression) AND strictly beats both parents in capability.
        #
        # We deliberately do NOT compare ``result.score`` to the parents'
        # scalar ``score`` (the old ``_beats_both`` check): under cluster-batch
        # each parent's nodes.score is measured on its OWN small claimed batch,
        # which is not comparable to the merged node's full-set eval score —
        # comparing them spuriously accepts (full ~0.80 > a 6-task 0.53) or
        # rejects. The scale-independent, correct definition of a successful
        # recombination is task-level union gain:
        #   (1) the merged child solves the entire union of both parents' wins
        #       (``_parent_win_loss_count == 0``), so it is >= each parent, AND
        #   (2) that union strictly exceeds each parent — i.e. each parent
        #       contributed >=1 unique win (complementary), so the merged child
        #       strictly beats both — OR the merge discovered a brand-new win
        #       neither parent had.
        if self._parent_win_loss_count(result) != 0:
            return False
        primary_unique = set(self.primary.solved_tasks) - set(self.secondary.solved_tasks)
        secondary_unique = set(self.secondary.solved_tasks) - set(self.primary.solved_tasks)
        strictly_beats_both = bool(primary_unique) and bool(secondary_unique)
        return strictly_beats_both or self._new_child_win_count(result) >= 1

    def _merge_result_is_better(
        self,
        candidate: eval_runner.EvalResult,
        incumbent: eval_runner.EvalResult,
    ) -> bool:
        candidate_key = (
            -self._parent_win_loss_count(candidate),
            candidate.score,
            self._new_child_win_count(candidate),
        )
        incumbent_key = (
            -self._parent_win_loss_count(incumbent),
            incumbent.score,
            self._new_child_win_count(incumbent),
        )
        return candidate_key > incumbent_key

    def _all_parent_solved(self) -> set:
        """Union of EVERY merged parent's wins: primary + the ORIGINAL secondary +
        all N-way extra secondaries. The fold loop overwrites ``self.secondary``,
        so relying on it alone would drop earlier parents' wins from the gate."""
        assert self.primary
        wins = set(self.primary.solved_tasks)
        sec0 = getattr(self, "_original_secondary", None) or self.secondary
        if sec0:
            wins |= set(sec0.solved_tasks)
        for extra in (getattr(self, "extra_secondaries", None) or []):
            wins |= set(extra.solved_tasks)
        return wins

    def _parent_win_loss_count(self, result: eval_runner.EvalResult) -> int:
        assert self.primary and self.secondary
        parent_wins = self._all_parent_solved()
        child_solved = set(result.solved_tasks)
        return len(parent_wins - child_solved)

    def _new_child_win_count(self, result: eval_runner.EvalResult) -> int:
        assert self.primary and self.secondary
        parent_wins = self._all_parent_solved()
        child_solved = set(result.solved_tasks)
        return len(child_solved - parent_wins)

    def _record_rejection_snapshot(
        self,
        label: str,
        result: eval_runner.EvalResult,
        *,
        validation_only: bool,
    ) -> dict[str, object]:
        if validation_only:
            parent_win_losses = len(self._lost_validation_parent_wins(result))
            assert self.primary and self.secondary
            parent_wins = set(self.primary.solved_tasks) | set(self.secondary.solved_tasks)
            validated = set(result.task_names)
            child_solved = set(result.solved_tasks)
            new_child_wins = len((child_solved - parent_wins) & validated)
        else:
            parent_win_losses = self._parent_win_loss_count(result)
            new_child_wins = self._new_child_win_count(result)
        snapshot: dict[str, object] = {
            "label": label,
            "score": result.score,
            "parent_win_losses": parent_win_losses,
            "new_child_wins": new_child_wins,
            "validation_only": validation_only,
        }
        self._rejection_history.append(snapshot)
        self._last_rejection = snapshot
        return snapshot

    def _should_stop_repair_after_rejection(
        self,
        repair_iteration: int,
        before_repair: dict[str, object] | None,
    ) -> bool:
        latest = self._last_rejection
        if not latest:
            return False
        latest_losses = int(latest.get("parent_win_losses") or 0)
        if latest_losses <= 0:
            return False
        if before_repair is None:
            return False
        before_losses = int(before_repair.get("parent_win_losses") or 0)
        before_new = int(before_repair.get("new_child_wins") or 0)
        latest_new = int(latest.get("new_child_wins") or 0)
        before_score = float(before_repair.get("score") or 0.0)
        latest_score = float(latest.get("score") or 0.0)
        non_improving = (
            latest_losses >= before_losses
            and latest_new <= before_new
            and latest_score <= before_score
        )
        if not non_improving:
            return False
        note = (
            f"repair {repair_iteration}: stopping repair loop because parent-win "
            f"losses persisted without score/new-win improvement "
            f"(before={before_repair}, after={latest})."
        )
        self.log.info(note)
        self.repair_notes.append(note)
        self._record_note({
            "merge_repair_early_stop": {
                "iteration": repair_iteration,
                "before": before_repair,
                "after": latest,
                "reason": "non_improving_parent_win_loss",
            }
        })
        return True

    def _remember_rejected_candidate(
        self,
        label: str,
        result: eval_runner.EvalResult,
        validation_metadata: dict | None,
    ) -> None:
        assert self.worktree
        commit_sha = worktree.head_sha(self.worktree.monet_dir)
        candidate = _RejectedMergeCandidate(
            label=label,
            result=result,
            commit_sha=commit_sha,
            validation_metadata=validation_metadata,
        )
        if self.best_rejected is None or self._merge_result_is_better(result, self.best_rejected.result):
            self.best_rejected = candidate
            self._record_note({
                "best_rejected_merge": {
                    "label": label,
                    "commit_sha": commit_sha,
                    "score": result.score,
                    "parent_win_losses": self._parent_win_loss_count(result),
                    "new_child_wins": self._new_child_win_count(result),
                }
            })

    def _restore_best_rejected_before_repair(self, iteration: int) -> None:
        if not self.best_rejected:
            return
        self.log.info(
            "restoring best rejected merge %s before repair %s",
            worktree.short_sha(self.best_rejected.commit_sha),
            iteration,
        )
        self._restore_best_merge_candidate(self.best_rejected.commit_sha)

    def _persist_best_rejected_candidate(self) -> None:
        if not self.best_rejected:
            return
        self._restore_best_merge_candidate(self.best_rejected.commit_sha)
        self._persist_eval(
            self.best_rejected.result,
            status="rejected",
            validation_metadata=self.best_rejected.validation_metadata,
            best_rejected_label=self.best_rejected.label,
        )

    def _merge_resolved_metadata(
        self,
        *,
        solved_tasks: list[str],
        validation_metadata: dict | None,
        validation_only: bool,
        best_rejected_label: str | None,
    ) -> list[dict]:
        assert self.primary and self.secondary and self.child
        primary_solved = set(self.primary.solved_tasks)
        secondary_solved = set(self.secondary.solved_tasks)
        parent_wins = primary_solved | secondary_solved
        child_solved = set(solved_tasks)
        parent_wins_considered = parent_wins
        if validation_only and validation_metadata:
            parent_wins_considered = parent_wins & set(validation_metadata.get("validated_tasks") or [])
        existing = [
            item for item in self.child.resolved_tasks
            if isinstance(item, dict)
            and "merge_delta" not in item
            and "merge_validation" not in item
            and "best_rejected_merge" not in item
        ]
        if validation_metadata is not None:
            existing.append({"merge_validation": validation_metadata})
        if best_rejected_label:
            existing.append({
                "best_rejected_merge": {
                    "label": best_rejected_label,
                    "commit_sha": self.best_rejected.commit_sha if self.best_rejected else None,
                    "score": self.best_rejected.result.score if self.best_rejected else None,
                }
            })
        existing.append({
            "merge_delta": {
                "validation_only": validation_only,
                "lost_parent_wins": sorted(parent_wins_considered - child_solved),
                "unvalidated_parent_wins": sorted(parent_wins - parent_wins_considered),
                "new_child_wins": sorted(child_solved - parent_wins),
                "lost_primary_wins": sorted((primary_solved & parent_wins_considered) - child_solved),
                "lost_secondary_wins": sorted((secondary_solved & parent_wins_considered) - child_solved),
                "preserved_both_parent_wins": sorted((primary_solved & secondary_solved) & child_solved),
            },
        })
        return existing

    def _restore_best_merge_candidate(self, best_commit: str) -> None:
        assert self.worktree
        current = worktree.head_sha(self.worktree.monet_dir)
        if current == best_commit:
            return
        proc = _git(
            ["revert", "--no-edit", f"{best_commit}..HEAD"],
            cwd=self.worktree.monet_dir,
            check=False,
        )
        if proc.returncode != 0:
            self.log.warning(
                "could not revert rejected merge repair to %s; hard-resetting local worktree: %s",
                worktree.short_sha(best_commit),
                proc.stderr,
            )
            worktree.reset_to(self.worktree.monet_dir, best_commit)
            return
        self._record_merge_commit("revert rejected merge repair")

    def _analyze_regression(self, i: int) -> bool:
        assert self.primary and self.secondary and self.child and self.worktree
        merge_context = self._repair_context_json()
        prompt = cursor_agent.render_prompt(
            self._template_path("merge_regression_analyze.md"),
            {
                "pipeline_id": self.pipeline_id,
                "iteration": i,
                "max_iters": self.cfg.max_repair_iters,
                "wt_dir": str(self.worktree.eval_dir),
                "primary_parent": self.primary.id,
                "secondary_parent": self.secondary.id,
                "primary_score": self.primary.score,
                "secondary_score": self.secondary.score,
                "child_score": self.child.score,
                "job_log_path": self.child.job_log_path,
                "primary_job_log_path": self.primary.job_log_path,
                "secondary_job_log_path": self.secondary.job_log_path,
                "parent_child_delta_json": self._parent_child_delta_json(),
                "merge_contract_json": self._current_merge_contract_json(),
                "merge_repair_context_json": merge_context,
            },
        )
        self._save_prompt(f"merge_repair_{i}_analyze", prompt)
        log_path = self.cursor_log_dir / f"merge_repair_{i}_analyze.log"
        result = meta_agent.run(
            prompt,
            workspace=self.worktree.eval_dir,
            log_path=log_path,
            model=self.base.meta_model,
            plan_mode=True,
            timeout_s=self.base.analyze_timeout_s,
            reasoning_effort=self.base.meta_effort,
        )
        if result.error:
            self._record_cursor_failure(f"merge_repair_{i}_analyze", log_path, result)
            return False
        self.repair_notes.append(f"repair {i} analysis:\n{result.text}")
        return "UNSOLVABLE" not in (result.text or "").upper()

    def _parent_child_delta_json(self) -> str:
        assert self.primary and self.secondary and self.child
        primary_solved = set(self.primary.solved_tasks)
        secondary_solved = set(self.secondary.solved_tasks)
        child_solved = set(self.child.solved_tasks)
        payload = {
            "primary_only_passed": sorted(primary_solved - secondary_solved),
            "secondary_only_passed": sorted(secondary_solved - primary_solved),
            "both_parents_passed": sorted(primary_solved & secondary_solved),
            "both_parents_failed": sorted(
                set(self.primary.failed_tasks) & set(self.secondary.failed_tasks)
            ),
            "child_lost_parent_wins": sorted((primary_solved | secondary_solved) - child_solved),
            "child_new_wins": sorted(child_solved - (primary_solved | secondary_solved)),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _current_merge_contract_json(self) -> str:
        if self.merge_contract_json is None:
            self.merge_contract_json = self._merge_contract_json()
        return self.merge_contract_json

    def _repair_context_json(self) -> str:
        assert self.primary and self.secondary and self.child
        latest_validation = None
        latest_delta = None
        for item in self.child.resolved_tasks:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("merge_validation"), dict):
                latest_validation = item["merge_validation"]
            if isinstance(item.get("merge_delta"), dict):
                latest_delta = item["merge_delta"]
        payload = {
            "pipeline_id": self.pipeline_id,
            "primary_parent": self.primary.id,
            "secondary_parent": self.secondary.id,
            "child": self.child.id,
            "latest_validation": latest_validation,
            "latest_delta": latest_delta,
            "best_rejected_merge": (
                {
                    "label": self.best_rejected.label,
                    "commit_sha": self.best_rejected.commit_sha,
                    "score": self.best_rejected.result.score,
                    "parent_win_losses": self._parent_win_loss_count(self.best_rejected.result),
                    "new_child_wins": self._new_child_win_count(self.best_rejected.result),
                }
                if self.best_rejected else None
            ),
            "repair_notes": self.repair_notes[-6:],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _repair_iteration(self, i: int) -> None:
        assert self.worktree
        analysis = self.repair_notes[-1] if self.repair_notes else ""
        merge_context = self._repair_context_json()
        plan_prompt = cursor_agent.render_prompt(
            self._template_path("merge_repair_plan.md"),
            {
                "pipeline_id": self.pipeline_id,
                "iteration": i,
                "max_iters": self.cfg.max_repair_iters,
                "wt_dir": str(self.worktree.eval_dir),
                "analysis": analysis,
                "merge_contract_json": self._current_merge_contract_json(),
                "merge_repair_context_json": merge_context,
            },
        )
        self._save_prompt(f"merge_repair_{i}_plan", plan_prompt)
        plan_log_path = self.cursor_log_dir / f"merge_repair_{i}_plan.log"
        plan_result = meta_agent.run(
            plan_prompt,
            workspace=self.worktree.eval_dir,
            log_path=plan_log_path,
            model=self.base.meta_model,
            plan_mode=True,
            timeout_s=self.base.analyze_timeout_s,
            reasoning_effort=self.base.meta_effort,
        )
        if plan_result.error:
            summary = self._record_cursor_failure(f"merge_repair_{i}_plan", plan_log_path, plan_result)
            raise MergeConflictResolutionFailed(
                f"repair {i} plan failed: {summary}",
            )
        plan_text = plan_result.text or ""
        if not plan_text.strip():
            raise MergeConflictResolutionFailed(
                f"repair {i} plan failed: empty plan",
            )
        self.repair_notes.append(f"repair {i} plan:\n{plan_text}")

        prompt = cursor_agent.render_prompt(
            self._template_path("merge_repair_implement.md"),
            {
                "pipeline_id": self.pipeline_id,
                "iteration": i,
                "max_iters": self.cfg.max_repair_iters,
                "wt_dir": str(self.worktree.eval_dir),
                "analysis": analysis,
                "plan_text": plan_text,
                "merge_contract_json": self._current_merge_contract_json(),
                "merge_repair_context_json": merge_context,
            },
        )
        self._save_prompt(f"merge_repair_{i}_implement", prompt)
        implement_log_path = self.cursor_log_dir / f"merge_repair_{i}_implement.log"
        result = meta_agent.run(
            prompt,
            workspace=self.worktree.eval_dir,
            log_path=implement_log_path,
            model=self.base.meta_model,
            timeout_s=self.base.implement_timeout_s,
            reasoning_effort=self.base.meta_effort,
        )
        if result.error:
            summary = self._record_cursor_failure(f"merge_repair_{i}_implement", implement_log_path, result)
            raise MergeConflictResolutionFailed(
                f"repair {i} failed: {summary}",
            )
        review_prompt = cursor_agent.render_prompt(
            self._template_path("merge_repair_review.md"),
            {
                "pipeline_id": self.pipeline_id,
                "iteration": i,
                "max_iters": self.cfg.max_repair_iters,
                "wt_dir": str(self.worktree.eval_dir),
                "analysis": analysis,
                "plan_text": plan_text,
                "merge_contract_json": self._current_merge_contract_json(),
                "merge_repair_context_json": merge_context,
            },
        )
        self._save_prompt(f"merge_repair_{i}_review", review_prompt)
        review_log_path = self.cursor_log_dir / f"merge_repair_{i}_review.log"
        review = meta_agent.run(
            review_prompt,
            workspace=self.worktree.eval_dir,
            log_path=review_log_path,
            model=self.base.meta_model,
            plan_mode=True,
            timeout_s=self.base.review_timeout_s,
            reasoning_effort=self.base.meta_effort,
        )
        if review.error:
            summary = self._record_cursor_failure(f"merge_repair_{i}_review", review_log_path, review)
            raise MergeConflictResolutionFailed(
                f"repair {i} self-review failed: {summary}",
            )
        self.repair_notes.append(f"repair {i} self-review:\n{review.text}")
        if _repair_review_blocks(review.text or ""):
            self._record_note({
                "merge_repair_blocked": {
                    "iteration": i,
                    "review": review.text,
                }
            })
            raise MergeConflictResolutionFailed(f"repair {i} blocked by self-review")
        self._record_merge_commit(f"merge regression repair {i}")

    def _merge_intermediate_pr(self) -> None:
        if not self.pr_url:
            return
        assert self.worktree
        self._push_branch(self.worktree.monet_branch)
        cmd = ["gh", "pr", "merge", self.pr_url, "--merge", "--delete-branch=false"]
        if self.base.monet_repo_url:
            cmd[2:2] = ["--repo", self.base.monet_repo_url]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.worktree.monet_dir),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if proc.returncode == 0:
                self._record_note({"merge_pr_merged": True, "merge_pr_url": self.pr_url})
            else:
                self.log.warning("gh pr merge failed: %s", proc.stderr)
                self._record_note({"merge_pr_merged": False, "merge_pr_url": self.pr_url})
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.log.warning("gh pr merge unavailable: %s", e)
            self._record_note({"merge_pr_merged": False, "merge_pr_url": self.pr_url})

    def _record_merge_commit(self, message: str) -> None:
        assert self.worktree and self.child
        before = set(self.child.commits)
        _git(["add", "-A"], cwd=self.worktree.monet_dir, check=True)
        diff = _git(["diff", "--cached", "--quiet"], cwd=self.worktree.monet_dir, check=False)
        if diff.returncode != 0:
            _git(["commit", "-m", message], cwd=self.worktree.monet_dir, check=True)
        latest = worktree.head_sha(self.worktree.monet_dir)
        commits = (
            worktree.commits_since(self.worktree.monet_dir, self.primary.commit_sha)
            if self.primary else [latest]
        )
        tree.update_node(
            self.conn, self.child.id,
            commit_sha=latest,
            commits_json=list(reversed(commits)),
        )
        self.child = tree.get_node(self.conn, self.child.id)
        added = [sha for sha in self.child.commits if sha not in before]
        self._record_note({
            "merge_commit": {
                "message": message,
                "head": latest,
                "commits": added,
            }
        })
        self._push_branch(self.worktree.monet_branch)

    def _push_branch(self, branch: str) -> None:
        assert self.worktree
        try:
            worktree.push_branch(self.worktree.monet_dir, branch)
        except Exception as e:
            self.log.warning("push branch %s failed: %s", branch, e)

    def _ensure_remote_branch(self, *, branch: str, commit_sha: str | None) -> bool:
        """Ensure a branch exists on origin, even if it is absent locally."""
        assert self.worktree
        remote_ref = f"refs/heads/{branch}"
        remote = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", remote_ref],
            cwd=str(self.worktree.monet_dir),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if remote.returncode == 0:
            return True

        local = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=str(self.worktree.monet_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if local.returncode == 0:
            self._push_branch(branch)
            return True

        if not commit_sha:
            self.log.warning("branch %s absent locally/remotely and no commit sha is known", branch)
            return False
        commit_exists = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
            cwd=str(self.worktree.monet_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if commit_exists.returncode != 0:
            self.log.warning(
                "branch %s absent and commit %s is not available",
                branch, worktree.short_sha(commit_sha),
            )
            return False
        pushed = subprocess.run(
            ["git", "push", "-u", "origin", f"{commit_sha}:{remote_ref}"],
            cwd=str(self.worktree.monet_dir),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if pushed.returncode != 0:
            self.log.warning("push branch %s via %s failed: %s", branch, worktree.short_sha(commit_sha), pushed.stderr)
            return False
        return True

    def _record_note(self, entry: dict) -> None:
        if not self.child:
            return
        resolved = self.child.resolved_tasks
        resolved.append(entry)
        tree.update_node(self.conn, self.child.id, resolved_tasks_json=resolved)
        self.child = tree.get_node(self.conn, self.child.id)

    def _record_experiences_from_eval(
        self,
        node_eval: tree.NodeEval,
        parent_eval: tree.NodeEval | None,
    ) -> None:
        if not self.child or parent_eval is None:
            return
        affected = (
            [("improved", task) for task in node_eval.improved_tasks]
            + [("regressed", task) for task in node_eval.regressed_tasks]
        )
        for kind, task in affected:
            try:
                tree.insert_task_experience(
                    self.conn,
                    campaign=self.base.campaign,
                    task=task,
                    node_id=self.child.id,
                    pipeline_id=self.pipeline_id,
                    worker_kind="merge",
                    commit_sha=self.child.commit_sha,
                    commit_number=None,
                    experience_kind=kind,
                    eval_kind=node_eval.eval_kind,
                    before_reward=parent_eval.task_rewards.get(task),
                    after_reward=node_eval.task_rewards.get(task),
                    analysis=f"{kind} after merge evaluation; inspect merge logs for attribution.",
                    code_change_summary="Automatically inferred from merge evaluation delta.",
                    artifact_paths=[p for p in [node_eval.job_log_path] if p],
                    confidence=0.6,
                    metadata={"source": "merge_eval_delta"},
                )
            except Exception:
                self.log.warning("could not record merge experience for %s", task, exc_info=True)

    def _record_cursor_failure(
        self,
        phase: str,
        log_path: Path,
        result: cursor_agent.CursorResult,
    ) -> str:
        summary = cursor_failures.cursor_failure_summary(log_path, result)
        self.repair_notes.append(f"{phase}: cursor failure: {summary}")
        self._record_note({
            "merge_cursor_failure": {
                "phase": phase,
                "summary": summary,
                "error": result.error,
                "exit_code": result.exit_code,
                "log_path": str(log_path),
            }
        })
        return summary

    def _write_notes(self) -> None:
        if not self.child or not self.child.effort_md_path:
            return
        with Path(self.child.effort_md_path).open("a") as f:
            f.write("\n## Merge Result\n\n")
            f.write(f"- intermediate PR: {self.pr_url or '(not created)'}\n")
            f.write(f"- final score: {self.final_score}\n")
            if self.repair_notes:
                f.write("\n### Repair Notes\n\n")
                for note in self.repair_notes:
                    f.write(note.strip() + "\n\n")

    def _mark_done(self, status: str) -> None:
        tree.update_pipeline(
            self.conn, self.pipeline_id,
            status=status, finished_at=tree.utcnow_iso(),
        )

    def _template_path(self, name: str) -> Path:
        return _prompt_path(name)

    def _save_prompt(self, kind: str, body: str) -> None:
        path = self.prompts_dir_path / f"{kind}.md"
        path.write_text(body)
        self._record_note({"merge_prompt": {"name": kind, "path": str(path)}})


def _git(
    args: list[str],
    *,
    cwd: Path,
    check: bool,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def _repair_review_blocks(text: str) -> bool:
    marker = "<<<REPAIR_REVIEW>>>"
    idx = text.find(marker)
    if idx < 0:
        return False
    tail = text[idx + len(marker):].strip().split()
    return bool(tail and tail[0].upper() == "BLOCK")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _task_validation_priority(
    task: str,
    stats: dict[str, tree.TaskOutcomeStats],
) -> float:
    task_stats = stats.get(task)
    if task_stats is None:
        return 1.0
    return (
        0.50 * task_stats.failure_rate
        + 0.75 * task_stats.regression_rate
        + 0.10 * task_stats.merge_failures
        + 1.0 / (1.0 + task_stats.total_evals)
    )


__all__ = ["MergePipelineConfig", "NodeMergePipeline"]
