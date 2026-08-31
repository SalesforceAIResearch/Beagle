"""Reward-aware, theme-level failure-mode classification for self-evolve.

This is a *complementary* lens to the per-trajectory ``IssueCategory`` fault
localization in :mod:`trace_analyzer.model`. Where ``IssueCategory`` answers
"what did the agent do wrong in this trajectory", this module answers the
cross-task, gradient/theme question: **why did this TRIAL fail, as a category we
can aggregate across tasks** — so the campaign can discover dominant themes
("most timeouts are dependency-setup", "most gaps are wrong-output") and steer
evolution / GATE synthesis at the theme rather than one task at a time.

Motivated by the 2026-06-22 monet-vs-cursor analysis: of the tasks where cursor
(0.825) beats monet (0.789) by >=0.4, ~5 are *wrong-output* (ran to completion,
failed the test = a correctness gap), ~3 are *timeout* (dominated by dependency
INSTALL, not code navigation — so LSP would not help), 1 infra. Conflating these
("it failed") loses the signal; this module keeps them separate.

Inputs come from what the eval already records per trial: the trajectory
(``*.trajectory.jsonl``), the ``error`` string, and the ``reward``. Per-command
*timing* is not yet logged by the harness, so the timeout sub-classification is
COMMAND-PATTERN based (which install/compile/compute commands ran); add
per-command duration to monet's bash tool to upgrade this to true budget-%.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class FailureMode(str, Enum):
    SOLVED = "solved"                       # reward >= pass threshold
    TIMEOUT_SETUP = "timeout-setup"         # timed out; dominated by dependency install
    TIMEOUT_COMPUTE = "timeout-compute"     # timed out; dominated by compile/train/sim/run
    TIMEOUT_EXPLORATION = "timeout-exploration"  # timed out; dominated by code nav (LSP could help)
    TIMEOUT_OTHER = "timeout-other"         # timed out; no dominant activity
    WRONG_OUTPUT = "wrong-output"           # ran to completion, failed the grader (correctness gap)
    INFRA = "infra"                         # XRLEnv / container / tunnel / gateway error (not the agent)
    TOOL_ERROR = "tool-error"               # non-timeout agent/tool crash (exit!=0, API 4xx)


_TIMEOUT_RE = re.compile(r"timeout|agenttimeout|timed out", re.I)
_INFRA_RE = re.compile(r"xrlenv|env.?start|container|tunnel|fetch failed|connection|reset by peer|unavailable|grpc", re.I)
_TOOLERR_RE = re.compile(r"exit|non.?zero|40[0-9]|invalid model|rc=1|crash", re.I)

_SETUP_RE = re.compile(r"apt-get|apt install|pip install|pip3 install|conda install|npm install|cargo install|\bgem install", re.I)
_COMPILE_RE = re.compile(r"\bgcc\b|\bg\+\+\b|\bmake\b|cargo build|nvcc|cmake|configure", re.I)
_COMPUTE_RE = re.compile(r"train|epoch|fasttext|\.fit\(|torch|tensorflow|\ba\.out\b|python3? |for .* in|while |timeout |\.sim|simulate|pytest", re.I)
_EXPLORE_TOOLS = {"grep", "file_read", "glob", "lsp", "codebase_search", "read", "file_search"}


@dataclass
class TrialClassification:
    failure_mode: FailureMode
    reward: float
    num_turns: int = 0
    bash_calls: int = 0
    explore_calls: int = 0
    activity: list[str] = field(default_factory=list)   # setup/compile/compute present
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "failure_mode": self.failure_mode.value,
            "reward": round(self.reward, 3),
            "num_turns": self.num_turns,
            "bash_calls": self.bash_calls,
            "explore_calls": self.explore_calls,
            "activity": self.activity,
            "note": self.note,
        }


def _read_trajectory_activity(traj_path: str | None) -> tuple[Counter, str]:
    """Return (tool-name counts, concatenated bash command text)."""
    tools: Counter = Counter()
    cmds: list[str] = []
    if not traj_path:
        return tools, ""
    try:
        with open(traj_path) as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if e.get("type") == "tool_start":
                    name = e.get("name") or e.get("toolName")
                    tools[name] += 1
                    if name == "bash":
                        cmds.append(str(e.get("input") or "").lower())
    except OSError:
        pass
    return tools, " ".join(cmds)


def _activity_tags(cmd_text: str) -> list[str]:
    tags = []
    if _SETUP_RE.search(cmd_text):
        tags.append("setup")
    if _COMPILE_RE.search(cmd_text):
        tags.append("compile")
    if _COMPUTE_RE.search(cmd_text):
        tags.append("compute")
    return tags


def classify_trial(
    *, reward: float, error: str | None, trajectory_path: str | None = None,
    num_turns: int = 0, pass_threshold: float = 1.0,
) -> TrialClassification:
    """Classify one trial into a :class:`FailureMode`."""
    err = (error or "").strip()
    tools, cmd_text = _read_trajectory_activity(trajectory_path)
    bash = tools.get("bash", 0)
    explore = sum(tools.get(k, 0) for k in _EXPLORE_TOOLS)
    tags = _activity_tags(cmd_text)
    base = dict(reward=reward, num_turns=num_turns, bash_calls=bash,
                explore_calls=explore, activity=tags)

    if reward >= pass_threshold:
        return TrialClassification(FailureMode.SOLVED, **base)

    # errored trials: timeout / infra / tool-error
    if _TIMEOUT_RE.search(err):
        # sub-classify the timeout by where the work went (command-pattern proxy)
        if explore > bash and explore >= 8:
            mode = FailureMode.TIMEOUT_EXPLORATION
        elif "setup" in tags:
            mode = FailureMode.TIMEOUT_SETUP
        elif "compile" in tags or "compute" in tags:
            mode = FailureMode.TIMEOUT_COMPUTE
        else:
            mode = FailureMode.TIMEOUT_OTHER
        return TrialClassification(mode, note="timeout; activity=" + ",".join(tags), **base)
    if _INFRA_RE.search(err):
        return TrialClassification(FailureMode.INFRA, note=err[:60], **base)
    if err and _TOOLERR_RE.search(err):
        return TrialClassification(FailureMode.TOOL_ERROR, note=err[:60], **base)

    # no error but didn't pass => ran to completion, failed the grader
    return TrialClassification(FailureMode.WRONG_OUTPUT,
                               note="completed but reward<pass", **base)


def summarize_run(run_json_path: str, *, pass_threshold: float = 1.0) -> dict:
    """Classify every trial in a harbor ``run.json`` and tally themes.

    Returns ``{"per_trial": [...], "theme_tally": {mode: n}, "by_task": {...}}``.
    """
    d = json.load(open(run_json_path))
    per_trial = []
    tally: Counter = Counter()
    by_task: dict[str, Counter] = {}
    for r in d.get("per_task_results", []):
        task = re.sub(r"__s\d+$", "", r.get("task_id", ""))
        c = classify_trial(
            reward=float(r.get("reward", 0) or 0),
            error=r.get("error"),
            trajectory_path=r.get("trajectory_path"),
            num_turns=int(r.get("num_turns", 0) or 0),
            pass_threshold=pass_threshold,
        )
        per_trial.append({"task_id": r.get("task_id"), **c.as_dict()})
        tally[c.failure_mode.value] += 1
        by_task.setdefault(task, Counter())[c.failure_mode.value] += 1
    return {
        "run": run_json_path,
        "theme_tally": dict(tally.most_common()),
        "per_trial": per_trial,
        "by_task": {k: dict(v) for k, v in by_task.items()},
    }


def _main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Classify failure modes in a harbor run.json")
    p.add_argument("run_json")
    p.add_argument("--pass-threshold", type=float, default=1.0)
    p.add_argument("--failures-only", action="store_true")
    args = p.parse_args(argv)
    s = summarize_run(args.run_json, pass_threshold=args.pass_threshold)
    print("THEME TALLY:")
    for mode, n in s["theme_tally"].items():
        print(f"  {n:4}  {mode}")
    print("\nPER-TRIAL (failures):" if args.failures_only else "\nPER-TRIAL:")
    for t in s["per_trial"]:
        if args.failures_only and t["failure_mode"] == "solved":
            continue
        print(f"  {t['task_id'][:34]:34} {t['failure_mode']:20} "
              f"r={t['reward']:.2f} turns={t['num_turns']:>3} "
              f"bash={t['bash_calls']:>3} expl={t['explore_calls']:>3} "
              f"{','.join(t['activity'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
