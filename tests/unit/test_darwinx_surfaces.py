"""A guard keyed to the wrong evolvee's paths is a guard that passes everything.

darwinx decides what KIND of edit a candidate made -- core, skill, prompt -- by matching path
substrings, and those defaults name monet's files. Measured 2026-08-26 on opencode: an 80-line
additive edit to ``packages/opencode/src/session/`` drew zero violations while the identical edit
to monet's ``src/query/loop.js`` was bounced pre-eval, and the proposer was told its only skill
surface was ``src/core/bundled-skills.js`` -- a file opencode does not have. It spent both of its
iterations editing the system prompt instead, the one shape every task pays for and none can be
credited for. These tests pin both halves: monet's arm is untouched, and a declared evolvee gets
guards that actually fire.
"""

import pytest

from beagle.algorithms.darwinx.config import DarwinXConfig
from beagle.algorithms.darwinx.vendor.evolve import generalization as G

OPENCODE_CORE = "packages/opencode/src/session/index.ts"
OPENCODE_PROMPT = "packages/opencode/src/session/prompt/verification.txt"
OPENCODE_SKILL = ".opencode/skills/verify/SKILL.md"
MONET_CORE = "src/query/loop.js"


def _diff(*paths, added=80):
    out = []
    for p in paths:
        out += [f"--- a/{p}", f"+++ b/{p}", f"@@ -1 +1,{added} @@"]
        out += [f"+line {i}" for i in range(added)]
    return "\n".join(out)


@pytest.fixture
def opencode_surfaces(monkeypatch):
    for var, val in DarwinXConfig(
        shared_core_paths="packages/opencode/src/session/,packages/opencode/src/tool/",
        prompt_paths="packages/opencode/src/session/prompt/",
        skill_target_doc="add a cue-gated skill at .opencode/skills/<name>/SKILL.md",
        prompt_rule_doc="Do not edit the prompt files.",
    ).to_driver_env().items():
        monkeypatch.setenv(var, val)


def test_monets_arm_is_unchanged_when_no_evolvee_is_declared():
    assert len(G.scan_diff_locality(_diff(MONET_CORE))) == 1
    assert G.scan_diff_locality(_diff(OPENCODE_PROMPT)) == []
    assert G.skill_target_doc().count("bundled-skills.js") == 1
    assert G.prompt_surface_rule() == ""


def test_a_declared_hot_surface_is_bounced_where_it_previously_passed(opencode_surfaces):
    kinds = [v.kind for v in G.scan_diff_locality(_diff(OPENCODE_CORE))]
    assert kinds == ["broad_shared_core_change"]


def test_a_prompt_edit_is_bounced_and_told_where_the_capability_belongs(opencode_surfaces):
    violations = G.scan_diff_locality(_diff(OPENCODE_PROMPT, added=3))
    assert [v.kind for v in violations] == ["prompt_surface_edit"]
    # The bounce is only useful if it names this evolvee's real skill path.
    assert ".opencode/skills/" in violations[0].pattern


def test_adding_a_skill_stays_allowed_so_the_proposer_has_a_valid_surface(opencode_surfaces):
    assert G.scan_diff_locality(_diff(OPENCODE_SKILL, added=200)) == []
    assert G.classify_diff_surface(_diff(OPENCODE_SKILL, added=3)) == "skill"


def test_the_proposers_instructions_name_this_evolvee(monkeypatch):
    monkeypatch.setenv("DARWINX_GATE_EVOLVEE_LABEL", "opencode")
    monkeypatch.setenv("DARWINX_GATE_CORE_PATH_DOC", "packages/opencode/src/")
    assert G.evolvee_label() == "opencode"
    assert G.core_path_doc() == "packages/opencode/src/"


def test_bouncing_the_prompt_requires_saying_where_to_go_instead():
    # Reverting a surface the instructions never forbade spends an iteration teaching a rule the
    # proposer was never given.
    with pytest.raises(ValueError, match="prompt_paths requires"):
        DarwinXConfig(prompt_paths="packages/opencode/src/session/prompt/")
    with pytest.raises(ValueError, match="prompt_paths requires"):
        DarwinXConfig(prompt_paths="a/", skill_target_doc="add a skill")


def test_surface_paths_reach_the_driver_under_the_names_the_guards_read():
    env = DarwinXConfig(shared_core_paths="a/", prompt_paths="b/",
                        skill_target_doc="doc", prompt_rule_doc="rule").to_driver_env()
    assert env["DARWINX_GATE_SHARED_CORE_PATHS"] == "a/"
    assert env["DARWINX_GATE_PROMPT_PATHS"] == "b/"
    assert "DARWINX_GATE_SKILL_PATH_MARKERS" not in env  # unset knobs keep the driver's defaults
