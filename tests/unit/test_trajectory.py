"""monet stream-json → harbor ATIF (agent/trajectory.json). Skipped without harbor
(the converter builds harbor's own pydantic models); when present, we assert the output
passes harbor's own validator + carries the right steps/tool-calls/observations."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("harbor")

from beagle.benchmarks.trajectory import write_trajectory_json  # noqa: E402

_STREAM = "\n".join([
    '{"type":"session_meta","session_id":"S1"}',
    '{"type":"text_delta","text":"Let me read the file."}',
    '{"type":"tool_use_start","index":0,"toolId":"c1","toolName":"file_read"}',
    '{"type":"tool_use_delta","index":0,"partialJson":"{\\"path\\":\\"a.py\\"}"}',
    '{"type":"tool_output","id":"c1","output":"line1\\nline2","isError":false}',
    '{"type":"message_delta","usage":{"inputTokens":100,"cacheReadTokens":10,"outputTokens":20}}',
    '{"type":"turn_complete","turn":0,"stopReason":"tool_use"}',
    '{"type":"text_delta","text":"Done."}',
    '{"type":"turn_complete","turn":1,"stopReason":"end_turn"}',
    '{"type":"usage","usage":{"inputTokens":250,"outputTokens":45}}',
    '{"type":"turn_done"}',
])


def test_monet_stream_to_valid_atif(tmp_path) -> None:
    (tmp_path / "monet.stream.jsonl").write_text(_STREAM)
    path = write_trajectory_json(
        tmp_path, trajectory_format="monet-stream-json", instruction="Fix the bug.",
        agent_name="monet", agent_version="abc123", model_name="gpt-5.5",
    )
    assert path is not None and path.name == "trajectory.json"

    # 1. Passes harbor's OWN validator (the contract).
    from harbor.utils.trajectory_validator import TrajectoryValidator
    v = TrajectoryValidator()
    assert v.validate(str(path)), v.get_errors()

    # 2. Right structure: a user step (the prompt) + one agent step per turn.
    d = json.loads(path.read_text())
    assert d["schema_version"] == "ATIF-v1.7"
    assert d["agent"] == {"name": "monet", "version": "abc123", "model_name": "gpt-5.5"}
    steps = d["steps"]
    assert [s["source"] for s in steps] == ["user", "agent", "agent"]
    assert steps[0]["message"] == "Fix the bug." and steps[0]["step_id"] == 1

    turn0 = steps[1]
    tc = turn0["tool_calls"][0]
    assert tc["function_name"] == "file_read" and tc["arguments"] == {"path": "a.py"}
    assert turn0["observation"]["results"][0]["content"] == "line1\nline2"
    assert turn0["metrics"]["prompt_tokens"] == 110  # input + cacheRead folded
    assert turn0["metrics"]["completion_tokens"] == 20

    assert steps[2]["message"] == "Done." and "tool_calls" not in steps[2]  # exclude_none drops it

    # final_metrics is the session total — accumulated by mirroring the canonical parser, so it must
    # equal parse_monet_usage EXACTLY (both usage-bearing events summed: message_delta 100/10/20 +
    # standalone usage 250/45), independent of where the trailing usage event sits vs turn_complete.
    from beagle.agents.monet._helpers import parse_monet_usage
    canon = parse_monet_usage(_STREAM).to_token_counts()
    fm = d["final_metrics"]
    assert fm["total_prompt_tokens"] == canon["prompt"] == 360
    assert fm["total_completion_tokens"] == canon["completion"] == 65
    assert fm["total_cached_tokens"] == canon["cache_read"] == 10


def test_no_converter_or_missing_stream_is_noop(tmp_path) -> None:
    assert write_trajectory_json(tmp_path, trajectory_format="unknown-fmt",
                                 instruction="x", agent_name="a") is None
    # registered format but no stream file present → no-op
    assert write_trajectory_json(tmp_path, trajectory_format="monet-stream-json",
                                 instruction="x", agent_name="monet") is None
    assert write_trajectory_json(tmp_path, trajectory_format="mini-swe",
                                 instruction="x", agent_name="mini-swe") is None


_MINI_TRAJ = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please solve: fix the bug"},
        {"role": "assistant", "content": "Let me look.",
         "extra": {"actions": [{"command": "ls -la"}],
                   "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}}},
        {"role": "user", "content": '{"returncode":0,"output":"file.py"}'},
        {"role": "assistant", "content": "Now I fix it.",
         "extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}],
                   "response": {"usage": {"prompt_tokens": 150, "completion_tokens": 10}}}},
    ]
}


# Reasoning mini-swe uses the Responses API (model_class=litellm_response): turns are stored
# `object=response` items (output + extra.actions, usage under top-level input/output_tokens) and
# observations are `type=function_call_output` items — no `role`. The converter must handle this.
_MINI_TRAJ_RESPONSES = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please solve: fix the bug"},
        {"object": "response", "output": [{"type": "reasoning", "summary": []},
                                          {"type": "function_call", "name": "bash"}],
         "usage": {"input_tokens": 100, "output_tokens": 20},
         "extra": {"actions": [{"command": "ls -la"}]}},
        {"type": "function_call_output", "call_id": "c1", "output": '{"returncode":0,"output":"file.py"}'},
        {"object": "response", "output": [{"type": "function_call", "name": "bash"}],
         "usage": {"input_tokens": 150, "output_tokens": 10},
         "extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]}},
        {"role": "exit", "content": ""},
    ]
}


def test_mini_swe_responses_api_traj_to_valid_atif(tmp_path) -> None:
    # Reasoning path: object=response turns + function_call_output observations → same ATIF shape as
    # the chat path (regression for the "only 2 steps" bug when mini uses the Responses API).
    (tmp_path / "mini.traj.json").write_text(json.dumps(_MINI_TRAJ_RESPONSES))
    path = write_trajectory_json(
        tmp_path, trajectory_format="mini-swe", instruction="Fix the bug.",
        agent_name="mini-swe", agent_version="v2.4.6", model_name="gpt-5.5")
    assert path is not None
    from harbor.utils.trajectory_validator import TrajectoryValidator
    assert TrajectoryValidator().validate(str(path))
    d = json.loads(path.read_text())
    steps = d["steps"]
    assert [s["source"] for s in steps] == ["system", "user", "agent", "agent"]   # not just [system,user]
    t0 = steps[2]
    assert t0["tool_calls"][0]["arguments"] == {"command": "ls -la"}
    assert t0["observation"]["results"][0]["content"] == '{"returncode":0,"output":"file.py"}'
    assert t0["metrics"] == {"prompt_tokens": 100, "completion_tokens": 20}   # input/output → prompt/completion
    assert "observation" not in steps[3]                                       # last turn: no trailing output
    assert d["final_metrics"]["total_steps"] == 4 and d["final_metrics"]["total_completion_tokens"] == 30


def test_mini_swe_traj_to_valid_atif(tmp_path) -> None:
    (tmp_path / "mini.traj.json").write_text(json.dumps(_MINI_TRAJ))
    path = write_trajectory_json(
        tmp_path, trajectory_format="mini-swe", instruction="Fix the bug.",
        agent_name="mini-swe", agent_version="v2.4.6", model_name="gpt-5.5")
    assert path is not None and path.name == "trajectory.json"

    from harbor.utils.trajectory_validator import TrajectoryValidator
    v = TrajectoryValidator()
    assert v.validate(str(path)), v.get_errors()

    d = json.loads(path.read_text())
    assert d["schema_version"] == "ATIF-v1.7"
    assert d["agent"] == {"name": "mini-swe", "version": "v2.4.6", "model_name": "gpt-5.5"}
    steps = d["steps"]
    # system prompt preserved as a system step, then the task (user) step, then one agent step/turn
    assert [s["source"] for s in steps] == ["system", "user", "agent", "agent"]
    assert steps[0]["message"] == "You are a helpful assistant."
    assert steps[1]["message"] == "Please solve: fix the bug"

    turn0 = steps[2]
    tc = turn0["tool_calls"][0]
    assert tc["function_name"] == "bash" and tc["arguments"] == {"command": "ls -la"}
    assert turn0["observation"]["results"][0]["content"] == '{"returncode":0,"output":"file.py"}'
    assert turn0["metrics"] == {"prompt_tokens": 100, "completion_tokens": 20}

    assert "observation" not in steps[3]                 # last turn: no trailing output
    assert d["final_metrics"]["total_prompt_tokens"] == 250
    assert d["final_metrics"]["total_completion_tokens"] == 30
    assert d["final_metrics"]["total_steps"] == 4         # system + user + 2 agent turns
