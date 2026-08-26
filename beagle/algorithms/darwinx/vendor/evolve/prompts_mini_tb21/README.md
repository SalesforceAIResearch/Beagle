# mini-swe-agent prompts, Terminal-Bench 2.1 layer

Layer this **on top of** `prompts_mini`, do not use it alone:

```sh
export DARWINX_GATE_PROMPT_DIR="$PIPE/self_evolve/prompts_mini_tb21:$PIPE/self_evolve/prompts_mini"
```

`_prompt_path` searches the list left to right and falls through to
`self_evolve/prompts` last, so only the files that actually differ live here.
`implement.md` and `review.md` are deliberately absent — they are surface-agnostic
and must keep coming from `prompts_mini`, so a fix there reaches both benchmarks.

## Why it exists

One substitution, in the two prompts that name the harness surface:

| route | what runs mini | prompt surface it reads |
|---|---|---|
| SWE-bench Verified | our host-side adapter, `agent.preset` | `config/benchmarks/swebench.yaml` |
| Terminal-Bench 2.1 | Harbor runs the `mini-swe-agent` console script | **`config/mini.yaml`** |

The console script is `minisweagent.run.mini:app`, whose `DEFAULT_CONFIG_FILE` is
`builtin_config_dir / "mini.yaml"`, and Harbor passes no `-c` override. So on
TB2.1 nothing reads `swebench.yaml` or `default.yaml`.

This is not a cosmetic difference. Measured on task `fix-git`: setting
`step_limit: 2` in `mini.yaml` turns a run that resolved in 99 s into
`Exit: LimitsExceeded`. The identical edit to `default.yaml` changes nothing.

A campaign pointed at TB2.1 with the unlayered `prompts_mini` would therefore
spend every prompt-surface iteration editing a file the agent never loads. The
candidates would diff cleanly, pass review, and behave identically to their
parent — a flat curve produced by a campaign that looks like it is working.
