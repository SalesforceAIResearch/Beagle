# Local Kata model validation — 2026-09-15

The opt-in [model smoke](../tests/smoke/kata_model.py) completed a real code repair
through Beagle's `MiniSweAgent.run_in`, with local inference and shell commands
inside Kata. [Reproduction instructions](kata-runtime.md#real-agent-and-model-repair-smoke)
describe the image preparation and required guest resources.

## Recorded environment

| Component | Value |
| --- | --- |
| Host | Ubuntu 24.04 under Windows WSL2/KVM |
| Docker / Kata / QEMU | 29.8.0 / Go runtime 4.1.0 / 11.0.1 |
| Host / guest kernels | 6.6.87.1-microsoft-standard-WSL2 / 6.18.35 |
| Guest configuration | 4 vCPUs, 8192 MiB RAM, CPU inference |
| Agent | mini-swe-agent 2.4.6, built-in `litellm_textbased` mode |
| Inference server | llama.cpp build 10975, commit `4c9233c03` |
| Model | Qwen2.5-Coder-7B-Instruct, Q4_K_M |
| Network | `none`; model endpoint on guest loopback only |

Model revision: `13fb94bfda8c8cf22497dc57b78f391a9acb426a` in the official
[Qwen GGUF repository](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF).
The downloaded file and runtime input were checked against SHA-256
`509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c`.
Opening the model input for writing from the guest was rejected.

## Result

Three of the five fixture tests failed before the agent ran. The agent submitted
this source change (the smoke did not provide the replacement implementation):

```diff
-    return max(lower, value)
+    return max(lower, min(value, upper))
```

The native trajectory ended with `Submitted`, recorded four assistant turns,
2,285 prompt tokens and 388 completion tokens. Beagle preserved the native
trajectory and patch and wrote its run record. No external model API was used.

Only the generated source file was copied into a fresh Kata VM. The original
tests there reported:

```text
test_above ... ok
test_below ... ok
test_float ... ok
test_inside ... ok
test_negative_interval ... ok
Ran 5 tests
OK
```

Both containers were absent from Docker afterward, and no associated QEMU
process remained. The local run retained `mini.traj.json`, `patch.diff`, model
logs, before/after test output, `evidence.json` and Beagle's `run.json`.

This is one functional agent/model/tool/lifecycle validation on a synthetic
fixture. It is not a SWE-bench score, general model-quality measurement, or
adversarial containment certification. The repository's default test suite is
validated separately from this opt-in model smoke.
