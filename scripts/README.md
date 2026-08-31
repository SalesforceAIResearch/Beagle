# scripts/

Operator tools, one topic per subfolder (each with its own `README.md`):

| topic | what | entry points |
|---|---|---|
| [`gateway/`](gateway/README.md) | reach LLM Gateway Express from the cluster (agent-agnostic tunnel) | `laptop.sh`, `login-node.sh`, `gateway_proxy.py` |
| [`onboard/`](onboard/README.md) | stand up an evolvable agent's experiment copy | `onboard_agent.py` |

Everything here is runnable standalone (`python3 scripts/<topic>/<name>.py …` or the
`.sh` wrappers); the onboard shim also runs as `python -m beagle.tools.onboard …`.
