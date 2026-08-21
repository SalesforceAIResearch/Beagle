# Overview

XRLEnv works best when the benchmark keeps owning task parsing,
grading, reports, and retry policy. The adapter should only replace
the part that touches containers.

| If the benchmark... | Use |
|---|---|
| Already uses the Python Docker SDK | {doc}`swe_bench` and the `xrlenv.from_env()` drop-in |
| Is Harbor-format (terminal-bench-2, seta-env, …) — loads an environment class from config | {doc}`harbor_framework` and the framework/harness adapter pattern |
| Is Pier-format (DeepSWE) — loads an environment class from config | {doc}`pier_framework` and {doc}`deep_swe` |
| Is Harbor-format with grade-from-artifact grading (FrontierSWE) | {doc}`frontier_swe` — rich `reward.json` schema + run-time oracle mode |
| Uses raw-container mode with a substrate image (WebArena-Infinity) | {doc}`webarena_infinity` — channel-tag scheme, rebuild workflow, downstream consumer config |
| Invokes `docker` as a raw CLI subprocess (EvoClaw) | {doc}`evoclaw` — in-process subprocess interceptor pattern |
| Has a different harness shape | {doc}`writing_your_own_adapter` to decide whether to use a drop-in, adapter, or direct API |
