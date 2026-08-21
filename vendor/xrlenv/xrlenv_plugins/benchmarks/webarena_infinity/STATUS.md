# WebArena-Infinity — onboarding status

Onboarding disposition for the **runner-shim** integration (§7.2): an answer-free
substrate image + three integration scripts that drive WAI's own `run_eval_parallel`
eval against xrlenv raw containers instead of local app-server ports. The onboarding
"gate" is that the substrate builds answer-free, acquires on the cluster, and WAI's eval
(agent → verifier) runs end-to-end producing `results.json` + `report.html`.

Unlike the harbor benchmarks there is no per-task oracle *reward*: WAI grades each task
with its own verifier inside the container (state / answer check). The proof is that the
substrate + integration are faithful (answers never co-resident with the agent at runtime —
audit H1/D6) and the eval completes on the cluster.

## Gate config (current)

`run_eval_parallel_xrlenv.py` (run from inside the WAI checkout): `--workers N` = N
containers in flight cluster-wide (reused across many tasks); `--web-app apps/<name>`
selects the real-tasks suite; `--model` picks the agent. Per-run lifecycle: acquire →
inject `evaluation/` → start the app server in-container → per task: agent (phase A) →
inject verifier+answer → verifier (phase B) → delete the answer → pull artifacts. The
substrate is pinned to WAI commit `1ca77813` (`WEBARENA_REF`), tagged `:dev` (channel tag;
resolved to the current registry digest per acquire).

## Results

> **Last known: all evaluated tasks passed** (Failed = 0) under the answer-free substrate,
> per the onboarding runs. WebArena-Infinity is graded by WAI's own per-task verifier (not a
> single oracle reward), so exact per-suite Passed/Total + run metadata refresh on the next
> `results.json` / `report.html` — update from there if a suite regresses.

| Item | Status | Meaning |
|---|---|---|
| Substrate image (answer-free) | ✅ built + pushed | `:dev` on the private registry; build fails if any verifier/oracle survives the `substrate` stage |
| Cluster acquire + integration | ✅ validated | raw-container acquire path; scripts inject + run WAI's eval in-container |
| Eval pass rate | ✅ **all passed** | every evaluated task passed WAI's verifier (last known — refresh from `results.json`) |

## Reproduce

```bash
# .env at the WAI repo root supplies XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN +
# XRLENV_PRIVATE_REGISTRY_HOST/_PORT + LLM keys (see README §Prerequisites).

# 1. build + push the answer-free substrate image (build host):
source .env
.venv/bin/python scripts/build_and_push_images.py \
    --plan xrlenv_plugins/benchmarks/webarena_infinity/build_plan.yaml \
    --registry "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}" --force

# 2. install the integration scripts into the WAI checkout, then 3. run the full sweep:
cp xrlenv_plugins/benchmarks/webarena_infinity/copy_to_call_site/* <path-to-webarena-infinity>/evaluation/
cd <path-to-webarena-infinity>
bash evaluation/run_full_sweep.sh                 # official 10-app real-tasks, --model oracle (the gate); APPS=all → full 13

# a real-agent run over one app (needs LLM keys in .env), or the underlying per-app runner:
#   MODEL=gemini-pro WEB_APP=apps/gmail bash evaluation/run_full_sweep.sh
#   python evaluation/run_eval_parallel_xrlenv.py --model gemini-pro --workers 8 --web-app apps/gmail
```

Output (`results.json` + `report.html`, multi-run merge, resume) is identical to WAI's
local runner.

## Notes

- **Answer-free at runtime.** The default `substrate` build stage strips every task verifier
  + oracle solver and fails if any survive; the answer is injected only for the verifier
  step and deleted before the next task.
- **Freshness.** `:dev` is a distribution *channel* tag — a rebuild re-pushes `:dev` and the
  CP tag→digest resolver serves the new digest per acquire, so rebuilds propagate without
  minting a new tag. On a cluster on older code `:dev` is a plain mutable tag (see the Sphinx
  page's freshness caveat).
