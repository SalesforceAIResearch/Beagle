# experiments/scripts — results dashboard

Interactive dashboard over every eval run under `experiments/results/`.

```bash
uv pip install -e '.[analysis]'                       # one-time (streamlit + pandas; NOT a runtime dep)
streamlit run experiments/scripts/dashboard.py        # opens http://localhost:8501
```

## What it shows
Two **sidebar pages** (`st.navigation`):

| page | contents |
|------|----------|
| **📊 Overview** | one row per (run × benchmark): score, resolved/tasks, total tokens, cached tokens, cache %, est. cost, median agent latency. Its own filters (benchmark · harness · model · run) + an editable price table |
| **🔍 Trajectory viewer** | pick a run → benchmark → trial → reward/tokens + the ATIF `trajectory.json` |

## How it stays fast
`results_data.py` does the one heavy pass: for each trial it runs **beagle's own usage parser**
(`parse_opencode_usage` / `parse_monet_usage` / mini's `_parse_trajectory` — the same parsers the
rollout pipeline uses, so token counts and the **cache split** are accurate) and reads reward + error
from `result.json`, then writes a compact `<run>/summary.json`. The app reads those summaries; it only
opens a raw trajectory when you use the viewer. Hit **🔄 Refresh (Rebuild Data)** after new runs land.

> Token source: harbor's `run.json` historically zeroed the cache split (a shim bug, now fixed for new
> runs) and ATIF `trajectory.json` aggregates inconsistently across harnesses — so the per-agent parser
> is the canonical source here. `summary.json` is a generated artifact (git-ignored under `results/`).

## Cost
beagle is pricing-agnostic, so cost is an **estimate** from a per-model price table
(`DEFAULT_PRICES` in `results_data.py`, `$/1M` tokens for fresh input / cached input / output). Edit it
there, or adjust live in the sidebar. `cost = (prompt − cached)·input + cached·cached + completion·output`.

## Extending
New agent harness → add a row to `_PARSERS` in `results_data.py` (`harness → (module, parser, stream
file, kind)`); unknown harnesses fall back to ATIF `final_metrics`. New page → add a `def *_page()` in
`dashboard.py` and register it in the `st.navigation([...])` list.
