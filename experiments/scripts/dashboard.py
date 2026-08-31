"""beagle eval-results dashboard (Streamlit).

    streamlit run experiments/scripts/dashboard.py

Reads the per-run ``summary.json`` files that ``results_data.py`` builds from each run's native
artifacts (tokens via beagle's own usage parsers — the cache split is accurate). Sidebar navigation
between two pages: an Overview table (with its own filters + editable pricing) and an ATIF trajectory
viewer. Click "Refresh" after new runs land (or a first run) to re-parse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_data as R  # noqa: E402

st.set_page_config(page_title="beagle · eval results", layout="wide", page_icon="🐕")


@st.cache_data(show_spinner="parsing trajectories…")
def _load(rebuild: bool, schema: int) -> list[dict]:   # `schema` is a cache key: a bump auto-invalidates
    return R.load_summaries(rebuild=rebuild)


# ── Overview page (filters + pricing live here — they scope this table) ───────────────────────────
def overview_page() -> None:
    st.title("📊 Overview")
    models = sorted({s["model"] for s in summaries})
    fc = st.columns(4)
    f_bench = fc[0].multiselect("Benchmark", sorted({b for s in summaries for b in s["benchmarks"]}))
    f_harness = fc[1].multiselect("Harness", sorted({s["harness"] for s in summaries}))
    f_model = fc[2].multiselect("Model", models)
    all_runs = sorted(s["runname"] for s in summaries)
    f_run = fc[3].multiselect("Run", all_runs)   # empty = all runs; select to narrow (keeps the UI clean)

    body = st.container()   # reserve the table's slot HERE (above pricing); filled after prices resolve

    with st.expander("💲 Pricing · $/1M tokens (edit to reprice)"):   # renders below the table
        price_df = pd.DataFrame(
            [{"model": m, **(R.DEFAULT_PRICES.get(m) or R.DEFAULT_PRICES["*"])} for m in models])
        edited = st.data_editor(price_df, hide_index=True, width="stretch", key="prices",
                                num_rows="fixed")
        prices = {row["model"]: {"input": row["input"], "cached": row["cached"], "output": row["output"]}
                  for _, row in edited.iterrows()}
        prices.setdefault("*", R.DEFAULT_PRICES["*"])

    def _keep(s: dict) -> bool:
        return ((not f_harness or s["harness"] in f_harness)
                and (not f_model or s["model"] in f_model)
                and (not f_run or s["runname"] in f_run))

    rows = [r for r in R.overview_rows([s for s in summaries if _keep(s)], prices)
            if not f_bench or r["Benchmark"] in f_bench]
    with body:
        if not rows:
            st.warning("No rows match the current filters.")
            return
        df = pd.DataFrame(rows).sort_values(["Benchmark", "Harness"], ignore_index=True)
        m = st.columns(4)
        m[0].metric("Total Runs", len(df))
        m[1].metric("Total tokens", f"{df['Total tokens'].sum()/1e6:,.1f}M")
        m[2].metric("Cached", f"{df['Cached tokens'].sum()/1e6:,.1f}M "
                              f"({100*df['Cached tokens'].sum()/max(1, df['Total tokens'].sum()):.0f}%)")
        m[3].metric("Est. cost", f"${df['Cost (est. $)'].sum():,.2f}")

        disp = df.drop(columns=["In progress"]).copy()
        disp["Score"] = (disp["Score"] * 100).round(1)                 # → percent
        disp["Total tokens"] = (disp["Total tokens"] / 1e6).round(2)   # → millions
        disp["Cached tokens"] = (disp["Cached tokens"] / 1e6).round(2)
        st.dataframe(
            disp, hide_index=True, width="stretch",
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%.1f%%", help="resolved / tasks"),
                "Total tokens": st.column_config.NumberColumn("Total (M)", format="%.2f"),
                "Cached tokens": st.column_config.NumberColumn("Cached (M)", format="%.2f"),
                "Cache %": st.column_config.NumberColumn(format="%.0f%%"),
                "Cost (est. $)": st.column_config.NumberColumn("Cost ($)", format="%.2f"),
                "Latency/task (s)": st.column_config.NumberColumn(
                    "Latency/task (s)", format="%.0f",
                    help="median agent-execution time per task (excludes setup + verifier)"),
                "Cost/task ($)": st.column_config.NumberColumn(
                    "Cost/task ($)", format="%.2f", help="median est. cost per task (at the prices below)"),
            })
        st.caption(
            "- **Latency/task** and **Cost/task** are medians across the benchmark's tasks.\n"
            "- Cost = (prompt−cached)·input-price + cached·cached-price + completion·output-price, at the prices below.")
        if df["In progress"].any():
            st.info("▶ " + ", ".join(df[df["In progress"]]["Run"].unique()) + " — in progress (no run.json yet)")


# ── Trajectory viewer page (independent — its own run/benchmark/trial pickers) ────────────────────
def trajectory_page() -> None:
    st.title("🔍 Trajectory viewer")
    pick = st.selectbox("Run", sorted(s["runname"] for s in summaries), key="traj_run")
    s = next(x for x in summaries if x["runname"] == pick)
    bench = st.selectbox("Benchmark", sorted(s["benchmarks"]), key="traj_bench")
    b = s["benchmarks"][bench]
    labels = {f'{"✅" if t["resolved"] else "❌"} {t["task_id"]}  ({t["error_type"]})': t
              for t in b["trials"]}
    if not labels:
        st.info("no trials for this benchmark")
        return
    t = labels[st.selectbox("Trial", list(labels), key="traj_trial")]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reward", "—" if t["reward"] is None else f'{t["reward"]:.2f}')
    m2.metric("Prompt tok", f'{t["tokens"]["prompt"]:,}')
    m3.metric("Cached tok", f'{t["tokens"]["cache_read"]:,}')
    m4.metric("Completion tok", f'{t["tokens"]["completion"]:,}')
    if t["error"]:
        st.error(t["error"])
    # ATIF (agent/trajectory.json) only — the canonical, harness-agnostic trajectory format.
    tj = R.RESULTS_DIR / pick / t["trajectory"]
    st.caption(f"ATIF · `{tj}`")
    if tj.exists():
        st.json(json.loads(tj.read_text()), expanded=False)
    else:
        st.info("no ATIF trajectory.json for this trial")


# ── shared setup (runs on every page) + sidebar page nav ──────────────────────────────────────────
# st.sidebar.title("🐕 Experiment results")
pg = st.navigation([
    st.Page(overview_page, title="Overview", icon="📊", default=True),
    st.Page(trajectory_page, title="Trajectory viewer", icon="🔍"),
])
if st.sidebar.button("🔄 Refresh (Rebuild Data)", width="stretch"):
    _load.clear()
    st.session_state["_rebuild"] = True
summaries = _load(rebuild=st.session_state.pop("_rebuild", False), schema=R._SCHEMA)
if not summaries:
    st.error(f"No runs found under {R.RESULTS_DIR}")
    st.stop()
pg.run()
