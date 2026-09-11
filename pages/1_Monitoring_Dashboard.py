"""Monitoring dashboard: retrieval quality and confidence trends over time.

This is the "monitoring" deliverable from the project description. It reads the
SQLite telemetry written by every pipeline run, so it reflects real queries
rather than a canned demo dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.store import clear, load_dataframe, summary_stats  # noqa: E402

st.set_page_config(page_title="Monitoring · Confidence-Aware RAG", page_icon="📊", layout="wide")
st.title("📊 Monitoring dashboard")
st.caption("Retrieval quality and confidence trends across every query the pipeline has served.")

limit = st.sidebar.slider("Queries to load", 20, 2000, 300, step=20)
run_tag = st.sidebar.text_input("Filter by run tag", value="", placeholder="e.g. squad2-dev")

df = load_dataframe(limit=limit, run_tag=run_tag or None)
stats = summary_stats()

if df.empty:
    st.info("No telemetry yet. Ask a few questions on the main page, then come back.")
    st.stop()

# --- Headline metrics --------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Queries logged", stats["total"])
c2.metric("Answered", stats["answered"])
c3.metric("Abstained", stats["abstained"], f"{stats['abstention_rate']:.0%} of all queries")
c4.metric("Flagged low-confidence", stats["flagged"])
c5.metric("Mean composite", f"{stats['mean_composite']:.3f}")

c6, c7, c8 = st.columns(3)
c6.metric("Evident conflicts caught", stats["total_conflicts"])
c7.metric("Baseless claims caught", stats["total_baseless"])
c8.metric("Mean latency", f"{stats['mean_latency_ms']:.0f} ms")

st.divider()

# --- Confidence over time ----------------------------------------------------
st.subheader("Confidence trend")
st.caption(
    "A downward drift in Gate A means the corpus is drifting away from what users "
    "are asking about — the signal that the knowledge base needs updating."
)

trend_cols = [c for c in ("gate_a_score", "gate_b_score", "composite") if c in df.columns]
if trend_cols:
    trend = df.set_index("id")[trend_cols]
    st.line_chart(trend, height=300)

# --- Verdict mix -------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Verdict distribution")
    counts = df["verdict"].value_counts().rename_axis("verdict").reset_index(name="count")
    st.bar_chart(counts.set_index("verdict"), height=280)

with right:
    st.subheader("Where confidence is lost")
    tax = {
        "supported claims": int(
            (df["n_claims"].fillna(0) - df["n_conflicts"].fillna(0) - df["n_baseless"].fillna(0)).sum()
        ),
        "baseless information": int(df["n_baseless"].fillna(0).sum()),
        "evident conflict": int(df["n_conflicts"].fillna(0).sum()),
    }
    import pandas as pd

    st.bar_chart(pd.Series(tax, name="claims"), height=280)
    st.caption("RAGTruth taxonomy, aggregated over all logged answers.")

st.divider()

# --- Retrieval quality -------------------------------------------------------
st.subheader("Retrieval quality")
rc1, rc2 = st.columns(2)
with rc1:
    if "top_similarity" in df.columns:
        st.caption("Top embedding similarity per query")
        st.area_chart(df.set_index("id")[["top_similarity", "mean_similarity"]], height=260)
with rc2:
    if "similarity_margin" in df.columns:
        st.caption("Similarity margin (top-1 − top-2). Narrow margins signal weak coverage.")
        st.bar_chart(df.set_index("id")[["similarity_margin"]], height=260)

# --- Latency budget ----------------------------------------------------------
st.subheader("Latency budget")
lat_cols = [c for c in ("retrieval_ms", "gate_a_ms", "generation_ms", "gate_b_ms") if c in df.columns]
if lat_cols:
    means = df[lat_cols].mean().round(1)
    st.caption(
        "Mean milliseconds per stage. The point of the local NLI scorers is that both "
        "gates together stay small next to generation."
    )
    st.bar_chart(means, height=260)
    st.dataframe(means.to_frame("mean ms").T, use_container_width=True)

st.divider()

# --- Benchmark comparison (guarded vs baseline) ------------------------------
# The evaluation chapter's central table, computed live from telemetry so it
# stays in sync with whatever sweep was last run.
bench_df = df[df["run_tag"].notna() & (df["run_tag"] != "smoke") & df["item_id"].notna()] if "item_id" in df.columns else df.iloc[0:0]
if not bench_df.empty and "arm" in bench_df.columns and bench_df["arm"].nunique() > 1:
    import pandas as pd

    st.subheader("Benchmark comparison: guarded vs baseline")
    st.caption(
        "Same questions, both arms (run_tag="
        + ", ".join(sorted(bench_df["run_tag"].unique()))
        + "). Hallucination = answered an unanswerable question; "
        "over-abstention = refused an answerable one."
    )

    rows = []
    for arm, sub in bench_df.groupby("arm"):
        rows.append(
            {
                "arm": arm,
                "n": len(sub),
                "answered": int((~sub["verdict"].str.startswith("abstain")).sum()),
                "abstained": int(sub["verdict"].str.startswith("abstain").sum()),
                "mean composite": round(sub["composite"].mean(), 3),
                "conflicts caught": int(sub["n_conflicts"].fillna(0).sum()),
                "baseless caught": int(sub["n_baseless"].fillna(0).sum()),
                "mean latency ms": round(sub["total_ms"].mean(), 0),
            }
        )
    comp = pd.DataFrame(rows).set_index("arm")
    st.dataframe(comp.T, use_container_width=True)

    verdicts = bench_df.groupby(["arm", "verdict"]).size().unstack(fill_value=0)
    st.bar_chart(verdicts, height=280)
    st.caption("Verdict mix per arm. The baseline arm has no gate enforcement, so it can only answer.")

st.divider()

# --- Raw log -----------------------------------------------------------------
st.subheader("Query log")
show = [
    c
    for c in (
        "id", "ts", "question", "verdict", "composite", "gate_a_score", "gate_b_score",
        "n_conflicts", "n_baseless", "total_ms", "provider", "scorer", "run_tag",
    )
    if c in df.columns
]
st.dataframe(df[show].sort_values("id", ascending=False), use_container_width=True, height=340)

st.download_button(
    "Download full log as CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name="confidence_rag_telemetry.csv",
    mime="text/csv",
)

with st.sidebar.expander("Danger zone"):
    st.caption("Wipes every logged query. Used to get a clean slate before a demo.")
    if st.button("Clear telemetry", type="secondary"):
        clear()
        st.rerun()


