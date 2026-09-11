"""Streamlit front end for the confidence-aware RAG pipeline.

Run with:  streamlit run app.py

The interface is built around making the confidence decision *visible*. A demo
that just prints an answer proves nothing; this one shows both gate scores, the
per-claim entailment breakdown, and why the pipeline chose to answer or abstain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings  # noqa: E402

st.set_page_config(
    page_title="Confidence-Aware RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERDICT_STYLE = {
    "answer": ("✅", "Answered", "#1a7f37"),
    "flag_low_confidence": ("⚠️", "Answered with low confidence", "#9a6700"),
    "abstain_no_context": ("🚫", "Abstained — retrieval insufficient", "#a40e26"),
    "abstain_ungrounded": ("🚫", "Abstained — answer not grounded", "#a40e26"),
}


@st.cache_resource(show_spinner="Loading vector store…")
def _store(collection: str):
    from src.retriever import get_store

    return get_store(collection)


def _sidebar() -> dict:
    st.sidebar.header("Configuration")

    provider = st.sidebar.selectbox(
        "LLM provider",
        ["ollama", "groq", "gemini"],
        index=["ollama", "groq", "gemini"].index(settings.llm.provider),
        help="Ollama runs locally with no rate limits. Groq and Gemini need an API key in .env.",
    )
    settings.llm.provider = provider  # type: ignore[assignment]

    scorer = st.sidebar.selectbox(
        "Confidence scorer",
        ["nli", "llm", "both"],
        index=["nli", "llm", "both"].index(settings.confidence.scorer),
        help=(
            "nli = local cross-encoders, milliseconds per query. "
            "llm = DeepEval judge, slower but the reference standard. "
            "both = run each and compare."
        ),
    )
    settings.confidence.scorer = scorer  # type: ignore[assignment]

    st.sidebar.subheader("Thresholds")
    settings.confidence.gate_a_threshold = st.sidebar.slider(
        "Gate A — retrieval sufficiency", 0.0, 1.0, settings.confidence.gate_a_threshold, 0.01
    )
    settings.confidence.gate_b_threshold = st.sidebar.slider(
        "Gate B — answer groundedness", 0.0, 1.0, settings.confidence.gate_b_threshold, 0.01
    )
    settings.confidence.composite_threshold = st.sidebar.slider(
        "Composite", 0.0, 1.0, settings.confidence.composite_threshold, 0.01
    )
    top_k = st.sidebar.slider("Passages retrieved (k)", 1, 12, settings.retrieval.top_k)

    st.sidebar.caption(
        "Thresholds calibrated on the SQuAD 2.0 dev sample (see "
        "reports/calibration.json); sliders override them for this session only."
    )

    baseline = st.sidebar.checkbox(
        "Baseline mode (gates measured, not enforced)",
        value=False,
        help="The control arm: what a conventional RAG system would return.",
    )
    return {"top_k": top_k, "baseline": baseline}


def _render_verdict(report) -> None:
    icon, label, colour = VERDICT_STYLE.get(report.verdict, ("•", report.verdict, "#57606a"))
    st.markdown(
        f"<div style='padding:0.65rem 0.9rem;border-radius:8px;"
        f"border-left:5px solid {colour};background:rgba(127,127,127,0.10);'>"
        f"<strong>{icon} {label}</strong><br>"
        f"<span style='font-size:0.9rem;opacity:0.85'>{report.explanation}</span></div>",
        unsafe_allow_html=True,
    )


def _render_scores(report) -> None:
    c1, c2, c3 = st.columns(3)
    a = report.gate_a
    b = report.gate_b

    c1.metric(
        "Gate A · retrieval",
        f"{a.score:.3f}",
        f"{'pass' if a.passed else 'fail'} @ {settings.confidence.gate_a_threshold:.2f}",
        delta_color="normal" if a.passed else "inverse",
    )
    if b is None:
        c2.metric("Gate B · groundedness", "—", "generation skipped")
    else:
        c2.metric(
            "Gate B · groundedness",
            f"{b.score:.3f}",
            f"{'pass' if b.passed else 'fail'} @ {settings.confidence.gate_b_threshold:.2f}",
            delta_color="normal" if b.passed else "inverse",
        )
    c3.metric(
        "Composite (weighted harmonic)",
        f"{report.composite:.3f}",
        f"threshold {settings.confidence.composite_threshold:.2f}",
    )

    with st.expander("Retrieval signals (features for threshold calibration)"):
        st.write(
            {
                "reranker relevancy (drives Gate A)": round(a.rerank_relevancy, 4),
                "top embedding similarity": round(a.top_similarity, 4),
                "mean embedding similarity": round(a.mean_similarity, 4),
                "similarity margin (top1 − top2)": round(a.similarity_margin, 4),
                "per-passage relevance": [round(x, 3) for x in a.per_chunk_relevance],
                "Gate A latency (ms)": round(a.latency_ms, 1),
            }
        )
        st.caption(
            "Embedding similarity is logged but does not drive the gate: MiniLM "
            "cosine values are not calibrated probabilities."
        )


CLAIM_BADGE = {
    "supported": ("✅", "supported"),
    "evident_conflict": ("❌", "evident conflict"),
    "baseless_information": ("⚠️", "baseless information"),
}


def _render_claims(report) -> None:
    b = report.gate_b
    if b is None or not b.claims:
        return

    counts = b.taxonomy_counts
    st.subheader("Claim-level analysis")
    st.caption(
        f"{counts['supported']} supported · {counts['evident_conflict']} evident conflict · "
        f"{counts['baseless_information']} baseless information "
        "(RAGTruth hallucination taxonomy)"
    )

    for i, c in enumerate(b.claims, 1):
        icon, label = CLAIM_BADGE[c.taxonomy_label]
        with st.expander(f"{icon} Claim {i}: {c.claim[:90]}{'…' if len(c.claim) > 90 else ''}"):
            st.write(f"**Label:** {label}")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Entailment", f"{c.entailment:.3f}")
            cc2.metric("Neutral", f"{c.neutral:.3f}")
            cc3.metric("Contradiction", f"{c.contradiction:.3f}")
            st.caption(f"Best-supporting passage: #{c.best_context_index + 1}")


def _render_passages(result) -> None:
    st.subheader("Retrieved passages")
    a = result.confidence.gate_a
    for i, chunk in enumerate(result.retrieval.chunks):
        rel = a.per_chunk_relevance[i] if i < len(a.per_chunk_relevance) else None
        header = f"[{i + 1}] {chunk.source} · similarity {chunk.similarity:.3f}"
        if rel is not None:
            header += f" · relevance {rel:.3f}"
        with st.expander(header):
            st.write(chunk.text)
            st.caption(str(chunk.metadata))


def _render_judge_notes(report) -> None:
    a, b = report.gate_a, report.gate_b
    notes = [n for n in (a.llm_reason, b.llm_reason if b else "") if n]
    if not notes:
        return
    with st.expander("LLM judge reasoning (DeepEval)"):
        if a.llm_relevancy is not None:
            st.write(f"**Contextual relevancy:** {a.llm_relevancy:.3f} — {a.llm_reason}")
        if b is not None and b.llm_faithfulness is not None:
            st.write(f"**Faithfulness:** {b.llm_faithfulness:.3f} — {b.llm_reason}")


SAMPLE_QUESTIONS = [
    "What are the five stages of a standard RAG pipeline?",
    "What is the difference between Baseless Information and Evident Conflict?",
    "Why can faithfulness only be computed after generation?",
    "What is the average annual rainfall in the Amazon basin?",
]


def main() -> None:
    opts = _sidebar()

    st.title("🛡️ Confidence-Aware RAG Pipeline")
    st.caption(
        "Two-gate hallucination detection: retrieval sufficiency is checked *before* "
        "generation, answer groundedness *after*. Low confidence abstains instead of guessing."
    )

    with st.expander("Index status", expanded=False):
        try:
            store = _store(settings.retrieval.collection_name)
            count = store._collection.count()
            if count:
                st.success(f"{count} chunks indexed in collection '{settings.retrieval.collection_name}'.")
            else:
                st.warning(
                    "The index is empty. Put PDFs/notes in `data/raw/` and run "
                    "`python scripts/build_index.py --reset`."
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Vector store unavailable: {exc}")
            store = None

    st.markdown("**Try one of these:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, q in zip(cols, SAMPLE_QUESTIONS, strict=True):
        if col.button(q[:34] + "…", key=q, use_container_width=True):
            st.session_state["question"] = q

    question = st.text_input(
        "Ask a question about the indexed documents",
        key="question",
        placeholder="e.g. Why does RAG still hallucinate?",
    )

    if not question:
        st.info("Enter a question above, or click one of the samples.")
        return

    from src.pipeline import answer_question

    with st.spinner("Retrieving, gating, generating…"):
        try:
            result = answer_question(
                question, top_k=opts["top_k"], store=store, baseline=opts["baseline"]
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
            return

    _render_verdict(result.confidence)
    st.markdown("### Answer")
    st.write(result.displayed_answer)
    if result.confidence.abstained and result.answer:
        with st.expander("What the model would have said (suppressed)"):
            st.write(result.answer)

    _render_scores(result.confidence)
    _render_claims(result.confidence)
    _render_judge_notes(result.confidence)
    _render_passages(result)

    timings = [f"retrieval {result.retrieval_ms:.0f}ms"]
    if result.generation:
        timings.append(f"generation {result.generation.latency_ms:.0f}ms")
    timings.append(f"Gate A {result.confidence.gate_a.latency_ms:.0f}ms")
    if result.confidence.gate_b:
        timings.append(f"Gate B {result.confidence.gate_b.latency_ms:.0f}ms")
    timings.append(f"total {result.total_ms:.0f}ms")
    st.caption(" · ".join(timings))
    st.caption(f"Sources: {', '.join(result.sources) or 'none'}")


if __name__ == "__main__":
    main()



