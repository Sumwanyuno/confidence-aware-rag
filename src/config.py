"""Central configuration for the confidence-aware RAG pipeline.

Every tunable lives here so that experiments are reproducible from a single
file and so the Streamlit UI has one place to read defaults from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BENCH_DIR = DATA_DIR / "benchmarks"
CHROMA_DIR = DATA_DIR / "chroma"
RUNS_DB = DATA_DIR / "runs.sqlite3"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, BENCH_DIR, CHROMA_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

Provider = Literal["ollama", "groq", "gemini"]


@dataclass
class LLMConfig:
    """Which model answers the question, and which model judges the answer.

    The judge is kept separate from the generator on purpose: a model grading
    its own output is a known source of self-preference bias, and keeping the
    two swappable lets us measure how much that bias costs us.
    """

    provider: Provider = os.getenv("RAG_PROVIDER", "ollama")  # type: ignore[assignment]

    # Generation
    generator_model: str = os.getenv("RAG_GENERATOR_MODEL", "llama3.1:8b")
    temperature: float = 0.0  # deterministic: hallucination results must repeat
    max_tokens: int = 512

    # Judging (LLM-as-judge path)
    judge_model: str = os.getenv("RAG_JUDGE_MODEL", "qwen2.5:7b-instruct")
    judge_temperature: float = 0.0

    # Networking
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    request_timeout: int = 180
    max_retries: int = 3


@dataclass
class RetrievalConfig:
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("RAG_EMBEDDING_DEVICE", "cuda")  # falls back to cpu
    collection_name: str = "cornerstone"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # Chroma returns squared-L2 distance for normalised embeddings; we convert
    # to cosine similarity in retriever.py so all scores are in [0, 1].
    distance_metric: str = "cosine"


@dataclass
class ConfidenceConfig:
    """Thresholds and weights for the two-gate confidence layer.

    The defaults below are *placeholders*. They are replaced by the values
    written to reports/calibration.json once scripts/calibrate.py has been run
    against a labelled dev split (see Milestone 3). Shipping uncalibrated
    thresholds and calling them tuned would be the exact overconfidence this
    project is about.
    """

    # Gate A - retrieval sufficiency, evaluated BEFORE generation.
    gate_a_threshold: float = 0.45
    # Gate B - answer groundedness, evaluated AFTER generation.
    gate_b_threshold: float = 0.60
    # Composite score below this is reported as low confidence even if both
    # individual gates squeaked through.
    composite_threshold: float = 0.55

    # Weighted harmonic mean weights: (retrieval_relevancy, faithfulness).
    # Faithfulness is weighted higher because a fluent-but-unsupported answer
    # is the failure mode we care most about catching.
    weights: tuple[float, float] = (0.4, 0.6)

    # Scorer backend for Gate A / Gate B.
    #   "nli"  -> local cross-encoders, milliseconds, no API calls
    #   "llm"  -> DeepEval LLM-as-judge, slower but the reference standard
    #   "both" -> run both and log the disagreement (used for the agreement study)
    scorer: Literal["nli", "llm", "both"] = os.getenv("RAG_SCORER", "nli")  # type: ignore[assignment]

    # Gate B backend: natural-language-inference cross-encoder. Premise is a
    # retrieved passage, hypothesis is one claim from the generated answer.
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    # Gate A backend: a relevance reranker, NOT an NLI model. Before generation
    # there is no hypothesis to entail, so the question is "does this passage
    # answer this query", which is a ranking task.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_device: str = os.getenv("RAG_NLI_DEVICE", "cuda")

    # A claim whose contradiction probability exceeds this is logged as
    # RAGTruth "Evident Conflict"; an unsupported non-contradicting claim is
    # logged as "Baseless Information".
    contradiction_threshold: float = 0.50
    # Score a contradicted answer is capped at. Deliberately an absolute value
    # rather than "just under gate_b_threshold": once calibration moves the
    # threshold the old formula would move the penalty with it, and a threshold
    # fitted near zero would turn the cap into a no-op. 0.15 sits below any
    # plausible fitted threshold and is low enough that the harmonic composite
    # collapses, so a contradicted answer can never read as merely borderline.
    contradiction_cap: float = 0.15
    # Below this entailment probability a claim counts as unsupported.
    support_threshold: float = 0.50

    # Abstention message shown to the user when a gate fails.
    abstain_message: str = (
        "I am not confident enough to answer this from the indexed documents. "
        "The retrieved passages do not appear to contain sufficient support."
    )


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)


settings = Settings()
