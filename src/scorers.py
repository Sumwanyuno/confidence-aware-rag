"""Scoring backends for the confidence layer.

Three independent ways to score the same thing, so the report can compare them
rather than assert that one is correct:

* ``RerankScorer``  - Gate A. Query-passage relevance via an ms-marco
  cross-encoder. Answers "does the corpus cover this question at all", which is
  a ranking problem and is decided *before* any tokens are generated.
* ``NLIScorer``     - Gate B. Claim-level entailment via a DeBERTa NLI
  cross-encoder. Answers "is each sentence of the answer actually supported by
  a retrieved passage", and distinguishes contradiction from mere absence.
* ``LLMJudgeScorer`` - reference standard. DeepEval's Faithfulness and
  Contextual Relevancy metrics. Accurate but costs LLM calls per answer, which
  is why the local scorers exist at all.

The local scorers run on the GPU in milliseconds, which is what makes a
several-thousand-question benchmark sweep feasible on a laptop.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass

from src.config import settings

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


def split_claims(answer: str, *, min_chars: int = 12) -> list[str]:
    """Split an answer into atomic claims.

    A sentence splitter is a deliberate simplification: proper claim extraction
    needs an LLM call per answer, which defeats the purpose of a cheap local
    scorer. Very short fragments are dropped because "Yes." or "See above."
    entail nothing and would drag the mean down without carrying meaning.
    """
    text = " ".join(answer.split())
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text)]
    claims = [p for p in parts if len(p) >= min_chars]
    return claims or ([text] if text else [])


def _device() -> str:
    requested = settings.confidence.nli_device
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return requested
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


@dataclass
class ClaimScore:
    """NLI verdict for one claim against the best-matching retrieved passage."""

    claim: str
    entailment: float
    neutral: float
    contradiction: float
    best_context_index: int

    @property
    def is_supported(self) -> bool:
        return self.entailment >= settings.confidence.support_threshold

    @property
    def is_contradicted(self) -> bool:
        return self.contradiction >= settings.confidence.contradiction_threshold

    @property
    def taxonomy_label(self) -> str:
        """RAGTruth-style label for this claim.

        Contradiction is checked first because a claim that conflicts with the
        source is a strictly worse failure than one the source is silent about.
        """
        if self.is_contradicted:
            return "evident_conflict"
        if not self.is_supported:
            return "baseless_information"
        return "supported"


@functools.lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    dev = _device()
    log.info("Loading cross-encoder %s on %s", model_name, dev)
    return CrossEncoder(model_name, device=dev, max_length=512)


class NLIScorer:
    """Gate B: claim-level groundedness via natural language inference."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.confidence.nli_model
        self._model = None
        self._label_order: list[str] | None = None

    def _ensure(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
            self._label_order = self._resolve_label_order(self._model)
        return self._model

    @staticmethod
    def _resolve_label_order(model) -> list[str]:
        """Read label order from the checkpoint instead of assuming it.

        NLI checkpoints disagree on ordering - some are
        (contradiction, entailment, neutral), others
        (entailment, neutral, contradiction). Hardcoding one silently inverts
        every score on the other, so the mapping is read from the config and
        only falls back to a documented default if that is unavailable.
        """
        id2label = getattr(getattr(model, "config", None), "id2label", None)
        if isinstance(id2label, dict) and id2label:
            try:
                return [str(id2label[i]).lower() for i in sorted(id2label)]
            except Exception:  # noqa: BLE001
                pass
        log.warning("Could not read id2label; assuming DeBERTa NLI default order.")
        return ["contradiction", "entailment", "neutral"]

    def _softmax_rows(self, logits):
        import numpy as np

        arr = np.asarray(logits, dtype=float)
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
        arr = arr - arr.max(axis=1, keepdims=True)
        exp = np.exp(arr)
        return exp / exp.sum(axis=1, keepdims=True)


    def score_claims(self, claims: list[str], contexts: list[str]) -> list[ClaimScore]:
        """Score every claim against every context, keeping the best support.

        Taking the max over contexts (rather than concatenating them into one
        premise) is deliberate: a claim supported by passage 3 alone is properly
        grounded, and concatenation would push the real premise past the
        model's 512-token window.
        """
        if not claims or not contexts:
            return []

        model = self._ensure()
        pairs = [(ctx, claim) for claim in claims for ctx in contexts]
        probs = self._softmax_rows(model.predict(pairs, show_progress_bar=False))

        order = self._label_order or ["contradiction", "entailment", "neutral"]
        idx = {name: i for i, name in enumerate(order)}
        e_i, n_i, c_i = idx.get("entailment", 1), idx.get("neutral", 2), idx.get("contradiction", 0)

        n_ctx = len(contexts)
        out: list[ClaimScore] = []
        for ci, claim in enumerate(claims):
            rows = probs[ci * n_ctx : (ci + 1) * n_ctx]
            best = int(max(range(n_ctx), key=lambda j: rows[j][e_i]))
            out.append(
                ClaimScore(
                    claim=claim,
                    entailment=float(rows[best][e_i]),
                    neutral=float(rows[best][n_i]),
                    # Worst-case contradiction across all passages: conflicting
                    # with any source is a conflict, even if another is neutral.
                    contradiction=float(max(r[c_i] for r in rows)),
                    best_context_index=best,
                )
            )
        return out

    def faithfulness(self, answer: str, contexts: list[str]) -> tuple[float, list[ClaimScore]]:
        """Mean entailment probability over the answer's claims.

        Not the *fraction* of claims that clear ``support_threshold``: a mean of
        probabilities degrades smoothly, so an answer that is 0.49-entailed
        throughout scores near 0.49 instead of collapsing to 0.0 the way a
        hard-counted fraction would. The per-claim labels in ``ClaimScore``
        still use the threshold; only this aggregate ignores it.
        """
        claims = split_claims(answer)
        scores = self.score_claims(claims, contexts)
        if not scores:
            return 0.0, []
        return sum(s.entailment for s in scores) / len(scores), scores


class RerankScorer:
    """Gate A: query-passage relevance, computed before generation."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.confidence.rerank_model
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def relevance(self, query: str, contexts: list[str]) -> list[float]:
        """Per-passage relevance in [0, 1].

        ms-marco cross-encoders emit an unbounded logit, so a sigmoid maps it
        into a probability-like range. The transform is monotonic, so it changes
        the calibration of the threshold but never the ranking.
        """
        if not contexts:
            return []
        import numpy as np

        raw = np.asarray(
            self._ensure().predict([(query, c) for c in contexts], show_progress_bar=False),
            dtype=float,
        ).ravel()
        return (1.0 / (1.0 + np.exp(-raw))).tolist()

    def contextual_relevancy(self, query: str, contexts: list[str]) -> tuple[float, list[float]]:
        """Aggregate Gate A signal.

        Uses the mean of the top-2 passages rather than the single best. One
        lucky high-scoring chunk is a weak basis for answering a question, and
        the max alone proved easy to saturate; the top-2 mean is the cheapest
        aggregate that resists both problems.
        """
        scores = self.relevance(query, contexts)
        if not scores:
            return 0.0, []
        ranked = sorted(scores, reverse=True)
        return float(sum(ranked[:2]) / min(2, len(ranked))), scores


def _build_deepeval_judge():
    """Create the DeepEval judge, subclassing its base class at runtime.

    Built dynamically because DeepEval's base class must be imported before it
    can be subclassed, and importing DeepEval at module scope would make the
    local NLI path depend on a package it never uses.
    """
    from deepeval.models.base_model import DeepEvalBaseLLM

    from src.llm import get_judge

    class Judge(DeepEvalBaseLLM):
        def __init__(self):
            self._chat = get_judge()

        def load_model(self):
            return self._chat

        def generate(self, prompt: str, *args, **kwargs) -> str:
            return str(self._chat.invoke(prompt).content)

        async def a_generate(self, prompt: str, *args, **kwargs) -> str:
            reply = await self._chat.ainvoke(prompt)
            return str(reply.content)

        def get_model_name(self) -> str:
            return f"{settings.llm.provider}:{settings.llm.judge_model}"

    return Judge()


class LLMJudgeScorer:
    """Reference standard: DeepEval's Faithfulness + Contextual Relevancy.

    Each call costs several LLM round-trips, so this path is used for the
    agreement study and for spot checks in the UI, never for a full sweep over
    thousands of benchmark questions.
    """

    def __init__(self):
        self._judge = None

    def _ensure(self):
        if self._judge is None:
            self._judge = _build_deepeval_judge()
        return self._judge

    def _measure(self, metric_cls, *, query: str, answer: str, contexts: list[str]):
        from deepeval.test_case import LLMTestCase

        metric = metric_cls(model=self._ensure(), include_reason=True, async_mode=False)
        case = LLMTestCase(input=query, actual_output=answer, retrieval_context=contexts)
        metric.measure(case)
        return float(metric.score or 0.0), str(getattr(metric, "reason", "") or "")

    def faithfulness(self, query: str, answer: str, contexts: list[str]) -> tuple[float, str]:
        from deepeval.metrics import FaithfulnessMetric

        return self._measure(FaithfulnessMetric, query=query, answer=answer, contexts=contexts)

    def contextual_relevancy(
        self, query: str, contexts: list[str], answer: str = "n/a"
    ) -> tuple[float, str]:
        from deepeval.metrics import ContextualRelevancyMetric

        return self._measure(
            ContextualRelevancyMetric, query=query, answer=answer, contexts=contexts
        )


@functools.lru_cache(maxsize=1)
def get_nli_scorer() -> NLIScorer:
    return NLIScorer()


@functools.lru_cache(maxsize=1)
def get_rerank_scorer() -> RerankScorer:
    return RerankScorer()


@functools.lru_cache(maxsize=1)
def get_llm_judge() -> LLMJudgeScorer:
    return LLMJudgeScorer()






