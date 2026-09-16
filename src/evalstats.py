"""Selective-prediction statistics: risk-coverage, AURC, and judge agreement.

Model-free maths separated from the scripts that use it, for the same reason
the harmonic mean lives in ``confidence.py``: every number that ends up in the
final report should be unit-testable without a GPU or a language model.

Definitions (Geifman & El-Yaniv, 2017, and the SAnthiA/DeepRisk formulation):
  coverage - the fraction of questions the system chooses to answer.
  risk     - the error rate among those answered questions.
  AURC     - area under the risk-coverage curve; lower is better. It averages
             the risk over all coverage levels, so a system that answers
             everything at 30% error is worse than one that answers half at 5%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    coverage: float
    risk: float


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = _ranks(xs)
    ry = _ranks(ys)
    return _pearson(rx, ry)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _binary_agreement(xs: list[bool], ys: list[bool]) -> dict:
    """Cohen's kappa and raw proportions for two binary labellings."""
    n = len(xs)
    if n == 0:
        return {"n": 0}
    yes_yes = sum(1 for a, b in zip(xs, ys, strict=True) if a and b)
    yes_no = sum(1 for a, b in zip(xs, ys, strict=True) if a and not b)
    no_yes = sum(1 for a, b in zip(xs, ys, strict=True) if not a and b)
    no_no = sum(1 for a, b in zip(xs, ys, strict=True) if not a and not b)
    po = (yes_yes + no_no) / n
    px = (yes_yes + yes_no) / n
    py = (yes_yes + no_yes) / n
    pe = px * py + (1 - px) * (1 - py)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {
        "n": n,
        "observed_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4) if not math.isnan(kappa) else None,
        "both_pass": yes_yes,
        "only_x": yes_no,
        "only_y": no_yes,
        "both_fail": no_no,
    }


def risk_coverage_curve(scores: list[float], correct: list[bool]) -> list[Point]:
    """The risk-coverage curve for one arm.

    Sorts questions by confidence descending, then walks down the ranking:
    answering the top k is every coverage level, and the risk at each level is
    the fraction of wrong answers among those k. Ties are handled by treating
    all members of a tie block as included together (a tie block must not
    produce a coverage point mid-block, or the curve depends on sort order).
    """
    if len(scores) != len(correct):
        raise ValueError("scores and correct must be the same length")
    if not scores:
        return []
    pairs = sorted(zip(scores, correct, strict=True), key=lambda p: -p[0])

    total = len(pairs)
    n_wrong = 0
    curve = [Point(0.0, 0.0)]
    i = 0
    while i < total:
        # Extend through the whole tie block.
        j = i
        while j + 1 < total and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            n_wrong += 0 if pairs[k][1] else 1
        n_ans = j + 1
        curve.append(Point(n_ans / total, n_wrong / n_ans))
        i = j + 1
    return curve


def aurc(curve: list[Point]) -> float:
    """Area under the risk-coverage curve via the trapezoidal rule.

    Risk at full coverage is included, so always-answering systems get their
    raw error rate as their AURC and selective systems can only do better.
    """
    if len(curve) < 2:
        return float("nan")
    area = 0.0
    for a, b in zip(curve[:-1], curve[1:], strict=True):
        area += (b.coverage - a.coverage) * (a.risk + b.risk) / 2
    return area


def risk_at_coverage(curve: list[Point], coverage: float) -> float:
    """Risk when answering at most ``coverage`` of the questions.

    The highest-coverage point at or below the target; if the target sits above
    every recorded point, the last point is returned.
    """
    if not curve:
        return float("nan")
    best = curve[0]
    for p in curve:
        if p.coverage <= coverage + 1e-9:
            best = p
    return best.risk


def coverage_at_risk(curve: list[Point], risk_cap: float) -> float:
    """The widest coverage whose risk stays at or below the cap.

    This is the operating point a deployment would actually choose: "answer as
    many questions as you can without exceeding X% error".
    """
    if not curve:
        return float("nan")
    best = curve[0]
    for p in curve:
        if p.risk <= risk_cap + 1e-9:
            best = max(best, p, key=lambda q: q.coverage)
    return best.coverage


def judge_agreement(
    nli_scores: list[float], llm_scores: list[float], threshold: float
) -> dict:
    """Agreement between two scorers on the same items at one threshold.

    ``threshold`` is applied to both, i.e. the local scorer's own calibrated
    threshold is used for both sides - the question is whether the judge would
    have made the same pass/fail call, not whether their scales coincide.
    """
    if len(nli_scores) != len(llm_scores):
        raise ValueError("score lists must be the same length")
    # NaN as well as None: reading telemetry through pandas turns SQL NULL into
    # float('nan'), which passes an ``is not None`` test and then silently
    # poisons every sum it touches. Rows where only one scorer ran are simply
    # not comparable, so they are dropped here rather than reported as NaN.
    pairs = [
        (nli, llm)
        for nli, llm in zip(nli_scores, llm_scores, strict=True)
        if nli is not None and llm is not None
        and not math.isnan(float(nli)) and not math.isnan(float(llm))
    ]
    if not pairs:
        return {"n": 0}
    ns = [float(p[0]) for p in pairs]
    ls = [float(p[1]) for p in pairs]
    return {
        "n": len(pairs),
        "pearson": round(_pearson(ns, ls), 4),
        "spearman": round(_spearman(ns, ls), 4),
        "mean_nli": round(sum(ns) / len(ns), 4),
        "mean_llm": round(sum(ls) / len(ls), 4),
        "decision_agreement": _binary_agreement(
            [s >= threshold for s in ns], [s >= threshold for s in ls]
        ),
    }
