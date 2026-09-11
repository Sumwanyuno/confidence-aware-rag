"""Unit tests for the SQuAD 2.0 scoring used by the benchmark evaluation.

Model-free like ``test_confidence.py``: the normalisation rules and the
no-answer convention are where a silent mistake would corrupt every headline
number in the report, so they are pinned against examples from the official
SQuAD 2.0 evaluation script.
"""

from __future__ import annotations

from src.squad_metrics import (
    exact_match_score,
    f1_score,
    is_abstention,
    normalize_answer,
)


class TestNormalisation:
    def test_official_examples(self):
        # Straight from the SQuAD evaluation reference implementation.
        assert normalize_answer("The Battery") == "battery"
        assert normalize_answer("A battery?") == "battery"
        assert normalize_answer("the battery") == "battery"
        assert normalize_answer("battery  ") == "battery"

    def test_punctuation_and_articles(self):
        # Hyphens are deleted without a space, per the reference implementation.
        assert normalize_answer("Saint-Denis!") == "saintdenis"
        assert normalize_answer("the 47th Legion") == "47th legion"

    def test_case_and_whitespace(self):
        assert normalize_answer("  mixed   CASE   text ") == "mixed case text"


class TestAbstentionDetection:
    def test_empty_and_blank_count_as_abstention(self):
        assert is_abstention("")
        assert is_abstention("   ")

    def test_gate_abstain_message_detected(self):
        assert is_abstention(
            "I am not confident enough to answer this from the indexed documents. "
            "The retrieved passages do not appear to contain sufficient support."
        )

    def test_generator_refusal_marker_detected(self):
        assert is_abstention("INSUFFICIENT_CONTEXT")

    def test_a_real_answer_is_not_an_abstention(self):
        assert not is_abstention("The Normans were descended from Vikings.")


class TestExactMatch:
    def test_matches_any_gold_answer(self):
        assert exact_match_score("a battery", ["battery", "cell"]) == 1.0
        assert exact_match_score("cell", ["battery", "cell"]) == 1.0

    def test_mismatch_is_zero(self):
        assert exact_match_score("the zebra", ["battery"]) == 0.0

    def test_unanswerable_abstention_scores_full_credit(self):
        assert exact_match_score("I am not confident enough to answer.", []) == 1.0
        assert exact_match_score("", []) == 1.0

    def test_unanswerable_answered_scores_zero(self):
        assert exact_match_score("Normandy", []) == 0.0

    def test_answerable_abstention_scores_zero(self):
        assert exact_match_score("I am not confident enough to answer.", ["Normandy"]) == 0.0


class TestF1:
    def test_identical_answers_score_one(self):
        assert f1_score("battery", ["battery"]) == 1.0

    def test_partial_overlap(self):
        # prediction: [in, the, 13th, century], gold: [13th, century]
        f1 = f1_score("In the 13th century.", ["13th century"])
        assert 0.0 < f1 < 1.0
        assert f1 == 0.8  # 2 * (2/4 * 2/2) / (2/4 + 2/2)

    def test_best_match_over_multiple_golds(self):
        assert f1_score("sixteenth century", ["13th century", "sixteenth century"]) == 1.0

    def test_unanswerable_abstention_scores_one(self):
        assert f1_score("not confident enough to answer this", []) == 1.0

    def test_unanswerable_answered_scores_zero(self):
        assert f1_score("Normandy", []) == 0.0

    def test_answerable_abstention_scores_zero(self):
        assert f1_score("not confident enough to answer this", ["Normandy"]) == 0.0
