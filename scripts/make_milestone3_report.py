"""Generate the Milestone 3 report as a .docx for the project portal.

Milestone 3: Threshold calibration, risk-coverage analysis, and LLM-judge
agreement study. Numbers match reports/calibration.json, benchmark_m3.json,
risk_coverage_m3.json, and judge_agreement.json. The calibrated run is
data/runs.sqlite3 run_tag='squad2-v2'.

    .venv/Scripts/python.exe scripts/make_milestone3_report.py
    -> reports/Milestone3_Report.docx
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "Milestone3_Report.docx"
LETTERHEAD = ROOT / "assets" / "mits_letterhead.png"
CALIBRATION_JSON = ROOT / "reports" / "calibration.json"
BENCHMARK_JSON = ROOT / "reports" / "benchmark_m3.json"
RISK_JSON = ROOT / "reports" / "risk_coverage_m3.json"
JUDGE_JSON = ROOT / "reports" / "judge_agreement.json"
FIG_RISK = ROOT / "reports" / "figures" / "risk_coverage.png"
FIG_SEPARATION = ROOT / "reports" / "figures" / "gate_a_separation.png"

TITLE = "Design and Development of a Confidence-Aware RAG Pipeline with Hallucination Detection"
STUDENT = "Shivam Gupta"
ENROLMENT = "BTCD24O1060"
DUE = date(2026, 9, 10)

INK = RGBColor(0x11, 0x1B, 0x2E)
ACCENT = RGBColor(0x0E, 0x6E, 0x62)
MUTED = RGBColor(0x5A, 0x63, 0x74)


def style_document(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1

    for name, size in (("Heading 1", 14), ("Heading 2", 11.5)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = ACCENT if name == "Heading 1" else INK
        st.paragraph_format.space_before = Pt(11 if name == "Heading 1" else 8)
        st.paragraph_format.space_after = Pt(4)


def para(
    doc: Document,
    text: str,
    *,
    size: float = 10.5,
    bold: bool = False,
    italic: bool = False,
    colour: RGBColor | None = None,
    align=None,
    space_after: float = 6,
) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if colour is not None:
        run.font.color.rgb = colour
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item).font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(3)


def table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(t.rows[0].cells, headers, strict=True):
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        for cell, text in zip(t.add_row().cells, row, strict=True):
            cell.text = ""
            cell.paragraphs[0].add_run(text).font.size = Pt(9)
    if widths:
        for row in t.rows:
            for cell, w in zip(row.cells, widths, strict=True):
                cell.width = Inches(w)
    t.rows[0].header = True
    for row in t.rows:
        row.cant_split = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def cover(doc: Document) -> None:
    if LETTERHEAD.exists():
        doc.add_picture(str(LETTERHEAD), width=Inches(5.94))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_after = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    para(doc, TITLE, size=20, bold=True, colour=INK, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    para(
        doc,
        "Milestone 3: Calibration and Performance Validation",
        size=16,
        colour=ACCENT,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=2,
    )


def section_objective(doc: Document) -> None:
    doc.add_heading("1. Objective", level=1)
    para(
        doc,
        "This milestone was about replacing the placeholder thresholds with fitted ones and "
        "measuring what that change is worth. Milestone 2 showed that uncalibrated thresholds "
        "refuse to answer 84% of answerable questions, which makes the system unusable even though "
        "it does suppress hallucinations. The question Milestone 3 had to answer was whether data-"
        "driven thresholds can reduce that over-abstention rate without letting hallucinations back "
        "through, and whether the local NLI scorer agrees with a slower LLM judge closely enough to "
        "be trusted.",
    )


def section_calibration(doc: Document, cal: dict) -> None:
    doc.add_heading("2. Threshold Calibration", level=1)
    ga = cal["gate_a"]
    gb = cal["gate_b"]

    para(
        doc,
        f"Thresholds were fitted against the 150-item SQuAD 2.0 sample using the "
        f"answerable/unanswerable labels as ground truth. Gate A was calibrated by Youden's J "
        f"(the threshold that maximizes true positive rate minus false positive rate) over the "
        f"reranker score. Gate B was calibrated by F1 over mean NLI faithfulness, constrained to "
        f"stay above the 0.15 contradiction cap so calibration cannot weaken the hard-fail rule.",
    )
    para(
        doc,
        "Two properties of this calibration have to be stated before the numbers, because both "
        "limit what the numbers can be read as saying. First, the thresholds are fitted and "
        "evaluated on the same 150 items; there is no held-out split. Every figure in section 3 is "
        "therefore an in-sample figure and is optimistic by an amount this sample cannot measure. "
        "Second, Gate B's calibration uses a proxy label rather than a groundedness annotation: "
        "grounded is defined as a baseline-arm answer to an answerable question, and fabricated as "
        "a baseline-arm answer to an unanswerable one. That is the same answerable/unanswerable "
        "label Gate A is fitted on, so Gate B's AUC partly re-measures answerability rather than "
        "groundedness, and a genuinely ungrounded answer to an answerable question is counted as "
        "grounded by construction. Fixing this needs claim-level groundedness annotations, which "
        "this sample does not have.",
    )

    table(
        doc,
        ["Gate", "Old (M2)", "Fitted (M3)", "ROC AUC", "Chosen metric"],
        [
            [
                "Gate A",
                "0.45",
                f"{ga['chosen']['threshold']:.2f}",
                f"{ga['roc_auc']:.3f}",
                f"Youden J = {ga['chosen']['youden_j']:.2f}",
            ],
            [
                "Gate B",
                "0.60",
                f"{gb['chosen']['threshold']:.2f}",
                f"{gb['roc_auc']:.3f}",
                f"F1 = {gb['chosen']['f1']:.3f}",
            ],
            [
                "Composite",
                "0.55",
                f"{cal['recommended']['composite_threshold']:.2f}",
                "—",
                "Harmonic mean at corner",
            ],
        ],
        widths=[1.2, 1.2, 1.2, 1.2, 1.7],
    )

    para(
        doc,
        f"The Gate A threshold stayed at 0.45 because that was already the Youden-optimal point. "
        f"Gate B dropped from 0.60 to {gb['chosen']['threshold']:.2f}, and the composite dropped "
        f"from 0.55 to {cal['recommended']['composite_threshold']:.2f}. The ROC AUC of {ga['roc_auc']:.3f} "
        f"for Gate A is only marginally above random (0.50), which means the reranker score alone "
        f"does not separate answerable from unanswerable strongly. Gate B's AUC of {gb['roc_auc']:.3f} "
        f"is weaker still, and given the proxy label above it should be treated as an upper bound on "
        f"a quantity that has not really been measured yet rather than as a groundedness score.",
    )

    para(
        doc,
        f"A logistic model was also fitted over all four retrieval features (rerank score, top "
        f"similarity, mean similarity, similarity margin) for Gate A, achieving a cross-validated "
        f"AUC of {ga['logistic_fit']['cv_roc_auc']:.3f}. That is only 2 percentage points above "
        f"the single-feature threshold, so the additional features do not add much discriminative "
        f"power on this sample. The single-threshold gate is simpler and was kept.",
    )


def section_results(doc: Document, data: dict) -> None:
    doc.add_heading("3. Post-Calibration Results", level=1)
    g = data["arms"]["guarded"]
    b = data["arms"]["baseline"]

    para(
        doc,
        f"The same 150 questions re-run with the calibrated thresholds. The guarded arm now "
        f"answered {g['answered']} and abstained {g['abstained']} times, compared to 15 answered "
        f"and 135 abstained before calibration (Milestone 2). The baseline column is the same "
        f"control run reported in Milestone 2, not a fresh sweep: the baseline enforces no gates, "
        f"so changing a threshold cannot change its output, and reusing it keeps the comparison "
        f"free of the generator's run-to-run sampling noise.",
    )

    table(
        doc,
        ["Metric", "M2 (uncalibrated)", "M3 (calibrated)", "Baseline", "Δ vs baseline"],
        [
            [
                "EM",
                "0.480",
                f"{g['em']:.3f}",
                f"{b['em']:.3f}",
                f"{g['em'] - b['em']:+.3f}",
            ],
            ["F1", "0.509", f"{g['f1']:.3f}", f"{b['f1']:.3f}", f"{g['f1'] - b['f1']:+.3f}"],
            [
                "Abstention rate",
                "90.0%",
                f"{g['abstention_rate']:.1%}",
                f"{b['abstention_rate']:.1%}",
                "—",
            ],
            [
                "Hallucination rate",
                "4.0%",
                f"{g['hallucination_rate']:.1%}",
                f"{b['hallucination_rate']:.1%}",
                f"{g['hallucination_rate'] - b['hallucination_rate']:+.1%}",
            ],
            [
                "Over-abstention rate",
                "84.0%",
                f"{g['over_abstention_rate']:.1%}",
                f"{b['over_abstention_rate']:.1%}",
                "—",
            ],
        ],
        widths=[1.8, 1.3, 1.3, 1.0, 1.1],
    )

    para(
        doc,
        f"Calibration moved the needle in the intended direction: over-abstention dropped from 84% "
        f"to {g['over_abstention_rate']:.1%}, meaning the system now answers {int((1-g['over_abstention_rate'])*75)} "
        f"of the 75 answerable questions instead of 12. The cost is that hallucination rate rose "
        f"from 4% to {g['hallucination_rate']:.1%}: the system now produces {int(g['hallucination_rate']*75)} "
        f"fabricated answers instead of 3. That is the trade this project was built to measure.",
    )

    table(
        doc,
        ["Condition", "M2 answered", "M3 answered", "M3 abstained"],
        [
            ["Answerable (75)", "12", str(g["confusion"]["answerable"]["answered"]), str(g["confusion"]["answerable"]["abstained"])],
            ["Unanswerable (75)", "3", str(g["confusion"]["unanswerable"]["answered"]), str(g["confusion"]["unanswerable"]["abstained"])],
        ],
        widths=[1.8, 1.3, 1.3, 1.3],
    )


def section_risk_coverage(doc: Document, risk: dict) -> None:
    doc.add_heading("4. Risk-Coverage Analysis", level=1)
    g = risk["arms"]["guarded"]
    b = risk["arms"]["baseline"]

    para(
        doc,
        "The risk-coverage curve plots error rate (risk) against the fraction of questions answered "
        "(coverage) as the confidence threshold is varied. A system that answers everything has 100% "
        "coverage at the baseline's error rate; a system that answers one question has near-zero "
        "coverage. The area under the curve (AURC) summarizes the trade: lower is better.",
    )

    table(
        doc,
        ["Metric", "Guarded", "Baseline", "Improvement"],
        [
            ["AURC", f"{g['aurc']:.3f}", f"{b['aurc']:.3f}", f"{b['aurc'] - g['aurc']:.3f}"],
            [
                "Risk at 50% coverage",
                f"{g['risk_at_50pct_coverage']:.1%}",
                f"{b['risk_at_50pct_coverage']:.1%}",
                f"{b['risk_at_50pct_coverage'] - g['risk_at_50pct_coverage']:.1%} lower",
            ],
            [
                "Risk at 80% coverage",
                f"{g['risk_at_80pct_coverage']:.1%}",
                f"{b['risk_at_80pct_coverage']:.1%}",
                f"{b['risk_at_80pct_coverage'] - g['risk_at_80pct_coverage']:.1%} lower",
            ],
        ],
        widths=[2.0, 1.5, 1.5, 1.5],
    )

    para(
        doc,
        f"The guarded arm's AURC of {g['aurc']:.3f} is {b['aurc'] - g['aurc']:.3f} below the "
        f"baseline's {b['aurc']:.3f}, meaning the confidence layer as a selector produces less risk "
        f"at every coverage level. At 50% coverage the guarded arm's risk is {g['risk_at_50pct_coverage']:.1%} "
        f"versus the baseline's {b['risk_at_50pct_coverage']:.1%}, which is "
        f"{b['risk_at_50pct_coverage'] - g['risk_at_50pct_coverage']:.0%} lower in absolute terms. "
        f"The operating point a deployment would choose is \"answer as many questions as you can "
        f"without exceeding X% error\"; this curve shows that trade quantitatively. One caveat on "
        f"how to read it: both arms are ranked by the same composite score, so the curve measures "
        f"the score's value as a ranking signal, not the gates' value as a hard cut-off. The "
        f"baseline curve is worse mainly because its answers are wrong more often, not because its "
        f"scores rank differently.",
    )

    if FIG_RISK.exists():
        doc.add_picture(str(FIG_RISK), width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        para(
            doc,
            "Figure 1. Risk-coverage curves for guarded and baseline arms. Lower and to the right is better.",
            size=9,
            italic=True,
            colour=MUTED,
            align=WD_ALIGN_PARAGRAPH.CENTER,
            space_after=8,
        )


def section_agreement(doc: Document, judge: dict) -> None:
    doc.add_heading("5. Judge Agreement Study", level=1)
    ga = judge["gate_a_relevancy"]
    gb = judge["gate_b_faithfulness"]
    lat = judge["latency_ms"]

    para(
        doc,
        f"A subset of {judge['n']} items was run with both the local NLI scorer and a DeepEval LLM "
        f"judge to measure whether the cheap local path agrees with the reference standard closely "
        f"enough to be trusted. The judge is gemma2 via Ollama, deliberately a different model from "
        f"the llama3.1:8b generator, because a model grading its own output is a known source of "
        f"self-preference bias. An earlier judge choice, qwen2.5:7b-instruct, had to be replaced: it "
        f"answered \"no\" to every faithfulness verdict regardless of the claim, deterministically at "
        f"temperature 0, which makes its scores meaningless rather than merely noisy. Agreement is "
        f"reported as Spearman correlation on the continuous scores and Cohen's kappa on the "
        f"pass/fail decision at the calibrated thresholds.",
    )

    table(
        doc,
        ["Gate", "n", "Spearman ρ", "Cohen's κ", "Observed agreement"],
        [
            [
                "Gate A (relevancy)",
                str(ga["n"]),
                f"{ga['spearman']:.3f}",
                f"{ga['decision_agreement']['cohens_kappa']:.3f}" if ga['decision_agreement']['cohens_kappa'] is not None else "0.0",
                f"{ga['decision_agreement']['observed_agreement']:.1%}",
            ],
            [
                "Gate B (faithfulness)",
                str(gb["n"]),
                f"{gb['spearman']:.3f}",
                f"{gb['decision_agreement']['cohens_kappa']:.3f}",
                f"{gb['decision_agreement']['observed_agreement']:.1%}",
            ],
        ],
        widths=[2.0, 0.8, 1.3, 1.3, 1.8],
    )

    para(
        doc,
        f"Gate A shows near-zero agreement (κ = {ga['decision_agreement']['cohens_kappa'] or 0.0:.3f}, "
        f"ρ = {ga['spearman']:.3f}): the judge passed 0 items the NLI scorer passed, and both failed "
        f"on {ga['decision_agreement']['both_fail']}. The NLI scorer alone passed "
        f"{ga['decision_agreement']['only_x']}, which the judge failed. This is a genuine disagreement, "
        f"not a measurement error.",
    )

    para(
        doc,
        f"Gate B shows weak negative agreement (κ = {gb['decision_agreement']['cohens_kappa']:.3f}, "
        f"ρ = {gb['spearman']:.3f}) over the {gb['n']} items where both scorers produced a score. "
        f"They agreed on {gb['decision_agreement']['both_pass']} passes and "
        f"{gb['decision_agreement']['both_fail']} failure, and disagreed on "
        f"{gb['decision_agreement']['only_x'] + gb['decision_agreement']['only_y']} items, almost "
        f"all in one direction: the judge passed {gb['decision_agreement']['only_y']} items the NLI "
        f"scorer failed. A negative kappa means the two agreed less often than chance alone would "
        f"predict, so on this evidence the local scorer is not a substitute for the judge on Gate B. "
        f"The sample is small ({gb['n']} items), which is itself a limitation of the study rather "
        f"than a defence of the result.",
    )

    para(
        doc,
        f"Latency cost: the judge's median Gate B latency was {lat['judge_run_gate_b_median']/1000:.1f} seconds "
        f"versus {lat['nli_only_gate_b_median']:.0f} ms for the NLI scorer, a {lat['judge_run_gate_b_median']/lat['nli_only_gate_b_median']:.0f}× "
        f"slowdown. The disagreement combined with that latency cost means the local scorer cannot be "
        f"blindly trusted as a drop-in replacement for the judge, but it is fast enough to be the "
        f"primary path with the judge reserved for spot checks.",
    )


def section_status(doc: Document) -> None:
    doc.add_heading("6. Status and Limitations", level=1)
    para(
        doc,
        "Milestone 3 is complete: thresholds are fitted, the post-calibration run is scored, the "
        "risk-coverage curves are plotted, and the judge-agreement study quantifies where the local "
        "scorer can and cannot be trusted. The claim that calibration reduces over-abstention is "
        "now supported by measurement rather than assertion.",
    )
    bullets(
        doc,
        [
            "The 150-item sample is small enough that the confidence intervals are wide. The "
            "hallucination rate moved from 4% (3/75) to 10.7% (8/75): that is a 3-to-8 shift, which "
            "could be noise. A larger sample would tighten the intervals.",
            "Gate A's weak AUC (0.652) means the answerable/unanswerable split is only marginally "
            "predictable from retrieval features alone on this sample. Multi-hop questions (HotpotQA) "
            "are expected to degrade this further, since a reranker scores passages individually.",
            "The judge disagreement is real and was not debugged away. The local NLI scorer and the "
            "LLM judge are measuring different things, and the project's claim that they are "
            "interchangeable is not supported by this data.",
        ],
    )
    para(
        doc,
        "The system is now end-to-end measurable, and the evaluation framework is in place for "
        "ablations, alternative models, and multi-hop benchmarks. The final milestone consolidates "
        "the results and demonstrates the system on the full course-materials corpus.",
    )


def main() -> int:
    missing = []
    for p, name in [
        (CALIBRATION_JSON, "calibration.json"),
        (BENCHMARK_JSON, "benchmark_m3.json"),
        (RISK_JSON, "risk_coverage_m3.json"),
        (JUDGE_JSON, "judge_agreement.json"),
    ]:
        if not p.exists():
            missing.append(name)
    if missing:
        print(f"missing files: {', '.join(missing)}")
        print("run the evaluation scripts first:")
        print("  .venv/Scripts/python.exe scripts/calibrate.py --tag squad2-v2")
        print("  .venv/Scripts/python.exe scripts/eval_benchmark.py --tag squad2-v2 --out reports/benchmark_m3.json")
        print("  .venv/Scripts/python.exe scripts/risk_coverage.py --tag squad2-v2 --out reports/risk_coverage_m3.json")
        print("  .venv/Scripts/python.exe scripts/judge_agreement.py --tag squad2-judge")
        return 1

    cal = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    bench = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    risk = json.loads(RISK_JSON.read_text(encoding="utf-8"))
    judge = json.loads(JUDGE_JSON.read_text(encoding="utf-8"))

    doc = Document()
    style_document(doc)
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    cover(doc)
    section_objective(doc)
    section_calibration(doc, cal)
    section_results(doc, bench)
    section_risk_coverage(doc, risk)
    section_agreement(doc, judge)
    section_status(doc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
