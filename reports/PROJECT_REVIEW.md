# Project review: Milestones 1–5

*Confidence-Aware RAG Pipeline with Hallucination Detection — audit of the five
milestone reports against the code and the saved results. 2 November 2026.*

Every number below comes from the saved results (`data/runs.sqlite3`,
`reports/*.json`, `reports/generation_hotpot_m4_rows.jsonl`). Nothing was
regenerated. To reproduce: `python scripts/audit_results.py`, which writes
`reports/audit_m5.json`.

---

## 1. Is the project complete?

**As software, yes. As an evaluation, not yet.** All the parts are built and
working. But three scoring bugs and one design flaw make several headline
claims wrong, and in some places the real result is *better* than what the
reports say.

| Milestone | Built and working | Evidence holds up? |
|---|---|---|
| M1 Architecture | Ingestion, retrieval, both gates, composite, telemetry, Streamlit app, dashboard | Yes. Tests pass (now 86). |
| M2 Harness + baseline | SQuAD 2.0 harness, two arms, per-item throwaway index | Counts are right. "Identical configurations" is wrong (see H3). |
| M3 Calibration | Thresholds, risk–coverage, judge study | Thresholds are right. EM headline and AURC reading are not (H1, H4, H5). |
| M4 HotpotQA | Retrieval ablation, generation sweep | Retrieval numbers are right. Generation metrics are mis-scored (H6–H9). |
| M5 Consolidation | Tests 75→86 pass, ruff clean, README | Repeats M3/M4 errors. The M4/M5 scripts were never committed (fixed on this branch). |

---

## 2. Logical holes, most serious first

### H1 — The guarded arm's answers were scored as wrong because of the "[1]" citation (critical)

The guarded prompt tells the model to cite passages, so it answers
`"1894 [1]."`. SQuAD normalisation removes the brackets but keeps the digit,
which gives `"1894 1"`. That never equals `"1894"`, so **every guarded answer
got EM = 0**. `benchmark_m3.json` shows `em_answerable: 0.0`.

| SQuAD, M3 calibrated | As reported | Re-scored |
|---|---|---|
| Guarded, answered answerable questions that were actually right | "EM 0" | **20 of 21 (95%)** |
| Baseline, answerable questions right (gold appears in the answer) | EM 25% | **68 of 75 (91%)** |

**What changes:** when the guarded pipeline answers, it is almost always right.
EM is also the wrong metric for a generative model that answers in full
sentences, so the audit reports *gold containment* (the gold answer appears in
the response) alongside EM and F1.

**Fixed in code:** `src/squad_metrics.py` now has `strip_citations()` and
`contains_gold()`, with tests.

### H2 — Most over-abstention comes from Gate B rejecting correct answers (critical)

The reports call 72% over-abstention the price of safety. M4 §4.4 says it is
"dominated by fragmentary LLM outputs, not by the NLI model misfiring". The
data shows the opposite:

| Guarded abstentions on the 75 answerable items | M2 | M3 |
|---|---|---|
| Gate A refused | 15 | 15 |
| Generator refused itself | 3 | 3 |
| **Gate B suppressed** | **45** | **36** |
| …of which the suppressed answer was actually correct | **45 / 45** | **35 / 36** |
| …of which the contradiction veto fired on a correct answer | 22 | 15 |

Real examples of suppressed answers: `"New Orleans [1]"`,
`"August 15, 1971 [1]"`, `"The Sky service [1]"`. Gate B scored them about
0.000–0.005.

**Why:** NLI checks whether a premise entails a *statement*. "New Orleans" is a
noun phrase, not a statement, so nothing can entail it, and the model returns
neutral or even contradiction. The short answer is correct. The hypothesis
Gate B tests is malformed. Lowering the threshold cannot fix this, because the
scores sit near zero.

**Correct wording:** "Over-abstention is driven mainly by Gate B: the NLI
hypothesis is the bare answer, and short answers are not propositions. In M3,
35 of the 36 answers Gate B suppressed were correct."

**The fix (future work, or M5 if there is time):** build the hypothesis from
the question and the answer together, e.g. "Q: When did the war occur? A: 1894"
or a declarative rewrite. This is the standard QA-to-claim conversion.

### H3 — The baseline is not an "identical configuration" (major)

M2 §2.2 says both arms are identical except for the gates. They are not. The
guarded arm uses a different prompt (`SYSTEM_PROMPT` in `src/generator.py`):
"answer only from context, reply INSUFFICIENT_CONTEXT if unsure, cite
passages". The baseline prompt is just "Answer the question using the context
passages provided."

So guarded vs baseline measures **prompt + gates together**. The split on the
75 unanswerable items in M3:

| Caught by | Count |
|---|---|
| Gate A (generation skipped) | 33 |
| **Generator's own refusal (the prompt, not the gates)** | **27** |
| Gate B | 7 |
| Got through (hallucinated) | 8 |

**Correct wording:** "The guarded pipeline (abstention-aware prompt plus both
gates) answered 8 of 75 unanswerable questions, against 75 of 75 for the plain
baseline. Of the 67 refusals, 40 came from the gates and 27 from the
generator's own refusal." Isolating the gates needs a third arm: the guarded
prompt with no gates. That is about 10 minutes of Ollama time.

### H4 — M3 and M5 use EM 0.447 as a headline, which M2 already warned against (major)

M2 §3.3 explains correctly that SQuAD 2.0 gives an abstention on an
unanswerable question full credit, so a system that abstains a lot gets an
inflated aggregate score. M3 then reports "EM +0.320 vs baseline", and the M5
executive summary leads with "achieving an exact match of 0.447". Combined
with H1, **all** of the guarded EM comes from abstentions. The examiner can
point at your own M2 paragraph.

**Correct wording:** drop aggregate EM as a headline. Lead with the 2×2 table
(answered/abstained × answerable/unanswerable) and with *selective accuracy*:
when the guarded system answers, it is right 95% of the time (20/21) and
hallucinates on 8 of 75 unanswerable questions.

### H5 — The risk–coverage comparison does not measure what M3 says (major)

`scripts/risk_coverage.py` ranks all 150 guarded rows, including abstentions,
by composite. It counts an abstention on an unanswerable item as a "correct
answer" and scores correctness with EM, which has the H1 bug. So the guarded
curve mostly measures "abstaining is rewarded on SQuAD 2.0". M3 itself admits
the baseline "is worse primarily because its generated answers are wrong more
often, not because its ranking signal differs". That sentence undercuts the
claim that the confidence layer "consistently produces less risk across every
coverage level".

**The clean test** uses the baseline arm only, because it answers everything
and records every gate score without acting on it. Then ask whether each score
ranks right answers above wrong ones:

| Signal (baseline arm, n = 150) | AUROC (0.5 = coin flip) | AURC (lower is better) |
|---|---|---|
| Random ranking | 0.50 | 0.547 |
| **Gate A (reranker)** | **0.63** | **0.460** |
| Composite | 0.58 | 0.506 |
| Gate B (NLI) | 0.54 | 0.524 |
| Perfect ranking | 1.00 | 0.149 |

**How to read it:** the confidence signal is real but modest, and Gate A
carries most of it. On SQuAD 2.0, Gate B is close to chance. See H10 for why
that is expected.

### H6 — M4's "contradiction rate" is really a Gate B failure rate (major)

`run_generation_hotpot.py` set `contradiction = gate_b.score < gate_b_threshold`.
That flag is true for *any* Gate B failure, and most failures are low
entailment ("baseless"), not contradictions. The contradiction veto clamps the
score to exactly 0.15. Counting rows at exactly 0.15 gives a **lower bound of
11–14%**, not 70–75%. M4 §4.2's "the contradiction head fires" is not
supported by the data.

**Correct wording:** "Gate B failed 71–75% of baseline answers (mostly low
entailment; the contradiction veto fired on at least 11–14%)."

**Fixed in code:** the run script now records `gate_b_failed` and a true
`contradiction` (any claim labelled evident_conflict) as separate fields.

### H7 — M4 says 70–75% of flags hit "mostly valid answers", but its own table says 89–91% are hallucinations (major)

Both statements are mis-scored. Re-scored by gold containment:

| HotpotQA baseline | Sentence | Chunk |
|---|---|---|
| Accuracy (gold in answer) | 43% | 59% |
| Gate B fails a **correct** answer | **62.5%** (40/64) | **65%** |
| Gate B fails an **incorrect** answer | 84% (72/86) | 79% |
| Gate B AUROC, correct vs incorrect | 0.62 | 0.57 |

**What this supports:** Gate B carries a weak signal on multi-hop answers, but
it rejects about two-thirds of correct ones. That is the evidence for your
multi-hop limitation, now with the right numbers.

### H8 — M4's guarded F1 includes answers Gate B suppressed, and its "hallucination" equals its answer rate (major)

- The script scored `result.answer`, which is the raw answer even when Gate B
  suppressed it. 30 (sentence) and 53 (chunk) suppressed answers earned F1
  credit. The corrected guarded F1 is 0.056 / 0.069, not 0.086 / 0.150.
- `hallucination = answered and not EM`. With citations, EM is always 0, so
  hallucination rate = answer rate (17% = 17%, 22% = 22% in the table). That
  is the giveaway.

**The finding M4 missed is positive:**

| HotpotQA | Wrong among answered (baseline) | Wrong among answered (guarded) | Coverage (guarded) |
|---|---|---|---|
| Sentence | 57% | **36%** | 17% |
| Chunk | 41% | **24%** | 22% |

The guarded pipeline does cut the error rate on HotpotQA. It just does so at
very low coverage.

### H9 — "Gate B is the bottleneck" and "Gate A transfers cleanly" are both overstated (moderate)

- Guarded abstentions on HotpotQA (sentence): Gate A 9, **generator
  self-refusal 80**, Gate B 36. On chunks: 5 / 54 / 58. On sentences, the
  *prompt* is the biggest source of abstention.
- Every HotpotQA question is answerable, so Gate A has no negatives to reject.
  A 94–97% pass rate is the right behaviour, but it says nothing about whether
  Gate A would still *block* bad retrievals. "Transfers cleanly" cannot be
  tested on this data. Also, M3 predicted that Gate A would *degrade* on
  multi-hop, and M4 says the opposite without referring back to that.

**Correct wording:** "Gate A passed 94–97% of HotpotQA items (all answerable).
Its false-pass rate on multi-hop could not be measured because the sample has
no unanswerable questions."

### H10 — Two different meanings of "hallucination" (moderate, conceptual)

- M1's problem statement means **unfaithful generation**: text the passages do
  not support. Gate B is built for this.
- The SQuAD 2.0 metric means **answering an unanswerable question**. SQuAD 2.0
  unanswerable questions are written to have a plausible-looking answer *in the
  passage*. The model's answer is usually entailed by the passage (Gate B ≈
  0.99 on several of the 8 that got through); it just answers a slightly
  different question.

So Gate B is *expected* to miss SQuAD-style hallucinations, and Gate A (is
this passage relevant to *this* question?) does the work. State this
distinction in the report. It turns a weakness into an insight.

### H11 — Sample-size claims need numbers (moderate)

| Rate | Value | 95% Wilson interval |
|---|---|---|
| M2 hallucination | 3/75 = 4.0% | 1.4% – 11.1% |
| M3 hallucination | 8/75 = 10.7% | 5.5% – 19.7% |
| M3 over-abstention | 54/75 = 72% | 61% – 81% |

M2→M3 change on the same 75 items: 5 items newly hallucinated, 0 recovered.
Exact McNemar test p = 0.0625, so it is **not significant at 0.05**. Say:
"suggestive, not significant (p = 0.06)".

### H12 — In-sample calibration: now measured, mostly fine (moderate)

M3 says the optimism "cannot be measured". It can. Over 200 random 50/50
splits, refitting Gate A on one half and testing on the other:

- 0.45 was chosen in 185 of 200 splits, so the threshold is stable.
- Held-out Youden J median 0.225 (90% range 0.08–0.34) vs 0.24 in-sample, so
  the optimism is small.
- Held-out AUROC median 0.65 (0.58–0.74).

**Gate B caveat:** the "F1-optimal" 0.25 is optimal only *within the allowed
range* (≥ 0.20, above the contradiction cap). Passing everything would score
F1 = 0.667, higher than 0.481. On this proxy label, Gate B as a filter does
worse than no filter. Say so; it agrees with H2.

### H13 — Smaller inconsistencies (minor)

- **Composite threshold.** M3 says "0.33 = harmonic mean at corner". The code
  uses 1.10 × corner (corner 0.304 → 0.3345; config 0.33). The
  `flag_low_confidence` verdict **never fired** in any run (0/150), so the
  composite threshold had no observable effect. Say so.
- **M5 smoke test.** The table says "LLM-enabled integration smoke test",
  while the text says it runs "without calling the generative LLM". The
  default `smoke_test.py` skips the LLM unless `--with-llm` is passed. Pick
  the one you actually ran.
- **Gate A AUC 0.652.** M3 calls it "weak", M5 calls it "moderate". Pick one
  ("modest" is fair).
- **M5 "Ruff passed / 75 tests".** True for the committed code. Now 86 tests
  pass.
- **Reproducibility.** `build_hotpot.py`, `run_retrieval_hotpot.py`,
  `run_generation_hotpot.py`, `diagnose_gate_b.py` and
  `make_milestone5_report.py` existed only in your working folder and were
  never committed. They are committed on this branch. The `.docx` files in
  `Downloads` also differ from the ones in `reports/`, so make sure you submit
  the right versions.

---

## 3. Corrected headline story (use this in M5)

1. **On single-hop questions (SQuAD 2.0)**, the guarded pipeline (abstention
   prompt + Gate A + Gate B) answered **8 of 75** unanswerable questions vs
   **75 of 75** for a plain RAG baseline. When it did answer an answerable
   question, it was right **95%** of the time (20/21).
2. **The cost is coverage:** it answered only 21 of 75 answerable questions.
   The main cause is **Gate B rejecting correct short answers** (35 of 36
   suppressions were correct), because NLI cannot evaluate a bare noun phrase.
   This is a fixable design flaw, not a law of nature.
3. **As a hallucination *detector*** (ranking the baseline's answers), the
   reranker (Gate A) carries the most signal (AUROC 0.63). The NLI gate is
   near chance on SQuAD 2.0, because SQuAD's "hallucinations" are about the
   question, not about faithfulness to the passage.
4. **On multi-hop questions (HotpotQA)**, chunking raised full supporting-fact
   recall from 31% to 66% and baseline accuracy from 43% to 59%. The guarded
   pipeline cut the error rate among answered questions (41% → 24% on chunks)
   but answered only 22%. Gate B rejected about two-thirds of correct
   multi-hop answers, which is consistent with single-premise NLI being unable
   to combine evidence across passages.

This story is honest, defensible, and stronger than the current one, because
every number survives an examiner recomputing it.

---

## 4. What was changed on this branch

| File | Change |
|---|---|
| `src/squad_metrics.py` | `strip_citations()` applied before EM/F1; new `contains_gold()` |
| `src/evalstats.py` | `wilson_interval()`, `mcnemar_exact()`, `auroc()` |
| `tests/` | 11 new tests (86 total, all passing) |
| `scripts/audit_results.py` | New: re-scores all saved runs, writes `reports/audit_m5.json` |
| `scripts/run_generation_hotpot.py` | Scores the *shown* answer; separates Gate B failure from contradiction; containment-based hallucination; records self-refusal |
| (previously uncommitted) | HotpotQA scripts, M5 report generator, pending README/calibration edits |

The old JSON result files (`benchmark_m2/m3.json`, `generation_hotpot_m4.json`,
…) were left untouched as the historical record. The corrected figures are in
`audit_m5.json`.

---

## 5. Understanding your own project: viva preparation

### The idea in one paragraph
A normal RAG system always answers, in the same confident tone, whether or not
the retrieved text supports it. This pipeline checks twice. **Gate A** (before
generation) asks whether the passages are even relevant to the question, using
a cross-encoder *reranker*. If not, it refuses without calling the LLM, which
is cheap. **Gate B** (after generation) asks whether each sentence of the
answer is *entailed* by the passages, using an NLI model. If either check
fails, the system abstains. The aim is to trade some coverage for far fewer
confident wrong answers.

### Key terms an examiner will use
- **Cross-encoder reranker vs bi-encoder.** The bi-encoder (MiniLM) embeds the
  question and passage separately and compares vectors: fast, used for
  retrieval. A cross-encoder reads question and passage *together*: slower,
  but much better at judging relevance. That is why it drives Gate A.
- **NLI (natural language inference).** Given a premise and a hypothesis, it
  classifies the pair as entailment, neutral, or contradiction. Gate B uses
  passage = premise and answer sentence = hypothesis.
- **Sigmoid / calibration.** Turning a raw model score into something that
  behaves like a probability. "Calibrated" means that among answers scored
  0.8, about 80% are right.
- **Weighted harmonic mean.** Dominated by the smaller value, so good
  retrieval cannot hide a bad answer: 0.95 and 0.05 → about 0.08, not 0.50.
- **Youden's J.** TPR − FPR, the threshold that best separates two classes,
  treating misses and false alarms equally.
- **AUROC.** The probability that a random positive outscores a random
  negative. 0.5 is a coin flip, 1.0 is perfect. Threshold-free.
- **Risk–coverage / AURC.** Sort answers by confidence and answer only the top
  k%. Risk is the error rate among those answered. AURC averages the risk over
  every k; lower is better. This is the standard way to evaluate *selective
  prediction* (systems allowed to abstain).
- **Selective accuracy.** Accuracy on the questions you chose to answer.
- **Cohen's kappa.** Agreement between two raters beyond chance. 0 means
  chance level.
- **Rate-matching.** Giving the judge whichever threshold makes it pass the
  same number of items as NLI, so you compare *rankings*, not scales.
- **McNemar test.** Compares two systems on the *same* items using only the
  items where they disagree.
- **Wilson interval.** A confidence interval for a proportion that behaves
  well near 0% and 100%.
- **In-sample optimism.** Performance measured on the data used to tune is
  biased upward. You measure it with held-out splits.
- **Proxy label.** A stand-in for the label you actually want. Gate B wanted
  "is this answer grounded?" and got "was the question answerable?".

### Likely questions and strong answers

**Q: Why two gates and not one score at the end?**
Faithfulness is undefined until an answer exists. A pre-generation gate is
cheaper (it skipped the LLM on 48 of 150 SQuAD items) and catches retrieval
failure. A post-generation gate catches invention when retrieval was fine.
Different questions need different models.

**Q: Your guarded arm abstains 81% of the time. Isn't that useless?**
At that operating point, yes, and I found out why: Gate B rejects correct
short answers because a bare noun phrase is not an NLI hypothesis (35 of 36
suppressions were correct). When the system does answer, it is right 95% of
the time. The fix is to condition the hypothesis on the question.

**Q: How much of the improvement comes from the prompt rather than the gates?**
Of 67 refusals on unanswerable questions, 27 came from the generator's own
INSUFFICIENT_CONTEXT refusal (the prompt) and 40 from the gates. A clean split
needs a third arm: the guarded prompt with gates off.

**Q: Why does Gate B barely work on SQuAD 2.0?**
SQuAD 2.0 unanswerable questions have a plausible answer span in the
paragraph, so the model's wrong answer is usually *entailed by the passage*. It
just answers a different question. That is a relevance failure, which Gate A
targets, not a faithfulness failure.

**Q: Why did calibration lower Gate B from 0.60 to 0.25? Doesn't that weaken it?**
The contradiction veto is a separate absolute cap (0.15) and the calibration
range was not allowed to go below it, so outright contradictions still fail.
But on the proxy label, Gate B as a filter scored below "pass everything"
(F1 0.48 vs 0.67). That is consistent with the fragment problem.

**Q: Your sample is only 150. Are the results reliable?**
The large effects are: 8/75 vs 75/75 hallucinated, CI 5.5–19.7% vs
95–100%. The M2→M3 change from 3 to 8 is not significant (McNemar p = 0.06).
The held-out splits show the Gate A threshold is stable (0.45 chosen in 185 of
200 splits).

**Q: Why does chunking beat sentences on HotpotQA?**
Partly a volume effect: 11.8 chunks vs 40.8 sentences per item, so the top 5
covers more text. Full recall rose from 31% to 66% and baseline accuracy from
43% to 59%. Note: no chunk contained both supporting facts, so this did not
test single-premise multi-hop entailment.

**Q: Why use local NLI instead of an LLM judge?**
It is about 860× faster (262 ms vs 224 s for Gate B). The judge's scores were
not usable as ground truth either: binary faithfulness, and relevancy never
above 0.45. Agreement was at chance level (κ ≈ 0), so the judge is not a
gold standard. Neither scorer validates the other.

---

## 6. Recommended next work, in priority order

1. **Rewrite the M5 report's results section** with the corrected story
   (section 3) and the wording fixes above. *Highest value, no compute.*
2. **Gate B v2: a question-conditioned hypothesis.** Score
   `"{question} {answer}"` (or a declarative rewrite) instead of the bare
   answer, then re-evaluate offline on the stored SQuAD answers. Expected to
   recover most of the 35 wrongly suppressed answers. About an hour.
3. **Third arm: the guarded prompt with no gates.** Separates the prompt's
   contribution from the gates'. About 10–15 minutes of Ollama time.
4. **Larger n** (e.g. 500 items) for tighter confidence intervals, if time
   allows.
