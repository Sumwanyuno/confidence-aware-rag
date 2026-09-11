# Confidence-Aware RAG Pipeline with Hallucination Detection

A retrieval-augmented generation system that **knows when it does not know**. Instead of
always producing a fluent answer, it scores retrieval reliability before generating and
answer groundedness after, and abstains when either check fails.

> Semester project (Jul–Dec 2026) · Artificial Intelligence & Machine Learning Systems

---

## The problem

A conventional RAG pipeline always answers. Vector search returns its top-k nearest
neighbours whether or not any of them are relevant, and a language model handed weak
context will still produce a confident, well-formed, wrong answer. Nearest is not the
same as sufficient.

## The approach: two gates, not one score

Most RAG evaluation happens offline, after the fact. This pipeline puts the check
*inline*, and splits it in two because the two questions are answerable at different
times:

```
question
   │
   ├─► retrieve top-k passages
   │
   ├─► GATE A · retrieval sufficiency         ← before generation
   │     cross-encoder relevance (ms-marco)
   │     fail ─────────────────────────────► abstain, no tokens spent
   │
   ├─► generate answer (context-restricted prompt)
   │
   ├─► GATE B · answer groundedness           ← after generation
   │     per-claim NLI entailment (DeBERTa)
   │     fail ─────────────────────────────► abstain, answer suppressed
   │
   └─► composite = weighted harmonic mean(A, B)
         below threshold ──────────────────► answer shown, flagged low-confidence
         otherwise ───────────────────────► answer shown
```

**Why two gates.** Faithfulness is undefined until an answer exists, so it cannot guard
generation. Retrieval relevance needs only the query and the passages, so it can — and
catching an unanswerable question at Gate A means the wrong answer is never produced at
all, rather than produced and then filtered.

**Why a harmonic mean.** Relevancy 0.95 with faithfulness 0.05 averages to a
respectable-looking 0.50. The weighted harmonic mean returns about 0.09. Both conditions
have to hold, the same way F1 refuses to reward precision without recall.

**Claim-level output.** Gate B splits the answer into claims and labels each one using
the RAGTruth taxonomy — *supported*, *baseless information* (the source is silent), or
*evident conflict* (the source disagrees). Any single contradiction fails the gate
outright, because averaging lets nine correct sentences hide one that conflicts.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
cp .env.example .env
```

Index the sample corpus and verify every component:

```bash
python scripts/build_index.py --reset
python scripts/smoke_test.py
```

`smoke_test.py` needs **no LLM** — retrieval and both confidence gates run entirely on
local cross-encoders. That is the point: the confidence layer is independently testable.

To enable generation, install [Ollama](https://ollama.com/download) and pull a model:

```bash
ollama pull llama3.1:8b
python scripts/smoke_test.py --with-llm
```

Launch the app:

```bash
streamlit run app.py
```

---

## Repository layout

| Path | What it does |
| --- | --- |
| `src/config.py` | Every tunable in one place: thresholds, weights, models, devices |
| `src/embeddings.py` | Single embedding entry point, with CUDA→CPU fallback |
| `src/ingest.py` | load → chunk → embed → persist to Chroma, with provenance metadata |
| `src/retriever.py` | Retrieval, plus explicit Chroma-distance → similarity conversion |
| `src/generator.py` | Context-restricted prompt; `generate_unguarded` is the control arm |
| `src/scorers.py` | Three scoring backends: reranker (Gate A), NLI (Gate B), DeepEval judge |
| `src/confidence.py` | Harmonic mean, gate thresholds, decision table, RAGTruth labels |
| `src/pipeline.py` | The orchestration the architecture diagram describes |
| `src/store.py` | SQLite telemetry backing the monitoring dashboard |
| `app.py` | Streamlit Q&A interface showing both gate scores and per-claim analysis |
| `pages/1_Monitoring_Dashboard.py` | Confidence trends, verdict mix, latency budget |
| `scripts/build_index.py` | Ingest `data/raw/` into ChromaDB |
| `scripts/smoke_test.py` | Staged component verification (24 checks, no LLM needed) |
| `scripts/bench_latency.py` | Warm per-stage latency of the confidence layer |
| `scripts/make_thumbnail.py` | Regenerates `assets/thumbnail.png` (architecture banner) |
| `scripts/make_milestone_report.py` | Regenerates the milestone report `.docx` |
| `tests/` | Unit tests for the scoring maths and the gate decision table |

---

## Configuration

Everything is driven by `.env` (see `.env.example`). The two settings that matter most:

| Variable | Options | Effect |
| --- | --- | --- |
| `RAG_PROVIDER` | `ollama` / `groq` / `gemini` | Which backend generates and judges |
| `RAG_SCORER` | `nli` / `llm` / `both` | Local cross-encoders, DeepEval judge, or both |

`RAG_SCORER=nli` is the default because it costs no API calls, which is what makes a
benchmark sweep over thousands of questions feasible on a laptop. `both` runs each and
logs the disagreement, which is the basis of the local-vs-LLM-judge agreement study.

---

## Status

**Milestone 1 complete** (due 27 Aug 2026). Environment, ingestion, retrieval, both gates,
composite scoring, telemetry, Streamlit app and monitoring dashboard all implemented and
verified:

| Check | Command | Result |
|---|---|---|
| Unit tests (model-free) | `python -m pytest tests/ -q` | 31 passed |
| Integration smoke test | `python scripts/smoke_test.py` | 24 passed, 0 failed, 0 skipped |
| Linter | `ruff check .` | no findings |
| Confidence-layer latency | `python scripts/bench_latency.py` | 134 ms median (Gate A 16 ms + Gate B 118 ms) |

Measured separation on the sample corpus: an in-corpus question scores 0.999 at Gate A
while an out-of-corpus question scores 0.000 and abstains before the generator is called;
a grounded answer scores 0.997 at Gate B while a fabricated one scores 0.000 and is
labelled *Evident Conflict*.

**Thresholds are uncalibrated placeholders.** They are replaced by values fitted on a
labelled dev split (SQuAD 2.0 answerable/unanswerable) in Milestone 3. Shipping a
guessed threshold and calling it tuned would be the exact overconfidence this project
is about.

### Remaining milestones (weekly)

| Milestone | Due | Focus |
|---|---|---|
| 2 | 3 Sep 2026 | Local generation end to end; SQuAD 2.0 harness; baseline control arm |
| 3 | 10 Sep 2026 | Threshold calibration; risk–coverage curves and AURC; NLI-vs-LLM-judge agreement |
| 4 | 17 Sep 2026 | HotpotQA multi-hop; ablations; RAGTruth taxonomy spot-check |
| 5 | 24 Sep 2026 | Results consolidation, reproducibility pass, final report and demo |

## Running tests

```bash
python -m pytest tests/ -q
```
