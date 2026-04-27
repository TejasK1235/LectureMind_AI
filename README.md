# LectureMind, NLP/AI Services

> LectureMind is a lecture intelligence platform that transforms raw lecture recordings into structured learning materials. This repository contains the NLP/AI services layer, built on Whisper, Groq LLMs, and sentence transformers responsible for transcription, tagging, summarization, question bank generation, and quiz generation.

---

## Overview

LectureMind is the AI/NLP backend for a lecture intelligence platform. It accepts audio or video lecture recordings and produces:

- **Tagged transcripts**, every segment classified as lecture content, administrative, or chatter
- **Revision notes**, chunked, deduplicated, LLM-rewritten notes formatted as a PDF
- **Question banks**, Bloom's taxonomy-distributed questions across up to 5 cognitive levels
- **Quizzes**, MCQ and extempore topic generation from question bank output
- **Slide integration**, optional PDF/PPTX ingestion to enrich outputs with content the lecturer may not have spoken aloud

This repository contains the Python/FastAPI NLP layer only. The Spring Boot REST backend and React frontend are in separate repositories.

---

## Architecture

```
Audio/Video File
      │
      ▼
┌─────────────────────────────┐
│  Whisper (OpenAI)           │  Speech-to-text transcription
│  + Segment Merging          │  Short segment merging by pause gap
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Tagging Pipeline           │  Rule-based pre-filter → LLM classification
│  (tagging2/)                │  Tags: LECTURE_CONTENT | ADMINISTRATIVE | OTHER_CHATTER
│                             │  + Emphasis propagation
└─────────────┬───────────────┘
              │
      ┌───────┴───────────────┐
      │                       │
      ▼                       ▼
┌─────────────────┐  ┌─────────────────────┐
│  Summarization  │  │  Question Bank      │
│  Pipeline       │  │  Pipeline           │
│                 │  │  (question_bank/)   │
│  TF-IDF scoring │  │                     │
│  + chunked LLM  │  │  Concept builder +  │
│  rewrite → PDF  │  │  Bloom QB           │
└─────────────────┘  └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  Quiz Generation    │
                     │  (quiz/)            │
                     │  • MCQ              │
                     │  • Extempore topics │
                     └─────────────────────┘
```

---

## Endpoints

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| POST | `/transcribe` | Audio/video file | Tagged segments JSON array |
| POST | `/summarize` | Tagged segments | PDF (revision notes) |
| POST | `/summarize/with-slides` | Tagged segments + slide file | PDF (slide-enriched notes) |
| POST | `/question-bank/generate` | Tagged segments + Bloom params | Question bank JSON |
| POST | `/question-bank/generate/with-slides` | Tagged segments + slide file + Bloom params | Question bank JSON |
| GET  | `/question-bank/result` | - | Fetch QB JSON |
| POST | `/quiz/mcq` | Question bank JSON | MCQ quiz JSON |
| POST | `/quiz/extempore` | Question bank JSON | Extempore topics JSON |
| GET | `/health` | - | `{status, whisper_model}` |

---

## Pipeline Details

### Tagging (`tagging2/`)

Two-stage classifier for lecture segment labelling:

1. **Rule-based pre-filter**, high-precision, low-recall. Fires only on unambiguous administrative vocabulary (exam dates, attendance, deadlines). Roughly 5–10% of segments resolved here.
2. **LLM classification**, remaining segments sent to `llama-3.1-8b-instant` via Groq in batches of 10. Tiebreaker rule biases toward `LECTURE_CONTENT` to minimise false negatives.
3. **Emphasis propagation**, segments adjacent to emphasised content are flagged for downstream scoring.

### Summarization (`summarization/`)

1. **Extractive stage**, TF-IDF scoring with position, first-mention, and length bonuses. Top N% of segments selected (ratio scales with lecture length: 60–80%).
2. **Chunked LLM rewrite**, selected segments split into batches of 50. Each batch independently produces bullet points. Bullets semantically deduplicated using `all-MiniLM-L6-v2` (cosine similarity threshold 0.82) before final merge call.
3. **Merge LLM call**, deduplicated bullets reorganised into headed revision notes.
4. **PDF generation**, markdown output converted to styled PDF via wkhtmltopdf.

### Question Bank (`question_bank/`)

1. **Concept builder**, segments grouped into concept blocks by topic boundary detection (silence gaps, ADMINISTRATIVE flush, 250-word cap). Optional slide text attached via Jaccard alignment (threshold 0.08).
2. **Concept scoring**, 60% word count + 40% emphasis count + recency bonus (0–0.1).
3. **Question generation**, `llama-3.1-8b-instant` generates questions per concept per Bloom level with 1.5× buffer multiplier.
4. **Evaluation**, two-stage deduplication: Jaccard pre-filter (threshold 0.65) → semantic check via `all-MiniLM-L6-v2` (threshold 0.85). Catches paraphrased duplicates the Jaccard pass misses.
5. **Assembly**, largest-remainder rounding ensures exact requested question counts per Bloom level.

### Quiz Generation (`quiz/`)

**MCQ**, draws from Remember and Understand Bloom levels only (higher levels lack single unambiguous correct answers). Generates distractors via `llama-3.3-70b-versatile`. Distractor validation: semantic similarity to correct answer checked via `all-MiniLM-L6-v2` (threshold 0.80), rejects distractors that are too close to the correct answer.

**Extempore**, generates presentation topic titles from high-scoring concepts (`score ≥ 0.25`). Cross-topic semantic deduplication applied after generation (threshold 0.82).

### Slide Ingestion (`question_bank/slide_ingestor.py`)

Supports PDF (PyMuPDF) and PPTX (python-pptx). Symbol normalisation converts Unicode (Greek letters, math operators, arrows, superscripts) to ASCII equivalents for clean Jaccard alignment and LLM processing.

**Known limitation:** image-embedded content, code screenshots, and equation images are not extracted (pixels only, no OCR). VLM integration planned for a future release.

---

## Models

| Pipeline | Model | Rationale |
|----------|-------|-----------|
| Transcription | `whisper-large-v3` (local) | Best open-source ASR accuracy |
| Tagging | `llama-3.1-8b-instant` | Low latency, high throughput for batch classification |
| Summarization | `llama-3.1-8b-instant` | Sufficient for rewrite tasks |
| Question Bank | `llama-3.1-8b-instant` | Adequate for structured generation |
| MCQ Distractors | `llama-3.3-70b-versatile` | Distractor generation requires reasoning about plausible misconceptions |
| Extempore | `llama-3.1-8b-instant` | Simple formatting task |
| Semantic tasks | `all-MiniLM-L6-v2` | Deduplication, alignment, validation |

All LLM calls routed via [Groq](https://groq.com) for low-latency inference.

---

## Tech Stack

- **Runtime**, Python 3.10+, FastAPI, Uvicorn
- **Transcription**, OpenAI Whisper (local inference)
- **LLM inference**, Groq API
- **Embeddings**, `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Slide parsing**, PyMuPDF, python-pptx
- **PDF generation**, wkhtmltopdf via pdfkit
- **Serving**, ngrok (development tunnelling)

---

## Project Structure

```
LectureMind/
├── api.py                        # FastAPI app, all endpoints
├── tagging2/
│   ├── config.py                 # Model names (single source of truth)
│   ├── schema.py                 # Class schemas for tagging pipeline
│   ├── pipeline/
│   │   ├── tagger.py             # Main tagging orchestration
│   │   └── segment.py            # UUID assignment, field normalisation
│   │   └── run.py                # Orchestrator for the entire tagging pipeline
│   ├── llm/
│   │   └── prompt.py             # Classification prompt
│   │   └── classify.py           # Mini orchestrator for LLM classification
│   │   └── llm_client.py         # LLM interaction
│   └── rules/
│       └── rule_based.py         # Administrative keyword lists
├── summarization/
│   ├── run.py                    # Pipeline entry point
│   ├── extractor.py              # TF-IDF scoring + segment selection
│   ├── scorer.py                 # Multi-factor segment scorer
│   ├── llm_summarizer.py         # Chunked LLM rewrite + semantic dedup
│   └── prompt.py                 # Summary and chunk prompts
│   └── utils.py                  # Utility functions for summarization module
├── question_bank/
│   ├── run.py                    # Pipeline entry point
│   ├── concept_builder.py        # Segment → concept grouping
│   ├── concept_scorer.py         # Concept scoring
│   ├── question_generator.py     # Per-concept per-Bloom generation
│   ├── evaluator.py              # Dedup + quality filtering
│   ├── qb_assembler.py           # Final assembly + rounding
│   ├── slide_ingestor.py         # PDF/PPTX parsing + symbol normalisation
│   └── prompts.py                # Bloom descriptions and starters
└── quiz/
    ├── run.py                    # MCQ and extempore entry points
    ├── mcq_generator.py          # MCQ generation with distractor validation
    ├── extempore_generator.py    # Topic title generation
    ├── distractor_validator.py   # Semantic distractor validation
    └── prompts.py                # MCQ and extempore prompts
```

---

## Setup

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Install wkhtmltopdf (PDF generation)
# Windows: https://wkhtmltopdf.org/downloads.html
# Linux: sudo apt-get install wkhtmltopdf

# Set environment variables
export GROQ_API_KEY=your_groq_api_key

# Run
uvicorn api:app --reload
```

### `requirements.txt`

```
fastapi
uvicorn[standard]
python-multipart
pydantic
openai-whisper
groq
scikit-learn
numpy
sentence-transformers
pymupdf
python-pptx
pdfkit
markdown
pypdf
torch
```

---

## Known Limitations & Planned Improvements

| Area | Current Limitation | Planned Fix |
|------|--------------------|-------------|
| Slide ingestion | Image-embedded formulas and code screenshots not extracted | Selective VLM integration for slides with <15 words extracted |
| Concept boundary detection | Rule-based flush on silence/word count, misses A→B→A topic patterns | Embedding-based semantic boundary detection |
| Emphasis detection | Text-only, rule-based | Prosodic features from raw audio (pitch, tempo, energy) |
| QB provenance | No per-question concept_id stored | Store question provenance for targeted quiz generation |
| Multi-lecture quiz pool | Each quiz draws from single lecture QB | Cumulative QB pool across lectures per course |
| Summarization scoring | TF-IDF blind to pedagogical importance (worked examples score low) | Embedding-based scoring with pedagogical anchor phrases |

---

## Team

LectureMind is built by a four-person team:

- **Tejas Kothavale** — NLP/AI Services (Python, FastAPI)  
  Repository: [LectureMind-AI](https://github.com/TejasK1235/LectureMind_AI)

- **Sanchit Khadkodkar** — Backend (Spring Boot REST API)  
  Repository: [LectureMind-Backend](https://github.com/sanchit100github/LectureMind)

- **Ratul Kulkarni** — Frontend (React)  
  Repository: [LectureMind-Frontend](https://github.com/RatulKulkarni/lecturemind)
  
- **[Anish Kulkarni](https://github.com/Anishkulkarni27)** — Research & Documentation
---