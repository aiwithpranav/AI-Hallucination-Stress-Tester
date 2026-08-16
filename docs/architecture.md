# Architecture — AI Hallucination Stress Tester

## Overview

This document describes the internal architecture of the AI Hallucination Stress Tester.

---

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                    │
│   Enters any free-form question in the frontend UI              │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /api/analyze
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                            │
│                                                                 │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │  LLM Module  │   │   RAG Module    │   │  Verification   │  │
│  │              │   │                 │   │  Module         │  │
│  │ Gemini /     │   │ Wikipedia API   │   │                 │  │
│  │ OpenAI       │   │ → Passages      │   │ Cosine sim      │  │
│  │              │   │ → Embeddings    │   │ → Status        │  │
│  │ Generates    │   │   (local model) │   │ → Risk          │  │
│  │ answer for   │   │ → FAISS rank    │   │ → Confidence    │  │
│  │ ANY question │   │ → Top evidence  │   │ → LLM judge     │  │
│  └──────┬───────┘   └────────┬────────┘   └────────┬────────┘  │
│         │ answer             │ evidence            │ analysis  │
│         └────────────────────┴─────────────────────┘           │
│                              │                                  │
│                     Structured JSON Response                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │ JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FRONTEND                                  │
│   Dynamically renders all result sections                       │
│   - Status, Risk, Confidence, Evidence, Source, Explanation     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### Frontend (`frontend/`)

| File | Responsibility |
|---|---|
| `index.html` | HTML structure, result section placeholders |
| `style.css` | Full design system, dark glassmorphism theme |
| `script.js` | Input validation, API call, dynamic DOM rendering |

The frontend is entirely static (no build step). It communicates with the backend via `fetch()` to `POST /api/analyze`.

---

### Backend (`backend/`)

#### `main.py`
- FastAPI application entry point
- Registers the `/api/analyze` route
- Configures CORS to allow frontend origin
- Global exception handler for structured error responses

#### `llm/generator.py`
- **Responsibility**: Generate an answer for any question using an LLM
- **Providers**: Google Gemini (default), OpenAI (fallback)
- **Configuration**: `LLM_PROVIDER`, `GEMINI_API_KEY`, `OPENAI_API_KEY` from env
- **Output**: Plain text answer string

#### `rag/retriever.py`
- **Responsibility**: Retrieve real-world evidence for any question
- **Source**: Wikipedia REST API (`en.wikipedia.org/api/rest_v1`)
- **Process**:
  1. Search Wikipedia for the question terms
  2. Fetch article summaries / intro sections
  3. Split into passage chunks
  4. Return list of `{ text, source, source_url, source_type }` dicts
- **No API key required**

#### `rag/embedder.py`
- **Responsibility**: Convert text to dense vector representations
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Output**: NumPy arrays (float32)
- **Note**: Model downloaded ~80MB on first run, then cached

#### `rag/vector_store.py`
- **Responsibility**: Rank evidence passages by similarity to the answer
- **Method**: FAISS `IndexFlatIP` (inner product, normalized = cosine similarity)
- **Per-request**: Index is built fresh per request (in-memory, no persistence)
- **Output**: Top-k ranked passages with similarity scores

#### `verification/verifier.py`
- **Responsibility**: Core analysis engine
- **Steps**:
  1. Embed the LLM-generated answer
  2. Embed all evidence passages
  3. Build FAISS index and retrieve top-k similar passages
  4. Compute average similarity score → base confidence
  5. Classify verification status based on thresholds
  6. Derive hallucination risk (inverse of confidence/support)
  7. Call LLM as judge → get natural-language explanation + recommendation
  8. Classify source reliability (Wikipedia = High, others = Medium)
  9. Return full structured result dict

---

## Verification Logic

### Status Thresholds (cosine similarity, 0–1)

| Confidence Range | Status |
|---|---|
| ≥ 0.75 | Verified |
| 0.50 – 0.74 | Partially Supported |
| 0.25 – 0.49 | Not Supported |
| < 0.25 (or no evidence) | Unable to Verify |

### Hallucination Risk Mapping

| Status + Confidence | Risk Level |
|---|---|
| Verified + conf ≥ 0.85 | NO Risk |
| Verified + conf < 0.85 | Very Low |
| Partially Supported + conf ≥ 0.65 | Very Low |
| Partially Supported + conf < 0.65 | Low |
| Not Supported | Medium |
| Unable to Verify | High |

### LLM Judge Prompt
The verifier sends a structured prompt to the LLM containing:
- The original question
- The generated answer
- The top evidence passage
- The similarity score

It requests:
- A concise risk explanation (1–2 sentences)
- A specific recommended action for the user

This explanation is based on the actual evidence comparison, not hardcoded text.

---

## API Contract

### POST `/api/analyze`

**Request:**
```json
{ "question": "string (required, 1–2000 chars)" }
```

**Success Response (200):**
```json
{
  "question": "string",
  "answer": "string",
  "status": "Verified | Partially Supported | Not Supported | Unable to Verify",
  "confidence": 0.78,
  "hallucinationRisk": "NO Risk | Very Low | Low | Medium | High",
  "evidenceText": "string",
  "source": "string",
  "sourceReliability": "High | Medium | Low",
  "sourceStatus": "Authoritative | General | Unverified",
  "riskExplanation": "string",
  "recommendation": "string",
  "summary": "string",
  "verificationMethod": "Semantic Similarity + LLM Judge",
  "evidenceStatus": "Supporting | Contradicting | Neutral | Insufficient",
  "verificationOutcome": "string",
  "date": "2024-01-01T12:00:00Z"
}
```

**Error Response (4xx/5xx):**
```json
{
  "error": "string",
  "detail": "string"
}
```

---

## Data Flow Sequence

```
1. POST /api/analyze { "question": "Q" }
2. generator.generate_answer("Q") → answer "A"
3. retriever.retrieve_evidence("Q") → [passage1, passage2, ...]
4. embedder.embed(answer) → vector_a
5. embedder.embed_batch(passages) → [vector_p1, vector_p2, ...]
6. vector_store.top_k(vector_a, passage_vectors) → ranked passages
7. verifier.compute_confidence(scores) → 0.78
8. verifier.classify_status(0.78) → "Partially Supported"
9. verifier.classify_risk("Partially Supported", 0.78) → "Low"
10. generator.llm_judge(Q, A, top_evidence, score) → {explanation, recommendation}
11. Return structured JSON → Frontend
```
