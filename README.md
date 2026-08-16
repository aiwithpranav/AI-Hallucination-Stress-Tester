# AI Hallucination Stress Tester

> A full-stack AI application that generates answers to any question using an LLM, retrieves real evidence via RAG, and performs evidence-based hallucination detection — producing a structured verification report with confidence score and risk classification.

---

## What It Does

1. **User asks any question** — no predefined questions or answers.
2. **LLM generates an answer** — dynamically, using Google Gemini (or OpenAI).
3. **RAG retrieves real evidence** — Wikipedia REST API provides relevant passages for any question.
4. **Verification engine compares** the answer against the retrieved evidence using semantic similarity (sentence-transformers + FAISS).
5. **Hallucination risk is classified** — NO Risk / Very Low / Low / Medium / High — derived from actual similarity scores, not hardcoded.
6. **Structured report is returned** — verification status, confidence, evidence, source reliability, risk explanation, and recommended action.

---

## Architecture

```
User
 ↓
Frontend (HTML + CSS + JS)
 ↓  POST /api/analyze  { "question": "..." }
FastAPI Backend
 ├── LLM Module (Gemini / OpenAI)
 │     └── Generates answer for any question
 ├── RAG Module (Wikipedia API + SentenceTransformers + FAISS)
 │     └── Retrieves + embeds + ranks real evidence
 └── Verification Module
       └── Cosine similarity scoring → status + risk + confidence + LLM judge
 ↓  JSON response
Frontend renders verification report
```

---

## Project Structure

```
AI-Hallucination-Stress-Tester/
│
├── frontend/
│   ├── index.html          # Main UI
│   ├── style.css           # Design system (dark glassmorphism, Apple-inspired)
│   └── script.js           # API communication + dynamic rendering
│
├── backend/
│   ├── main.py             # FastAPI app, /api/analyze route, CORS
│   ├── llm/
│   │   ├── __init__.py
│   │   └── generator.py    # Gemini + OpenAI provider abstraction
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py    # Wikipedia API search + passage extraction
│   │   ├── embedder.py     # SentenceTransformer embedding wrapper
│   │   └── vector_store.py # FAISS in-memory similarity search
│   ├── verification/
│   │   ├── __init__.py
│   │   └── verifier.py     # Scoring, classification, LLM judge
│   └── requirements.txt
│
├── docs/
│   └── architecture.md
│
├── .env.example            # Variable names only — never commit real keys
├── .gitignore
└── README.md
```

---

## Frontend

- Pure HTML + CSS + JavaScript (no frameworks)
- Premium dark glassmorphism aesthetic (Apple-inspired)
- Google Fonts: Inter
- Responsive, mobile-friendly
- Result sections:
  - Verification Date / Status / Summary
  - Hallucination Risk / Confidence Score
  - Evidence Text / Source / Source Reliability
  - Risk Explanation / Recommendation
  - Status Guide / Verification Method

Serves as a static file — open `frontend/index.html` directly in a browser, or serve via any static file server. Communicates with the backend at `http://localhost:8000/api/analyze`.

---

## Backend

Built with **FastAPI** (Python). Async, fast, self-documenting (`/docs`).

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/analyze` | Full LLM + RAG + verification pipeline |
| GET | `/health` | Health check |

### Modules

**`llm/generator.py`** — Provider abstraction:
- Reads `LLM_PROVIDER` from env (`gemini` or `openai`)
- Sends question to LLM → returns answer string
- Never exposes API keys to frontend

**`rag/retriever.py`** — Dynamic evidence retrieval:
- Searches Wikipedia REST API for the user's question
- Extracts and cleans relevant passages
- Returns list of `{ text, source, source_type }` dicts

**`rag/embedder.py`** — Local embeddings:
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Downloads ~80MB on first run, then cached locally
- No API key required

**`rag/vector_store.py`** — Similarity ranking:
- Builds a temporary FAISS index per request
- Returns top-k most relevant evidence chunks

**`verification/verifier.py`** — Analysis engine:
- Computes cosine similarity (answer vs evidence)
- Classifies verification status, hallucination risk, confidence
- Calls LLM as judge for natural-language risk explanation

---

## LLM

### Google Gemini (Default)
- Model: `gemini-1.5-flash` (configurable)
- Get your API key: https://aistudio.google.com/app/apikey
- Set `GEMINI_API_KEY` in `.env`

### OpenAI (Fallback)
- Model: `gpt-3.5-turbo` (configurable)
- Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`

---

## RAG

Evidence is retrieved dynamically from **Wikipedia REST API** per question. This means:
- No pre-ingestion required
- Any question can be answered with relevant real-world evidence
- Sources are real Wikipedia articles (with URLs)

The retrieval pipeline:
1. Query Wikipedia with the user's question
2. Extract article summaries and intro sections
3. Split into passages
4. Embed with sentence-transformers
5. Rank by similarity to the LLM-generated answer
6. Top passages become evidence for verification

---

## Verification Flow

```
answer (from LLM)
    ↓  embed
evidence chunks (from RAG)
    ↓  embed
cosine similarity scores
    ↓
average top-k score → confidence (0.0–1.0)
    ↓
status classification:
  ≥ 0.75  → Verified
  0.50–0.74 → Partially Supported
  0.25–0.49 → Not Supported
  < 0.25  → Unable to Verify
    ↓
hallucination risk (inverse):
  Verified + conf ≥ 0.85 → NO Risk
  Partially + conf ≥ 0.65 → Very Low
  Partially + conf ≥ 0.50 → Low
  Not Supported         → Medium
  Unable to Verify      → High
    ↓
LLM judge: generate riskExplanation + recommendation
```

---

## Environment Variables

Copy `.env.example` → `.env` and fill in your values.

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `gemini` or `openai` |
| `GEMINI_API_KEY` | If using Gemini | Google AI Studio key |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI platform key |
| `GEMINI_MODEL` | No | Default: `gemini-1.5-flash` |
| `OPENAI_MODEL` | No | Default: `gpt-3.5-turbo` |
| `BACKEND_PORT` | No | Default: `8000` |
| `EMBEDDING_MODEL` | No | Default: `all-MiniLM-L6-v2` |
| `RAG_TOP_K` | No | Number of evidence passages (default: 5) |

---

## How to Run

### Prerequisites
- Python 3.9+
- A Gemini or OpenAI API key

### 1. Clone / open the project folder

```bash
cd d:\AI\LLMs\Hallucination
```

### 2. Set up backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cd ..                          # back to project root
copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux
# Edit .env — add your GEMINI_API_KEY
```

### 4. Start the backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend is now running at: http://localhost:8000
API docs: http://localhost:8000/docs

### 5. Open the frontend

Open `frontend/index.html` directly in your browser.

> If you get CORS errors, serve the frontend with a simple server:
> ```bash
> cd frontend
> python -m http.server 3000
> ```
> Then open http://localhost:3000

---

## Testing

Test with multiple unrelated questions to verify the system is dynamic:

| Question Type | Example |
|---|---|
| Factual | "What is the capital of France?" |
| Scientific | "How does CRISPR gene editing work?" |
| Historical | "Who signed the Magna Carta?" |
| Technical | "What is the speed of light?" |
| Current | "What is quantum computing?" |

Each question should return a **different** answer, evidence, confidence, and risk.

---

## Current Limitations

- Wikipedia is the sole evidence source — questions with no Wikipedia articles may return low confidence / `Unable to Verify`
- FAISS index is built per-request (in-memory) — not persisted between requests
- Embedding model downloads ~80MB on first run (cached after)
- No user authentication or session management (out of scope)
- No PDF/document upload UI (out of scope for v1)

---

## Future Extension Points

- Swap Wikipedia for a pre-ingested document corpus (research papers, company docs)
- Persistent FAISS / ChromaDB / Pinecone vector store
- Multiple evidence sources (web search, ArXiv, PubMed)
- Batch question testing mode
- Export verification report as PDF
- Fine-tuned verification model
- Streaming responses for LLM output

---

## License

MIT — for portfolio and educational purposes.
