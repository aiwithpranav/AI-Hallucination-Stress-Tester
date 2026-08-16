"""
AI Hallucination Stress Tester — Backend Entry Point

FastAPI application providing:
  POST /api/analyze  - Full LLM + RAG + verification pipeline
  GET  /health       - Health check

Usage:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm.generator import LLMGenerator
from rag.retriever import EvidenceRetriever
from verification.verifier import Verifier

# ─── Load environment variables ───────────────────────────────────────────────
# Look for .env in the project root (parent of backend/) first, then CWD.
_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / ".env"
load_dotenv(dotenv_path=_env_file if _env_file.exists() else None, override=False)
load_dotenv(override=False)  # Also load from CWD as fallback

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Hallucination Stress Tester",
    description="LLM + RAG + Semantic Verification API",
    version="1.0.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin] if frontend_origin != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

# ─── Module Singletons ────────────────────────────────────────────────────────
# Instantiated once at startup; LLMGenerator loads API key from env.
llm_generator = LLMGenerator()
evidence_retriever = EvidenceRetriever()
verifier = Verifier()

# ─── Request / Response Models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to analyze for hallucination risk",
    )


class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    status: str
    confidence: float
    hallucinationRisk: str
    evidenceText: str
    source: str
    sourceReliability: str
    sourceStatus: str
    riskExplanation: str
    recommendation: str
    summary: str
    verificationMethod: str
    evidenceStatus: str
    verificationOutcome: str
    date: str


# ─── Global Exception Handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a structured JSON error for any unhandled exception."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
    }


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    tags=["Verification"],
    summary="Analyze a question for hallucination risk",
    responses={
        200: {"description": "Verification result"},
        422: {"description": "Validation error — question too short/long"},
        500: {"description": "Internal error (LLM/RAG/verification failure)"},
        503: {"description": "LLM or retrieval service unavailable"},
    },
)
async def analyze(request: AnalyzeRequest):
    """
    Full pipeline:
    1. Generate an answer using the LLM.
    2. Retrieve relevant evidence via RAG (Wikipedia API).
    3. Verify the answer against the evidence.
    4. Return structured hallucination risk assessment.
    """
    question = request.question.strip()

    # Step 1: Generate answer via LLM
    try:
        answer = await llm_generator.generate_answer(question)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": "LLM generation failed",
                "detail": (
                    f"Could not generate an answer: {exc}. "
                    "Check that your API key is correctly set in .env "
                    "and that the LLM service is reachable."
                ),
            },
        )

    # Step 2: Retrieve evidence via RAG
    try:
        evidence_passages = await evidence_retriever.retrieve(question)
    except Exception as exc:
        # Retrieval failure degrades to Unable-to-Verify, not a hard crash
        evidence_passages = []
        retrieval_error = str(exc)
    else:
        retrieval_error = None

    # Step 3: Verify answer against evidence
    try:
        result = await verifier.verify(
            question=question,
            answer=answer,
            evidence_passages=evidence_passages,
            retrieval_error=retrieval_error,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Verification failed",
                "detail": str(exc),
            },
        )

    # Step 4: Return structured response
    return AnalyzeResponse(
        question=question,
        answer=answer,
        date=datetime.now(timezone.utc).isoformat(),
        **result,
    )
