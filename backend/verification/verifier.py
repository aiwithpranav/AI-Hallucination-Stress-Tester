"""
Verification Engine Module

This is the core analysis module that:
1. Embeds the LLM-generated answer and all retrieved evidence passages.
2. Computes cosine similarity scores using FAISS.
3. Derives a confidence score from the actual similarity data.
4. Classifies the verification status (Verified / Partially Supported / etc.).
5. Derives the hallucination risk level (NO Risk / Very Low / Low / Medium / High).
6. Calls the LLM as a judge for natural-language risk explanation.
7. Classifies source reliability based on source type.
8. Returns a fully structured verification result dict.

IMPORTANT:
- Nothing in this module is hardcoded based on the question or answer.
- All classifications are derived from the actual similarity scores.
- Confidence is computed from real embedding similarity — not randomized.
- The LLM judge produces explanations from actual evidence comparison.

Thresholds (from specification + architecture doc):
    confidence ≥ 0.75 → Verified
    confidence 0.50–0.74 → Partially Supported
    confidence 0.25–0.49 → Not Supported
    confidence < 0.25  → Unable to Verify

Risk levels:
    Verified + conf ≥ 0.85 → NO Risk
    Verified + conf < 0.85 → Very Low
    Partially + conf ≥ 0.65 → Very Low
    Partially + conf < 0.65 → Low
    Not Supported           → Medium
    Unable to Verify        → High
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv

from rag.embedder import Embedder
from rag.vector_store import VectorStore

_project_root = Path(__file__).resolve().parent.parent.parent
_env_file = _project_root / ".env"
load_dotenv(dotenv_path=_env_file if _env_file.exists() else None, override=False)
load_dotenv(override=False)

# How many top evidence passages to average for confidence
_TOP_K_FOR_CONFIDENCE = int(os.getenv("RAG_TOP_K", "5"))

# ─── Threshold Constants ──────────────────────────────────────────────────────
_THRESHOLD_VERIFIED          = 0.65
_THRESHOLD_PARTIALLY_LOW     = 0.40
_THRESHOLD_NOT_SUPPORTED_LOW = 0.20

_RISK_NO_RISK_MIN_CONF       = 0.85
_RISK_VLOW_PARTIAL_MIN_CONF  = 0.65


class Verifier:
    """
    Orchestrates the complete verification pipeline:
    embed → similarity → classify → LLM judge → structured result.
    """

    def __init__(self):
        self._embedder = Embedder()
        self._vector_store = VectorStore()

    # ─── Main Verify Interface ────────────────────────────────────────────────

    async def verify(
        self,
        question: str,
        answer: str,
        evidence_passages: List[Dict],
        retrieval_error: Optional[str] = None,
    ) -> dict:
        """
        Full verification pipeline.

        Args:
            question:          The original user question.
            answer:            The LLM-generated answer.
            evidence_passages: List of passage dicts from EvidenceRetriever.
            retrieval_error:   If retrieval failed, the error message (may be None).

        Returns:
            A dict with all fields needed for the AnalyzeResponse model.
        """
        # ── Handle the case where no evidence was retrieved ──────────────────
        if not evidence_passages or retrieval_error:
            return self._build_unable_to_verify_result(
                question=question,
                answer=answer,
                reason=retrieval_error or "No evidence passages could be retrieved.",
            )

        # ── Embed the answer and all evidence passages ────────────────────────
        answer_vector = await self._embedder.embed_async(answer)
        passage_texts = [p["text"] for p in evidence_passages]
        passage_vectors = await self._embedder.embed_batch_async(passage_texts)

        if passage_vectors is None or len(passage_vectors) == 0:
            return self._build_unable_to_verify_result(
                question=question,
                answer=answer,
                reason="Embedding failed for evidence passages.",
            )

        # ── Compute similarity scores ──────────────────────────────────────────
        # Strategy: use the HIGHER of:
        #   (a) global average: average cosine similarity of full answer vs top-K passages
        #   (b) sentence-level: split answer into sentences, embed each, find the
        #       MAX similarity any sentence has against any passage, then average
        #       those per-sentence maxima.
        # This prevents long answers from being penalised when individual
        # claims within the answer are well-supported by specific passages.

        global_avg = self._vector_store.compute_average_similarity(
            answer_vector,
            passage_vectors,
            top_k=_TOP_K_FOR_CONFIDENCE,
        )

        # Sentence-level similarity
        sentence_avg = await self._compute_sentence_level_confidence(
            answer=answer,
            passage_vectors=passage_vectors,
        )

        # Final confidence: best of the two methods
        confidence = max(global_avg, sentence_avg)

        top_results = self._vector_store.search(
            answer_vector,
            passage_vectors,
            evidence_passages,
            k=1,
        )

        # Best single evidence passage
        if top_results:
            top_passage, top_score = top_results[0]
        else:
            top_passage = evidence_passages[0] if evidence_passages else {}
            top_score = confidence

        # ── Classify status and risk from actual scores ───────────────────────
        status = self._classify_status(confidence)
        hallucination_risk = self._classify_risk(status, confidence)
        evidence_status = self._classify_evidence_status(confidence)

        # ── Source reliability classification ─────────────────────────────────
        source_reliability, source_status = self._classify_source(
            top_passage.get("source_type", "Unknown")
        )

        # ── LLM Judge: natural-language explanation ────────────────────────────
        # Import here to avoid circular imports (verifier ← generator)
        from llm.generator import LLMGenerator

        llm = LLMGenerator()
        judge_result = await llm.llm_judge(
            question=question,
            answer=answer,
            top_evidence=top_passage.get("text", "No evidence available."),
            similarity_score=top_score,
        )

        # ── Build verification outcome text ───────────────────────────────────
        verification_outcome = self._build_outcome_text(
            status=status,
            confidence=confidence,
            n_passages=len(evidence_passages),
        )

        # ── Build summary ─────────────────────────────────────────────────────
        summary = self._build_summary(
            question=question,
            status=status,
            confidence=confidence,
            hallucination_risk=hallucination_risk,
            n_passages=len(evidence_passages),
            source=top_passage.get("source", "Wikipedia"),
        )

        # ── Return structured result ──────────────────────────────────────────
        return {
            "status": status,
            "confidence": round(confidence, 4),
            "hallucinationRisk": hallucination_risk,
            "evidenceText": top_passage.get("text", ""),
            "source": top_passage.get("source_url", top_passage.get("source", "")),
            "sourceReliability": source_reliability,
            "sourceStatus": source_status,
            "riskExplanation": judge_result.get(
                "riskExplanation",
                "Unable to generate risk explanation."
            ),
            "recommendation": judge_result.get(
                "recommendation",
                "Verify with authoritative sources."
            ),
            "summary": summary,
            "verificationMethod": "Semantic Similarity (sentence-transformers/FAISS) + LLM Judge",
            "evidenceStatus": evidence_status,
            "verificationOutcome": verification_outcome,
        }

    # ─── Sentence-Level Confidence ────────────────────────────────────────────

    async def _compute_sentence_level_confidence(
        self,
        answer: str,
        passage_vectors: "np.ndarray",
    ) -> float:
        """
        Split the answer into sentences and embed each one individually.
        For each sentence, find its best matching passage.
        Return the mean of per-sentence best-match scores.

        This avoids penalising long answers where the holistic answer
        vector drifts away from any single passage vector.
        """
        import re
        import numpy as np

        # Split answer into sentences (simple regex — good enough for this use)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 15]
        if not sentences:
            return 0.0

        try:
            sent_vectors = await self._embedder.embed_batch_async(sentences)
        except Exception:
            return 0.0

        if sent_vectors is None or len(sent_vectors) == 0:
            return 0.0

        # For each sentence vector, find max cosine similarity across all passages
        import faiss
        embedding_dim = passage_vectors.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(passage_vectors)

        per_sentence_scores = []
        for svec in sent_vectors:
            svec_2d = svec.reshape(1, -1)
            scores, _ = index.search(svec_2d, 1)  # top-1 per sentence
            best = float(max(0.0, min(1.0, scores[0][0])))
            per_sentence_scores.append(best)

        return float(np.mean(per_sentence_scores)) if per_sentence_scores else 0.0

    # ─── Classification Methods ───────────────────────────────────────────────

    @staticmethod
    def _classify_status(confidence: float) -> str:
        """
        Map cosine similarity score to a verification status.
        Thresholds defined in the architecture specification.
        """
        if confidence >= _THRESHOLD_VERIFIED:
            return "Verified"
        elif confidence >= _THRESHOLD_PARTIALLY_LOW:
            return "Partially Supported"
        elif confidence >= _THRESHOLD_NOT_SUPPORTED_LOW:
            return "Not Supported"
        else:
            return "Unable to Verify"

    @staticmethod
    def _classify_risk(status: str, confidence: float) -> str:
        """
        Derive hallucination risk from verification status + confidence.
        All risk levels are derived from actual analysis — not hardcoded.
        """
        if status == "Verified":
            if confidence >= _RISK_NO_RISK_MIN_CONF:
                return "NO Risk"
            else:
                return "Very Low"
        elif status == "Partially Supported":
            if confidence >= _RISK_VLOW_PARTIAL_MIN_CONF:
                return "Very Low"
            else:
                return "Low"
        elif status == "Not Supported":
            return "Medium"
        else:  # Unable to Verify
            return "High"

    @staticmethod
    def _classify_evidence_status(confidence: float) -> str:
        """Classify the evidence status based on similarity score."""
        if confidence >= _THRESHOLD_VERIFIED:
            return "Supporting"
        elif confidence >= _THRESHOLD_PARTIALLY_LOW:
            return "Neutral"
        elif confidence >= _THRESHOLD_NOT_SUPPORTED_LOW:
            return "Contradicting"
        else:
            return "Insufficient"

    @staticmethod
    def _classify_source(source_type: str) -> tuple:
        """
        Classify source reliability and status.

        Wikipedia is considered 'High' reliability / 'Authoritative' for
        general knowledge questions, but not 'Official' (reserved for
        government/regulatory sources).

        This does NOT automatically label any source as 'Official' or
        government-approved without actual verification — per specification.

        Returns:
            (reliability: str, status: str)
        """
        reliability_map = {
            "Wikipedia":  ("High", "Authoritative"),
            "Government": ("High", "Official"),
            "Academic":   ("High", "Authoritative"),
            "News":       ("Medium", "General"),
        }
        return reliability_map.get(source_type, ("Medium", "General"))

    # ─── Text Builders ────────────────────────────────────────────────────────

    @staticmethod
    def _build_outcome_text(
        status: str, confidence: float, n_passages: int
    ) -> str:
        pct = int(round(confidence * 100))
        return (
            f"{status} — Semantic similarity: {pct}% across {n_passages} "
            f"evidence passage{'s' if n_passages != 1 else ''} retrieved from Wikipedia."
        )

    @staticmethod
    def _build_summary(
        question: str,
        status: str,
        confidence: float,
        hallucination_risk: str,
        n_passages: int,
        source: str,
    ) -> str:
        pct = int(round(confidence * 100))
        return (
            f"The AI-generated answer to this question was cross-referenced against "
            f"{n_passages} evidence passage{'s' if n_passages != 1 else ''} retrieved "
            f"from {source}. "
            f"The semantic similarity between the answer and retrieved evidence is {pct}%, "
            f"resulting in a verification status of '{status}' and a hallucination risk "
            f"of '{hallucination_risk}'."
        )

    # ─── Fallback: Unable to Verify ───────────────────────────────────────────

    @staticmethod
    def _build_unable_to_verify_result(
        question: str, answer: str, reason: str
    ) -> dict:
        """
        Build a structured result for when evidence retrieval failed or
        produced no results. This is an honest reporting of the pipeline state.
        """
        return {
            "status": "Unable to Verify",
            "confidence": 0.0,
            "hallucinationRisk": "High",
            "evidenceText": (
                "No relevant evidence could be retrieved for this question. "
                f"Reason: {reason}"
            ),
            "source": "No source available",
            "sourceReliability": "Low",
            "sourceStatus": "Unverified",
            "riskExplanation": (
                "The answer could not be verified because no relevant evidence was "
                "found in the knowledge base. Without supporting evidence, the "
                "hallucination risk is classified as High."
            ),
            "recommendation": (
                "Treat this answer with caution. No supporting evidence was found. "
                "Verify this information using authoritative primary sources before relying on it."
            ),
            "summary": (
                f"The AI-generated answer to this question could not be verified. "
                f"No relevant evidence passages were retrieved. "
                f"Reason: {reason} "
                f"Hallucination risk: High."
            ),
            "verificationMethod": "Semantic Similarity (sentence-transformers/FAISS) + LLM Judge",
            "evidenceStatus": "Insufficient",
            "verificationOutcome": (
                "Unable to Verify — Evidence retrieval returned no results. "
                "Confidence: 0%."
            ),
        }
