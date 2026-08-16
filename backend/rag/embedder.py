"""
RAG Embedder Module

Wraps sentence-transformers to produce dense vector embeddings for
both the LLM-generated answer and evidence passages.

The embedding model runs locally — no API key required.
Model is downloaded once (~80MB) and cached by the sentence-transformers library.

Usage:
    embedder = Embedder()
    vector = embedder.embed("Some text")
    vectors = embedder.embed_batch(["Text 1", "Text 2"])
"""

import os
import asyncio
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
_env_file = _project_root / ".env"
load_dotenv(dotenv_path=_env_file if _env_file.exists() else None, override=False)
load_dotenv(override=False)

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class Embedder:
    """
    Singleton-safe embedding wrapper for sentence-transformers.

    The model is loaded lazily on first use to avoid startup delays
    when the model is not needed (e.g., health-check requests).
    """

    _instance = None
    _model = None

    def __new__(cls):
        # Singleton — share the loaded model across requests
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        """Lazily load the sentence-transformer model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

        print(f"[Embedder] Loading model: {_MODEL_NAME} (first-run download may take a moment)")
        self._model = SentenceTransformer(_MODEL_NAME)
        print(f"[Embedder] Model loaded successfully.")

    # ─── Public Interface ─────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a dense vector.

        Args:
            text: Input text (sentence, paragraph, etc.)

        Returns:
            numpy float32 array of shape (embedding_dim,).
        """
        self._load_model()
        vector = self._model.encode(
            text,
            normalize_embeddings=True,  # L2 normalize → cosine = dot product
            show_progress_bar=False,
        )
        return vector.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts efficiently.

        Args:
            texts: List of input texts.

        Returns:
            numpy float32 array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32)

        self._load_model()
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    async def embed_async(self, text: str) -> np.ndarray:
        """
        Async wrapper for embed() — runs in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, text)

    async def embed_batch_async(self, texts: List[str]) -> np.ndarray:
        """
        Async wrapper for embed_batch() — runs in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)
