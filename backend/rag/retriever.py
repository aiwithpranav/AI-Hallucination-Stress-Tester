"""
RAG Evidence Retriever Module

Retrieves real-world evidence for any user question using the Wikipedia REST API.

No API key required. Every query is genuinely dynamic — there are no
hardcoded question→evidence mappings.

Evidence is returned as a list of passage dicts:
    {
        "text":        str,   # The passage text
        "source":      str,   # Wikipedia article title
        "source_url":  str,   # Full Wikipedia article URL
        "source_type": str,   # "Wikipedia" — for reliability classification
    }

Usage:
    retriever = EvidenceRetriever()
    passages = await retriever.retrieve("How does CRISPR work?")
"""

import os
import re
import asyncio
from pathlib import Path
from typing import List, Dict

import httpx
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
_env_file = _project_root / ".env"
load_dotenv(dotenv_path=_env_file if _env_file.exists() else None, override=False)
load_dotenv(override=False)

# Wikipedia REST API base URL
_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

# Configuration
_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
_PASSAGE_MAX_CHARS = int(os.getenv("RAG_PASSAGE_MAX_CHARS", "500"))

# Request timeout (seconds)
_TIMEOUT = 15.0


class EvidenceRetriever:
    """
    Retrieves real evidence passages from Wikipedia for any question.
    Uses the Wikipedia search API + page summary API.
    """

    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={
                "User-Agent": (
                    "AIHallucinationStressTester/1.0 "
                    "(portfolio project; educational use only)"
                )
            },
            follow_redirects=True,
        )

    async def retrieve(self, question: str) -> List[Dict]:
        """
        Retrieve relevant evidence passages for the given question.

        Strategy:
          1. Search Wikipedia for the question.
          2. Fetch summaries of the top search results.
          3. Split summaries into passages.
          4. Return up to RAG_TOP_K passages.

        Args:
            question: The user's free-form question.

        Returns:
            List of passage dicts (may be empty if no results found).
        """
        # Extract key terms from the question for better search results
        search_query = self._extract_search_query(question)

        # Search Wikipedia
        search_results = await self._search_wikipedia(search_query)
        if not search_results:
            return []

        # Fetch summaries for top results
        passages = []
        fetch_tasks = [
            self._fetch_article_passages(title)
            for title in search_results[:3]  # Fetch top 3 articles
        ]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                passages.extend(result)
            # Silently skip articles that failed to fetch

        return passages[:_RAG_TOP_K]

    # ─── Search Wikipedia ─────────────────────────────────────────────────────

    async def _search_wikipedia(self, query: str) -> List[str]:
        """
        Search Wikipedia and return a list of matching article titles.

        Args:
            query: Search query string.

        Returns:
            List of article titles (strings).
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "format": "json",
            "utf8": 1,
        }

        try:
            response = await self._client.get(_WIKI_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            return [result["title"] for result in search_results]
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Wikipedia search API returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Wikipedia search request failed: {exc}"
            ) from exc

    # ─── Fetch Article Passages ───────────────────────────────────────────────

    async def _fetch_article_passages(self, title: str) -> List[Dict]:
        """
        Fetch the summary of a Wikipedia article and split into passages.

        Args:
            title: Wikipedia article title.

        Returns:
            List of passage dicts.
        """
        encoded_title = title.replace(" ", "_")
        url = f"{_WIKI_SUMMARY_URL}/{encoded_title}"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception:
            # Individual article fetch failure is non-critical
            return []

        # Extract relevant text fields
        extract = data.get("extract", "")
        article_title = data.get("title", title)
        article_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        if not extract:
            return []

        # Clean and split into passages
        cleaned = self._clean_text(extract)
        passage_texts = self._split_into_passages(cleaned)

        return [
            {
                "text": passage,
                "source": f"Wikipedia: {article_title}",
                "source_url": article_url or f"https://en.wikipedia.org/wiki/{encoded_title}",
                "source_type": "Wikipedia",
            }
            for passage in passage_texts
            if passage.strip()
        ]

    # ─── Text Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_search_query(question: str) -> str:
        """
        Extract a clean search query from a natural-language question.
        Removes common question words for better Wikipedia search results.
        """
        # Remove leading question words
        cleaned = re.sub(
            r"^(what is|what are|who is|who was|how does|how do|when did|"
            r"where is|why is|explain|tell me about|describe|define|"
            r"what was|how was|can you|could you|please)\s+",
            "",
            question.lower().strip(),
            flags=re.IGNORECASE,
        )
        # Remove trailing punctuation
        cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()
        return cleaned if cleaned else question

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove Wikipedia markup artifacts and excessive whitespace."""
        # Remove parenthetical pronunciation guides: (UK: /ˈ.../)
        text = re.sub(r"\((?:UK|US)?:?\s*/[^)]+/\)", "", text)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _split_into_passages(text: str, max_chars: int = _PASSAGE_MAX_CHARS) -> List[str]:
        """
        Split text into passages of at most max_chars characters,
        respecting sentence boundaries where possible.
        """
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        passages = []
        current = ""

        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current) + len(sentence) + 1 <= max_chars:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    passages.append(current)
                # If a single sentence is too long, truncate it
                if len(sentence) > max_chars:
                    passages.append(sentence[:max_chars].rsplit(" ", 1)[0] + "…")
                    current = ""
                else:
                    current = sentence

        if current:
            passages.append(current)

        return passages

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.aclose()
