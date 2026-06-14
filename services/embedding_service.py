"""
Embedding Service (Gemini)

Generates text embeddings via Google's free Gemini embedding API
for use with ChromaDB. Kept separate from chat generation, which
is handled by OpenRouter (see openrouter_service.py).
"""

import google.generativeai as genai
from loguru import logger

from config import get_settings


class EmbeddingService:
    """Service for generating text embeddings via Gemini."""

    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.gemini_api_key)

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> list[float]:
        """
        Generate an embedding vector for a single piece of text.

        task_type options: 'retrieval_document', 'retrieval_query'
        """
        result = genai.embed_content(
            model=self.settings.gemini_embedding_model,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text, task_type=task_type))
        return embeddings