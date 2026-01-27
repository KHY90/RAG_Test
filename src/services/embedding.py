"""Embedding service using sentence-transformers."""

from typing import Optional

from src.config import settings


class EmbeddingService:
    """Service for generating text embeddings using multilingual-e5-base."""

    def __init__(self):
        self.model = None
        self.model_name = settings.embedding_model
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    async def load_model(self) -> None:
        """Load the embedding model."""
        if self._is_loaded:
            return

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        self._is_loaded = True

    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """
        Encode texts into embeddings.

        Args:
            texts: List of texts to encode
            is_query: If True, prepend "query: " prefix (for e5 models)

        Returns:
            List of embedding vectors
        """
        if not self._is_loaded:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        # For e5 models, add appropriate prefix
        if "e5" in self.model_name.lower():
            if is_query:
                texts = [f"query: {t}" for t in texts]
            else:
                texts = [f"passage: {t}" for t in texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query text."""
        result = self.encode([query], is_query=True)
        return result[0]

    def encode_documents(self, documents: list[str]) -> list[list[float]]:
        """Encode document texts (passages)."""
        return self.encode(documents, is_query=False)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if not self._is_loaded:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")
        return self.model.get_sentence_embedding_dimension()
