"""Local embedding service using TF-IDF (fallback when API not available)."""

from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """Local TF-IDF based embedding service (no API required)."""

    def __init__(self, vector_size: int = 128) -> None:
        """Initialize with vector size.

        Args:
            vector_size: Dimension of output vectors (must be power of 2)
        """
        self.vector_size = vector_size
        self._vocab: dict[str, int] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> list[str]:
        """Simple Chinese/English tokenization."""
        # Remove punctuation and normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text)

        tokens = []
        for part in text.split():
            # English words
            if re.match(r'^[a-z]+$', part):
                if len(part) >= 2:
                    tokens.append(part)
            else:
                # Chinese: char-level + 2-grams
                chars = list(part)
                for c in chars:
                    if len(c.strip()):
                        tokens.append(c)
                # Add 2-grams
                for i in range(len(chars) - 1):
                    bigram = chars[i] + chars[i + 1]
                    tokens.append(bigram)

        return tokens

    def _hash_feature(self, token: str) -> int:
        """Hash token to feature index."""
        return hash(token) % self.vector_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate TF-IDF embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Tokenize all texts
        all_tokens: list[list[str]] = []
        doc_freq: dict[str, int] = {}

        for text in texts:
            tokens = self._tokenize(text)
            all_tokens.append(tokens)

            # Document frequency
            seen = set()
            for t in tokens:
                if t not in seen:
                    doc_freq[t] = doc_freq.get(t, 0) + 1
                    seen.add(t)

        n_docs = len(texts)

        # Generate TF-IDF vectors
        embeddings = []
        for tokens in all_tokens:
            vector = [0.0] * self.vector_size

            if not tokens:
                embeddings.append(vector)
                continue

            # Term frequency
            tf: dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            # Normalize TF
            max_tf = max(tf.values()) if tf else 1
            for t in tf:
                tf[t] = 0.5 + 0.5 * tf[t] / max_tf  # Augmented normalized TF

            # Build vector using hashing trick
            for t, freq in tf.items():
                # IDF
                idf = math.log((n_docs + 1) / (doc_freq.get(t, 1) + 1)) + 1

                # TF-IDF
                weight = freq * idf

                # Hash to multiple bins for better distribution
                h1 = hash(t) % self.vector_size
                h2 = hash(t + "_salt") % self.vector_size

                vector[h1] += weight
                vector[h2] += weight * 0.5

            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vector))
            if norm > 0:
                vector = [x / norm for x in vector]

            embeddings.append(vector)

        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product: float = float(sum(a * b for a, b in zip(vec1, vec2)))
        norm1: float = float(sum(a * a for a in vec1) ** 0.5)
        norm2: float = float(sum(b * b for b in vec2) ** 0.5)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        denom = norm1 * norm2
        if denom == 0:
            return 0.0
        return dot_product / denom
