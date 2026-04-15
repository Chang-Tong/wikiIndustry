"""Embedding service for generating text vectors using LLM API or local fallback."""

from __future__ import annotations

import logging

import httpx

from app.core.settings import settings
from app.services.local_embedding import LocalEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings via Ollama (强制使用)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        use_local_fallback: bool = False,  # 默认不使用 fallback，强制 Ollama
        use_ollama: bool = True,
    ) -> None:
        """Initialize embedding service.

        强制使用 Ollama 进行向量化，如果 Ollama 未配置或不可用则报错。

        Args:
            base_url: OpenAI-compatible API base URL (可选，用于覆盖默认 Ollama URL)
            api_key: API key (Ollama 不需要)
            model: Embedding model name (可选，用于覆盖默认模型)
            use_local_fallback: 是否允许使用本地 TF-IDF fallback (默认不允许)
            use_ollama: 是否使用 Ollama (必须为 True)
        """
        self.use_local_fallback = use_local_fallback
        self._local_service: LocalEmbeddingService | None = None

        # 检查是否强制使用 Ollama
        if settings.require_ollama_embedding:
            if not settings.ollama_base_url:
                raise RuntimeError(
                    "REQUIRE_OLLAMA_EMBEDDING=true 但未配置 OLLAMA_BASE_URL"
                )
            self.base_url = base_url or settings.ollama_base_url
            self.api_key = "ollama"
            self.model = model or settings.ollama_embedding_model or "qwen3-embedding:0.6b"
            logger.info(f"[强制] 使用 Ollama 进行向量化: {self.base_url}, model: {self.model}")
            return

        # 以下是非强制模式（向后兼容）
        # Priority 1: Explicit parameters
        if base_url:
            self.base_url = base_url
            self.api_key = api_key or ""
            self.model = model or "text-embedding-3-small"
            return

        # Priority 2: Ollama (for embeddings)
        if use_ollama and settings.ollama_base_url:
            self.base_url = settings.ollama_base_url
            self.api_key = "ollama"
            self.model = model or settings.ollama_embedding_model or "qwen3-embedding:0.6b"
            logger.info(f"Using Ollama for embeddings: {self.base_url}, model: {self.model}")
            return

        # Priority 3: OpenAI from settings
        if settings.openai_base_url and settings.openai_api_key:
            self.base_url = settings.openai_base_url
            self.api_key = settings.openai_api_key
            self.model = model or "text-embedding-3-small"
            return

        # No API configured
        self.base_url = ""
        self.api_key = ""
        self.model = model or "local"

    def _get_local_service(self) -> LocalEmbeddingService:
        """Lazy init local embedding service."""
        if self._local_service is None:
            self._local_service = LocalEmbeddingService(vector_size=128)
        return self._local_service

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts using Ollama (强制).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: 如果 Ollama 调用失败且强制使用 Ollama
        """
        # Filter out empty texts
        valid_texts = [t for t in texts if t and len(t.strip()) > 0]
        if not valid_texts:
            return []

        # 强制使用 Ollama
        if settings.require_ollama_embedding:
            if not self.base_url:
                raise RuntimeError("Ollama URL 未配置，无法生成 embedding")
            try:
                logger.info(f"调用 Ollama embedding API: {self.model}, 文本数: {len(valid_texts)}")
                return await self._embed_api(valid_texts)
            except Exception as e:
                logger.error(f"Ollama embedding 失败: {e}")
                raise RuntimeError(
                    f"Ollama 向量化失败 (model={self.model})。"
                    f"请检查: 1) Ollama 服务是否运行 2) 模型 {self.model} 是否已下载"
                ) from e

        # 以下是非强制模式（向后兼容）
        # Try API first if configured
        if self.base_url and self.api_key:
            try:
                return await self._embed_api(valid_texts)
            except Exception as e:
                logger.warning(f"API embedding failed: {e}")
                if not self.use_local_fallback:
                    raise

        # Fall back to local TF-IDF
        if self.use_local_fallback:
            logger.info("Using local TF-IDF embedding fallback")
            return self._get_local_service().embed(valid_texts)

        raise RuntimeError("Embedding failed and no fallback available")

    async def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """Call external API for embeddings."""
        # Truncate long texts
        truncated = []
        for t in texts:
            if len(t) > 8000:
                t = t[:8000]
            truncated.append(t)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": truncated,
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            result = response.json()

            embeddings: list[list[float]] = []
            data = result.get("data", [])
            if not isinstance(data, list):
                return embeddings
            for item in data:
                if not isinstance(item, dict):
                    embeddings.append([])
                    continue
                emb = item.get("embedding", [])
                if not isinstance(emb, list):
                    embeddings.append([])
                    continue
                embeddings.append([float(x) for x in emb])

            return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed([text])
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
