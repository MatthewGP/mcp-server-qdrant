import logging
from typing import List

import httpx
from openai import AsyncOpenAI

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import EmbeddingProviderSettings

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(EmbeddingProvider):
    """
    OpenAI-compatible embedding provider for local servers like LM Studio, Ollama, etc.
    Supports both OpenAI-compatible API format and Ollama native format.
    """

    def __init__(self, settings: EmbeddingProviderSettings):
        if not settings.api_base_url:
            raise ValueError("api_base_url is required for OpenAI-compatible provider")

        self.settings = settings
        self.model_name = settings.model_name
        self.client = AsyncOpenAI(
            api_key=settings.api_key or "not-needed",  # Some servers require any key
            base_url=settings.api_base_url,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

        # Cache for detected vector size
        self._detected_vector_size: int | None = None
        self._api_format: str | None = None  # "openai" or "ollama"

    async def _detect_api_format(self) -> str:
        """Detect whether the endpoint uses OpenAI or Ollama API format."""
        if self._api_format:
            return self._api_format

        try:
            # Try OpenAI format first
            await self.client.embeddings.create(model=self.model_name, input=["test"])
            self._api_format = "openai"
            logger.info("Detected OpenAI-compatible API format")
            return "openai"
        except Exception as e:
            logger.debug(f"OpenAI format detection failed: {e}")

            # Try Ollama format
            try:
                async with httpx.AsyncClient(timeout=self.settings.timeout) as client:
                    httpx_response = await client.post(
                        f"{self.settings.api_base_url}/embeddings",
                        json={"model": self.model_name, "prompt": "test"},
                    )
                    if httpx_response.status_code == 200:
                        self._api_format = "ollama"
                        logger.info("Detected Ollama API format")
                        return "ollama"
            except Exception as ollama_e:
                logger.debug(f"Ollama format detection failed: {ollama_e}")

        raise ValueError(
            "Unable to determine API format. Please check your server configuration."
        )

    async def _embed_with_openai_format(self, texts: List[str]) -> List[List[float]]:
        """Embed using OpenAI-compatible API format."""
        response = await self.client.embeddings.create(
            model=self.model_name, input=texts
        )

        # Detect vector size if not set
        if self.settings.vector_size is None and self._detected_vector_size is None:
            self._detected_vector_size = len(response.data[0].embedding)
            logger.info(f"Auto-detected vector size: {self._detected_vector_size}")

        return [item.embedding for item in response.data]

    async def _embed_with_ollama_format(self, texts: List[str]) -> List[List[float]]:
        """Embed using Ollama native API format."""
        async with httpx.AsyncClient(timeout=self.settings.timeout) as client:
            embeddings = []
            for text in texts:
                response = await client.post(
                    f"{self.settings.api_base_url}/embeddings",
                    json={"model": self.model_name, "prompt": text},
                )
                response.raise_for_status()
                data = await response.json()
                embeddings.append(data["embedding"])

            # Detect vector size if not set
            if self.settings.vector_size is None and self._detected_vector_size is None:
                self._detected_vector_size = len(embeddings[0])
                logger.info(f"Auto-detected vector size: {self._detected_vector_size}")

            return embeddings

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        if not documents:
            return []

        # Detect API format if not already done
        if not self._api_format:
            await self._detect_api_format()

        try:
            if self._api_format == "openai":
                return await self._embed_with_openai_format(documents)
            else:  # ollama
                return await self._embed_with_ollama_format(documents)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        if not query:
            raise ValueError("Query cannot be empty")

        # Detect API format if not already done
        if not self._api_format:
            await self._detect_api_format()

        try:
            if self._api_format == "openai":
                result = await self._embed_with_openai_format([query])
                return result[0]
            else:  # ollama
                result = await self._embed_with_ollama_format([query])
                return result[0]
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Uses a sanitized version of the model name.
        """
        # Sanitize model name for use in collection name
        sanitized = (
            self.model_name.lower()
            .replace("/", "-")
            .replace("_", "-")
            .replace(".", "-")
        )
        return f"openai-{sanitized}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        if self.settings.vector_size is not None:
            return self.settings.vector_size

        if self._detected_vector_size is not None:
            return self._detected_vector_size

        # If size hasn't been detected yet, we need to trigger an embedding
        # This is a fallback that shouldn't normally be called
        raise ValueError(
            "Vector size not configured. Either set EMBEDDING_VECTOR_SIZE environment variable "
            "or call embed_documents/embed_query first to auto-detect."
        )
