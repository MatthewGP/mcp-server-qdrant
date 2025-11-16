from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_server_qdrant.embeddings.openai_compatible import OpenAICompatibleProvider
from mcp_server_qdrant.settings import EmbeddingProviderSettings


@pytest.fixture
def mock_settings():
    """Create mock embedding provider settings."""
    return EmbeddingProviderSettings(
        _env_file=None,
        **{
            "EMBEDDING_PROVIDER": "openai_compatible",
            "EMBEDDING_MODEL": "text-embedding-test-model",
            "EMBEDDING_API_BASE_URL": "http://localhost:2345/v1",
            "EMBEDDING_API_KEY": "test-key",
            "EMBEDDING_VECTOR_SIZE": "384",
            "EMBEDDING_TIMEOUT": "30",
            "EMBEDDING_MAX_RETRIES": "3",
        },
    )


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6]),
    ]
    return mock_response


@pytest.fixture
def mock_ollama_response():
    """Create a mock Ollama API response."""
    return {"embedding": [0.1, 0.2, 0.3]}


@pytest.mark.asyncio
class TestOpenAICompatibleProvider:
    """Integration tests for OpenAICompatibleProvider."""

    async def test_initialization_with_valid_settings(self, mock_settings):
        """Test that the provider can be initialized with valid settings."""
        provider = OpenAICompatibleProvider(mock_settings)
        assert provider.model_name == "text-embedding-test-model"
        assert provider.settings == mock_settings
        assert provider.client is not None

    async def test_initialization_without_api_base_url_fails(self):
        """Test that initialization fails without api_base_url."""
        settings = EmbeddingProviderSettings(
            _env_file=None,
            **{
                "EMBEDDING_PROVIDER": "openai_compatible",
                "EMBEDDING_MODEL": "test-model",
                # Missing EMBEDDING_API_BASE_URL
            },
        )
        with pytest.raises(ValueError, match="api_base_url is required"):
            OpenAICompatibleProvider(settings)

    @patch("mcp_server_qdrant.embeddings.openai_compatible.AsyncOpenAI")
    async def test_detect_openai_api_format(
        self, mock_openai_client, mock_settings, mock_openai_response
    ):
        """Test detection of OpenAI-compatible API format."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = mock_openai_response
        mock_openai_client.return_value = mock_client

        provider = OpenAICompatibleProvider(mock_settings)

        # Trigger format detection
        format_type = await provider._detect_api_format()

        assert format_type == "openai"
        assert provider._api_format == "openai"
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-test-model", input=["test"]
        )

    @patch("httpx.AsyncClient")
    async def test_detect_ollama_api_format(
        self, mock_httpx_client, mock_settings, mock_ollama_response
    ):
        """Test detection of Ollama API format as fallback."""
        # Mock OpenAI client to fail for provider initialization
        with patch(
            "mcp_server_qdrant.embeddings.openai_compatible.AsyncOpenAI"
        ) as mock_openai_client_init:
            # Create a mock client that will fail format detection
            mock_openai_client = AsyncMock()
            mock_openai_client.embeddings.create.side_effect = Exception(
                "OpenAI format failed"
            )
            mock_openai_client_init.return_value = mock_openai_client

            provider = OpenAICompatibleProvider(mock_settings)

            # Mock httpx for Ollama API
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ollama_response
            mock_client.post.return_value = mock_response
            mock_httpx_client.return_value.__aenter__.return_value = mock_client

            format_type = await provider._detect_api_format()

            assert format_type == "ollama"
            assert provider._api_format == "ollama"

    @patch("mcp_server_qdrant.embeddings.openai_compatible.AsyncOpenAI")
    async def test_embed_documents_openai_format(
        self, mock_openai_client, mock_settings, mock_openai_response
    ):
        """Test embedding documents using OpenAI format."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = mock_openai_response
        mock_openai_client.return_value = mock_client

        provider = OpenAICompatibleProvider(mock_settings)
        provider._api_format = "openai"  # Skip format detection

        documents = ["This is a test.", "Another test document."]
        embeddings = await provider.embed_documents(documents)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-test-model", input=documents
        )

    @patch("httpx.AsyncClient")
    async def test_embed_documents_ollama_format(
        self, mock_httpx_client, mock_settings
    ):
        """Test embedding documents using Ollama format."""
        provider = OpenAICompatibleProvider(mock_settings)
        provider._api_format = "ollama"  # Skip format detection

        # Mock httpx for Ollama API
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},  # First document
            {"embedding": [0.4, 0.5, 0.6]},  # Second document
        ]
        # raise_for_status is a regular method, not async
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        documents = ["This is a test.", "Another test document."]
        embeddings = await provider.embed_documents(documents)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

        # Verify two separate calls were made (Ollama processes one document at a time)
        assert mock_client.post.call_count == 2

    @patch("mcp_server_qdrant.embeddings.openai_compatible.AsyncOpenAI")
    async def test_embed_query_openai_format(
        self, mock_openai_client, mock_settings, mock_openai_response
    ):
        """Test embedding a query using OpenAI format."""
        mock_openai_response.data = [Mock(embedding=[0.7, 0.8, 0.9])]
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = mock_openai_response
        mock_openai_client.return_value = mock_client

        provider = OpenAICompatibleProvider(mock_settings)
        provider._api_format = "openai"  # Skip format detection

        query = "This is a test query."
        embedding = await provider.embed_query(query)

        assert embedding == [0.7, 0.8, 0.9]

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-test-model", input=[query]
        )

    @patch("httpx.AsyncClient")
    async def test_embed_query_ollama_format(
        self, mock_httpx_client, mock_settings, mock_ollama_response
    ):
        """Test embedding a query using Ollama format."""
        provider = OpenAICompatibleProvider(mock_settings)
        provider._api_format = "ollama"  # Skip format detection

        # Mock httpx for Ollama API
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_response
        # raise_for_status is a regular method, not async
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        query = "This is a test query."
        embedding = await provider.embed_query(query)

        assert embedding == [0.1, 0.2, 0.3]

        mock_client.post.assert_called_once_with(
            f"{mock_settings.api_base_url}/embeddings",
            json={"model": "text-embedding-test-model", "prompt": query},
        )

    async def test_get_vector_name(self, mock_settings):
        """Test that vector name is generated correctly."""
        provider = OpenAICompatibleProvider(mock_settings)
        vector_name = provider.get_vector_name()

        assert vector_name == "openai-text-embedding-test-model"

    async def test_get_vector_name_sanitization(self):
        """Test that vector name is properly sanitized."""
        settings = EmbeddingProviderSettings(
            _env_file=None,
            **{
                "EMBEDDING_PROVIDER": "openai_compatible",
                "EMBEDDING_MODEL": "text/embedding_model-v2.0",
                "EMBEDDING_API_BASE_URL": "http://localhost:2345/v1",
            },
        )
        provider = OpenAICompatibleProvider(settings)
        vector_name = provider.get_vector_name()

        assert vector_name == "openai-text-embedding-model-v2-0"

    async def test_get_vector_size_configured(self, mock_settings):
        """Test getting vector size when manually configured."""
        provider = OpenAICompatibleProvider(mock_settings)
        vector_size = provider.get_vector_size()

        assert vector_size == 384  # From mock_settings

    async def test_get_vector_size_detected(self):
        """Test getting vector size when auto-detected."""
        settings = EmbeddingProviderSettings(
            _env_file=None,
            **{
                "EMBEDDING_PROVIDER": "openai_compatible",
                "EMBEDDING_MODEL": "test-model",
                "EMBEDDING_API_BASE_URL": "http://localhost:2345/v1",
                # No EMBEDDING_VECTOR_SIZE set
            },
        )
        provider = OpenAICompatibleProvider(settings)
        provider._detected_vector_size = 512  # Simulate auto-detection

        vector_size = provider.get_vector_size()
        assert vector_size == 512

    async def test_get_vector_size_not_detected_fails(self):
        """Test that getting vector size fails when neither configured nor detected."""
        settings = EmbeddingProviderSettings(
            _env_file=None,
            **{
                "EMBEDDING_PROVIDER": "openai_compatible",
                "EMBEDDING_MODEL": "test-model",
                "EMBEDDING_API_BASE_URL": "http://localhost:2345/v1",
                # No EMBEDDING_VECTOR_SIZE set
            },
        )
        provider = OpenAICompatibleProvider(settings)

        with pytest.raises(ValueError, match="Vector size not configured"):
            provider.get_vector_size()

    @patch("mcp_server_qdrant.embeddings.openai_compatible.AsyncOpenAI")
    async def test_auto_detect_vector_size(self, mock_openai_client):
        """Test that vector size is auto-detected during embedding."""
        settings = EmbeddingProviderSettings(
            _env_file=None,
            **{
                "EMBEDDING_PROVIDER": "openai_compatible",
                "EMBEDDING_MODEL": "test-model",
                "EMBEDDING_API_BASE_URL": "http://localhost:2345/v1",
                # No EMBEDDING_VECTOR_SIZE set
            },
        )

        # Mock response with 512 dimensions
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.0] * 512)]
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_client.return_value = mock_client

        provider = OpenAICompatibleProvider(settings)
        provider._api_format = "openai"  # Skip format detection

        await provider.embed_documents(["test"])

        assert provider._detected_vector_size == 512
        assert provider.get_vector_size() == 512

    async def test_embed_empty_documents(self, mock_settings):
        """Test embedding empty documents list."""
        provider = OpenAICompatibleProvider(mock_settings)
        provider._api_format = "openai"  # Skip format detection

        embeddings = await provider.embed_documents([])
        assert embeddings == []

    async def test_embed_empty_query_fails(self, mock_settings):
        """Test that embedding empty query fails."""
        provider = OpenAICompatibleProvider(mock_settings)
        provider._api_format = "openai"  # Skip format detection

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await provider.embed_query("")
