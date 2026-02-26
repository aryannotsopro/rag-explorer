"""
tests/test_rag_engine.py - Unit tests for RAGEngine with mocked external calls.

All Groq, HuggingFace, and Pinecone interactions are mocked — no API keys required.
Run with: pytest tests/test_rag_engine.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import (
    DocumentCategory,
    DocumentChunk,
    DocumentMetadata,
    MetadataFilter,
    SearchQuery,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_chunk(text: str = "Sample text", source: str = "test.txt") -> DocumentChunk:
    """Factory for test DocumentChunk objects."""
    meta = DocumentMetadata(
        source=source,
        category=DocumentCategory.TECHNOLOGY,
        chunk_index=0,
        total_chunks=1,
    )
    return DocumentChunk(text=text, metadata=meta)


def make_mock_pinecone_match(id_="id-1", score=0.85, text="chunk text", category="technology"):
    """Create a mock Pinecone QueryMatch object."""
    match = MagicMock()
    match.id = id_
    match.score = score
    match.metadata = {
        "text": text,
        "source": "test.txt",
        "category": category,
        "page_number": 1,
        "chunk_index": 0,
        "total_chunks": 1,
        "ingested_at": "2024-01-01T00:00:00",
        "document_date": "",
        "author": "",
        "file_type": "txt",
        "word_count": 2,
        "tags": [],
    }
    return match


# --------------------------------------------------------------------------- #
# RAGEngine tests
# --------------------------------------------------------------------------- #
class TestRAGEngineInit:
    """Test RAGEngine initialization (no index calls)."""

    @patch("rag_engine.Pinecone")
    @patch("rag_engine.AsyncGroq")
    @patch("rag_engine.get_settings")
    def test_init(self, mock_settings, mock_groq, mock_pc):
        """Engine should initialise without error when settings are present."""
        cfg = MagicMock()
        cfg.pinecone_api_key = "test-key"
        cfg.groq_api_key = "test-groq-key"
        cfg.huggingface_api_key = "test-hf-key"
        cfg.embedding_model = "hf-model"
        cfg.llm_model = "llama-3-8b"
        cfg.max_concurrent_requests = 2
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine
        engine = RAGEngine()
        assert engine is not None
        mock_pc.assert_called_once_with(api_key="test-key")
        mock_groq.assert_called_once_with(api_key="test-groq-key")
        assert engine._hf_endpoint == "https://api-inference.huggingface.co/models/hf-model"


class TestEmbedTexts:
    """Tests for the embed_texts async method."""

    @pytest.mark.asyncio
    @patch("rag_engine.Pinecone")
    @patch("rag_engine.AsyncGroq")
    @patch("rag_engine.get_settings")
    @patch("rag_engine.httpx.AsyncClient.post")
    async def test_embed_returns_vectors(self, mock_post, mock_settings, mock_groq, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.groq_api_key = "k"
        cfg.huggingface_api_key = "k"
        cfg.embedding_batch_size = 100
        cfg.max_concurrent_requests = 2
        mock_settings.return_value = cfg

        # Mock httpx response
        fake_embedding = [0.1] * 384
        mock_response = MagicMock()
        mock_response.json.return_value = [fake_embedding]
        mock_post.return_value = mock_response

        from rag_engine import RAGEngine
        engine = RAGEngine()

        result = await engine.embed_texts(["Hello world"])
        assert len(result) == 1
        assert len(result[0]) == 384


class TestSearch:
    """Tests for the search method."""

    @pytest.mark.asyncio
    @patch("rag_engine.Pinecone")
    @patch("rag_engine.get_settings")
    @patch("rag_engine.httpx.AsyncClient.post")
    async def test_search_returns_results(self, mock_post, mock_settings, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.groq_api_key = "k"
        cfg.huggingface_api_key = "k"
        cfg.embedding_dimension = 384
        cfg.max_concurrent_requests = 2
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine
        engine = RAGEngine()

        # Mock Pinecone index
        mock_index = MagicMock()
        mock_query_response = MagicMock()
        mock_query_response.matches = [make_mock_pinecone_match()]
        mock_index.query.return_value = mock_query_response
        engine._index = mock_index

        # Mock embed
        fake_emb = [0.1] * 384
        mock_response = MagicMock()
        mock_response.json.return_value = [fake_emb]
        mock_post.return_value = mock_response

        sq = SearchQuery(query_text="foo", top_k=5)
        results, latency = await engine.search(sq)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "id-1"
        assert results[0].score == 0.85
        assert results[0].chunk.text == "chunk text"
        assert latency >= 0


class TestGenerateResponse:
    """Tests for the generate_response method."""

    @pytest.mark.asyncio
    @patch("rag_engine.Pinecone")
    @patch("rag_engine.get_settings")
    async def test_generate_response_structure(self, mock_settings, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.groq_api_key = "k"
        cfg.huggingface_api_key = "k"
        cfg.llm_model = "llama-3"
        cfg.llm_temperature = 0.1
        cfg.llm_max_tokens = 100
        cfg.max_concurrent_requests = 2
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine
        engine = RAGEngine()

        # Mock Groq client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "This is the answer."
        mock_completion.usage.total_tokens = 120
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 20
        
        engine._groq_client = AsyncMock()
        engine._groq_client.chat.completions.create.return_value = mock_completion

        chunk = make_chunk("Context block 1")
        from models import SearchResult
        results = [SearchResult(chunk=chunk, score=0.9, rank=1)]

        response = await engine.generate_response("query", results, retrieval_latency_ms=42.0)

        assert response.query == "query"
        assert response.answer == "This is the answer."
        assert len(response.sources) == 1
        assert response.tokens_used == 120
        assert response.retrieval_latency_ms == 42.0
        assert response.generation_latency_ms >= 0


class TestFullQuery:
    """Tests the full orchestrating `query` async method."""

    @pytest.mark.asyncio
    @patch("rag_engine.Pinecone")
    @patch("rag_engine.get_settings")
    async def test_query_no_results(self, mock_settings, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.groq_api_key = "k"
        cfg.huggingface_api_key = "k"
        cfg.max_concurrent_requests = 2
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine
        engine = RAGEngine()
        
        # Override embed and search to return empty
        async def mock_search(*args, **kwargs):
            return [], 10.0
        engine.search = mock_search
        # Generation should NOT be called
        engine.generate_response = AsyncMock()

        sq = SearchQuery(query_text="nothing", top_k=5)
        response = await engine.query(sq)

        assert "couldn't find any relevant documents" in response.answer
        assert len(response.sources) == 0
        assert response.retrieval_latency_ms == 10.0
        engine.generate_response.assert_not_called()
