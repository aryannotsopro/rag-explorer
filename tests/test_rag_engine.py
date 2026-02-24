"""
tests/test_rag_engine.py - Unit tests for RAGEngine with mocked external calls.

All OpenAI and Pinecone interactions are mocked — no API keys required.
Run with: pytest tests/test_rag_engine.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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
    @patch("rag_engine.OpenAI")
    @patch("rag_engine.AsyncOpenAI")
    @patch("rag_engine.get_settings")
    def test_init(self, mock_settings, mock_async_oai, mock_oai, mock_pc):
        """Engine should initialise without error when settings are present."""
        cfg = MagicMock()
        cfg.pinecone_api_key = "test-key"
        cfg.openai_api_key = "test-oai-key"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.llm_model = "gpt-4o-mini"
        cfg.pinecone_index_name = "test-index"
        cfg.max_concurrent_requests = 2
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine
        engine = RAGEngine()
        assert engine is not None
        mock_pc.assert_called_once_with(api_key="test-key")


class TestEmbedTexts:
    """Tests for the embed_texts async method."""

    @patch("rag_engine.Pinecone")
    @patch("rag_engine.AsyncOpenAI")
    @patch("rag_engine.get_settings")
    def test_embed_returns_vectors(self, mock_settings, mock_async_oai, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.openai_api_key = "k"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.embedding_batch_size = 100
        cfg.max_concurrent_requests = 2
        cfg.pinecone_index_name = "idx"
        mock_settings.return_value = cfg

        # Mock the sync embed call
        fake_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_embedding)]

        from rag_engine import RAGEngine
        engine = RAGEngine()
        engine._sync_client = MagicMock()
        engine._sync_client.embeddings.create.return_value = mock_response

        result = asyncio.run(engine.embed_texts(["Hello world"]))
        assert len(result) == 1
        assert len(result[0]) == 1536

    @patch("rag_engine.Pinecone")
    @patch("rag_engine.AsyncOpenAI")
    @patch("rag_engine.get_settings")
    def test_embed_batches_correctly(self, mock_settings, mock_async_oai, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.openai_api_key = "k"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.embedding_batch_size = 2   # force batching
        cfg.max_concurrent_requests = 2
        cfg.pinecone_index_name = "idx"
        mock_settings.return_value = cfg

        fake_emb = [0.0] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_emb), MagicMock(embedding=fake_emb)]

        from rag_engine import RAGEngine
        engine = RAGEngine()
        engine._sync_client = MagicMock()
        engine._sync_client.embeddings.create.return_value = mock_response

        texts = ["a", "b", "c", "d"]   # 4 texts, batch size 2 → 2 calls
        result = asyncio.run(engine.embed_texts(texts))
        assert len(result) == 4
        assert engine._sync_client.embeddings.create.call_count == 2


class TestSearch:
    """Tests for the search method."""

    @patch("rag_engine.Pinecone")
    @patch("rag_engine.AsyncOpenAI")
    @patch("rag_engine.get_settings")
    def test_search_returns_results(self, mock_settings, mock_async_oai, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.openai_api_key = "k"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.embedding_batch_size = 100
        cfg.embedding_dimension = 1536
        cfg.max_concurrent_requests = 2
        cfg.pinecone_index_name = "idx"
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine

        engine = RAGEngine()
        engine._sync_client = MagicMock()

        # Mock embed
        fake_emb = [0.1] * 1536
        engine._sync_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_emb)]
        )

        # Mock Pinecone query
        mock_match = make_mock_pinecone_match(id_="v1", score=0.9, text="AI is great")
        mock_query_response = MagicMock()
        mock_query_response.matches = [mock_match]
        engine._index = MagicMock()
        engine._index.query.return_value = mock_query_response

        query = SearchQuery(query_text="What is AI?")
        results, latency = asyncio.run(engine.search(query))

        assert len(results) == 1
        assert results[0].rank == 1
        assert results[0].score == pytest.approx(0.9)
        assert results[0].chunk.text == "AI is great"
        assert latency > 0

    @patch("rag_engine.Pinecone")
    @patch("rag_engine.AsyncOpenAI")
    @patch("rag_engine.get_settings")
    def test_search_passes_filter(self, mock_settings, mock_async_oai, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.openai_api_key = "k"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.embedding_batch_size = 100
        cfg.embedding_dimension = 1536
        cfg.max_concurrent_requests = 2
        cfg.pinecone_index_name = "idx"
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine

        engine = RAGEngine()
        engine._sync_client = MagicMock()
        engine._sync_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )

        engine._index = MagicMock()
        engine._index.query.return_value = MagicMock(matches=[])

        query = SearchQuery(
            query_text="Roman empire",
            filters=MetadataFilter(category=DocumentCategory.HISTORY),
        )
        asyncio.run(engine.search(query))

        call_kwargs = engine._index.query.call_args[1]
        assert call_kwargs["filter"] == {"category": {"$eq": "history"}}


class TestGenerateResponse:
    """Tests for the generate_response method."""

    @patch("rag_engine.Pinecone")
    @patch("rag_engine.AsyncOpenAI")
    @patch("rag_engine.get_settings")
    def test_generate_response_structure(self, mock_settings, mock_async_oai, mock_pc):
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.openai_api_key = "k"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.embedding_batch_size = 100
        cfg.embedding_dimension = 1536
        cfg.max_concurrent_requests = 2
        cfg.pinecone_index_name = "idx"
        cfg.llm_model = "gpt-4o-mini"
        cfg.llm_temperature = 0.1
        cfg.llm_max_tokens = 512
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine
        from models import SearchResult

        engine = RAGEngine()
        chunk = make_chunk("Quantum physics is fascinating.")
        result = SearchResult(chunk=chunk, score=0.88, rank=1)

        # Mock the LLM response
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Great answer about quantum."
        mock_completion.usage.total_tokens = 120
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 20
        engine._sync_client = MagicMock()
        engine._sync_client.chat.completions.create.return_value = mock_completion

        response = asyncio.run(
            engine.generate_response("What is quantum physics?", [result])
        )
        assert response.answer == "Great answer about quantum."
        assert response.tokens_used == 120
        assert response.prompt_tokens == 100
        assert response.completion_tokens == 20
        assert len(response.sources) == 1

    @patch("rag_engine.Pinecone")
    @patch("rag_engine.AsyncOpenAI")
    @patch("rag_engine.get_settings")
    def test_query_no_results(self, mock_settings, mock_async_oai, mock_pc):
        """If search returns nothing, engine should return a polite no-results message."""
        cfg = MagicMock()
        cfg.pinecone_api_key = "k"
        cfg.openai_api_key = "k"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.embedding_batch_size = 100
        cfg.embedding_dimension = 1536
        cfg.max_concurrent_requests = 2
        cfg.pinecone_index_name = "idx"
        mock_settings.return_value = cfg

        from rag_engine import RAGEngine

        engine = RAGEngine()
        engine._sync_client = MagicMock()
        engine._sync_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.0] * 1536)]
        )
        engine._index = MagicMock()
        engine._index.query.return_value = MagicMock(matches=[])

        query = SearchQuery(query_text="Completely unknown topic xyz123")
        response = asyncio.run(engine.query(query))
        assert "couldn't find" in response.answer.lower() or "relevant" in response.answer.lower()
        assert response.sources == []
