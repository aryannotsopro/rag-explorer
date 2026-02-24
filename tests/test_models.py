"""
tests/test_models.py - Unit tests for Pydantic models.

No API calls. Tests validation, serialisation, and business logic.
Run with: pytest tests/test_models.py -v
"""

import pytest
from datetime import date
from pydantic import ValidationError

from models import (
    DocumentCategory,
    DocumentChunk,
    DocumentMetadata,
    FileType,
    IngestionReport,
    MetadataFilter,
    QueryHistoryItem,
    RAGResponse,
    SearchQuery,
    SearchResult,
)


# --------------------------------------------------------------------------- #
# DocumentMetadata
# --------------------------------------------------------------------------- #
class TestDocumentMetadata:

    def test_basic_creation(self):
        meta = DocumentMetadata(source="doc.pdf", category=DocumentCategory.TECHNOLOGY)
        assert meta.source == "doc.pdf"
        assert meta.category == DocumentCategory.TECHNOLOGY
        assert meta.chunk_index == 0
        assert meta.total_chunks == 1
        assert meta.ingested_at  # auto-set

    def test_date_coercion_from_date_object(self):
        meta = DocumentMetadata(source="x.txt", document_date=date(2024, 6, 15))
        assert meta.document_date == "2024-06-15"

    def test_date_coercion_from_string(self):
        meta = DocumentMetadata(source="x.txt", document_date="2024-06-15")
        assert meta.document_date == "2024-06-15"

    def test_to_pinecone_metadata_types(self):
        meta = DocumentMetadata(
            source="test.pdf",
            category=DocumentCategory.SCIENCE,
            page_number=3,
            author="Dr. Who",
            file_type=FileType.PDF,
            tags=["quantum", "physics"],
        )
        pm = meta.to_pinecone_metadata()
        # All values must be Pinecone-compatible types
        for k, v in pm.items():
            assert isinstance(v, (str, int, float, bool, list)), \
                f"Key '{k}' has non-Pinecone type {type(v)}"
        assert pm["category"] == "science"
        assert pm["author"] == "Dr. Who"
        assert pm["tags"] == ["quantum", "physics"]

    def test_missing_source_raises(self):
        with pytest.raises(ValidationError):
            DocumentMetadata()  # source is required

    def test_invalid_category_raises(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(source="f.txt", category="not_a_category")

    def test_page_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(source="f.txt", page_number=0)


# --------------------------------------------------------------------------- #
# DocumentChunk
# --------------------------------------------------------------------------- #
class TestDocumentChunk:

    def _meta(self, source="doc.txt"):
        return DocumentMetadata(source=source)

    def test_basic_creation(self):
        chunk = DocumentChunk(text="Hello world", metadata=self._meta())
        assert chunk.text == "Hello world"
        assert chunk.chunk_id  # UUID auto-generated
        assert chunk.embedding is None

    def test_text_is_stripped(self):
        chunk = DocumentChunk(text="  trimmed  ", metadata=self._meta())
        assert chunk.text == "trimmed"

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            DocumentChunk(text="", metadata=self._meta())

    def test_word_count_property(self):
        chunk = DocumentChunk(text="one two three four five", metadata=self._meta())
        assert chunk.word_count == 5

    def test_unique_ids(self):
        c1 = DocumentChunk(text="text one", metadata=self._meta())
        c2 = DocumentChunk(text="text two", metadata=self._meta())
        assert c1.chunk_id != c2.chunk_id


# --------------------------------------------------------------------------- #
# MetadataFilter
# --------------------------------------------------------------------------- #
class TestMetadataFilter:

    def test_empty_filter_returns_none(self):
        f = MetadataFilter()
        assert f.to_pinecone_filter() is None

    def test_category_filter(self):
        f = MetadataFilter(category=DocumentCategory.HISTORY)
        pf = f.to_pinecone_filter()
        assert pf == {"category": {"$eq": "history"}}

    def test_source_filter(self):
        f = MetadataFilter(source="report.pdf")
        pf = f.to_pinecone_filter()
        assert pf == {"source": {"$eq": "report.pdf"}}

    def test_date_range_filter(self):
        f = MetadataFilter(date_from="2024-01-01", date_to="2024-12-31")
        pf = f.to_pinecone_filter()
        assert "ingested_at" in pf
        assert pf["ingested_at"]["$gte"] == "2024-01-01"
        assert "2024-12-31" in pf["ingested_at"]["$lte"]

    def test_tags_filter(self):
        f = MetadataFilter(tags=["ai", "nlp"])
        pf = f.to_pinecone_filter()
        assert pf == {"tags": {"$in": ["ai", "nlp"]}}

    def test_combined_filters(self):
        f = MetadataFilter(
            category=DocumentCategory.TECHNOLOGY,
            source="guide.pdf",
            author="Alice",
        )
        pf = f.to_pinecone_filter()
        assert len(pf) == 3
        assert pf["category"] == {"$eq": "technology"}
        assert pf["source"] == {"$eq": "guide.pdf"}
        assert pf["author"] == {"$eq": "Alice"}


# --------------------------------------------------------------------------- #
# SearchQuery
# --------------------------------------------------------------------------- #
class TestSearchQuery:

    def test_basic_query(self):
        q = SearchQuery(query_text="What is quantum computing?")
        assert q.query_text == "What is quantum computing?"
        assert q.top_k == 5  # default

    def test_query_stripped(self):
        q = SearchQuery(query_text="  spaced  ")
        assert q.query_text == "spaced"

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            SearchQuery(query_text="")

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            SearchQuery(query_text="hello", top_k=0)
        with pytest.raises(ValidationError):
            SearchQuery(query_text="hello", top_k=21)

    def test_with_filter(self):
        q = SearchQuery(
            query_text="Tell me about Rome",
            top_k=3,
            filters=MetadataFilter(category=DocumentCategory.HISTORY),
        )
        pf = q.filters.to_pinecone_filter()
        assert pf["category"]["$eq"] == "history"


# --------------------------------------------------------------------------- #
# RAGResponse
# --------------------------------------------------------------------------- #
class TestRAGResponse:

    def test_auto_total_tokens(self):
        r = RAGResponse(
            query="Q",
            answer="A",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert r.tokens_used == 150

    def test_auto_total_latency(self):
        r = RAGResponse(
            query="Q",
            answer="A",
            retrieval_latency_ms=30.0,
            generation_latency_ms=200.0,
        )
        assert r.total_latency_ms == pytest.approx(230.0)

    def test_from_rag_response_history(self):
        r = RAGResponse(query="Test query", answer="A" * 300, tokens_used=42)
        item = QueryHistoryItem.from_rag_response(r)
        assert item.query == "Test query"
        assert len(item.answer_preview) <= 200
        assert item.tokens_used == 42


# --------------------------------------------------------------------------- #
# IngestionReport
# --------------------------------------------------------------------------- #
class TestIngestionReport:

    def test_success_property(self):
        report = IngestionReport(
            source="test.pdf",
            category=DocumentCategory.SCIENCE,
            file_type=FileType.PDF,
            total_chunks=10,
            vectors_upserted=10,
            ingestion_time_ms=500.0,
        )
        assert report.success is True

    def test_partial_failure(self):
        report = IngestionReport(
            source="test.pdf",
            category=DocumentCategory.SCIENCE,
            file_type=FileType.PDF,
            total_chunks=10,
            vectors_upserted=8,  # 2 failed
            ingestion_time_ms=500.0,
            errors=["Upsert batch 2 failed"],
        )
        assert report.success is False
