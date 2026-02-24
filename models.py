"""
models.py - Pydantic v2 data models for the RAG application.

All data flowing through the system is validated through these models,
providing type safety and clear contracts between components.
"""

from __future__ import annotations

import uuid
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class DocumentCategory(str, Enum):
    """Controlled vocabulary for document categories (used as Pinecone metadata)."""
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    HISTORY = "history"
    BUSINESS = "business"
    HEALTH = "health"
    GENERAL = "general"
    OTHER = "other"


class FileType(str, Enum):
    """Supported document file types."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"


# --------------------------------------------------------------------------- #
# Core document models
# --------------------------------------------------------------------------- #
class DocumentMetadata(BaseModel):
    """
    Rich metadata attached to every chunk stored in Pinecone.

    These fields become Pinecone metadata and can be used in filter expressions.
    Keep values JSON-serialisable (str, int, float, bool, list[str]).
    """
    source: str = Field(..., description="Original filename or URL")
    category: DocumentCategory = Field(
        default=DocumentCategory.GENERAL,
        description="Document topic category for filtering",
    )
    page_number: Optional[int] = Field(
        default=None, ge=1, description="Page number within the source document"
    )
    chunk_index: int = Field(
        default=0, ge=0, description="Zero-based position of this chunk within the document"
    )
    total_chunks: int = Field(
        default=1, ge=1, description="Total number of chunks in the parent document"
    )
    # Store as ISO string so Pinecone (which doesn't support datetime natively) can filter on it
    ingested_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC ISO-8601 timestamp of when the chunk was ingested",
    )
    document_date: Optional[str] = Field(
        default=None,
        description="Optional document creation/publication date (ISO-8601 date string)",
    )
    author: Optional[str] = Field(default=None, description="Document author")
    file_type: Optional[FileType] = Field(default=None, description="Source file type")
    word_count: Optional[int] = Field(
        default=None, ge=0, description="Approximate word count of this chunk"
    )
    tags: List[str] = Field(default_factory=list, description="Free-form tags")

    @field_validator("document_date", mode="before")
    @classmethod
    def coerce_date(cls, v):
        """Accept date objects and convert them to ISO strings."""
        if isinstance(v, (date, datetime)):
            return v.isoformat()[:10]
        return v

    def to_pinecone_metadata(self) -> Dict[str, Any]:
        """
        Flatten the model to a dict suitable for Pinecone metadata storage.
        Pinecone accepts str, int, float, bool, and list-of-str values only.
        """
        return {
            "source": self.source,
            "category": self.category.value,
            "page_number": self.page_number or 0,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "ingested_at": self.ingested_at,
            "document_date": self.document_date or "",
            "author": self.author or "",
            "file_type": self.file_type.value if self.file_type else "",
            "word_count": self.word_count or 0,
            "tags": self.tags,
        }


class DocumentChunk(BaseModel):
    """A single text chunk ready for embedding and upsert to Pinecone."""

    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID used as the Pinecone vector ID",
    )
    text: str = Field(..., min_length=1, description="Raw chunk text")
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = Field(
        default=None, description="Dense embedding vector (populated after embed step)"
    )

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# --------------------------------------------------------------------------- #
# Search / query models
# --------------------------------------------------------------------------- #
class MetadataFilter(BaseModel):
    """
    Optional metadata filter applied to Pinecone queries.

    All fields are optional; only non-None fields are sent to Pinecone.
    Multiple fields are combined with logical AND.
    """
    category: Optional[DocumentCategory] = None
    source: Optional[str] = None
    date_from: Optional[str] = Field(
        default=None,
        description="ISO date string — only return chunks ingested on or after this date",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="ISO date string — only return chunks ingested on or before this date",
    )
    author: Optional[str] = None
    tags: Optional[List[str]] = Field(
        default=None,
        description="Return chunks that contain ANY of these tags",
    )

    def to_pinecone_filter(self) -> Optional[Dict[str, Any]]:
        """Translate to a Pinecone filter dict (MongoDB-style operators)."""
        conditions: Dict[str, Any] = {}

        if self.category:
            conditions["category"] = {"$eq": self.category.value}
        if self.source:
            conditions["source"] = {"$eq": self.source}
        if self.author:
            conditions["author"] = {"$eq": self.author}
        if self.tags:
            conditions["tags"] = {"$in": self.tags}

        # Date range on the ingested_at ISO string — lexicographic comparison works
        if self.date_from or self.date_to:
            date_cond: Dict[str, Any] = {}
            if self.date_from:
                date_cond["$gte"] = self.date_from
            if self.date_to:
                date_cond["$lte"] = self.date_to + "T23:59:59"
            conditions["ingested_at"] = date_cond

        return conditions if conditions else None


class SearchQuery(BaseModel):
    """Validated search query submitted by the user."""

    query_text: str = Field(..., min_length=1, max_length=2000, description="User's natural-language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filters: MetadataFilter = Field(
        default_factory=MetadataFilter,
        description="Optional metadata filters to narrow the search space",
    )

    @field_validator("query_text")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #
class SearchResult(BaseModel):
    """A single retrieved chunk with its relevance score."""

    chunk: DocumentChunk
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score (0–1)")
    rank: int = Field(..., ge=1, description="1-based rank in the result list")


class RAGResponse(BaseModel):
    """Complete response returned to the UI after a RAG query."""

    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="LLM-generated answer")
    sources: List[SearchResult] = Field(
        default_factory=list,
        description="Ranked list of chunks used as context",
    )
    tokens_used: int = Field(default=0, ge=0, description="Total OpenAI tokens consumed")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    retrieval_latency_ms: float = Field(default=0.0, description="Time spent on Pinecone retrieval (ms)")
    generation_latency_ms: float = Field(default=0.0, description="Time spent on LLM generation (ms)")
    total_latency_ms: float = Field(default=0.0, description="End-to-end wall-clock time (ms)")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC timestamp of this response",
    )

    @model_validator(mode="after")
    def compute_total_tokens(self) -> "RAGResponse":
        if self.tokens_used == 0:
            self.tokens_used = self.prompt_tokens + self.completion_tokens
        if self.total_latency_ms == 0.0:
            self.total_latency_ms = self.retrieval_latency_ms + self.generation_latency_ms
        return self


class QueryHistoryItem(BaseModel):
    """Persisted entry in the sidebar query history."""

    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer_preview: str = Field(
        ..., description="First 200 characters of the answer for quick display"
    )
    sources_count: int = Field(default=0, description="Number of source chunks retrieved")
    tokens_used: int = Field(default=0)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def from_rag_response(cls, response: RAGResponse) -> "QueryHistoryItem":
        return cls(
            query=response.query,
            answer_preview=response.answer[:200],
            sources_count=len(response.sources),
            tokens_used=response.tokens_used,
            timestamp=response.timestamp,
        )


# --------------------------------------------------------------------------- #
# Ingestion report
# --------------------------------------------------------------------------- #
class IngestionReport(BaseModel):
    """Summary returned after ingesting a document."""

    source: str
    category: DocumentCategory
    file_type: FileType
    total_chunks: int
    vectors_upserted: int
    ingestion_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    errors: List[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.vectors_upserted == self.total_chunks
