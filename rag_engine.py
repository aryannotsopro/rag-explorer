"""
rag_engine.py - Core RAG orchestration layer.

Responsibilities:
  1. Manage the Pinecone index (create if absent, query, upsert, delete).
  2. Generate OpenAI embeddings in batches with retry logic and rate limiting.
  3. Build a prompt from retrieved context and call GPT-4o-mini.
  4. Return fully-typed RAGResponse objects.

All heavy I/O uses async where possible; synchronous callers in Streamlit can
use asyncio.run() or the _sync_* wrappers provided at the bottom.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import openai
from openai import AsyncOpenAI, OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import get_settings
from models import (
    DocumentChunk,
    MetadataFilter,
    RAGResponse,
    SearchQuery,
    SearchResult,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _batched(iterable: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def _ms(start: float) -> float:
    """Elapsed milliseconds since *start* (from time.perf_counter())."""
    return (time.perf_counter() - start) * 1000


# --------------------------------------------------------------------------- #
# RAGEngine
# --------------------------------------------------------------------------- #

class RAGEngine:
    """
    Central orchestrator for embedding, retrieval, and generation.

    Usage (sync):
        engine = RAGEngine()
        engine.ensure_index()
        report = asyncio.run(engine.upsert_chunks(chunks))
        response = asyncio.run(engine.query(SearchQuery(query_text="...")))
    """

    # System prompt injected before every RAG context block
    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. Answer the user's question **only** using "
        "the context passages provided below. Each passage is labelled with [Source N]. "
        "Cite the relevant source labels inline (e.g. [Source 1]) when you use information "
        "from them. If the context does not contain enough information to answer, say so "
        "honestly — do not fabricate facts."
    )

    def __init__(self) -> None:
        cfg = get_settings()
        self.cfg = cfg

        # Pinecone client ------------------------------------------------------
        self._pc = Pinecone(api_key=cfg.pinecone_api_key)
        self._index = None  # lazy — created in ensure_index()

        # OpenAI clients -------------------------------------------------------
        self._sync_client = OpenAI(api_key=cfg.openai_api_key)
        self._async_client = AsyncOpenAI(api_key=cfg.openai_api_key)

        # Rate-limiting semaphore for concurrent embed / chat calls
        self._sem = asyncio.Semaphore(cfg.max_concurrent_requests)

        logger.info(
            "RAGEngine initialised | index=%s | embed=%s | llm=%s",
            cfg.pinecone_index_name,
            cfg.embedding_model,
            cfg.llm_model,
        )

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #

    def ensure_index(self) -> None:
        """
        Create the Pinecone serverless index if it does not already exist,
        then bind self._index to it.
        """
        cfg = self.cfg
        existing = [i.name for i in self._pc.list_indexes()]

        if cfg.pinecone_index_name not in existing:
            logger.info("Creating Pinecone index '%s' …", cfg.pinecone_index_name)
            self._pc.create_index(
                name=cfg.pinecone_index_name,
                dimension=cfg.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cfg.pinecone_cloud,
                    region=cfg.pinecone_environment,
                ),
            )
            # Wait until the index is ready
            while not self._pc.describe_index(cfg.pinecone_index_name).status["ready"]:
                time.sleep(1)
            logger.info("Index '%s' created and ready.", cfg.pinecone_index_name)
        else:
            logger.info("Index '%s' already exists.", cfg.pinecone_index_name)

        self._index = self._pc.Index(cfg.pinecone_index_name)

    @property
    def index(self):
        if self._index is None:
            self.ensure_index()
        return self._index

    def index_stats(self) -> Dict[str, Any]:
        """Return Pinecone index statistics (total vector count, etc.)."""
        return self.index.describe_index_stats()

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts synchronously (used internally)."""
        response = self._sync_client.embeddings.create(
            model=self.cfg.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed *texts* using the configured OpenAI model, batching automatically.

        Returns a list of embedding vectors in the same order as *texts*.
        """
        cfg = self.cfg
        all_embeddings: List[List[float]] = []

        for batch in _batched(texts, cfg.embedding_batch_size):
            async with self._sem:
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self._embed_batch_sync, batch
                )
            all_embeddings.extend(embeddings)

        return all_embeddings

    # ------------------------------------------------------------------ #
    # Upsert
    # ------------------------------------------------------------------ #

    async def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Embed and upsert *chunks* into Pinecone.

        Returns the number of vectors successfully upserted.
        """
        if not chunks:
            return 0

        logger.info("Embedding %d chunks …", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = await self.embed_texts(texts)

        # Attach embeddings to chunks
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Batch upsert to Pinecone
        vectors = [
            {
                "id": chunk.chunk_id,
                "values": chunk.embedding,
                "metadata": {
                    **chunk.metadata.to_pinecone_metadata(),
                    "text": chunk.text,  # store raw text for retrieval
                },
            }
            for chunk in chunks
        ]

        upserted = 0
        for batch in _batched(vectors, self.cfg.upsert_batch_size):
            self.index.upsert(vectors=batch)
            upserted += len(batch)
            logger.debug("Upserted %d / %d vectors", upserted, len(vectors))

        logger.info("Upsert complete: %d vectors in Pinecone.", upserted)
        return upserted

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: SearchQuery,
        *,
        embed_query_text: Optional[str] = None,
    ) -> tuple[List[SearchResult], float]:
        """
        Embed the query and search Pinecone with optional metadata filters.

        Returns (results, retrieval_latency_ms).
        """
        t0 = time.perf_counter()
        query_text = embed_query_text or query.query_text
        [query_vector] = await self.embed_texts([query_text])

        pinecone_filter = query.filters.to_pinecone_filter()
        logger.debug(
            "Querying Pinecone | top_k=%d | filter=%s", query.top_k, pinecone_filter
        )

        response = self.index.query(
            vector=query_vector,
            top_k=query.top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )

        latency = _ms(t0)
        results: List[SearchResult] = []

        for rank, match in enumerate(response.matches, start=1):
            meta = match.metadata or {}
            # Reconstruct DocumentChunk from stored metadata
            from models import DocumentMetadata, DocumentCategory, FileType

            doc_meta = DocumentMetadata(
                source=meta.get("source", "unknown"),
                category=DocumentCategory(meta.get("category", "general")),
                page_number=meta.get("page_number") or None,
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 1),
                ingested_at=meta.get("ingested_at", ""),
                document_date=meta.get("document_date") or None,
                author=meta.get("author") or None,
                file_type=FileType(meta["file_type"]) if meta.get("file_type") else None,
                word_count=meta.get("word_count") or None,
                tags=meta.get("tags", []),
            )
            chunk = DocumentChunk(
                chunk_id=match.id,
                text=meta.get("text", ""),
                metadata=doc_meta,
            )
            results.append(SearchResult(chunk=chunk, score=match.score, rank=rank))

        logger.info("Search returned %d results in %.1f ms", len(results), latency)
        return results, latency

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _build_prompt(self, query: str, results: List[SearchResult]) -> str:
        """Construct the user-turn message with numbered context passages."""
        context_blocks = []
        for r in results:
            header = (
                f"[Source {r.rank}] "
                f"(file: {r.chunk.metadata.source}, "
                f"category: {r.chunk.metadata.category.value}, "
                f"score: {r.score:.3f})"
            )
            context_blocks.append(f"{header}\n{r.chunk.text}")

        context_str = "\n\n---\n\n".join(context_blocks)
        return (
            f"Context passages:\n\n{context_str}\n\n"
            f"---\n\nQuestion: {query}\n\nAnswer:"
        )

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _chat_sync(self, messages: list) -> openai.types.chat.ChatCompletion:
        return self._sync_client.chat.completions.create(
            model=self.cfg.llm_model,
            messages=messages,
            temperature=self.cfg.llm_temperature,
            max_tokens=self.cfg.llm_max_tokens,
        )

    async def generate_response(
        self,
        query: str,
        results: List[SearchResult],
        *,
        retrieval_latency_ms: float = 0.0,
    ) -> RAGResponse:
        """
        Given a user query and retrieved context chunks, call the LLM
        and return a fully-typed RAGResponse.
        """
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(query, results)},
        ]

        async with self._sem:
            completion = await asyncio.get_event_loop().run_in_executor(
                None, self._chat_sync, messages
            )

        gen_latency = _ms(t0)
        usage = completion.usage
        answer = completion.choices[0].message.content or ""

        logger.info(
            "Generation complete | tokens=%d | latency=%.1f ms",
            usage.total_tokens,
            gen_latency,
        )

        return RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            tokens_used=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=gen_latency,
        )

    # ------------------------------------------------------------------ #
    # Full RAG pipeline
    # ------------------------------------------------------------------ #

    async def query(self, search_query: SearchQuery) -> RAGResponse:
        """
        End-to-end RAG:  embed → retrieve → generate → return RAGResponse.
        This is the primary entry point called by the Streamlit UI.
        """
        results, retrieval_latency = await self.search(search_query)

        if not results:
            return RAGResponse(
                query=search_query.query_text,
                answer=(
                    "I couldn't find any relevant documents matching your query"
                    + (" with the applied filters." if search_query.filters.to_pinecone_filter() else ".")
                ),
                sources=[],
                retrieval_latency_ms=retrieval_latency,
            )

        return await self.generate_response(
            search_query.query_text,
            results,
            retrieval_latency_ms=retrieval_latency,
        )

    # ------------------------------------------------------------------ #
    # Deletion
    # ------------------------------------------------------------------ #

    def delete_document(self, source: str) -> int:
        """
        Delete all vectors whose source metadata matches *source*.

        Returns the number of IDs deleted (Pinecone does not expose this
        directly, so we query-first then delete).
        """
        # Pinecone fetch-by-filter is available with namespace; use query trick
        dummy_vector = [0.0] * self.cfg.embedding_dimension
        results = self.index.query(
            vector=dummy_vector,
            top_k=10_000,
            filter={"source": {"$eq": source}},
            include_metadata=False,
        )
        ids = [m.id for m in results.matches]
        if ids:
            self.index.delete(ids=ids)
            logger.info("Deleted %d vectors for source '%s'.", len(ids), source)
        return len(ids)

    # ------------------------------------------------------------------ #
    # Synchronous convenience wrappers (for Streamlit / non-async callers)
    # ------------------------------------------------------------------ #

    def sync_upsert(self, chunks: List[DocumentChunk]) -> int:
        return asyncio.run(self.upsert_chunks(chunks))

    def sync_query(self, search_query: SearchQuery) -> RAGResponse:
        return asyncio.run(self.query(search_query))
