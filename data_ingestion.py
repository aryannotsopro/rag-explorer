"""
data_ingestion.py - Document loading, chunking, and embedding pipeline.

Supports PDF, TXT, and DOCX files. Uses LangChain loaders and
RecursiveCharacterTextSplitter. Returns validated DocumentChunk objects
that are ready to be passed directly to RAGEngine.upsert_chunks().
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from config import get_settings
from models import (
    DocumentCategory,
    DocumentChunk,
    DocumentMetadata,
    FileType,
    IngestionReport,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Categorisation heuristic (keyword-based, easily extensible)
# --------------------------------------------------------------------------- #
_CATEGORY_KEYWORDS: dict[DocumentCategory, list[str]] = {
    DocumentCategory.TECHNOLOGY: [
        "software", "hardware", "algorithm", "machine learning", "ai", "neural",
        "cloud", "api", "database", "programming", "code", "data science",
        "computing", "internet", "cybersecurity",
    ],
    DocumentCategory.SCIENCE: [
        "biology", "chemistry", "physics", "experiment", "hypothesis", "research",
        "quantum", "molecular", "genome", "climate", "environment", "astronomy",
        "particle", "theory", "scientific",
    ],
    DocumentCategory.HISTORY: [
        "ancient", "medieval", "century", "war", "empire", "civilization",
        "revolution", "dynasty", "archaeological", "historical", "battle",
        "treaty", "colonization", "independence", "monarchy",
    ],
    DocumentCategory.BUSINESS: [
        "market", "revenue", "profit", "startup", "finance", "investment",
        "enterprise", "strategy", "management", "acquisition", "ipo", "stock",
        "economy", "trade", "commerce",
    ],
    DocumentCategory.HEALTH: [
        "medicine", "disease", "treatment", "clinical", "patient", "therapy",
        "drug", "hospital", "wellness", "nutrition", "mental health", "surgery",
        "vaccine", "diagnosis", "healthcare",
    ],
}


def infer_category(text_sample: str) -> DocumentCategory:
    """
    Guess the document category based on keyword frequency in the first 2000 chars.
    Falls back to GENERAL if no category exceeds a threshold.
    """
    sample = text_sample[:2000].lower()
    scores: dict[DocumentCategory, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(sample.count(kw) for kw in keywords)

    best_cat, best_score = max(scores.items(), key=lambda x: x[1])
    return best_cat if best_score >= 2 else DocumentCategory.GENERAL


# --------------------------------------------------------------------------- #
# DocumentProcessor
# --------------------------------------------------------------------------- #

class DocumentProcessor:
    """
    Load a document file → split into chunks → return DocumentChunk list.

    Example:
        processor = DocumentProcessor()
        chunks = processor.process_file(
            file_path="report.pdf",
            category=DocumentCategory.SCIENCE,
            progress_callback=lambda pct, msg: print(f"{pct:.0%} {msg}"),
        )
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        cfg = get_settings()
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.debug(
            "DocumentProcessor ready | chunk_size=%d | overlap=%d",
            self.chunk_size,
            self.chunk_overlap,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_file(
        self,
        file_path: Union[str, Path],
        *,
        category: Optional[DocumentCategory] = None,
        author: Optional[str] = None,
        document_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[DocumentChunk]:
        """
        Process a file on disk and return a list of DocumentChunk objects.

        Args:
            file_path: Path to the file (PDF, TXT, or DOCX).
            category: Override auto-detected category.
            author: Optional author name.
            document_date: Optional ISO date string (YYYY-MM-DD).
            tags: Optional list of free-form tags.
            progress_callback: Callable(fraction: float, message: str) for UI updates.

        Returns:
            List of DocumentChunk ready for RAGEngine.upsert_chunks().
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._detect_file_type(path)
        _progress(progress_callback, 0.05, f"Loading {path.name} …")

        raw_docs = self._load(path, file_type)
        _progress(progress_callback, 0.25, f"Loaded {len(raw_docs)} page(s). Splitting …")

        chunks = self._split_and_package(
            raw_docs=raw_docs,
            source_name=path.name,
            file_type=file_type,
            category=category,
            author=author,
            document_date=document_date,
            tags=tags or [],
            progress_callback=progress_callback,
        )
        _progress(progress_callback, 1.0, f"Done — {len(chunks)} chunk(s) ready.")
        return chunks

    def process_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        *,
        category: Optional[DocumentCategory] = None,
        author: Optional[str] = None,
        document_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[DocumentChunk]:
        """
        Process a file provided as raw bytes (e.g. from Streamlit st.file_uploader).
        Writes bytes to a temp file, processes, then cleans up.
        """
        import tempfile, os

        suffix = Path(filename).suffix or ".tmp"
        _progress(progress_callback, 0.02, f"Saving uploaded file '{filename}' …")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            return self.process_file(
                tmp_path,
                category=category,
                author=author,
                document_date=document_date,
                tags=tags,
                progress_callback=progress_callback,
            )
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_file_type(path: Path) -> FileType:
        ext = path.suffix.lower().lstrip(".")
        try:
            return FileType(ext)
        except ValueError:
            raise ValueError(
                f"Unsupported file type '.{ext}'. Supported: pdf, txt, docx."
            )

    def _load(self, path: Path, file_type: FileType) -> list:
        """Load document pages using the appropriate LangChain loader."""
        str_path = str(path)
        try:
            if file_type == FileType.PDF:
                loader = PyPDFLoader(str_path)
            elif file_type == FileType.TXT:
                loader = TextLoader(str_path, encoding="utf-8", autodetect_encoding=True)
            elif file_type == FileType.DOCX:
                loader = Docx2txtLoader(str_path)
            else:
                raise ValueError(f"No loader for {file_type}")
            return loader.load()
        except Exception as exc:
            logger.error("Failed to load '%s': %s", path, exc)
            raise

    def _split_and_package(
        self,
        raw_docs: list,
        source_name: str,
        file_type: FileType,
        category: Optional[DocumentCategory],
        author: Optional[str],
        document_date: Optional[str],
        tags: List[str],
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> List[DocumentChunk]:
        """Split raw LangChain Documents into DocumentChunk objects."""

        # Determine category from first page content if not provided
        first_text = raw_docs[0].page_content if raw_docs else ""
        resolved_category = category or infer_category(first_text)

        langchain_chunks = self._splitter.split_documents(raw_docs)
        total = len(langchain_chunks)

        logger.info(
            "Split '%s' into %d chunks (size=%d, overlap=%d)",
            source_name,
            total,
            self.chunk_size,
            self.chunk_overlap,
        )

        doc_chunks: List[DocumentChunk] = []
        for i, lc_chunk in enumerate(langchain_chunks):
            if i % 10 == 0:
                _progress(
                    progress_callback,
                    0.25 + 0.65 * (i / max(total, 1)),
                    f"Packaging chunk {i + 1}/{total} …",
                )

            text = lc_chunk.page_content.strip()
            if not text:
                continue

            page_num = lc_chunk.metadata.get("page", None)
            if page_num is not None:
                page_num = int(page_num) + 1  # LangChain uses 0-index for pages

            metadata = DocumentMetadata(
                source=source_name,
                category=resolved_category,
                page_number=page_num,
                chunk_index=i,
                total_chunks=total,
                ingested_at=datetime.utcnow().isoformat(),
                document_date=document_date,
                author=author,
                file_type=file_type,
                word_count=len(text.split()),
                tags=tags,
            )
            doc_chunks.append(DocumentChunk(text=text, metadata=metadata))

        return doc_chunks


# --------------------------------------------------------------------------- #
# IngestionPipeline — convenience wrapper that ties Processor + RAGEngine
# --------------------------------------------------------------------------- #

class IngestionPipeline:
    """
    High-level pipeline: process a file AND upsert to Pinecone in one call.

    Returns an IngestionReport.
    """

    def __init__(self, rag_engine=None) -> None:
        # Import here to avoid circular dependency at module load time
        from rag_engine import RAGEngine
        self.processor = DocumentProcessor()
        self.engine = rag_engine or RAGEngine()

    def ingest_file(
        self,
        file_path: Union[str, Path],
        *,
        category: Optional[DocumentCategory] = None,
        author: Optional[str] = None,
        document_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> IngestionReport:
        t0 = time.perf_counter()
        path = Path(file_path)
        file_type = DocumentProcessor._detect_file_type(path)
        errors: List[str] = []
        chunks: List[DocumentChunk] = []
        upserted = 0

        try:
            chunks = self.processor.process_file(
                path,
                category=category,
                author=author,
                document_date=document_date,
                tags=tags,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            errors.append(f"Processing error: {exc}")
            logger.exception("Processing failed for '%s'", path)

        if chunks:
            try:
                upserted = self.engine.sync_upsert(chunks)
            except Exception as exc:
                errors.append(f"Upsert error: {exc}")
                logger.exception("Upsert failed for '%s'", path)

        resolved_category = (
            chunks[0].metadata.category if chunks else DocumentCategory.GENERAL
        )

        return IngestionReport(
            source=path.name,
            category=resolved_category,
            file_type=file_type,
            total_chunks=len(chunks),
            vectors_upserted=upserted,
            ingestion_time_ms=(time.perf_counter() - t0) * 1000,
            errors=errors,
        )

    def ingest_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        *,
        category: Optional[DocumentCategory] = None,
        author: Optional[str] = None,
        document_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> IngestionReport:
        import tempfile, os
        from pathlib import Path as _Path

        suffix = _Path(filename).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return self.ingest_file(
                tmp_path,
                category=category,
                author=author,
                document_date=document_date,
                tags=tags,
                progress_callback=progress_callback,
            )
        finally:
            os.unlink(tmp_path)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _progress(
    callback: Optional[Callable[[float, str], None]],
    fraction: float,
    message: str,
) -> None:
    if callback:
        try:
            callback(fraction, message)
        except Exception:
            pass  # never let a UI callback crash the pipeline
