"""
tests/test_data_ingestion.py - Unit tests for DocumentProcessor.

Uses in-memory string fixtures to avoid real file I/O.
No API calls needed — embedding is not invoked here.
Run with: pytest tests/test_data_ingestion.py -v
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from models import DocumentCategory, FileType


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
SAMPLE_TECH_TEXT = """\
Artificial intelligence is transforming industries worldwide. Machine learning algorithms
enable computers to learn from data without being explicitly programmed. Deep learning,
a subset of machine learning, uses neural networks with many layers to process complex patterns.

Natural language processing (NLP) allows computers to understand human language. Large language
models (LLMs) like GPT have demonstrated remarkable capabilities in text generation and reasoning.

Cloud computing provides scalable infrastructure for AI workloads. APIs allow developers to
integrate AI capabilities into their applications with just a few lines of code.

The future of software development involves AI-assisted coding tools that can generate, review,
and refactor code automatically. These tools are already boosting developer productivity significantly.
"""

SAMPLE_HISTORY_TEXT = """\
The ancient civilizations of Mesopotamia arose between the Tigris and Euphrates rivers.
Sumerians developed one of the earliest known writing systems, called cuneiform, around 3400 BC.

The Egyptian civilization thrived along the Nile for over 3000 years. The pyramids of Giza
are among the most iconic monuments in human history, built as tombs for pharaohs.

The Roman Empire expanded across Europe, North Africa, and the Middle East, leaving an enduring
legacy in law, governance, language, and architecture.

Medieval Europe saw the rise of feudalism, the Catholic Church's influence, and the Crusades.
The Renaissance brought a revival of arts and sciences, laying the foundation for the modern world.
"""


# --------------------------------------------------------------------------- #
# DocumentProcessor tests
# --------------------------------------------------------------------------- #
class TestDocumentProcessor:

    @patch("data_ingestion.get_settings")
    def _make_processor(self, mock_settings):
        cfg = MagicMock()
        cfg.chunk_size = 500
        cfg.chunk_overlap = 50
        mock_settings.return_value = cfg
        from data_ingestion import DocumentProcessor
        return DocumentProcessor(chunk_size=500, chunk_overlap=50)

    def test_processor_initialises(self):
        with patch("data_ingestion.get_settings") as ms:
            cfg = MagicMock()
            cfg.chunk_size = 500
            cfg.chunk_overlap = 50
            ms.return_value = cfg
            from data_ingestion import DocumentProcessor
            proc = DocumentProcessor()
            assert proc.chunk_size == 500
            assert proc.chunk_overlap == 50

    def test_detect_file_type_pdf(self, tmp_path):
        from data_ingestion import DocumentProcessor
        p = tmp_path / "doc.pdf"
        p.touch()
        ft = DocumentProcessor._detect_file_type(p)
        assert ft == FileType.PDF

    def test_detect_file_type_txt(self, tmp_path):
        from data_ingestion import DocumentProcessor
        p = tmp_path / "notes.txt"
        p.touch()
        ft = DocumentProcessor._detect_file_type(p)
        assert ft == FileType.TXT

    def test_detect_file_type_docx(self, tmp_path):
        from data_ingestion import DocumentProcessor
        p = tmp_path / "report.docx"
        p.touch()
        ft = DocumentProcessor._detect_file_type(p)
        assert ft == FileType.DOCX

    def test_detect_unsupported_raises(self, tmp_path):
        from data_ingestion import DocumentProcessor
        p = tmp_path / "doc.xlsx"
        p.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentProcessor._detect_file_type(p)

    def test_process_txt_file(self, tmp_path):
        """End-to-end: process a real TXT file and check chunk properties."""
        txt_file = tmp_path / "tech.txt"
        txt_file.write_text(SAMPLE_TECH_TEXT, encoding="utf-8")

        with patch("data_ingestion.get_settings") as ms:
            cfg = MagicMock()
            cfg.chunk_size = 300
            cfg.chunk_overlap = 50
            ms.return_value = cfg
            from data_ingestion import DocumentProcessor
            proc = DocumentProcessor(chunk_size=300, chunk_overlap=50)

        chunks = proc.process_file(
            txt_file,
            category=DocumentCategory.TECHNOLOGY,
            author="Test Author",
            tags=["ai", "ml"],
        )

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text.strip()  # no empty chunks
            assert chunk.metadata.source == "tech.txt"
            assert chunk.metadata.category == DocumentCategory.TECHNOLOGY
            assert chunk.metadata.author == "Test Author"
            assert "ai" in chunk.metadata.tags
            assert chunk.metadata.file_type == FileType.TXT
            assert chunk.metadata.word_count > 0
            assert chunk.chunk_id  # UUID assigned

    def test_process_txt_respects_chunk_size(self, tmp_path):
        """Chunks should not dramatically exceed the configured chunk size."""
        # Write a long text document
        long_text = "This is a sentence about technology and computing. " * 200
        txt_file = tmp_path / "long.txt"
        txt_file.write_text(long_text, encoding="utf-8")

        with patch("data_ingestion.get_settings") as ms:
            cfg = MagicMock()
            cfg.chunk_size = 400
            cfg.chunk_overlap = 50
            ms.return_value = cfg
            from data_ingestion import DocumentProcessor
            proc = DocumentProcessor(chunk_size=400, chunk_overlap=50)

        chunks = proc.process_file(txt_file, category=DocumentCategory.TECHNOLOGY)
        # Each chunk should be <= chunk_size * 1.1 (small tolerance for splitter)
        for chunk in chunks:
            assert len(chunk.text) <= 400 * 1.5, f"Chunk too large: {len(chunk.text)} chars"

    def test_process_nonexistent_file_raises(self):
        with patch("data_ingestion.get_settings") as ms:
            cfg = MagicMock()
            cfg.chunk_size = 500
            cfg.chunk_overlap = 50
            ms.return_value = cfg
            from data_ingestion import DocumentProcessor
            proc = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            proc.process_file("/nonexistent/path/doc.txt")

    def test_progress_callback_called(self, tmp_path):
        txt_file = tmp_path / "prog.txt"
        txt_file.write_text(SAMPLE_TECH_TEXT, encoding="utf-8")

        calls = []

        def cb(fraction, message):
            calls.append((fraction, message))

        with patch("data_ingestion.get_settings") as ms:
            cfg = MagicMock()
            cfg.chunk_size = 300
            cfg.chunk_overlap = 50
            ms.return_value = cfg
            from data_ingestion import DocumentProcessor
            proc = DocumentProcessor(chunk_size=300, chunk_overlap=50)

        proc.process_file(txt_file, progress_callback=cb)
        assert len(calls) >= 2
        fractions = [c[0] for c in calls]
        assert fractions[-1] == 1.0  # always ends at 100%

    def test_auto_category_inference_technology(self, tmp_path):
        txt_file = tmp_path / "ai.txt"
        txt_file.write_text(SAMPLE_TECH_TEXT, encoding="utf-8")

        with patch("data_ingestion.get_settings") as ms:
            cfg = MagicMock()
            cfg.chunk_size = 600
            cfg.chunk_overlap = 50
            ms.return_value = cfg
            from data_ingestion import DocumentProcessor
            proc = DocumentProcessor(chunk_size=600, chunk_overlap=50)

        # Do NOT pass category — let it auto-infer
        chunks = proc.process_file(txt_file)
        assert chunks[0].metadata.category == DocumentCategory.TECHNOLOGY

    def test_auto_category_inference_history(self, tmp_path):
        txt_file = tmp_path / "hist.txt"
        txt_file.write_text(SAMPLE_HISTORY_TEXT, encoding="utf-8")

        with patch("data_ingestion.get_settings") as ms:
            cfg = MagicMock()
            cfg.chunk_size = 600
            cfg.chunk_overlap = 50
            ms.return_value = cfg
            from data_ingestion import DocumentProcessor
            proc = DocumentProcessor(chunk_size=600, chunk_overlap=50)

        chunks = proc.process_file(txt_file)
        assert chunks[0].metadata.category == DocumentCategory.HISTORY

    def test_chunk_indices_sequential(self, tmp_path):
        long_text = "Machine learning and neural networks are transforming computing. " * 100
        txt_file = tmp_path / "idx.txt"
        txt_file.write_text(long_text, encoding="utf-8")

        with patch("data_ingestion.get_settings") as ms:
            cfg = MagicMock()
            cfg.chunk_size = 200
            cfg.chunk_overlap = 20
            ms.return_value = cfg
            from data_ingestion import DocumentProcessor
            proc = DocumentProcessor(chunk_size=200, chunk_overlap=20)

        chunks = proc.process_file(txt_file, category=DocumentCategory.TECHNOLOGY)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# --------------------------------------------------------------------------- #
# Infer category unit tests
# --------------------------------------------------------------------------- #
class TestInferCategory:
    def test_technology_text(self):
        from data_ingestion import infer_category
        cat = infer_category("This document covers machine learning algorithms and neural networks for AI.")
        assert cat == DocumentCategory.TECHNOLOGY

    def test_history_text(self):
        from data_ingestion import infer_category
        cat = infer_category("The ancient Roman Empire conquered vast territories in medieval times during war.")
        assert cat == DocumentCategory.HISTORY

    def test_science_text(self):
        from data_ingestion import infer_category
        cat = infer_category("This biology experiment tests the hypothesis about molecular chemistry and quantum physics.")
        assert cat == DocumentCategory.SCIENCE

    def test_ambiguous_falls_back_to_general(self):
        from data_ingestion import infer_category
        cat = infer_category("Hello world. This is a short test.")
        assert cat == DocumentCategory.GENERAL
