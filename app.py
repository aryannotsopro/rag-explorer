"""
app.py - Main Streamlit application for the RAG system.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from io import StringIO
from typing import List, Optional

import pandas as pd
import streamlit as st

# ── Page config must be the very first Streamlit call ─────────────────────── #
st.set_page_config(
    page_title="RAG Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/rag-explorer",
        "Report a bug": "https://github.com/your-org/rag-explorer/issues",
    },
)

from config import get_settings
from data_ingestion import IngestionPipeline
from models import (
    DocumentCategory,
    MetadataFilter,
    QueryHistoryItem,
    RAGResponse,
    SearchQuery,
)
from rag_engine import RAGEngine

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Custom CSS — premium dark glassmorphism theme
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Background ── */
        .stApp {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
            color: #e6edf3;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: rgba(22, 27, 34, 0.95);
            border-right: 1px solid rgba(48, 54, 61, 0.8);
            backdrop-filter: blur(20px);
        }

        /* ── Cards ── */
        .rag-card {
            background: rgba(22, 27, 34, 0.8);
            border: 1px solid rgba(48, 54, 61, 0.6);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(12px);
            transition: border-color 0.2s ease, transform 0.15s ease;
        }
        .rag-card:hover {
            border-color: rgba(88, 166, 255, 0.5);
            transform: translateY(-1px);
        }

        /* ── Score badge ── */
        .score-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        .score-high  { background: rgba(46,160,67,0.25); color: #3fb950; border: 1px solid rgba(46,160,67,0.4); }
        .score-mid   { background: rgba(210,153,34,0.25); color: #d29922; border: 1px solid rgba(210,153,34,0.4); }
        .score-low   { background: rgba(248,81,73,0.25);  color: #f85149; border: 1px solid rgba(248,81,73,0.4); }

        /* ── Category chips ── */
        .category-chip {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 500;
            background: rgba(88,166,255,0.15);
            color: #58a6ff;
            border: 1px solid rgba(88,166,255,0.35);
            margin-right: 4px;
        }

        /* ── Metric boxes ── */
        .metric-row {
            display: flex;
            gap: 0.75rem;
            margin-top: 0.5rem;
        }
        .metric-box {
            flex: 1;
            background: rgba(30,37,47,0.7);
            border: 1px solid rgba(48,54,61,0.6);
            border-radius: 8px;
            padding: 0.65rem 1rem;
            text-align: center;
        }
        .metric-label { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
        .metric-value { font-size: 1.1rem; font-weight: 600; color: #e6edf3; font-family: 'JetBrains Mono', monospace; }

        /* ── Answer block ── */
        .answer-block {
            background: rgba(14, 22, 35, 0.7);
            border-left: 3px solid #58a6ff;
            border-radius: 0 10px 10px 0;
            padding: 1.1rem 1.4rem;
            margin: 0.75rem 0;
            line-height: 1.7;
        }

        /* ── History item ── */
        .history-item {
            background: rgba(22,27,34,0.6);
            border: 1px solid rgba(48,54,61,0.5);
            border-radius: 8px;
            padding: 0.5rem 0.8rem;
            margin-bottom: 0.4rem;
            font-size: 0.8rem;
            cursor: pointer;
            transition: background 0.15s ease;
        }
        .history-item:hover { background: rgba(48,54,61,0.7); }

        /* ── Headings ── */
        h1, h2, h3 { color: #e6edf3 !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
        .stButton > button {
            background: linear-gradient(135deg, #238636, #2ea043);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            transition: opacity 0.2s;
        }
        .stButton > button:hover { opacity: 0.85; }

        div[data-testid="stStatusWidget"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    defaults = {
        "query_history": [],          # list[QueryHistoryItem]
        "last_response": None,        # RAGResponse | None
        "engine": None,               # RAGEngine | None  (cached)
        "pipeline": None,             # IngestionPipeline | None (cached)
        "replay_query": "",           # text to pre-fill query box
        "api_keys_ok": False,         # True once keys validated
        "index_stats": None,          # dict from Pinecone
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# --------------------------------------------------------------------------- #
# Lazy-init singletons (cached in session state to survive re-runs)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_engine() -> RAGEngine:
    engine = RAGEngine()
    engine.ensure_index()
    return engine


@st.cache_resource(show_spinner=False)
def get_pipeline(_engine: RAGEngine) -> IngestionPipeline:
    return IngestionPipeline(rag_engine=_engine)


# --------------------------------------------------------------------------- #
# Sidebar rendering
# --------------------------------------------------------------------------- #
def render_sidebar() -> MetadataFilter:
    """Render the full sidebar and return the active MetadataFilter."""
    with st.sidebar:
        # Logo / title
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 1.5rem;">
                <div style="font-size:2.5rem;">🔍</div>
                <h2 style="margin:0; font-size:1.3rem; font-weight:700; color:#58a6ff;">RAG Explorer</h2>
                <p style="margin:0; font-size:0.75rem; color:#8b949e;">Retrieval Augmented Generation</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── API / Index status ────────────────────────────────────────── #
        st.markdown("**⚙️ System Status**")
        try:
            engine = get_engine()
            stats = engine.index_stats()
            total_vecs = stats.get("total_vector_count", 0)
            st.success(f"✅ Pinecone connected — **{total_vecs:,}** vectors")
            st.session_state.api_keys_ok = True
            st.session_state.index_stats = stats
        except Exception as exc:
            st.error(f"❌ Pinecone error: {exc}")
            st.session_state.api_keys_ok = False

        st.divider()

        # ── Metadata filters ─────────────────────────────────────────── #
        st.markdown("**🏷️ Search Filters**")

        category_options = ["All"] + [c.value.title() for c in DocumentCategory]
        selected_cat_label = st.selectbox("Category", category_options, key="filter_cat")
        selected_cat = (
            DocumentCategory(selected_cat_label.lower())
            if selected_cat_label != "All"
            else None
        )

        source_filter = st.text_input("Source filename", placeholder="e.g. report.pdf", key="filter_source") or None

        author_filter = st.text_input("Author", placeholder="e.g. Jane Smith", key="filter_author") or None

        use_date = st.checkbox("Filter by ingestion date", key="use_date_filter")
        date_from = date_to = None
        if use_date:
            col1, col2 = st.columns(2)
            with col1:
                date_from = st.date_input("From", value=datetime.utcnow().date() - timedelta(days=30), key="df")
            with col2:
                date_to = st.date_input("To", value=datetime.utcnow().date(), key="dt")

        top_k = st.slider("Max results", min_value=1, max_value=20, value=5, key="top_k")

        st.divider()

        # ── Query history ─────────────────────────────────────────────── #
        st.markdown("**🕑 Query History**")
        history: List[QueryHistoryItem] = st.session_state.query_history

        if not history:
            st.caption("No queries yet — ask something!")
        else:
            for item in reversed(history[-20:]):
                ts = item.timestamp[:16].replace("T", " ")
                if st.button(
                    f"**{item.query[:50]}{'…' if len(item.query) > 50 else ''}**\n\n_{ts}_",
                    key=f"h_{item.item_id}",
                    use_container_width=True,
                ):
                    st.session_state.replay_query = item.query

        st.divider()

        # ── Export ───────────────────────────────────────────────────── #
        st.markdown("**📤 Export**")
        col_a, col_b = st.columns(2)
        with col_a:
            if history and st.button("JSON", use_container_width=True):
                payload = [h.model_dump() for h in history]
                st.download_button(
                    "⬇ Download",
                    data=json.dumps(payload, indent=2),
                    file_name=f"rag_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="dl_json",
                )
        with col_b:
            if history and st.button("CSV", use_container_width=True):
                df = pd.DataFrame([h.model_dump() for h in history])
                st.download_button(
                    "⬇ Download",
                    data=df.to_csv(index=False),
                    file_name=f"rag_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="dl_csv",
                )

    return MetadataFilter(
        category=selected_cat,
        source=source_filter,
        author=author_filter,
        date_from=date_from.isoformat() if date_from else None,
        date_to=date_to.isoformat() if date_to else None,
    )


# --------------------------------------------------------------------------- #
# Upload tab
# --------------------------------------------------------------------------- #
def render_upload_tab() -> None:
    st.markdown("### 📂 Upload & Ingest Documents")
    st.caption(
        "Upload PDF, TXT, or DOCX files. Each file will be chunked, embedded, "
        "and stored in Pinecone with metadata for semantic search."
    )

    if not st.session_state.api_keys_ok:
        st.warning("⚠️ Connect to Pinecone first (check your `.env` file).")
        return

    col_up, col_meta = st.columns([1.4, 1])

    with col_up:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="file_uploader",
        )

    with col_meta:
        st.markdown("**Document Metadata**")
        category_label = st.selectbox(
            "Category",
            [c.value.title() for c in DocumentCategory],
            key="upload_cat",
        )
        author = st.text_input("Author (optional)", key="upload_author") or None
        doc_date = st.date_input("Document date (optional)", value=None, key="upload_date")
        tags_raw = st.text_input("Tags (comma-separated, optional)", key="upload_tags")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    if uploaded_files and st.button("🚀 Ingest Documents", type="primary", key="ingest_btn"):
        category = DocumentCategory(category_label.lower())
        engine = get_engine()
        pipeline = get_pipeline(engine)

        for uploaded_file in uploaded_files:
            st.markdown(f"**Processing `{uploaded_file.name}` …**")
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def make_callback(pb, st_):
                def _cb(fraction: float, message: str):
                    pb.progress(min(fraction, 1.0))
                    st_.caption(message)
                return _cb

            cb = make_callback(progress_bar, status_text)

            with st.spinner(f"Ingesting {uploaded_file.name} …"):
                try:
                    report = pipeline.ingest_bytes(
                        file_bytes=uploaded_file.read(),
                        filename=uploaded_file.name,
                        category=category,
                        author=author,
                        document_date=doc_date.isoformat() if doc_date else None,
                        tags=tags,
                        progress_callback=cb,
                    )
                    progress_bar.progress(1.0)
                    if report.success:
                        st.success(
                            f"✅ **{report.source}** ingested successfully — "
                            f"**{report.vectors_upserted}** vectors in "
                            f"{report.ingestion_time_ms / 1000:.1f}s"
                        )
                    else:
                        st.error(
                            f"⚠️ Partial ingestion for **{report.source}**: "
                            + "; ".join(report.errors)
                        )
                except Exception as exc:
                    st.error(f"❌ Failed to ingest `{uploaded_file.name}`: {exc}")
                    logger.exception("Ingestion failed for %s", uploaded_file.name)

        # Invalidate cached index stats
        st.cache_resource.clear()
        st.rerun()


# --------------------------------------------------------------------------- #
# Search tab
# --------------------------------------------------------------------------- #
def render_search_tab(active_filter: MetadataFilter) -> None:
    st.markdown("### 🔎 Semantic Search")
    top_k = st.session_state.get("top_k", 5)

    # Pre-fill from history replay
    default_query = st.session_state.pop("replay_query", "")

    query_text = st.text_area(
        "Ask a question",
        value=default_query,
        placeholder="e.g. What are the key principles of quantum computing?",
        height=100,
        key="query_input",
    )

    col_search, col_clear = st.columns([1, 5])
    with col_search:
        search_clicked = st.button("🔍 Search", type="primary", key="search_btn", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear history", key="clear_history", use_container_width=False):
            st.session_state.query_history = []
            st.session_state.last_response = None
            st.rerun()

    if search_clicked:
        if not query_text.strip():
            st.warning("Please enter a question.")
            return
        if not st.session_state.api_keys_ok:
            st.error("Pinecone is not connected. Check your `.env` file.")
            return

        with st.spinner("🔍 Retrieving and generating answer …"):
            engine = get_engine()
            search_q = SearchQuery(
                query_text=query_text,
                top_k=top_k,
                filters=active_filter,
            )
            try:
                response: RAGResponse = engine.sync_query(search_q)
                st.session_state.last_response = response
                # Save to history
                hist_item = QueryHistoryItem.from_rag_response(response)
                st.session_state.query_history.append(hist_item)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                logger.exception("Query failed")
                return

    # Display last response
    response: Optional[RAGResponse] = st.session_state.last_response
    if response:
        _render_response(response)


def _render_response(response: RAGResponse) -> None:
    """Render a RAGResponse to the Streamlit UI."""
    # ── Performance metrics ── #
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Retrieval</div>
                <div class="metric-value">{response.retrieval_latency_ms:.0f} ms</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Generation</div>
                <div class="metric-value">{response.generation_latency_ms:.0f} ms</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total</div>
                <div class="metric-value">{response.total_latency_ms:.0f} ms</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Tokens Used</div>
                <div class="metric-value">{response.tokens_used:,}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Sources</div>
                <div class="metric-value">{len(response.sources)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Answer ── #
    st.markdown("#### 🤖 Answer")
    st.markdown(
        f'<div class="answer-block">{response.answer}</div>',
        unsafe_allow_html=True,
    )

    # ── Sources ── #
    if response.sources:
        st.markdown(f"#### 📚 Retrieved Sources ({len(response.sources)})")
        for result in response.sources:
            score = result.score
            if score >= 0.80:
                badge_cls = "score-high"
            elif score >= 0.60:
                badge_cls = "score-mid"
            else:
                badge_cls = "score-low"

            meta = result.chunk.metadata
            with st.expander(
                f"[Source {result.rank}] {meta.source}"
                + (f" — page {meta.page_number}" if meta.page_number else ""),
                expanded=(result.rank <= 2),
            ):
                st.markdown(
                    f"""
                    <div class="rag-card">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.6rem;">
                            <span class="score-badge {badge_cls}">Score: {score:.3f}</span>
                            <span class="category-chip">{meta.category.value}</span>
                            {"<span class='category-chip'>" + meta.author + "</span>" if meta.author else ""}
                        </div>
                        <div style="font-size:0.83rem; color:#8b949e; margin-bottom:0.5rem;">
                            📄 <b>{meta.source}</b>
                            {f" · Page {meta.page_number}" if meta.page_number else ""}
                            · Chunk {meta.chunk_index + 1}/{meta.total_chunks}
                            · {meta.word_count or "?"} words
                        </div>
                        <hr style="border-color:rgba(48,54,61,0.5); margin: 0.5rem 0;">
                        <div style="font-size:0.88rem; line-height:1.65; color:#c9d1d9;">
                            {result.chunk.text[:1200]}{"…" if len(result.chunk.text) > 1200 else ""}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------- #
# Manage tab
# --------------------------------------------------------------------------- #
def render_manage_tab() -> None:
    st.markdown("### 🗄️ Index Management")

    if not st.session_state.api_keys_ok:
        st.warning("Connect to Pinecone to manage the index.")
        return

    engine = get_engine()
    stats = engine.index_stats()
    total_vecs = stats.get("total_vector_count", 0)

    col1, col2 = st.columns(2)
    col1.metric("Total Vectors", f"{total_vecs:,}")
    ns_stats = stats.get("namespaces", {})
    col2.metric("Namespaces", len(ns_stats))

    if ns_stats:
        st.markdown("**Namespace breakdown:**")
        ns_df = pd.DataFrame(
            [{"Namespace": k or "(default)", "Count": v.get("vector_count", 0)} for k, v in ns_stats.items()]
        )
        st.dataframe(ns_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**🗑️ Delete a document**")
    st.caption("Deletes all vectors whose `source` metadata matches the given filename.")
    del_source = st.text_input("Source filename to delete", placeholder="report.pdf", key="del_src")
    if st.button("Delete", type="secondary", key="del_btn") and del_source:
        with st.spinner(f"Deleting vectors for '{del_source}' …"):
            deleted = engine.delete_document(del_source)
        if deleted:
            st.success(f"Deleted {deleted} vector(s) for '{del_source}'.")
        else:
            st.info(f"No vectors found for source '{del_source}'.")
        st.cache_resource.clear()
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()
    init_session_state()

    # Sidebar (returns active filter)
    active_filter = render_sidebar()

    # Header
    st.markdown(
        """
        <div style="padding: 1.5rem 0 0.5rem;">
            <h1 style="margin:0; font-size:1.8rem; font-weight:700;">
                🔍 RAG Explorer
                <span style="font-size:0.9rem; font-weight:400; color:#8b949e; margin-left:12px;">
                    Semantic search with metadata filtering
                </span>
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs
    tab_search, tab_upload, tab_manage = st.tabs(
        ["🔎 Search", "📂 Upload", "🗄️ Manage"]
    )

    with tab_search:
        render_search_tab(active_filter)

    with tab_upload:
        render_upload_tab()

    with tab_manage:
        render_manage_tab()


if __name__ == "__main__":
    main()
