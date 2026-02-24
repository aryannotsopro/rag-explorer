# 📄 RAG Explorer: Technical Deep-Dive & Architecture Report

This report provides a comprehensive breakdown of the **RAG Explorer** project, explaining the design decisions, architectural patterns, and production-grade features implemented.

---

## 1. Project Objective
The goal was to build a **Production-Quality Retrieval Augmented Generation (RAG)** system that goes beyond a simple demo by incorporating:
- **Strict Data Validation**: Using Pydantic v2 for all data structures.
- **Advanced Retrieval**: Semantic search combined with multi-dimensional metadata filtering.
- **Scalability**: Async I/O for API calls and connection pooling.
- **Resilience**: Automatic retries and rate-limiting.
- **Professional UI**: A premium, responsive dashboard for end-users.

---

## 2. Technical Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.12 | Core runtime |
| **Interface** | Streamlit | Frontend UI & application server |
| **Data Validation** | Pydantic v2 | Schema enforcement & Settings management |
| **Vector DB** | Pinecone | Serverless vector index with metadata storage |
| **LLM / Embed** | OpenAI | GPT-4o-mini & text-embedding-3-small |
| **Orchestration** | LangChain | Document loading and text splitting |
| **Resilience** | Tenacity | Exponential backoff for API failures |
| **Container** | Docker | Environment parity and deployment |

---

## 3. Component Architecture

### 🛡️ `config.py` (The Brain)
- **Settings Management**: Uses `pydantic-settings` to load configuration from `.env` (local) or `st.secrets` (Streamlit Cloud).
- **Singleton Pattern**: The `get_settings()` function uses `@lru_cache` to ensure settings are only parsed once.
- **Secrets Injection**: Features a custom `_inject_streamlit_secrets` helper that maps Streamlit's secret store into environment variables for seamless cloud deployment.

### 🏗️ `models.py` (The Schema)
- **Type Safety**: Defines the "Contract" for the whole app. 
- **Pinecone Integration**: Includes a `to_pinecone_filter()` method that converts complex Python objects into the specific MongoDB-style JSON syntax Pinecone requires for filtering.
- **Calculated Fields**: The `RAGResponse` model automatically calculates total tokens and formatting based on prompt/completion counts.

### ⚙️ `rag_engine.py` (The Engine)
- **Batched Embeddings**: Efficiently processes hundreds of text chunks in parallel batches.
- **Concurrency Control**: Uses an `asyncio.Semaphore` (default: 5) to prevent "429 Too Many Requests" errors from OpenAI when processing large documents.
- **The RAG Loop**: 
    1.  Embed Query Text.
    2.  Query Pinecone with Vector + Filters.
    3.  Format retrieved chunks into a context block.
    4.  Prompt LLM with the context and system rules.

### 📂 `data_ingestion.py` (The Pipeline)
- **Smart Loading**: Switches loaders (PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader) based on file suffix.
- **Chunking Strategy**: Uses `RecursiveCharacterTextSplitter` which respects paragraph and sentence boundaries, preventing "cut-off" sentences.
- **Category Inference**: A lightweight keyword-matching algorithm that automatically tags documents (Technology, Science, etc.) if the user doesn't provide a category.

---

## 4. Production-Grade Features Explained

### ✅ Advanced Metadata Filtering
Unlike basic RAG, this system allows users to filter by **Category, Source, Author, and Date Range**. These filters are applied **at the database level** (inside Pinecone), meaning we only retrieve chunks from the specific documents you care about. This significantly reduces "hallucinations" by preventing irrelevant context from reaching the LLM.

### 🔄 Resilience & Retries
The code uses the `tenacity` library to wrap all external API calls. If the internet flickers or GPT is overloaded, the app will pause, wait (exponential backoff), and try again up to 5 times before failing.

### 🧪 Robust Testing
Located in `tests/`, the project includes:
- **Mocked Search**: Tests the engine without actually spending money on API calls.
- **Data Fidelity Tests**: Ensures that document chunking never loses text.
- **Configuration Tests**: Verifies that the app fails gracefully if an API key is missing.

---

## 5. Deployment Workflow

### Local Development
- **Venv**: Isolated environment using `requirements.txt`.
- **.env**: Local keys stored in a git-ignored file.

### Cloud Deployment (Streamlit)
- **CI/CD**: The app is connected to GitHub. Every `git push` triggers a live update.
- **Secrets**: API keys are injected at runtime via Streamlit's encrypted Key-Value store.

---

## 6. Future Enhancements

- **Hybrid Search**: Combining Keyword search (BM25) with Vector search for better medical/legal term matching.
- **Re-ranking**: Adding a "Cross-Encoder" pass after retrieval to sort the top 5 results by even higher precision.
- **Streaming**: Implementing `st.write_stream` for typewriter-style LLM responses.

---

*Report generated on 2026-02-24*
