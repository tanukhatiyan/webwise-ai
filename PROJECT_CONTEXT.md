# WebWise AI — Project Context Report

**Generated:** October 2026  
**Status:** Completed & Verified  
**System:** Windows, Python 3.11.5, Ollama Local Server (`qwen3:1.7b`)  
**Project:** WebWise AI — Real-Time Web Summarizer Chrome Extension  

---

## 1. Project Purpose & Functionality

WebWise AI is a complete, privacy-first, hybrid (local Ollama + optional Groq Cloud) web summarization and content analysis tool consisting of:
1. **Chrome Extension (Manifest V3):** Extracts readable text and title from the active browser tab, allows selecting between Local Ollama and Groq Cloud, and displays an interactive popup with real-time streaming summaries.
2. **Python FastAPI Backend:** Validates input, guards against prompt injection, manages Server-Sent Events (SSE) streaming, and routes requests to either local Qwen-Agent/Ollama or Groq Cloud.
3. **Dual AI Provider Architecture:**
   - **Local Ollama (`qwen3:1.7b`):** 100% private, zero internet required.
   - **Groq Cloud (`llama-3.3-70b-versatile`):** Ultra-fast cloud inference with secure backend API key handling.

### Supported Task Modes
- `summarize`: 3–5 sentence concise summary of the main topic and conclusions.
- `key_points`: 5 distinct, bulleted key takeaways (`•`).
- `explain_simply`: Plain English, beginner-friendly explanation avoiding jargon.
- `generate_questions`: 5 numbered, open-ended discussion questions.

---

## 2. Complete Project Structure

```
e:\webwiseAi\
├── .gitignore                     # Git ignore rules for venv, caches, env
├── PROJECT_CONTEXT.md             # Architecture report & context for AI sessions
├── README.md                      # Comprehensive user setup, testing, and API guide
├── pytest.ini                     # Root pytest configuration
├── backend\
│   ├── .env.example               # Environment variables configuration template
│   ├── config.py                  # Pydantic BaseSettings configuration
│   ├── schemas.py                 # TaskType Enum, SummarizeRequest, HealthStatus schemas
│   ├── agent.py                   # Qwen-Agent integration and streaming generator
│   ├── groq_provider.py           # Groq Cloud integration with streaming generator
│   ├── app.py                     # FastAPI server, CORS, /health, /summarize_stream_status
│   ├── test_agent_standalone.py   # Standalone runner for testing the 4 task modes
│   ├── pytest.ini                 # Backend-specific pytest configuration
│   ├── requirements.txt           # Python package dependencies
│   └── tests\
│       ├── __init__.py
│       └── test_api.py            # Unit and live integration tests (29 test cases)
└── extension\
    ├── manifest.json              # Chrome Extension Manifest V3 configuration
    ├── popup.html                 # Extension popup HTML UI layout
    ├── popup.css                  # Extension styling & animations
    ├── popup.js                   # Extension controller, DOM extractor & SSE parser
    └── icons\
        ├── icon16.png
        ├── icon48.png
        └── icon128.png
```

---

## 3. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     Active Browser Tab                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ chrome.scripting.executeScript
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Chrome Extension (popup.js)                 │
│  - Checks backend & model health (GET /health)              │
│  - Extracts main readable content & title                   │
│  - Sends POST /summarize_stream_status                      │
│  - Reads Server-Sent Events (SSE) via fetch ReadableStream   │
│  - Updates popup DOM in real time with progressive tokens   │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP POST /summarize_stream_status
                               │ (SSE text/event-stream)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend (app.py)                    │
│  - Pydantic validation (SummarizeRequest)                   │
│  - Pre-flight Ollama health check                           │
│  - asyncio.Queue bridge for thread-pool execution           │
│  - SSE event formatting (chunk, done, error)                │
│  - Cooperative stop_event cancellation                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ ThreadPoolExecutor
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Qwen-Agent (agent.py)                     │
│  - System prompt enforcing strict anti-prompt-injection     │
│  - OpenAI-compatible client configuration                   │
│  - Assistant agent yielding accumulated text deltas         │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP POST /v1/chat/completions (Stream)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Local Ollama Server (11434)                 │
│  - Model: qwen3:1.7b                                        │
│  - Local GPU/CPU inference without cloud dependencies       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Technologies & Dependencies

- **Language & Runtime:** Python 3.11.5, JavaScript (ES2022, Manifest V3).
- **Backend Framework:** FastAPI 0.141.1, Starlette, Uvicorn 0.30+.
- **AI Agent Framework:** `qwen-agent` 0.0.34 (Assistant mode, no tool calling/code interpreter).
- **Validation & Settings:** Pydantic v2, `pydantic-settings`.
- **HTTP Client:** HTTPX 0.27+ (async health probes).
- **Local Inference:** Ollama server (`http://127.0.0.1:11434`), model `qwen3:1.7b`.
- **Testing:** Pytest 8.3.5, `pytest-asyncio` 0.26.0, `anyio`.
- **Frontend / Extension:** Chrome Extension Manifest V3, Vanilla HTML5/CSS3/JavaScript.

---

## 5. Verification Status & Executed Tests

### Unit Tests
Executed via `.\venv\Scripts\python.exe -m pytest`:
- **Result:** 41 passed, 1 deselected, 0 failures.
- **Coverage:**
  - Health endpoint structure, types, model name, Ollama URL, Groq fields (`/health`).
  - Request validation (missing content, empty string, whitespace only, invalid task type, invalid provider, oversized content, long titles).
  - SSE streaming format (`text/event-stream`, `data: ` framing, valid JSON, `chunk` event structure, terminal `done` event).
  - Client early disconnection handling and cancellation cleanup.
  - Text chunking tests (`split_text_into_chunks`): paragraph splitting, sentence boundary splitting, and length limit enforcement.
  - Error handling (Ollama offline -> HTTP 503, model missing -> HTTP 503 with pull instructions).
  - Groq provider tests: unconfigured API key handling (HTTP 503), streaming SSE chunks, authentication error handling (401), rate limit handling (429), and strict no-silent-fallback enforcement.
  - Root endpoint (`/`).

### Live End-to-End Integration Tests
Executed via `.\venv\Scripts\python.exe -m pytest -m integration`:
- **Result:** 1 passed in 18.19s against live Ollama (`qwen3:1.7b`).
- **Verified:** Real end-to-end stream from Ollama through FastAPI to client stream reader.

### Standalone Agent Tests
Executed via `python test_agent_standalone.py`:
- **Result:** 4/4 passed:
  - `summarize`: 107 chunks, 579 characters.
  - `key_points`: 91 chunks, 444 characters.
  - `explain_simply`: 199 chunks, 910 characters.
  - `generate_questions`: 151 chunks, 797 characters.

### Live Server HTTP & SSE Streaming Test
Executed against live `uvicorn` instance on `http://127.0.0.1:7864`:
- `GET /health` returned HTTP 200 with all healthy flags.
- `OPTIONS /summarize_stream_status` with `Origin: chrome-extension://...` returned 200 with proper CORS headers.
- `POST /summarize_stream_status` streamed 65 token chunks and finished with `{"type": "done"}`.

### Frontend Validation
- `node --check extension/popup.js`: Syntax valid (ES2022).
- `extension/manifest.json`: JSON syntax valid, configured for Chrome Manifest V3.
- All icons (`icon16.png`, `icon48.png`, `icon128.png`) verified present.

---

## 6. Key Changes Made During Completion

1. **Frontend Implementation:**
   - Created `extension/popup.js`: In-tab DOM content extraction, backend health monitoring, dynamic button updates, SSE parser via `ReadableStreamDefaultReader`, `AbortController` cancellation, copy to clipboard, long-webpage chunking, state persistence via `chrome.storage.local`, persistent detached window mode, and offline multi-page PDF generation via locally bundled jsPDF.
   - Created `extension/popup.css`: Modern, polished card-based UI with animated status indicator, responsive scroll container, clean typography, loading spinner, toast notifications, and detached window styling.
   - Updated `extension/manifest.json`: Added `storage` permission for result persistence across popup reopens; enhanced host permissions and CSP for local development.
   - Bundled `extension/libs/jspdf.umd.min.js`: Zero remote scripts at runtime, fully offline and compliant with Manifest V3 CSP.

2. **Backend Enhancements:**
   - Added cooperative cancellation with `threading.Event`: Stops local GPU/CPU inference immediately when client cancels or closes the popup.
   - Fixed Windows console encoding in `test_agent_standalone.py` and `app.py` (`sys.stdout.reconfigure(encoding="utf-8")`) and replaced fragile console characters to prevent `UnicodeEncodeError`.
   - Added `__test__ = False` to `test_agent_standalone.py` and created root `pytest.ini` to prevent pytest from misidentifying standalone scripts as unit tests.
   - Added `test_client_early_disconnect_handled_gracefully` to unit test suite.

3. **Documentation:**
   - Created root `README.md` with complete architecture diagram, setup instructions, testing steps, and API docs.
   - Created and updated `PROJECT_CONTEXT.md`.
