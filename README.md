# WebWise AI — Real-Time Web Summarizer Chrome Extension

> **100% Local · Privacy-First · Real-Time Streaming**  
> AI-powered webpage summarization running directly on your machine using FastAPI, Qwen-Agent, and local Qwen models via Ollama. No data ever leaves your computer.

---

## Architecture Overview

```
┌────────────────────────────────┐
│      Active Chrome Tab         │<img width="1927" height="1087" alt="Screenshot 2026-10-22 010217" src="https://github.com/user-attachments/assets/154ade86-d711-4d00-b05c-6036307c3963" />

└───────────────┬────────────────┘
                │ chrome.scripting.executeScript
                ▼
┌────────────────────────────────┐
│   WebWise Chrome Extension     │
│   (Vanilla HTML / CSS / JS)    │
│   - Provider selector (Ollama/ │
│     Groq Cloud)                │
│   - Extracts clean page text   │
│   - Consumes SSE via fetch()   │
│   - Renders progressive tokens │
└───────────────┬────────────────┘
                │ POST /summarize_stream_status
                │ { content, task, provider }
                ▼
┌────────────────────────────────┐
│     FastAPI Local Backend      │
│     - Pydantic validation      │
│     - Secure GROQ_API_KEY      │
│     - Prompt injection defense │
│     - Async SSE generator      │
└───────┬────────────────┬───────┘
        │                │
        │ provider="ollama" provider="groq"
        ▼                ▼
┌───────────────┐ ┌──────────────────────┐
│  Qwen-Agent   │ │ Groq Cloud SDK       │
│  (0.0.34)     │ │ (groq-python)        │
└───────┬───────┘ └──────────┬───────────┘
        │                    │
        ▼                    ▼
┌───────────────┐ ┌──────────────────────┐
│ Local Ollama  │ │ Groq Cloud API       │
│ (qwen3:1.7b)  │ │ (llama-3.3-70b)      │
│ Port: 11434   │ │ Ultra-Fast Inference │
└───────────────┘ └──────────────────────┘
```

---

## Key Features

- **4 Analysis Modes:**
  - 📝 **Summarize:** Crisp 3–5 sentence overview of the article, main takeaways, and conclusion.
  - 🎯 **Key Points:** 5 distinct, bulleted key takeaways (`•`).
  - 💡 **Explain Simply:** Plain English, beginner-friendly explanation avoiding jargon.
  - ❓ **Generate Questions:** 5 open-ended discussion questions to foster deeper critical thinking.
- **Genuine Incremental Streaming:** Progressive token generation via Server-Sent Events (SSE); tokens appear in real time as the local model generates them.
- **Robust In-Page Content Extractor:** Prioritizes `<article>`, `<main>`, or core content containers, stripping advertising, scripts, styles, navigations, and footers.
- **Strict Anti-Prompt-Injection Safeguards:** Page text is isolated in untrusted user delimiters (`<webpage_content>`) with system prompt enforcement.
- **Cooperative Early Cancellation:** Abort ongoing inference on demand via the "Cancel" button, freeing GPU/CPU resources immediately.
- **📄 Offline Multi-Page PDF Export:** Download the complete summary as a formatted PDF with source URL, title, date, headings, and bullet points using a 100% locally bundled jsPDF library (no remote calls).
- **📌 Persistent Detached Window:** Click the pop-out button to detach the interface into a standalone floating window that stays open side-by-side while browsing.
- **💾 Automatic State Persistence:** Results are saved in `chrome.storage.local` so you never lose your generated summary when clicking away or reopening the extension.
- **Clipboard Integration:** One-click copy with toast feedback.
- **Visual Health Indicator:** Real-time pulse dot reflecting backend connectivity and Ollama model status (Green = Ready, Yellow = Model Missing, Red = Offline).

---

## Prerequisites

1. **Python 3.10+** (Python 3.11 recommended).
2. **Ollama:** Installed from [ollama.com](https://ollama.com).
3. **Google Chrome** (or any Chromium-based browser like Brave, Edge).

---

## Quick Start Guide

### Step 1: Start Ollama and Pull the Model

In a terminal:

```bash
# Start Ollama service (if not already running as a background service)
ollama serve

# Pull the default Qwen3 model (1.7B parameter version)
ollama pull qwen3:1.7b
```

Verify the model is downloaded:

```bash
ollama list
```

### (Optional) Configure Groq Cloud Provider

If you want ultra-fast cloud inference alongside local Ollama:
1. Create a free account at [console.groq.com](https://console.groq.com/keys) and generate an API key.
2. Create or edit `.env` in the backend or root folder:
   ```bash
   GROQ_API_KEY=gsk_your_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```
3. If `GROQ_API_KEY` is not set, WebWise AI continues running 100% locally with Ollama.
4. Your API key remains securely on your local backend and is **never** sent to the browser or extension.

### Step 2: Set Up and Start the FastAPI Backend

In a new terminal:

```bash
# Navigate to backend directory
cd backend

# If using existing venv at project root:
..\venv\Scripts\activate
# (On macOS/Linux: source ../venv/bin/activate)

# Install dependencies (if setting up fresh):
pip install -r requirements.txt

# Start the FastAPI server (listens on http://127.0.0.1:7864)
python -m uvicorn app:app --host 127.0.0.1 --port 7864 --reload
```

Verify backend health in your browser or curl:

```bash
curl http://127.0.0.1:7864/health
```

Expected output:
```json
{
  "api_healthy": true,
  "ollama_reachable": true,
  "model_available": true,
  "model_name": "qwen3:1.7b",
  "ollama_url": "http://127.0.0.1:11434",
  "message": "All systems ready. Model 'qwen3:1.7b' is available."
}
```

### Step 3: Load the Chrome Extension

1. Open Google Chrome and go to `chrome://extensions/`.
2. Toggle **Developer mode** on (top-right switch).
3. Click **Load unpacked** (top-left button).
4. Select the `extension/` folder in this repository:
   ```
   e:\webwiseAi\extension
   ```
5. The **WebWise AI** extension icon will appear in your Chrome toolbar.
6. Pin it to your toolbar for quick access!

---

## Usage Instructions

1. Open any article, blog post, or webpage (e.g. Wikipedia, documentation).
2. Click the **WebWise AI** extension icon.
3. Verify the status dot in the upper-right corner is **green** (connected to local backend).
4. Select your desired analysis mode from the dropdown:
   - *Summarize*
   - *Key Points*
   - *Explain Simply*
   - *Generate Questions*
5. Click **Summarize** (or the corresponding action button).
6. Watch the summary stream in real time!
7. Use **Copy** to copy the text to your clipboard, or **Cancel** to stop generation early.

---

## Testing & Verification

### Running Unit Tests (Fast, Mocked)

The unit test suite covers health checks, Pydantic validations, whitespace handling, character bounds, SSE formatting, and client cancellation.

Run from the project root:

```bash
.\venv\Scripts\python.exe -m pytest
```

### Running Live End-to-End Integration Tests

Integration tests run real inference through the local Ollama instance:

```bash
.\venv\Scripts\python.exe -m pytest -m integration
```

### Running the Standalone Agent Verification Tool

Directly tests the Qwen-Agent streaming output across all 4 modes in a terminal:

```bash
cd backend
..\venv\Scripts\python.exe test_agent_standalone.py
```

---

## API Documentation

Interactive Swagger documentation is available locally at:
- **Swagger UI:** [http://127.0.0.1:7864/docs](http://127.0.0.1:7864/docs)
- **ReDoc:** [http://127.0.0.1:7864/redoc](http://127.0.0.1:7864/redoc)

### `GET /health`
Returns system status for backend, Ollama server reachability, and model availability.

### `POST /summarize_stream_status`
Accepts webpage content and streams responses using Server-Sent Events (`text/event-stream`).

**Request Body:**
```json
{
  "content": "Visible webpage text...",
  "task": "summarize",
  "title": "Optional Article Title"
}
```

**SSE Events Streamed:**
- `data: {"type": "chunk", "content": "..."}\n\n`
- `data: {"type": "done"}\n\n`
- `data: {"type": "error", "message": "..."}\n\n`

---

## Project Structure

```
webwiseAi/
├── backend/
│   ├── app.py                   # FastAPI server & SSE streaming endpoint
│   ├── agent.py                 # Qwen-Agent Assistant & streaming generator
│   ├── config.py                # Pydantic configuration & environment settings
│   ├── schemas.py               # Pydantic models (SummarizeRequest, HealthStatus)
│   ├── test_agent_standalone.py # Standalone runner for all 4 task types
│   ├── requirements.txt         # Python dependencies
│   ├── pytest.ini               # Backend test runner configuration
│   └── tests/
│       ├── __init__.py
│       └── test_api.py          # 29 unit & integration test cases
├── extension/
│   ├── manifest.json            # Chrome Manifest V3 configuration
│   ├── popup.html               # Extension UI markup
│   ├── popup.css                # Modern stylesheet & layout
│   ├── popup.js                 # Extraction, SSE client, & UI controller
│   └── icons/
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
├── PROJECT_CONTEXT.md           # Architecture report & project context
├── pytest.ini                   # Root pytest configuration
└── README.md                    # Project documentation
```

---

## License & Privacy

- **100% Private:** No telemetry, no external API calls, no third-party tracking. All model weights and requests remain entirely on localhost (`127.0.0.1`).
