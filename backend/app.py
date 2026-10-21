"""
app.py — WebWise AI FastAPI Application
========================================
This is the main entry point for the backend server.

TEACHING NOTE — What FastAPI Does:
  FastAPI is a modern Python web framework built on top of:
    - Starlette: the ASGI framework (handles HTTP, WebSockets, middleware)
    - Pydantic: handles request/response validation
    - Python type hints: used to generate OpenAPI docs automatically

  When you run this file with uvicorn, it:
    1. Creates an ASGI application (handles async HTTP)
    2. Registers routes (endpoints like /health, /summarize_stream_status)
    3. Listens for HTTP requests on 127.0.0.1:7864

TEACHING NOTE — ASGI vs WSGI:
  Old Python web frameworks (Flask, Django) used WSGI — synchronous.
  One request blocked one thread. Needed many threads for concurrency.

  FastAPI uses ASGI — asynchronous. One thread can handle many requests
  simultaneously using Python's asyncio event loop. While one request waits
  for the AI model to respond, other requests are processed.

TEACHING NOTE — CORS (Cross-Origin Resource Sharing):
  Web browsers enforce a security rule: JavaScript on page A cannot fetch
  from server B unless server B explicitly permits it via CORS headers.

  Our Chrome extension popup runs from chrome-extension://<id>/popup.html
  The backend runs at http://127.0.0.1:7864

  Without CORS headers, the browser blocks the extension's fetch() call.
  We add CORSMiddleware to allow requests from chrome-extension:// origins.

  SECURITY: We only allow chrome-extension:// and local origins.
  We never allow '*' (all origins) — that would expose our Ollama endpoint.
"""

import asyncio
import json
import logging
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

# Ensure backend directory is in sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import settings
from schemas import HealthStatus, SummarizeRequest, TaskType, ProviderType
from agent import run_agent_streaming
from groq_provider import run_groq_streaming, check_groq_health

# ── Encoding setup for Windows consoles ───────────────────────────────────────
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan: startup and shutdown logic ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    TEACHING NOTE — FastAPI Lifespan:
      This is where you put startup/shutdown code.
      The code before `yield` runs at startup.
      The code after `yield` runs at shutdown.

      Old FastAPI used @app.on_event("startup") — that's deprecated.
      The modern way is the lifespan context manager.
    """
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("WebWise AI Backend starting...")
    logger.info("Model: %s | Ollama: %s", settings.model_name, settings.ollama_base_url)
    logger.info("Backend: http://%s:%d", settings.backend_host, settings.backend_port)
    logger.info("=" * 60)

    # Verify Ollama is reachable at startup (non-fatal — just warn)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if settings.model_name in models:
                    logger.info("[OK] Ollama ready | Model '%s' available", settings.model_name)
                else:
                    logger.warning(
                        "[WARN] Ollama running but model '%s' not found. Run: ollama pull %s",
                        settings.model_name, settings.model_name
                    )
    except Exception as e:
        logger.warning("[WARN] Ollama not reachable at startup: %s", e)
        logger.warning("  The backend will start but local Ollama tasks will fail until Ollama is running.")

    # Verify Groq Cloud configuration at startup
    if settings.is_groq_configured:
        logger.info("[OK] Groq Cloud provider configured | Model '%s'", settings.groq_model)
    else:
        logger.info("[INFO] Groq Cloud not configured (GROQ_API_KEY not set). Using Ollama provider.")

    yield  # ← The application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("WebWise AI Backend shutting down...")


# ── App creation ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="WebWise AI Backend",
    description=(
        "Real-time webpage summarization using Qwen3 via Ollama. "
        "All inference is local — no cloud APIs required."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI at http://127.0.0.1:7864/docs
    redoc_url="/redoc",    # ReDoc UI at http://127.0.0.1:7864/redoc
)


# ── CORS Middleware ───────────────────────────────────────────────────────────
# TEACHING NOTE — Why we need this:
#   Chrome extensions make HTTP requests from chrome-extension://<id>/popup.html
#   Browsers block these by default (Same-Origin Policy).
#   We must explicitly allow chrome-extension:// origins.
#
# SECURITY NOTE:
#   We use allow_origin_regex to match all chrome-extension:// origins.
#   Extension IDs are random, so we can't whitelist a specific ID.
#   We do NOT use allow_origins=["*"] — that would be too permissive.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:7864",
        "http://localhost:7864",
    ],
    allow_origin_regex=r"chrome-extension://.*",  # All Chrome extensions
    allow_credentials=False,  # No cookies/auth needed
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    expose_headers=["X-Content-Type-Options"],
    max_age=3600,
)


# ── Helper: Check Ollama health ───────────────────────────────────────────────
async def check_ollama_health() -> tuple[bool, bool]:
    """
    Returns (ollama_reachable, model_available).

    TEACHING NOTE — async with httpx:
      httpx is an async HTTP client. Using it inside an async function
      with `async with` means the request doesn't block other async tasks.
      Compare with requests.get() which BLOCKS the thread.

      async with httpx.AsyncClient() as client:
          resp = await client.get(url)  ← suspends this coroutine,
                                           lets other requests run
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code != 200:
                return False, False
            models = [m["name"] for m in resp.json().get("models", [])]
            model_available = settings.model_name in models
            return True, model_available
    except Exception:
        return False, False


# ── Endpoint: GET /health ─────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthStatus,
    summary="Backend and model health check",
    tags=["Health"],
)
async def health_check() -> HealthStatus:
    """
    Returns the health of the backend, Ollama model availability, and Groq status.
    """
    ollama_reachable, model_available = await check_ollama_health()
    groq_ok, groq_msg = check_groq_health()

    available_providers = []
    if ollama_reachable and model_available:
        available_providers.append("ollama")
    if groq_ok:
        available_providers.append("groq")

    # Construct descriptive summary message
    parts = []
    if ollama_reachable and model_available:
        parts.append(f"Ollama ready ({settings.model_name})")
    elif not ollama_reachable:
        parts.append(f"Ollama offline ({settings.ollama_base_url})")
    elif not model_available:
        parts.append(f"Ollama missing model '{settings.model_name}'")

    if groq_ok:
        parts.append(f"Groq Cloud ready ({settings.groq_model})")
    else:
        parts.append("Groq Cloud unconfigured (set GROQ_API_KEY)")

    msg = " | ".join(parts)

    return HealthStatus(
        api_healthy=True,
        ollama_reachable=ollama_reachable,
        model_available=model_available,
        model_name=settings.model_name,
        ollama_url=settings.ollama_base_url,
        message=msg,
        groq_configured=settings.is_groq_configured,
        groq_model=settings.groq_model if settings.is_groq_configured else None,
        default_provider=settings.default_provider,
        available_providers=available_providers if available_providers else ["ollama"],
    )


# ── Endpoint: POST /summarize_stream_status ───────────────────────────────────
@app.post(
    "/summarize_stream_status",
    summary="Stream a webpage analysis task",
    tags=["Summarization"],
    responses={
        200: {
            "description": "Server-Sent Events stream of text chunks",
            "content": {"text/event-stream": {}},
        },
        422: {"description": "Validation error (content too long, invalid task, etc.)"},
        503: {"description": "Selected AI provider not available"},
    },
)
async def summarize_stream(request_body: SummarizeRequest, request: Request):
    """
    Accepts webpage content and streams an analysis using Server-Sent Events.
    Supports both Local Ollama (qwen3:1.7b) and Groq Cloud (llama-3.3-70b-versatile).
    """
    # ── Resolve selected AI provider ───────────────────────────────────────
    selected_provider = request_body.provider
    if selected_provider is None:
        try:
            selected_provider = ProviderType(settings.default_provider.lower())
        except (ValueError, AttributeError):
            selected_provider = ProviderType.OLLAMA

    # ── Pre-flight provider validation (No silent switching) ───────────────
    if selected_provider == ProviderType.OLLAMA:
        ollama_reachable, model_available = await check_ollama_health()
        if not ollama_reachable:
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start it with: ollama serve",
            )
        if not model_available:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model '{settings.model_name}' is not available in Ollama. "
                    f"Pull it with: ollama pull {settings.model_name}"
                ),
            )
    elif selected_provider == ProviderType.GROQ:
        if not settings.is_groq_configured:
            raise HTTPException(
                status_code=503,
                detail="Groq API key is not configured. Please add GROQ_API_KEY to your backend .env file or environment.",
            )

    logger.info(
        "Summarize request | provider=%s | task=%s | content_length=%d chars | title=%s",
        selected_provider.value,
        request_body.task.value,
        len(request_body.content),
        # Log title presence, not the title itself — minor privacy measure
        "present" if request_body.title else "absent",
    )

    async def sse_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        SENTINEL = object()  # A unique object marking "generator is done"
        stop_event = threading.Event()

        def run_sync_generator():
            try:
                if selected_provider == ProviderType.GROQ:
                    gen = run_groq_streaming(
                        content=request_body.content,
                        task=request_body.task,
                        title=request_body.title,
                        stop_event=stop_event,
                    )
                else:
                    try:
                        gen = run_agent_streaming(
                            content=request_body.content,
                            task=request_body.task,
                            title=request_body.title,
                            stop_event=stop_event,
                        )
                    except TypeError:
                        gen = run_agent_streaming(
                            content=request_body.content,
                            task=request_body.task,
                            title=request_body.title,
                        )

                for chunk in gen:
                    if stop_event.is_set():
                        break
                    # asyncio.Queue is thread-safe for put/get calls
                    asyncio.run_coroutine_threadsafe(
                        queue.put(chunk), loop
                    ).result()  # Wait for put to complete
            except Exception as e:
                if not stop_event.is_set():
                    asyncio.run_coroutine_threadsafe(
                        queue.put(e), loop
                    ).result()
            finally:
                asyncio.run_coroutine_threadsafe(
                    queue.put(SENTINEL), loop
                ).result()

        # Start the sync generator in a thread pool
        future = loop.run_in_executor(None, run_sync_generator)

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected — stopping generator")
                    stop_event.set()
                    future.cancel()
                    break

                try:
                    # Wait for next chunk (with timeout to check disconnection)
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # No chunk yet, loop back and check disconnection

                if item is SENTINEL:
                    # Generator finished normally
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                elif isinstance(item, Exception):
                    # Generator raised an exception
                    logger.error("Generator error: %s", str(item))
                    yield f"data: {json.dumps({'type': 'error', 'message': str(item)})}\n\n"
                    break
                else:
                    # Normal text chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': item})}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE generator cancelled")
            stop_event.set()
            future.cancel()
        except Exception as e:
            logger.error("SSE generator unexpected error: %s", e)
            stop_event.set()
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal server error'})}\n\n"
        finally:
            stop_event.set()

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            # Tell browsers/proxies not to buffer the response
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind proxy
            "Connection": "keep-alive",
        },
    )


# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """API root — shows available endpoints."""
    return {
        "service": "WebWise AI Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "GET /health",
            "summarize": "POST /summarize_stream_status",
        },
    }


# ── Entry point for direct execution ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,  # Auto-reload on code changes during development
        app_dir=str(_BACKEND_DIR),
        log_level=settings.log_level.lower(),
    )
