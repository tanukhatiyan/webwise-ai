"""
groq_provider.py — WebWise AI Groq Cloud Provider Integration
==============================================================
Provides ultra-fast cloud LLM inference using the official Groq Python SDK.
All existing task prompts, system guardrails, anti-prompt-injection tags,
and SSE chunk streaming formats are fully preserved.

SECURITY NOTE:
  GROQ_API_KEY is loaded strictly on the backend from settings / environment.
  It is NEVER returned in API responses or exposed to the Chrome extension.
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Generator, Optional

# Ensure backend directory is in sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from config import settings, SUPPORTED_GROQ_MODELS
from schemas import TaskType
from agent import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)


class GroqConfigurationError(Exception):
    """Raised when Groq API key or model configuration is missing or invalid."""
    pass


class GroqInferenceError(Exception):
    """Raised when Groq API encounters a runtime error (auth, rate limit, timeout)."""
    pass


def get_groq_client():
    """
    Instantiate and return an official Groq client instance.
    Raises GroqConfigurationError if GROQ_API_KEY is missing.
    """
    if not settings.is_groq_configured:
        raise GroqConfigurationError(
            "Groq API key is not configured. Please add GROQ_API_KEY to your backend .env file or environment."
        )

    try:
        from groq import Groq
    except ImportError:
        raise GroqConfigurationError(
            "The 'groq' package is not installed. Install it with: pip install groq"
        )

    return Groq(
        api_key=settings.groq_api_key.strip(),
        timeout=settings.request_timeout,
    )


def check_groq_health() -> tuple[bool, str]:
    """
    Validate Groq Cloud configuration.
    Returns (is_ready, message).
    """
    if not settings.is_groq_configured:
        return False, "Groq API key not set. Add GROQ_API_KEY to .env to enable Groq Cloud."

    if not settings.is_groq_model_supported:
        supported_list = ", ".join(SUPPORTED_GROQ_MODELS[:4])
        return (
            True,
            f"Configured model '{settings.groq_model}' may not be standard. Known models: {supported_list}...",
        )

    return True, f"Groq Cloud ready. Model: '{settings.groq_model}'"


def run_groq_streaming(
    content: str,
    task: TaskType,
    title: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
) -> Generator[str, None, None]:
    """
    Run Groq Cloud completion with streaming and yield text token deltas.

    Uses the same SYSTEM_PROMPT and build_user_message as local Qwen-Agent to
    guarantee anti-prompt-injection protection and consistent output quality.

    Args:
        content: Extracted webpage text (untrusted input)
        task: Analysis task type
        title: Optional page title
        stop_event: Cooperative cancellation event

    Yields:
        str: Individual text chunks as received from Groq Cloud
    """
    import groq

    client = get_groq_client()
    user_prompt = build_user_message(content, task, title)

    logger.info(
        "Starting Groq streaming | model=%s | task=%s | content_len=%d",
        settings.groq_model,
        task.value,
        len(content),
    )

    try:
        stream = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.temperature,
            max_tokens=min(settings.max_tokens, 800),
            stream=True,
        )

        for chunk in stream:
            if stop_event and stop_event.is_set():
                logger.info("Groq stream interrupted by cooperative stop_event")
                break

            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    except groq.AuthenticationError:
        err_msg = (
            "Invalid Groq API key. Please check that GROQ_API_KEY in your backend .env file is correct."
        )
        logger.error("Groq AuthenticationError: %s", err_msg)
        raise GroqInferenceError(err_msg)

    except groq.RateLimitError:
        err_msg = "Groq Cloud rate limit reached. Please wait a moment and try again."
        logger.error("Groq RateLimitError: %s", err_msg)
        raise GroqInferenceError(err_msg)

    except groq.APITimeoutError:
        err_msg = "Groq Cloud request timed out. Please check your network connection."
        logger.error("Groq APITimeoutError: %s", err_msg)
        raise GroqInferenceError(err_msg)

    except groq.APIConnectionError:
        err_msg = "Could not connect to Groq Cloud. Please check your internet connection."
        logger.error("Groq APIConnectionError: %s", err_msg)
        raise GroqInferenceError(err_msg)

    except groq.NotFoundError:
        err_msg = (
            f"Groq model '{settings.groq_model}' was not found or is discontinued. "
            f"Please update GROQ_MODEL in your backend .env file."
        )
        logger.error("Groq NotFoundError: %s", err_msg)
        raise GroqInferenceError(err_msg)

    except groq.BadRequestError as e:
        err_msg = f"Groq request error: {e.message if hasattr(e, 'message') else str(e)}"
        logger.error("Groq BadRequestError: %s", err_msg)
        raise GroqInferenceError(err_msg)

    except GroqConfigurationError:
        raise

    except Exception as e:
        err_msg = f"Groq service error: {str(e)}"
        logger.error("Unexpected Groq error: %s", err_msg)
        raise GroqInferenceError(err_msg)
