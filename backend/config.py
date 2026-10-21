"""
config.py — WebWise AI Backend Configuration
=============================================
All configurable values live here. We use pydantic's BaseSettings so that
every value can be overridden by an environment variable or a .env file.

TEACHING NOTE:
  pydantic BaseSettings reads values in this priority order:
    1. Environment variables (highest priority — good for production)
    2. .env file values
    3. Default values defined here (lowest priority — good for development)

  This means you can run the backend with defaults during development and
  override any value by setting an env variable — no code changes needed.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent

# ── Supported Groq Cloud Models ─────────────────────────────────────────────
# Verified models available on Groq Cloud:
SUPPORTED_GROQ_MODELS: list[str] = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
]


class Settings(BaseSettings):
    """
    Application settings — all values are read from environment variables
    or the .env file if present. Defaults work for local development.
    """

    model_config = SettingsConfigDict(
        env_file=(_ROOT_DIR / ".env", _BACKEND_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,     # OLLAMA_BASE_URL == ollama_base_url
        extra="ignore",           # Ignore unknown env vars (don't crash)
        protected_namespaces=(),  # Avoid pydantic v2 warning about model_ namespace
    )

    # ── Ollama / Local Model settings ──────────────────────────────────────
    ollama_base_url: str = "http://127.0.0.1:11434"
    """Base URL for the local Ollama server."""

    model_name: str = "qwen3:1.7b"
    """
    The Ollama model to use. Must be pulled before starting the backend.
    Override with: MODEL_NAME=qwen3:8b python -m uvicorn app:app
    """

    # ── Groq Cloud settings ────────────────────────────────────────────────
    groq_api_key: Optional[str] = None
    """
    Groq Cloud API key. Set via GROQ_API_KEY environment variable or .env.
    NEVER expose this in frontend, version control, or client responses.
    """

    groq_model: str = "qwen/qwen3.8-27b"
    """
    The Groq model to use. Default is qwen/qwen3.8-27b.
    Override with: GROQ_MODEL=openai/gpt-oss-120b
    """

    default_provider: str = "ollama"
    """
    Default AI provider: 'ollama' or 'groq'.
    Can be overridden per-request via the 'provider' field.
    """

    @property
    def is_groq_configured(self) -> bool:
        """Return True if a non-empty Groq API key is present."""
        return bool(self.groq_api_key and self.groq_api_key.strip())

    @property
    def is_groq_model_supported(self) -> bool:
        """Check if the configured Groq model is in the supported models list."""
        return self.groq_model in SUPPORTED_GROQ_MODELS

    # ── Backend server settings ────────────────────────────────────────────
    backend_host: str = "127.0.0.1"
    """Bind address. NEVER change to 0.0.0.0 — keeps Ollama local-only."""

    backend_port: int = 7864
    """Port the FastAPI backend listens on."""

    # ── Security / Input limits ────────────────────────────────────────────
    max_content_chars: int = 50_000
    """
    Maximum webpage text length accepted (characters, not tokens).
    ~50k chars ≈ ~12k tokens — fits comfortably in qwen3:1.7b's context.
    Requests exceeding this are rejected with HTTP 422 before reaching the model.
    """

    max_title_chars: int = 500
    """Maximum length for the optional webpage title field."""

    # ── Model generation parameters ────────────────────────────────────────
    temperature: float = 0.7
    """
    Controls randomness. 0.0 = deterministic, 1.0 = very creative.
    0.7 is a good balance for summarization tasks.
    """

    max_tokens: int = 1024
    """Maximum tokens the model generates per response."""

    request_timeout: int = 120
    """Seconds before we give up waiting for the model. Prevents hung requests."""

    # ── CORS settings ─────────────────────────────────────────────────────
    allowed_origins: list[str] = [
        # Chrome extensions use the chrome-extension:// scheme.
        # We allow all extension origins at the pattern level in app.py.
        # This list is for explicit extra origins during development.
        "http://127.0.0.1:7864",
        "http://localhost:7864",
    ]

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    """Python logging level: DEBUG, INFO, WARNING, ERROR"""


# Singleton instance — import this everywhere instead of re-instantiating
settings = Settings()
