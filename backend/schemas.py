"""
schemas.py — WebWise AI Request & Response Schemas
===================================================
Pydantic models define the exact shape of data coming in and going out.

TEACHING NOTE — What Pydantic Does:
  Without Pydantic, you'd manually write:
      data = request.json()
      content = data.get("content")
      if not content or not isinstance(content, str):
          raise ValueError("content must be a non-empty string")
      ...and so on for every field.

  Pydantic automates ALL of that. You describe the shape once,
  and FastAPI automatically:
    1. Parses the incoming JSON body
    2. Validates every field (type, length, allowed values)
    3. Returns a structured 422 Unprocessable Entity error if validation fails
    4. Generates the OpenAPI/Swagger documentation automatically

  This is one of the biggest productivity wins in modern Python API development.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from config import settings


class TaskType(str, Enum):
    """
    The four task modes the agent supports.
    Using an Enum means the API will reject any other string value automatically.
    FastAPI will also show these as a dropdown in Swagger UI.
    """

    SUMMARIZE = "summarize"
    KEY_POINTS = "key_points"
    EXPLAIN_SIMPLY = "explain_simply"
    GENERATE_QUESTIONS = "generate_questions"


class ProviderType(str, Enum):
    """
    Supported AI providers for content analysis.
    - OLLAMA: Local inference using Ollama and local weights (100% private)
    - GROQ: Ultra-fast cloud inference via Groq Cloud API
    """

    OLLAMA = "ollama"
    GROQ = "groq"


class SummarizeRequest(BaseModel):
    """
    The JSON body that the /summarize_stream_status endpoint expects.

    TEACHING NOTE — Field():
      Field() adds metadata to a Pydantic field:
        - description: shown in Swagger UI
        - min_length / max_length: Pydantic validates these BEFORE your code runs
        - examples: shown in Swagger UI to help API users

    TEACHING NOTE — Prompt Injection Defense:
      The `content` field is treated as UNTRUSTED TEXT throughout the backend.
      The system prompt in agent.py explicitly tells the model to ignore
      instructions found in the page content. The content is placed in a
      USER message, never injected into the system prompt template itself.
      We also enforce a hard length limit here to reduce the attack surface.
    """

    content: str = Field(
        ...,  # ... means "required" in Pydantic
        min_length=1,
        max_length=settings.max_content_chars,
        description=(
            "Extracted visible text from the webpage. "
            "Must be non-empty and within the configured character limit."
        ),
        examples=["Python is a high-level programming language..."],
    )

    task: TaskType = Field(
        default=TaskType.SUMMARIZE,
        description="The analysis task to perform on the content.",
        examples=["summarize", "key_points", "explain_simply", "generate_questions"],
    )

    title: Optional[str] = Field(
        default=None,
        max_length=settings.max_title_chars,
        description="Optional: the webpage's <title> tag value.",
        examples=["Python (programming language) - Wikipedia"],
    )

    provider: Optional[ProviderType] = Field(
        default=None,
        description="The AI provider to use ('ollama' or 'groq'). Defaults to server's configured default provider.",
        examples=["ollama", "groq"],
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_whitespace_only(cls, v: str) -> str:
        """
        Reject requests where content is only spaces/newlines.
        min_length=1 passes for " " (a single space), so we add this check.
        """
        if not v.strip():
            raise ValueError(
                "content must contain actual text, not only whitespace."
            )
        return v.strip()

    @field_validator("title")
    @classmethod
    def sanitize_title(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace from the title if provided."""
        if v is not None:
            return v.strip() or None
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "Python is a high-level, general-purpose programming language...",
                    "task": "summarize",
                    "title": "Python - Wikipedia",
                }
            ]
        }
    }


class HealthStatus(BaseModel):
    """
    Response schema for the GET /health endpoint.

    TEACHING NOTE — Why separate API health from model health?
      - api_healthy: True = FastAPI is running. Your load balancer can rely on this.
      - ollama_reachable: True = Ollama server responded. But the model might not be loaded.
      - model_available: True = The specific model (qwen3:1.7b) is pulled and ready.

      This distinction matters because:
        - The API can be healthy even if Ollama is down (we can still serve other endpoints)
        - Ollama can be running but the model might not be pulled yet
    """

    # Suppress pydantic v2 warning about model_ prefix in field names.
    # The default protected_namespaces=('model_',) is meant to avoid conflicts
    # with Pydantic's own model_fields / model_config etc. Our field names
    # like model_name and model_available are intentional and safe here.
    model_config = {"protected_namespaces": ()}

    api_healthy: bool = Field(description="True if the FastAPI backend is running.")
    ollama_reachable: bool = Field(
        description="True if the Ollama server responded to a ping."
    )
    model_available: bool = Field(
        description="True if the configured model is listed in Ollama."
    )
    model_name: str = Field(description="The configured model name.")
    ollama_url: str = Field(description="The Ollama base URL being used.")
    message: str = Field(description="Human-readable status summary.")

    # ── Groq Cloud Provider Status ─────────────────────────────────────────
    groq_configured: bool = Field(
        default=False,
        description="True if GROQ_API_KEY is configured on the backend.",
    )
    groq_model: Optional[str] = Field(
        default=None,
        description="The configured Groq model name.",
    )
    default_provider: str = Field(
        default="ollama",
        description="The default AI provider ('ollama' or 'groq').",
    )
    available_providers: list[str] = Field(
        default_factory=lambda: ["ollama"],
        description="List of currently usable AI providers.",
    )
