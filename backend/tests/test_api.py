# -*- coding: utf-8 -*-
"""
tests/test_api.py — WebWise AI Backend Test Suite
===================================================
These tests cover:
  1. Health endpoint (structure + fields)
  2. Request validation (Pydantic errors surfaced as HTTP 422)
  3. Input size enforcement
  4. Invalid task rejection
  5. Empty / whitespace content rejection
  6. SSE streaming format validation (with Ollama mocked)

TEACHING NOTE — Why TestClient?
  FastAPI's TestClient wraps the app in a requests-compatible interface.
  This means you test the FULL stack (routing, validation, middleware, responses)
  WITHOUT needing a real running server.

  Under the hood, TestClient uses httpx to send requests to the ASGI app
  directly in memory — no network needed.

  For streaming tests we use stream=True (httpx streaming mode).

TEACHING NOTE — Mocking with unittest.mock:
  Some tests mock the Ollama health check and agent so tests are:
    - Fast (no real model inference)
    - Deterministic (same result every time)
    - Independent (don't require Ollama to be running)

  The tests marked with @pytest.mark.integration require a live Ollama instance.
  Run them with: pytest -m integration
"""

import json
import sys
import os

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add backend directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from config import settings
from agent import split_text_into_chunks

# ── Test client ───────────────────────────────────────────────────────────────
client = TestClient(app, raise_server_exceptions=False)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_content():
    """Short, safe article text for testing."""
    return (
        "Python is a high-level programming language created by Guido van Rossum "
        "and first released in 1991. It emphasizes code readability and simplicity."
    )


@pytest.fixture
def mock_ollama_healthy():
    """Mock Ollama health check to return (reachable=True, model_available=True)."""
    with patch("app.check_ollama_health", return_value=(True, True)):
        yield


@pytest.fixture
def mock_ollama_down():
    """Mock Ollama health check to return (reachable=False, model_available=False)."""
    with patch("app.check_ollama_health", return_value=(False, False)):
        yield


@pytest.fixture
def mock_agent_streaming():
    """Mock run_agent_streaming to return predictable chunks without real inference."""
    def fake_generator(content, task, title=None, *args, **kwargs):
        yield "This "
        yield "is "
        yield "a "
        yield "test "
        yield "summary."
    with patch("app.run_agent_streaming", side_effect=fake_generator):
        yield


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Health Endpoint Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self):
        """Health endpoint must always return 200 (the API itself is healthy)."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        """Response must contain all required HealthStatus fields."""
        response = client.get("/health")
        data = response.json()

        required_fields = {
            "api_healthy", "ollama_reachable", "model_available",
            "model_name", "ollama_url", "message"
        }
        assert required_fields.issubset(data.keys()), (
            f"Missing fields: {required_fields - data.keys()}"
        )

    def test_health_api_always_healthy(self):
        """api_healthy must be True — the API is running if we can call it."""
        response = client.get("/health")
        data = response.json()
        assert data["api_healthy"] is True

    def test_health_returns_configured_model_name(self):
        """model_name must match the configured setting."""
        response = client.get("/health")
        data = response.json()
        assert data["model_name"] == settings.model_name

    def test_health_returns_configured_ollama_url(self):
        """ollama_url must match the configured Ollama base URL."""
        response = client.get("/health")
        data = response.json()
        assert data["ollama_url"] == settings.ollama_base_url

    def test_health_message_is_string(self):
        """message field must be a non-empty string."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0

    def test_health_fields_are_correct_types(self):
        """Validate field types match the HealthStatus schema."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["api_healthy"], bool)
        assert isinstance(data["ollama_reachable"], bool)
        assert isinstance(data["model_available"], bool)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Request Validation Tests (HTTP 422)
# ═════════════════════════════════════════════════════════════════════════════

class TestRequestValidation:
    """
    Tests that Pydantic validation works correctly.

    TEACHING NOTE — HTTP 422 Unprocessable Entity:
      FastAPI automatically returns 422 when the request body doesn't match
      the Pydantic schema. You don't write any validation code — Pydantic
      and FastAPI handle it automatically based on the field definitions.
    """

    def test_missing_content_returns_422(self):
        """Omitting required 'content' field should return 422."""
        response = client.post(
            "/summarize_stream_status",
            json={"task": "summarize"},  # No content!
        )
        assert response.status_code == 422

    def test_empty_content_returns_422(self, mock_ollama_healthy):
        """Empty string content should fail min_length=1 validation."""
        response = client.post(
            "/summarize_stream_status",
            json={"content": "", "task": "summarize"},
        )
        assert response.status_code == 422

    def test_whitespace_only_content_returns_422(self, mock_ollama_healthy):
        """Content with only whitespace should fail the custom validator."""
        response = client.post(
            "/summarize_stream_status",
            json={"content": "   \n\t  ", "task": "summarize"},
        )
        assert response.status_code == 422

    def test_invalid_task_returns_422(self, sample_content, mock_ollama_healthy):
        """An unrecognized task value should fail the TaskType enum validation."""
        response = client.post(
            "/summarize_stream_status",
            json={"content": sample_content, "task": "hack_the_planet"},
        )
        assert response.status_code == 422

    def test_oversized_content_returns_422(self, mock_ollama_healthy):
        """Content exceeding max_content_chars should be rejected."""
        # Generate content that's definitely over the limit
        oversized = "A" * (settings.max_content_chars + 1)
        response = client.post(
            "/summarize_stream_status",
            json={"content": oversized, "task": "summarize"},
        )
        assert response.status_code == 422

    def test_valid_task_types_accepted(self, sample_content, mock_ollama_healthy, mock_agent_streaming):
        """All four valid task types must be accepted (not return 422)."""
        valid_tasks = ["summarize", "key_points", "explain_simply", "generate_questions"]
        for task in valid_tasks:
            with client.stream("POST", "/summarize_stream_status",
                               json={"content": sample_content, "task": task}) as r:
                assert r.status_code == 200, f"Task '{task}' returned {r.status_code}"

    def test_optional_title_accepted(self, sample_content, mock_ollama_healthy, mock_agent_streaming):
        """Providing an optional title should work fine."""
        with client.stream("POST", "/summarize_stream_status",
                           json={
                               "content": sample_content,
                               "task": "summarize",
                               "title": "Test Article",
                           }) as r:
            assert r.status_code == 200

    def test_title_too_long_returns_422(self, sample_content, mock_ollama_healthy):
        """Title exceeding max_title_chars should be rejected."""
        long_title = "T" * (settings.max_title_chars + 1)
        response = client.post(
            "/summarize_stream_status",
            json={"content": sample_content, "task": "summarize", "title": long_title},
        )
        assert response.status_code == 422

    def test_default_task_is_summarize(self, sample_content, mock_ollama_healthy, mock_agent_streaming):
        """Omitting task should default to 'summarize' (not return error)."""
        with client.stream("POST", "/summarize_stream_status",
                           json={"content": sample_content}) as r:
            assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: SSE Streaming Format Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestStreamingFormat:
    """
    Tests that the SSE stream format is correct.

    TEACHING NOTE — Server-Sent Events format:
      Each SSE event looks like:
        data: {"type": "chunk", "content": "some text"}\n\n

      The double newline (\n\n) marks the end of an event.
      The "data: " prefix is mandatory.
      The payload after "data: " is arbitrary (we use JSON).
    """

    def test_streaming_returns_event_stream_content_type(
        self, sample_content, mock_ollama_healthy, mock_agent_streaming
    ):
        """Response Content-Type must be text/event-stream for SSE."""
        with client.stream("POST", "/summarize_stream_status",
                           json={"content": sample_content, "task": "summarize"}) as r:
            assert r.status_code == 200
            content_type = r.headers.get("content-type", "")
            assert "text/event-stream" in content_type

    def test_streaming_chunks_are_valid_sse(
        self, sample_content, mock_ollama_healthy, mock_agent_streaming
    ):
        """Each SSE line must start with 'data: ' and contain valid JSON."""
        with client.stream("POST", "/summarize_stream_status",
                           json={"content": sample_content, "task": "summarize"}) as r:
            assert r.status_code == 200
            lines = []
            for line in r.iter_lines():
                if line:  # skip blank lines between events
                    lines.append(line)

        assert len(lines) > 0, "Expected at least one SSE event"

        for line in lines:
            assert line.startswith("data: "), f"SSE line missing 'data: ' prefix: {line!r}"
            payload = line[len("data: "):]
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                pytest.fail(f"SSE payload is not valid JSON: {payload!r}")
            assert "type" in data, f"SSE event missing 'type' field: {data}"

    def test_streaming_includes_chunk_events(
        self, sample_content, mock_ollama_healthy, mock_agent_streaming
    ):
        """Must receive at least one 'chunk' type event with non-empty content."""
        chunks = []
        with client.stream("POST", "/summarize_stream_status",
                           json={"content": sample_content, "task": "summarize"}) as r:
            for line in r.iter_lines():
                if line and line.startswith("data: "):
                    data = json.loads(line[len("data: "):])
                    if data.get("type") == "chunk":
                        chunks.append(data.get("content", ""))

        assert len(chunks) > 0, "Expected at least one 'chunk' event"
        assert any(c for c in chunks), "Expected at least one non-empty chunk"

    def test_streaming_ends_with_done_event(
        self, sample_content, mock_ollama_healthy, mock_agent_streaming
    ):
        """Stream must end with a 'done' type event."""
        events = []
        with client.stream("POST", "/summarize_stream_status",
                           json={"content": sample_content, "task": "summarize"}) as r:
            for line in r.iter_lines():
                if line and line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))

        assert len(events) > 0
        last_event = events[-1]
        assert last_event.get("type") == "done", (
            f"Last event should be 'done', got: {last_event}"
        )

    def test_chunk_events_have_content_field(
        self, sample_content, mock_ollama_healthy, mock_agent_streaming
    ):
        """Each 'chunk' event must have a 'content' string field."""
        with client.stream("POST", "/summarize_stream_status",
                           json={"content": sample_content, "task": "summarize"}) as r:
            for line in r.iter_lines():
                if line and line.startswith("data: "):
                    data = json.loads(line[len("data: "):])
                    if data.get("type") == "chunk":
                        assert "content" in data, "chunk event missing 'content' field"
                        assert isinstance(data["content"], str), "'content' must be a string"

    def test_client_early_disconnect_handled_gracefully(
        self, sample_content, mock_ollama_healthy, mock_agent_streaming
    ):
        """When client closes connection early, stream terminates cleanly without errors."""
        with client.stream("POST", "/summarize_stream_status",
                           json={"content": sample_content, "task": "summarize"}) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line and "chunk" in line:
                    break  # Early break simulates client disconnect


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Error Handling Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Tests for error states: Ollama down, model missing."""

    def test_ollama_down_returns_503(self, sample_content, mock_ollama_down):
        """When Ollama is not running, the endpoint must return 503."""
        response = client.post(
            "/summarize_stream_status",
            json={"content": sample_content, "task": "summarize"},
        )
        assert response.status_code == 503

    def test_ollama_down_error_message_is_helpful(self, sample_content, mock_ollama_down):
        """503 error detail must mention 'ollama serve' to guide the user."""
        response = client.post(
            "/summarize_stream_status",
            json={"content": sample_content, "task": "summarize"},
        )
        assert response.status_code == 503
        detail = response.json().get("detail", "").lower()
        assert "ollama" in detail, f"Error message should mention Ollama: {detail!r}"

    def test_model_not_available_returns_503(self, sample_content):
        """When Ollama runs but model is missing, return 503."""
        with patch("app.check_ollama_health", return_value=(True, False)):
            response = client.post(
                "/summarize_stream_status",
                json={"content": sample_content, "task": "summarize"},
            )
        assert response.status_code == 503

    def test_model_missing_error_mentions_ollama_pull(self, sample_content):
        """503 for missing model must mention 'ollama pull' to guide the user."""
        with patch("app.check_ollama_health", return_value=(True, False)):
            response = client.post(
                "/summarize_stream_status",
                json={"content": sample_content, "task": "summarize"},
            )
        detail = response.json().get("detail", "").lower()
        assert "ollama pull" in detail, f"Should mention 'ollama pull': {detail!r}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Root Endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestRootEndpoint:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_service_name(self):
        response = client.get("/")
        data = response.json()
        assert "WebWise AI" in data.get("service", "")

    def test_root_contains_endpoints_info(self):
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Text Chunking Tests (Handling Long Webpages)
# ═════════════════════════════════════════════════════════════════════════════

class TestTextChunking:
    """Tests for split_text_into_chunks utility."""

    def test_short_text_returns_single_chunk(self):
        text = "This is a short article."
        chunks = split_text_into_chunks(text, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty_list(self):
        assert split_text_into_chunks("", max_chars=1000) == []
        assert split_text_into_chunks("   \n\t  ", max_chars=1000) == []

    def test_paragraphs_split_cleanly(self):
        p1 = "Paragraph one with some text content."
        p2 = "Paragraph two with more text content."
        p3 = "Paragraph three with even more content."
        text = f"{p1}\n\n{p2}\n\n{p3}"
        # Set max_chars small enough so that each paragraph is in its own chunk
        chunks = split_text_into_chunks(text, max_chars=45)
        assert len(chunks) == 3
        assert chunks[0] == p1
        assert chunks[1] == p2
        assert chunks[2] == p3

    def test_all_chunks_stay_strictly_under_limit(self):
        # Generate ~100,000 characters of text across many paragraphs
        paragraphs = [f"Paragraph {i}: " + ("content word " * 50) for i in range(150)]
        long_text = "\n\n".join(paragraphs)
        assert len(long_text) > 80_000

        chunks = split_text_into_chunks(long_text, max_chars=35_000)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 35_000
            assert len(chunk.strip()) > 0

    def test_giant_single_paragraph_splits_at_sentence_boundaries(self):
        # A single paragraph with no newlines exceeding max_chars
        sentences = [f"Sentence {i} explains an important concept in detail." for i in range(50)]
        giant_para = " ".join(sentences)
        assert len(giant_para) > 2000

        chunks = split_text_into_chunks(giant_para, max_chars=500)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 500


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Integration tests (require live Ollama — skipped by default)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that require a live Ollama instance.
    Run with: pytest -m integration

    These are skipped in normal CI runs because they're slow (~20s each)
    and require external infrastructure (Ollama running with the model).
    """

    def test_real_summarize_produces_output(self):
        """Real end-to-end: actual model produces a non-empty summary."""
        short_content = (
            "The Python programming language was created by Guido van Rossum "
            "in 1991. It is known for its simple, readable syntax."
        )
        chunks_received = []

        with client.stream(
            "POST",
            "/summarize_stream_status",
            json={"content": short_content, "task": "summarize"},
            timeout=120.0,
        ) as r:
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            for line in r.iter_lines():
                if line and line.startswith("data: "):
                    data = json.loads(line[len("data: "):])
                    if data.get("type") == "chunk":
                        chunks_received.append(data["content"])

        full_text = "".join(chunks_received)
        assert len(full_text) > 20, f"Summary too short: {full_text!r}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: Groq Cloud Provider Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestGroqProvider:
    """Tests for the Groq Cloud provider integration."""

    def test_health_includes_groq_status_fields(self):
        """Health response should include groq_configured, groq_model, etc."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "groq_configured" in data
        assert isinstance(data["groq_configured"], bool)
        assert "default_provider" in data
        assert "available_providers" in data
        assert isinstance(data["available_providers"], list)

    def test_invalid_provider_returns_422(self, sample_content):
        """Passing an invalid provider name should fail schema validation."""
        response = client.post(
            "/summarize_stream_status",
            json={"content": sample_content, "provider": "chatgpt_pro"},
        )
        assert response.status_code == 422

    def test_groq_missing_api_key_returns_503(self, sample_content):
        """When provider='groq' but GROQ_API_KEY is not set, return 503."""
        with patch.object(settings, "groq_api_key", None):
            response = client.post(
                "/summarize_stream_status",
                json={"content": sample_content, "provider": "groq"},
            )
            assert response.status_code == 503
            detail = response.json().get("detail", "")
            assert "groq" in detail.lower()
            assert "groq_api_key" in detail.lower()

    def test_groq_streaming_success(self, sample_content):
        """When Groq is configured and succeeds, SSE stream yields chunks and terminates with done."""
        def fake_groq_generator(content, task, title=None, *args, **kwargs):
            yield "Groq "
            yield "fast "
            yield "summary."

        with patch.object(settings, "groq_api_key", "gsk_test_mock_key_12345"):
            with patch("app.run_groq_streaming", side_effect=fake_groq_generator):
                with client.stream(
                    "POST",
                    "/summarize_stream_status",
                    json={"content": sample_content, "provider": "groq"},
                ) as r:
                    assert r.status_code == 200
                    events = []
                    for line in r.iter_lines():
                        if line and line.startswith("data: "):
                            events.append(json.loads(line[len("data: "):]))

                    types = [e.get("type") for e in events]
                    assert "chunk" in types
                    assert types[-1] == "done"
                    content_text = "".join(e.get("content", "") for e in events if e.get("type") == "chunk")
                    assert content_text == "Groq fast summary."

    def test_groq_auth_error_handling(self, sample_content):
        """When Groq raises GroqInferenceError for invalid auth, stream yields error event with message."""
        from groq_provider import GroqInferenceError

        def failing_groq_generator(*args, **kwargs):
            raise GroqInferenceError("Invalid Groq API key. Please check that GROQ_API_KEY in your backend .env file is correct.")
            yield ""

        with patch.object(settings, "groq_api_key", "gsk_invalid_key"):
            with patch("app.run_groq_streaming", side_effect=failing_groq_generator):
                with client.stream(
                    "POST",
                    "/summarize_stream_status",
                    json={"content": sample_content, "provider": "groq"},
                ) as r:
                    assert r.status_code == 200
                    events = []
                    for line in r.iter_lines():
                        if line and line.startswith("data: "):
                            events.append(json.loads(line[len("data: "):]))

                    error_events = [e for e in events if e.get("type") == "error"]
                    assert len(error_events) > 0
                    assert "Invalid Groq API key" in error_events[0].get("message", "")

    def test_groq_rate_limit_error_handling(self, sample_content):
        """When Groq raises rate limit error, client receives clear rate limit message."""
        from groq_provider import GroqInferenceError

        def rate_limit_generator(*args, **kwargs):
            raise GroqInferenceError("Groq Cloud rate limit reached. Please wait a moment and try again.")
            yield ""

        with patch.object(settings, "groq_api_key", "gsk_valid_key"):
            with patch("app.run_groq_streaming", side_effect=rate_limit_generator):
                with client.stream(
                    "POST",
                    "/summarize_stream_status",
                    json={"content": sample_content, "provider": "groq"},
                ) as r:
                    assert r.status_code == 200
                    events = [json.loads(line[len("data: "):]) for line in r.iter_lines() if line and line.startswith("data: ")]
                    error_events = [e for e in events if e.get("type") == "error"]
                    assert len(error_events) > 0
                    assert "rate limit" in error_events[0].get("message", "").lower()

    def test_no_silent_fallback_when_groq_fails(self, sample_content):
        """If Groq fails, Ollama agent must NEVER be called (no silent fallback)."""
        from groq_provider import GroqInferenceError

        def failing_groq(*args, **kwargs):
            raise GroqInferenceError("Groq connection failure")
            yield ""

        mock_ollama_agent = MagicMock()

        with patch.object(settings, "groq_api_key", "gsk_valid_key"):
            with patch("app.run_groq_streaming", side_effect=failing_groq):
                with patch("app.run_agent_streaming", mock_ollama_agent):
                    with client.stream(
                        "POST",
                        "/summarize_stream_status",
                        json={"content": sample_content, "provider": "groq"},
                    ) as r:
                        assert r.status_code == 200
                        events = [json.loads(line[len("data: "):]) for line in r.iter_lines() if line and line.startswith("data: ")]
                        assert any(e.get("type") == "error" for e in events)
                        # Ollama agent must NEVER have been called!
                        assert mock_ollama_agent.call_count == 0
