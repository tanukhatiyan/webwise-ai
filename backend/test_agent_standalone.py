# -*- coding: utf-8 -*-
"""
test_agent_standalone.py — Phase 2 Agent Integration Test
==========================================================
Run this script to verify that:
  1. Qwen-Agent initializes correctly with Ollama config
  2. Ollama is reachable and the model is available
  3. The streaming generator produces real tokens
  4. All four task types work

Run with:
  python test_agent_standalone.py
"""

# Tell pytest not to collect this module as a unit test suite
__test__ = False

import sys
import time
import urllib.request
import urllib.error
import json
import os

# Ensure safe UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Adjust path so we can import our modules ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from schemas import TaskType
from agent import run_agent_streaming, create_agent


def check_ollama() -> bool:
    """Quick health check before running the expensive model call."""
    try:
        url = f"{settings.ollama_base_url}/api/tags"
        req = urllib.request.urlopen(url, timeout=5)
        data = json.loads(req.read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"  ✅ Ollama reachable at {settings.ollama_base_url}")
        print(f"  ✅ Available models: {models}")
        if settings.model_name in models:
            print(f"  ✅ Configured model '{settings.model_name}' is available")
            return True
        else:
            print(f"  ❌ Model '{settings.model_name}' NOT found. Run: ollama pull {settings.model_name}")
            return False
    except urllib.error.URLError as e:
        print(f"  ❌ Cannot reach Ollama at {settings.ollama_base_url}: {e}")
        print("     Make sure Ollama is running: ollama serve")
        return False


def test_agent_init():
    """Test that the agent initializes without errors."""
    print("\n[TEST 1] Agent Initialization")
    print("-" * 40)
    try:
        agent = create_agent()
        print("  ✅ Agent created successfully")
        print(f"  ✅ Agent type: {type(agent).__name__}")
        return True
    except Exception as e:
        print(f"  ❌ Agent creation failed: {e}")
        return False


def test_streaming(task: TaskType, content: str, title: str = None) -> bool:
    """Test a single streaming task and print tokens as they arrive."""
    print(f"\n[TEST] Task: {task.value.upper()}")
    print("-" * 40)
    print(f"  Content: '{content[:80]}...'")
    print("\n  Streaming output:")
    print("  " + "─" * 50)
    print("  ", end="", flush=True)

    start_time = time.time()
    chunk_count = 0
    total_chars = 0

    try:
        for chunk in run_agent_streaming(content=content, task=task, title=title):
            print(chunk, end="", flush=True)
            chunk_count += 1
            total_chars += len(chunk)

        elapsed = time.time() - start_time
        print(f"\n  " + "─" * 50)
        print(f"  ✅ Done | {chunk_count} chunks | {total_chars} chars | {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"\n  ❌ Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  WebWise AI — Phase 2 Agent Integration Test")
    print("=" * 60)

    # ── Step 1: Check Ollama ───────────────────────────────────────────────
    print("\n[PREFLIGHT] Checking Ollama...")
    if not check_ollama():
        print("\n❌ Cannot proceed — Ollama is not ready.")
        sys.exit(1)

    # ── Step 2: Agent init ─────────────────────────────────────────────────
    if not test_agent_init():
        print("\n❌ Cannot proceed — agent failed to initialize.")
        sys.exit(1)

    # ── Step 3: Test content (short, safe article excerpt) ────────────────
    sample_content = """
    The Python programming language was created by Guido van Rossum and first
    released in 1991. Python emphasizes code readability and simplicity, using
    significant whitespace and a clean syntax. It supports multiple programming
    paradigms including procedural, object-oriented, and functional programming.
    Python has become one of the most popular programming languages in the world,
    widely used in web development, data science, artificial intelligence,
    scientific computing, and automation. Its large standard library and active
    community have contributed thousands of third-party packages available
    through the Python Package Index (PyPI). Python 3, released in 2008,
    introduced many improvements over Python 2, which reached end-of-life in 2020.
    Major companies like Google, Instagram, Dropbox, and NASA use Python extensively.
    """.strip()

    sample_title = "Python (programming language) — Overview"

    # ── Step 4: Test only SUMMARIZE first (fastest feedback) ──────────────
    results = {}
    results["summarize"] = test_streaming(
        TaskType.SUMMARIZE, sample_content, sample_title
    )

    # Only test others if summarize worked
    if results["summarize"]:
        results["key_points"] = test_streaming(
            TaskType.KEY_POINTS, sample_content, sample_title
        )
        results["explain_simply"] = test_streaming(
            TaskType.EXPLAIN_SIMPLY, sample_content, sample_title
        )
        results["generate_questions"] = test_streaming(
            TaskType.GENERATE_QUESTIONS, sample_content, sample_title
        )

    # ── Step 5: Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 TEST RESULTS")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  🎉 All tests passed! Ready for Phase 3 (FastAPI).")
    else:
        print("  ⚠️  Some tests failed. Fix issues before proceeding.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
