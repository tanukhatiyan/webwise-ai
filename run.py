"""
WebWise AI — Application Launcher
===================================
Run this script to start the WebWise AI FastAPI backend server:
    python run.py
"""

import sys
import os
from pathlib import Path

# Set up paths
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure Windows console encoding is UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_ollama(ollama_url: str, model_name: str) -> tuple[bool, bool]:
    """Check if Ollama is running and if the required model is available."""
    import urllib.request
    import urllib.error
    import json

    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", headers={"User-Agent": "WebWise-Launcher"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # Match name exactly or by tag prefix (e.g. qwen3:1.7b)
                model_present = any(m == model_name or m.startswith(f"{model_name}:") for m in models)
                return True, model_present
    except Exception:
        pass
    return False, False


def main():
    print("=" * 65)
    print("  WebWise AI — Real-Time Web Summarizer Launcher")
    print("=" * 65)

    from config import settings

    print(f"[*] Checking Ollama server at {settings.ollama_base_url}...")
    ollama_ok, model_ok = check_ollama(settings.ollama_base_url, settings.model_name)

    if not ollama_ok:
        print(f"[!] WARNING: Ollama server is NOT running at {settings.ollama_base_url}.")
        print("    Please start it in another terminal with:  ollama serve")
    elif not model_ok:
        print(f"[!] WARNING: Ollama is running, but model '{settings.model_name}' was not found.")
        print(f"    Please pull it with:  ollama pull {settings.model_name}")
    else:
        print(f"[OK] Ollama is active | Model '{settings.model_name}' is ready!")

    print(f"[*] Checking Groq Cloud configuration...")
    if settings.is_groq_configured:
        print(f"[OK] Groq Cloud configured | Model: '{settings.groq_model}'")
    else:
        print(f"[i] Groq Cloud not configured (GROQ_API_KEY not set). Using local Ollama provider.")

    print("-" * 65)
    print(f"[*] Starting FastAPI Server on http://{settings.backend_host}:{settings.backend_port}")
    print(f"[*] API Health Check:  http://{settings.backend_host}:{settings.backend_port}/health")
    print(f"[*] Interactive Docs:  http://{settings.backend_host}:{settings.backend_port}/docs")
    print("-" * 65)
    print("[*] Chrome Extension Setup:")
    print("    1. Open chrome://extensions/ in Chrome")
    print("    2. Enable 'Developer mode' (top-right toggle)")
    print("    3. Click 'Load unpacked' and select the extension folder:")
    print(f"       {ROOT_DIR / 'extension'}")
    print("=" * 65 + "\n")

    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
        app_dir=str(BACKEND_DIR),
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
