"""
Unified LLM provider for Agent 1's reasoning text and the GNN forensic
rationale. Replaces the earlier Gemini-only integration (Gemini is not
named anywhere in the paper -- it's referred to generically as "cognitive
Large Language Model (LLM) agents" -- so this swap needs no paper changes).

Two providers, tried in order:
  1. Groq (cloud, primary) -- free tier, OpenAI-compatible REST API, has
     meaningfully more generous rate limits than Gemini's free tier (which
     was hit live during development: 15 requests/minute).
  2. Ollama (local, fallback) -- runs a model on this machine, zero network
     calls, zero rate limits. Can't fail from an API quota during a live
     demo because there's no external dependency at all. Requires the
     model already pulled (`ollama pull llama3.2`) and the Ollama daemon
     running locally.

If both are unavailable or fail, callers fall back to their own static
template reasoning -- this module never raises, it returns (None, "template")
so nothing ever hard-fails a scan over an LLM being unreachable.

Configure via environment variables (see .env.example):
  LLM_PROVIDER   "groq" (default) or "ollama" -- which to try first.
  GROQ_API_KEY   from https://console.groq.com/keys
  GROQ_MODEL     defaults to "llama-3.3-70b-versatile"
  OLLAMA_HOST    defaults to "http://localhost:11434"
  OLLAMA_MODEL   defaults to "llama3.2"
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_groq(prompt: str, max_retries: int = 2) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.4}

    delay = 1.0
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(_GROQ_URL, headers=headers, json=payload, timeout=15)
            if resp.status_code == 429 and attempt < max_retries:
                print(f"[!] Groq rate-limited, retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_error = e
    raise last_error


def _call_ollama(prompt: str, timeout: int = 20) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    text = resp.json().get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return text


def generate(prompt: str):
    """
    Tries the configured provider first, the other second, template last.
    Returns (text, source) where source is "groq", "ollama", or "template"
    (text is None when source == "template" -- caller supplies its own
    static fallback text).
    """
    providers = [("groq", _call_groq), ("ollama", _call_ollama)]
    if LLM_PROVIDER == "ollama":
        providers = list(reversed(providers))

    for name, fn in providers:
        try:
            return fn(prompt), name
        except Exception as e:
            print(f"[!] {name} LLM call failed: {e}")
            continue

    return None, "template"


def is_configured() -> bool:
    """True if at least one provider looks reachable/configured."""
    if GROQ_API_KEY:
        return True
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return True
    except Exception:
        return False
