"""
ollama_analyzer.py — LLM-powered photo analysis via Ollama (LLaVA vision model)

Provides:
  - Full structured photo analysis (scene, aesthetic critique, captions, categories)
  - Human-readable rejection explanations
  - Auto-categorisation / tagging
  - Custom-prompt analysis (user-defined prompts)
  - Ollama health-check utility

Requires:  pip install requests pillow
           ollama running locally (https://ollama.com)
           ollama pull llava:13b   (or your chosen vision model)
"""

import base64
import json
import logging
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

log = logging.getLogger(__name__)

_ollama_lock = threading.Lock()


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class LLMAnalysis:
    """Structured result from LLM vision analysis."""
    scene_description: str = ""
    aesthetic_critique: str = ""
    suggested_caption: str = ""
    categories: list[str] = field(default_factory=list)
    rejection_explanation: str = ""
    suggested_profile: str = ""
    quality_opinion: str = ""          # "good", "mediocre", "poor"
    confidence: float = 0.0
    custom_response: str = ""          # response to user's custom prompt
    raw_response: str = ""
    error: Optional[str] = None


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llava:13b"
    timeout: int = 120                  # seconds per request
    enabled: bool = True
    max_image_size: int = 1024          # resize longest edge before sending
    temperature: float = 0.3            # low = more deterministic
    custom_prompt: str = ""             # user's free-form prompt (applied per photo)


# ===========================================================================
# Image encoding
# ===========================================================================

def _encode_image(image_path: str, max_size: int = 1024) -> str:
    """Read image, resize if needed, return base64 JPEG string."""
    img = Image.open(image_path)

    # Resize to limit tokens / bandwidth
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===========================================================================
# Low-level Ollama API call
# ===========================================================================

def _call_ollama(
    cfg: OllamaConfig,
    prompt: str,
    image_b64: Optional[str] = None,
) -> str:
    """Send a single prompt (optionally with image) to Ollama. Thread-safe."""
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": 1024,
        },
    }
    if image_b64:
        payload["images"] = [image_b64]

    with _ollama_lock:
        try:
            resp = requests.post(
                f"{cfg.base_url}/api/generate",
                json=payload,
                timeout=cfg.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as exc:
            log.error("Ollama call failed: %s", exc)
            raise


# ===========================================================================
# Prompt templates
# ===========================================================================

_ANALYSIS_PROMPT = """\
You are a professional photo editor evaluating image quality.
Analyze this photo and respond ONLY with a JSON object (no markdown, no extra text):

{{
  "scene_description": "One sentence describing what is in the photo",
  "aesthetic_critique": "2-3 sentences on composition, lighting, focus quality",
  "suggested_caption": "A short caption suitable for a photo album",
  "categories": ["list", "of", "relevant", "tags"],
  "quality_opinion": "good" or "mediocre" or "poor",
  "suggested_profile": "portrait" or "group" or "landscape",
  "confidence": 0.0 to 1.0
}}

Be honest and specific. Focus on technical quality, not artistic preference."""

_REJECTION_PROMPT = """\
You are a professional photo editor.
This photo was rejected by automated quality checks with these scores:
{breakdown}

And these notes:
{notes}

Write a single friendly, human-readable sentence explaining why this photo
didn't pass quality checks and what could be improved. Be specific.
Respond with ONLY the explanation sentence, nothing else."""

_CATEGORIZE_PROMPT = """\
Look at this photo and classify it into relevant categories.
Respond ONLY with a JSON array of category strings, e.g.:
["family", "outdoor", "birthday", "children"]

Use lowercase tags. Include: event type, setting, mood, number of people,
activity, and season if detectable. Maximum 8 tags."""


# ===========================================================================
# Helper: strip markdown fences from LLM output
# ===========================================================================

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that LLMs sometimes add."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


# ===========================================================================
# Public analysis functions
# ===========================================================================

def analyze_photo(image_path: str, cfg: OllamaConfig) -> LLMAnalysis:
    """Full structured analysis of a single photo."""
    if not cfg.enabled:
        return LLMAnalysis(error="LLM analysis disabled")

    try:
        img_b64 = _encode_image(image_path, cfg.max_image_size)
    except Exception as exc:
        return LLMAnalysis(error=f"Image encode failed: {exc}")

    try:
        raw = _call_ollama(cfg, _ANALYSIS_PROMPT, img_b64)
    except Exception as exc:
        return LLMAnalysis(error=str(exc))

    # Parse JSON from response
    result = LLMAnalysis(raw_response=raw)
    try:
        text = _strip_markdown_fences(raw)
        data = json.loads(text)
        result.scene_description = data.get("scene_description", "")
        result.aesthetic_critique = data.get("aesthetic_critique", "")
        result.suggested_caption = data.get("suggested_caption", "")
        result.categories = data.get("categories", [])
        result.quality_opinion = data.get("quality_opinion", "")
        result.suggested_profile = data.get("suggested_profile", "")
        result.confidence = float(data.get("confidence", 0))
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Failed to parse LLM JSON: %s", exc)
        # Still keep raw_response for the report
        result.scene_description = raw[:200]

    return result


def explain_rejection(
    image_path: str,
    breakdown: dict,
    notes: list[str],
    cfg: OllamaConfig,
) -> str:
    """Generate a human-friendly rejection reason using the vision model."""
    if not cfg.enabled:
        return ""
    try:
        img_b64 = _encode_image(image_path, cfg.max_image_size)
        prompt = _REJECTION_PROMPT.format(
            breakdown=json.dumps(breakdown, indent=2),
            notes="\n".join(notes),
        )
        return _call_ollama(cfg, prompt, img_b64).strip()
    except Exception as exc:
        log.warning("Rejection explanation failed: %s", exc)
        return ""


def categorize_photo(image_path: str, cfg: OllamaConfig) -> list[str]:
    """Auto-tag a photo with category labels using the vision model."""
    if not cfg.enabled:
        return []
    try:
        img_b64 = _encode_image(image_path, cfg.max_image_size)
        raw = _call_ollama(cfg, _CATEGORIZE_PROMPT, img_b64)
        text = _strip_markdown_fences(raw)
        tags = json.loads(text)
        if isinstance(tags, list):
            return [str(t).lower().strip() for t in tags[:8]]
        return []
    except Exception:
        return []


def run_custom_prompt(
    image_path: str,
    custom_prompt: str,
    cfg: OllamaConfig,
) -> str:
    """Run the user's free-form prompt against a single photo."""
    if not cfg.enabled or not custom_prompt.strip():
        return ""
    try:
        img_b64 = _encode_image(image_path, cfg.max_image_size)
        return _call_ollama(cfg, custom_prompt.strip(), img_b64).strip()
    except Exception as exc:
        log.warning("Custom prompt failed: %s", exc)
        return f"Error: {exc}"


# ===========================================================================
# Health check
# ===========================================================================

def check_service(cfg: OllamaConfig) -> dict:
    """Health check — returns status dict for the GUI."""
    try:
        resp = requests.get(f"{cfg.base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            model_base = cfg.model.split(":")[0]
            found = any(model_base in m for m in models)
            return {
                "online": True,
                "model_available": found,
                "models": models,
            }
        return {"online": False, "model_available": False, "models": []}
    except Exception:
        return {"online": False, "model_available": False, "models": []}
