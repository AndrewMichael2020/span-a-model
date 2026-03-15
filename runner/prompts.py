"""
Prompts for Phi-4-Reasoning-Vision-15B inference.

The model supports two reasoning modes:
  <nothink>  — Fast direct answer, no chain-of-thought (default for speed)
  <think>    — Full CoT reasoning before answering (best for complex OCR/docs)

The mode is injected as the first token of the assistant turn in the
OpenAI messages API, which is how vLLM exposes it.
"""

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise image analysis assistant. Your job is to extract structured \
metadata from photographs and documents. You always respond with valid JSON only \
— no markdown fences, no prose, no explanation outside the JSON object.\
"""

# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
Analyze this image and return a JSON object with these exact keys:

{
  "ocr_text":    "<all text visible in the image, verbatim, or null>",
  "summary":     "<2-3 sentence description of what this image shows>",
  "tags":        ["<keyword>", ...],
  "objects":     ["<physical object>", ...],
  "place_clues": "<location hints: landmarks, street signs, geography, or null>",
  "scene":       "<one of: indoor | outdoor | document | screenshot | map | other>",
  "image_notes": "<notable quality issues, context clues, or null>"
}

Rules:
- ocr_text must be exact verbatim transcription of any text in the image
- tags should be 3-10 concise keywords useful for search
- objects should list distinct physical objects, not descriptions
- If a field has no applicable content, use null (not empty string or [])
- Respond ONLY with the JSON object, nothing else\
"""

# ── Thinking-mode prefix injected at start of assistant turn ──────────────────

NOTHINK_PREFIX = "<nothink>"   # disables chain-of-thought → fast
THINK_PREFIX = ""              # empty = model decides (usually thinks for complex images)

# ── vLLM messages builder ──────────────────────────────────────────────────────

def build_messages(image_b64: str, thinking_mode: str = "nothink") -> list[dict]:
    """
    Build the OpenAI-format messages list for one image.

    Args:
        image_b64: Base64-encoded JPEG/PNG data (no data: URI prefix needed here,
                   we add it below).
        thinking_mode: "nothink" (fast) or "think" (CoT, best accuracy).

    Returns:
        messages list ready for openai.chat.completions.create()
    """
    assistant_prefix = NOTHINK_PREFIX if thinking_mode == "nothink" else THINK_PREFIX

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": EXTRACTION_PROMPT,
                },
            ],
        },
    ]

    # Inject thinking-mode prefix as a partial assistant turn.
    # vLLM supports this via the "prefix" mechanism — we set it as
    # an initial assistant message that the model continues from.
    if assistant_prefix:
        messages.append({
            "role": "assistant",
            "content": assistant_prefix,
        })

    return messages
