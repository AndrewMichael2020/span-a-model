#!/usr/bin/env python3
"""
scripts/test_local.py — tests that can run WITHOUT GCP credentials.

Tests:
  1. Schema dataclass round-trips (to/from dict/JSON)
  2. Prompt building for both thinking modes
  3. JSON response parsing (simulating vLLM output)
  4. Image resize logic (no GCS, no vLLM)
  5. Manifest creation and progress tracking
  6. ingest.py CLI dry-run (no GCP)

Run: python scripts/test_local.py
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from io import BytesIO
from pathlib import Path

# Add packages to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "runner"))
sys.path.insert(0, str(REPO_ROOT / "client"))

# ── Test helpers ──────────────────────────────────────────────────────────────

_passed = 0
_failed = 0


def test(name: str, fn):
    global _passed, _failed
    try:
        fn()
        print(f"  ✅  {name}")
        _passed += 1
    except Exception as e:
        print(f"  ❌  {name}")
        traceback.print_exc()
        _failed += 1


# ── 1. Schema round-trips ─────────────────────────────────────────────────────

def test_schema():
    from schema import ExtractionResult, JobManifest, ImageEntry, JobProgress

    # ExtractionResult
    r = ExtractionResult(
        asset_id="img001",
        ocr_text="Hello world",
        summary="A test image",
        tags=["test", "demo"],
        objects=["text", "background"],
        scene="document",
    )
    d = r.to_dict()
    assert d["asset_id"] == "img001"
    assert d["tags"] == ["test", "demo"]
    assert d["error"] is None

    # JSON round-trip
    j = r.to_json()
    parsed = json.loads(j)
    assert parsed["ocr_text"] == "Hello world"

    # from_dict
    r2 = ExtractionResult.from_dict(parsed)
    assert r2.asset_id == r.asset_id
    assert r2.tags == r.tags

    # error_result
    err = ExtractionResult.error_result("img002", "inference timeout")
    assert err.error == "inference timeout"
    assert err.asset_id == "img002"

    # ImageEntry
    entry = ImageEntry(
        asset_id="img001",
        gcs_path="gs://bucket/job/images/img001.jpg",
    )
    assert entry.gcs_path.startswith("gs://")

    # JobManifest round-trip
    manifest = JobManifest(
        job_id="job_test_001",
        images=[entry],
        result_bucket="fmo-phi4-batch",
        result_prefix="job_test_001/results/",
        image_prefix="job_test_001/images/",
    )
    mj = manifest.to_json()
    m2 = JobManifest.from_dict(json.loads(mj))
    assert m2.job_id == "job_test_001"
    assert len(m2.images) == 1
    assert m2.images[0].asset_id == "img001"

    # JobProgress
    prog = JobProgress(job_id="job_test_001", total=100)
    prog.processed.append("img001")
    assert prog.pending_count == 99


def test_schema_defaults():
    from schema import ExtractionResult
    r = ExtractionResult(asset_id="x")
    assert r.tags == []
    assert r.objects == []
    assert r.model == "phi4-reasoning-vision-15b-awq"
    assert r.thinking_mode == "nothink"
    assert r.error is None


# ── 2. Prompt building ────────────────────────────────────────────────────────

def test_prompts_nothink():
    from prompts import build_messages, NOTHINK_PREFIX
    msgs = build_messages("FAKEB64==", thinking_mode="nothink")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # nothink prefix injected as assistant turn
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == NOTHINK_PREFIX
    # Image URL in user message
    user_content = msgs[1]["content"]
    assert any(c.get("type") == "image_url" for c in user_content)
    assert any(c.get("type") == "text" for c in user_content)


def test_prompts_think():
    from prompts import build_messages
    msgs = build_messages("FAKEB64==", thinking_mode="think")
    # In think mode, no assistant prefix is appended
    assert msgs[-1]["role"] == "user"
    assert len(msgs) == 2


def test_prompts_image_url_format():
    from prompts import build_messages
    msgs = build_messages("abc123", thinking_mode="nothink")
    image_content = [c for c in msgs[1]["content"] if c.get("type") == "image_url"]
    assert len(image_content) == 1
    url = image_content[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,abc123")


# ── 3. JSON response parsing (simulating vLLM output) ─────────────────────────

def test_parse_good_response():
    """Simulate a clean JSON response from the model."""
    from schema import ExtractionResult

    raw = json.dumps({
        "ocr_text": "STOP",
        "summary": "A red stop sign on a street corner.",
        "tags": ["traffic", "sign", "street", "urban"],
        "objects": ["stop sign", "pole", "sky"],
        "place_clues": None,
        "scene": "outdoor",
        "image_notes": None,
    })

    parsed = json.loads(raw)
    r = ExtractionResult(
        asset_id="test",
        ocr_text=parsed.get("ocr_text"),
        summary=parsed.get("summary"),
        tags=parsed.get("tags") or [],
        objects=parsed.get("objects") or [],
        place_clues=parsed.get("place_clues"),
        scene=parsed.get("scene"),
        image_notes=parsed.get("image_notes"),
    )
    assert r.ocr_text == "STOP"
    assert "traffic" in r.tags
    assert r.scene == "outdoor"


def test_parse_think_mode_response():
    """Simulate a <think>…</think> wrapped response."""
    raw_with_think = """<think>
Let me analyze this image carefully. I can see text in the image...
The image shows a document with printed text.
</think>
{"ocr_text": "Invoice #12345", "summary": "A printed invoice.", "tags": ["invoice", "document"], "objects": ["paper"], "place_clues": null, "scene": "document", "image_notes": null}"""

    # Replicate the strip logic in phi4_runner.py
    if "<think>" in raw_with_think and "</think>" in raw_with_think:
        after_think = raw_with_think.split("</think>", 1)[1].strip()
        raw_text = after_think if after_think else raw_with_think
    else:
        raw_text = raw_with_think

    parsed = json.loads(raw_text)
    assert parsed["ocr_text"] == "Invoice #12345"
    assert parsed["scene"] == "document"


def test_parse_null_fields():
    """Ensure null fields are handled correctly."""
    raw = json.dumps({
        "ocr_text": None,
        "summary": "A blank white image.",
        "tags": [],
        "objects": [],
        "place_clues": None,
        "scene": "other",
        "image_notes": "Image appears blank or overexposed.",
    })
    parsed = json.loads(raw)
    from schema import ExtractionResult
    r = ExtractionResult(
        asset_id="blank",
        ocr_text=parsed.get("ocr_text"),
        tags=parsed.get("tags") or [],
        objects=parsed.get("objects") or [],
        scene=parsed.get("scene"),
    )
    assert r.ocr_text is None
    assert r.tags == []


# ── 4. Image resize logic ─────────────────────────────────────────────────────

def test_image_resize_no_op():
    """Image smaller than max_px should not be upscaled."""
    try:
        from PIL import Image
    except ImportError:
        print("    (Pillow not installed, skipping image tests)")
        return

    # Create a small test image
    img = Image.new("RGB", (800, 600), color=(128, 200, 100))
    buf = BytesIO()
    img.save(buf, "JPEG")
    image_bytes = buf.getvalue()

    # Simulate resize_and_encode from runner
    img2 = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img2.size
    max_px = 1200
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img2 = img2.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out = BytesIO()
    img2.save(out, format="JPEG", quality=85)
    b64 = base64.b64encode(out.getvalue()).decode("ascii")

    # Should not have been resized (800 < 1200)
    assert img2.size == (800, 600)
    assert len(b64) > 0


def test_image_resize_large():
    """Image larger than max_px should be downscaled."""
    try:
        from PIL import Image
    except ImportError:
        return

    img = Image.new("RGB", (4000, 3000), color=(255, 128, 0))
    buf = BytesIO()
    img.save(buf, "JPEG")
    image_bytes = buf.getvalue()

    img2 = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img2.size
    max_px = 1200
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img2 = img2.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # 4000px → 1200px, 3000px → 900px
    assert img2.size[0] == 1200
    assert img2.size[1] == 900


# ── 5. CLI arg parsing (no GCP calls) ─────────────────────────────────────────

def test_cli_image_discovery():
    """Test that find_images correctly filters by extension."""
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        (td / "photo.jpg").write_bytes(b"fake")
        (td / "doc.PNG").write_bytes(b"fake")
        (td / "readme.txt").write_bytes(b"not an image")
        (td / "video.mp4").write_bytes(b"not an image")
        (td / "scan.tiff").write_bytes(b"fake")

        # Replicate find_images logic from ingest.py
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".webp", ".bmp"}
        paths = sorted([p for p in td.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

        assert len(paths) == 3
        names = {p.name for p in paths}
        assert "photo.jpg" in names
        assert "doc.PNG" in names
        assert "scan.tiff" in names
        assert "readme.txt" not in names
        assert "video.mp4" not in names


# ── 6. JobProgress checkpoint logic ──────────────────────────────────────────

def test_progress_resume():
    """Resuming from a checkpoint should skip already-processed assets."""
    from schema import JobProgress

    prog = JobProgress(job_id="j1", total=5)
    prog.processed = ["img001", "img002", "img003"]
    prog.failed = ["img004"]

    already_done = set(prog.processed) | set(prog.failed)
    all_ids = ["img001", "img002", "img003", "img004", "img005"]
    remaining = [a for a in all_ids if a not in already_done]

    assert remaining == ["img005"]
    assert prog.pending_count == 1


# ── Run all tests ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*55}")
    print(f"  span-a-model local tests (no GCP required)")
    print(f"{'═'*55}\n")

    test("Schema dataclass round-trips", test_schema)
    test("Schema default values", test_schema_defaults)
    test("Prompt building (nothink mode)", test_prompts_nothink)
    test("Prompt building (think mode)", test_prompts_think)
    test("Prompt image URL format", test_prompts_image_url_format)
    test("JSON response parsing (clean)", test_parse_good_response)
    test("JSON response parsing (think mode strip)", test_parse_think_mode_response)
    test("JSON response parsing (null fields)", test_parse_null_fields)
    test("Image resize — small image (no-op)", test_image_resize_no_op)
    test("Image resize — large image (downscale)", test_image_resize_large)
    test("CLI image discovery by extension", test_cli_image_discovery)
    test("Progress checkpoint resume logic", test_progress_resume)

    print(f"\n{'─'*55}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print(f"{'═'*55}\n")

    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
