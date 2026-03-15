"""
phi4_runner.py — runs INSIDE the GCP Spot VM.

Lifecycle:
  1. Wait for vLLM server to be healthy (it's started by the startup script)
  2. Poll GCS for pending job manifests
  3. For each image: download → infer → write result JSON to GCS
  4. Checkpoint progress after each image (resume-safe on spot preemption)
  5. When all jobs complete: stop the VM (returns to TERMINATED, costs $0)

Run: python phi4_runner.py --project PROJECT_ID --batch-bucket fmo-phi4-batch
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from google.cloud import storage
from openai import OpenAI
from PIL import Image

from prompts import build_messages
from schema import ExtractionResult, ImageEntry, JobManifest, JobProgress

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phi4_runner")

# ── Constants ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_HEALTH_URL = "http://127.0.0.1:8000/health"
VLLM_STARTUP_TIMEOUT_S = 600       # 10 min to load model
VLLM_HEALTH_POLL_S = 10

MANIFEST_BLOB_SUFFIX = "/manifest.json"
PROGRESS_BLOB_SUFFIX = "/progress.json"

MAX_IMAGE_PX = 1200                # max dimension before sending to model
JPEG_QUALITY = 85

POLL_INTERVAL_S = 15               # how often to check for new jobs when idle
MAX_IDLE_POLLS = 4                 # stop after N consecutive idle polls (~1 min)

# ── Graceful shutdown flag ─────────────────────────────────────────────────────

_shutdown_requested = False

def _handle_signal(sig, frame):
    global _shutdown_requested
    log.warning("Received signal %s — will stop after current image", sig)
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── GCS helpers ───────────────────────────────────────────────────────────────

def gcs_read_json(client: storage.Client, bucket_name: str, blob_name: str) -> Optional[dict]:
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return json.loads(blob.download_as_text())
    except Exception:
        return None


def gcs_write_text(client: storage.Client, bucket_name: str, blob_name: str, text: str) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(text, content_type="application/json")


def gcs_download_bytes(client: storage.Client, gcs_uri: str) -> bytes:
    """Download from gs://bucket/path and return raw bytes."""
    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got: {gcs_uri}"
    parts = gcs_uri[5:].split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def list_pending_manifests(client: storage.Client, bucket_name: str) -> list[str]:
    """Return blob names of manifests that have no corresponding progress file."""
    bucket = client.bucket(bucket_name)
    # List all manifest blobs
    manifests = [
        b.name for b in bucket.list_blobs(match_glob="*/manifest.json")
    ]
    # Filter: skip those that already have a completed progress file
    pending = []
    for manifest_blob in manifests:
        job_prefix = manifest_blob.replace("/manifest.json", "")
        progress_blob = f"{job_prefix}/progress.json"
        prog_data = gcs_read_json(client, bucket_name, progress_blob)
        if prog_data and prog_data.get("completed"):
            log.debug("Job %s already completed, skipping", job_prefix)
            continue
        pending.append(manifest_blob)
    return pending


# ── Image preprocessing ───────────────────────────────────────────────────────

def resize_and_encode(image_bytes: bytes, max_px: int = MAX_IMAGE_PX) -> str:
    """Resize image to max_px on longest side, return base64-encoded JPEG."""
    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── vLLM inference ────────────────────────────────────────────────────────────

def wait_for_vllm(timeout_s: int = VLLM_STARTUP_TIMEOUT_S) -> None:
    log.info("Waiting for vLLM to be ready (timeout %ds)…", timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(VLLM_HEALTH_URL, timeout=5)
            if r.status_code == 200:
                log.info("vLLM is ready ✓")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(VLLM_HEALTH_POLL_S)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s}s")


def infer_image(
    openai_client: OpenAI,
    image_b64: str,
    asset_id: str,
    thinking_mode: str = "nothink",
    model_name: str = "phi4-rv",
) -> ExtractionResult:
    """Call vLLM, parse JSON response, return ExtractionResult."""
    messages = build_messages(image_b64, thinking_mode)

    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )
        raw_text = response.choices[0].message.content or ""

        # Strip <think>…</think> block if present (think mode)
        if "<think>" in raw_text and "</think>" in raw_text:
            after_think = raw_text.split("</think>", 1)[1].strip()
            raw_text = after_think if after_think else raw_text

        # Parse JSON
        parsed = json.loads(raw_text.strip())
        return ExtractionResult(
            asset_id=asset_id,
            ocr_text=parsed.get("ocr_text"),
            summary=parsed.get("summary"),
            tags=parsed.get("tags") or [],
            objects=parsed.get("objects") or [],
            place_clues=parsed.get("place_clues"),
            scene=parsed.get("scene"),
            image_notes=parsed.get("image_notes"),
            ai_raw=raw_text,
            thinking_mode=thinking_mode,
        )

    except json.JSONDecodeError as e:
        log.warning("JSON parse error for %s: %s", asset_id, e)
        return ExtractionResult.error_result(asset_id, f"json_parse_error: {e}")
    except Exception as e:
        log.error("Inference error for %s: %s", asset_id, e)
        return ExtractionResult.error_result(asset_id, str(e))


# ── Job processor ─────────────────────────────────────────────────────────────

def process_job(
    gcs: storage.Client,
    openai_client: OpenAI,
    batch_bucket: str,
    manifest_blob_name: str,
    model_name: str,
) -> None:
    manifest_data = gcs_read_json(gcs, batch_bucket, manifest_blob_name)
    if not manifest_data:
        log.error("Could not read manifest: %s", manifest_blob_name)
        return

    manifest = JobManifest.from_dict(manifest_data)
    job_id = manifest.job_id
    log.info("Processing job %s (%d images, mode=%s)", job_id, len(manifest.images), manifest.thinking_mode)

    # Load or initialise progress
    progress_blob = f"{job_id}/progress.json"
    prog_data = gcs_read_json(gcs, batch_bucket, progress_blob)
    if prog_data:
        progress = JobProgress.from_dict(prog_data)
        log.info("Resuming from checkpoint: %d/%d done", len(progress.processed), progress.total)
    else:
        progress = JobProgress(
            job_id=job_id,
            total=len(manifest.images),
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    already_done = set(progress.processed) | set(progress.failed)

    for entry in manifest.images:
        if _shutdown_requested:
            log.warning("Shutdown requested, stopping job %s at checkpoint", job_id)
            gcs_write_text(gcs, batch_bucket, progress_blob, progress.to_json())
            return

        if entry.asset_id in already_done:
            continue

        thinking_mode = entry.thinking_mode or manifest.thinking_mode

        try:
            log.info("  → %s [%s]", entry.asset_id, thinking_mode)
            image_bytes = gcs_download_bytes(gcs, entry.gcs_path)
            image_b64 = resize_and_encode(image_bytes)
            result = infer_image(openai_client, image_b64, entry.asset_id, thinking_mode, model_name)
        except Exception as e:
            log.error("Failed to process %s: %s", entry.asset_id, e)
            result = ExtractionResult.error_result(entry.asset_id, str(e))

        # Write result
        result_blob = f"{manifest.result_prefix}{entry.asset_id}.json"
        gcs_write_text(gcs, batch_bucket, result_blob, result.to_json())

        # Update progress
        if result.error:
            progress.failed.append(entry.asset_id)
        else:
            progress.processed.append(entry.asset_id)

        # Checkpoint after every image
        gcs_write_text(gcs, batch_bucket, progress_blob, progress.to_json())

    progress.completed = True
    progress.finished_at = datetime.now(timezone.utc).isoformat()
    gcs_write_text(gcs, batch_bucket, progress_blob, progress.to_json())

    log.info(
        "Job %s complete: %d ok, %d failed",
        job_id, len(progress.processed), len(progress.failed)
    )

    # Optional callback
    if manifest.callback_url:
        try:
            requests.post(manifest.callback_url, json=progress.to_json(), timeout=10)
        except Exception as e:
            log.warning("Callback to %s failed: %s", manifest.callback_url, e)


# ── Self-terminate ────────────────────────────────────────────────────────────

def self_stop_vm(project_id: str, zone: str, instance_name: str) -> None:
    """Stop this VM (not delete — returns to TERMINATED state, costs $0)."""
    log.info("All jobs done. Stopping VM %s…", instance_name)
    try:
        subprocess.run(
            [
                "gcloud", "compute", "instances", "stop", instance_name,
                f"--zone={zone}",
                f"--project={project_id}",
                "--quiet",
            ],
            check=True,
            timeout=60,
        )
    except Exception as e:
        log.error("Failed to stop VM: %s", e)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phi-4-RV batch inference runner")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--batch-bucket", required=True, help="GCS bucket name for batch jobs")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--instance-name", default="phi4-worker")
    parser.add_argument("--model-name", default="phi4-rv", help="Model name served by vLLM")
    parser.add_argument("--no-self-stop", action="store_true", help="Don't stop VM after completion (for testing)")
    args = parser.parse_args()

    # Wait for vLLM to be ready
    wait_for_vllm()

    gcs = storage.Client(project=args.project)
    openai_client = OpenAI(base_url=VLLM_BASE_URL, api_key="unused")

    idle_polls = 0
    while not _shutdown_requested:
        pending = list_pending_manifests(gcs, args.batch_bucket)

        if not pending:
            idle_polls += 1
            log.info("No pending jobs (%d/%d idle polls)", idle_polls, MAX_IDLE_POLLS)
            if idle_polls >= MAX_IDLE_POLLS:
                log.info("Idle limit reached, shutting down")
                break
            time.sleep(POLL_INTERVAL_S)
            continue

        idle_polls = 0  # reset on work found
        for manifest_blob in pending:
            if _shutdown_requested:
                break
            process_job(gcs, openai_client, args.batch_bucket, manifest_blob, args.model_name)

    if not args.no_self_stop:
        self_stop_vm(args.project, args.zone, args.instance_name)
    else:
        log.info("--no-self-stop set, VM will keep running")


if __name__ == "__main__":
    main()
