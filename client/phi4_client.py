"""
phi4_client.py — programmatic client library (Mac / server side).

Used by:
  - ingest.py CLI prototype
  - media-organizer's future phi4_gcp provider adapter

High-level API:
    client = Phi4Client.from_env()
    job_id = client.submit_batch(image_paths, thinking_mode="nothink")
    results = client.wait_for_results(job_id)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterator, Optional

from google.cloud import pubsub_v1, storage
from PIL import Image

log = logging.getLogger("phi4_client")

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MAX_PX = 1200
DEFAULT_JPEG_QUALITY = 85
DEFAULT_POLL_INTERVAL_S = 20
DEFAULT_TIMEOUT_S = 7200      # 2 hours max wait


class Phi4Client:
    """
    Client for the span-a-model batch inference service.

    Environment variables (or pass explicitly):
      PHI4_GCP_PROJECT   — GCP project ID
      PHI4_BATCH_BUCKET  — GCS bucket name (e.g. fmo-phi4-batch)
      PHI4_PUBSUB_TOPIC  — Pub/Sub topic (e.g. phi4-job-requests)
    """

    def __init__(
        self,
        project: str,
        batch_bucket: str,
        pubsub_topic: str,
        max_px: int = DEFAULT_MAX_PX,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        self.project = project
        self.batch_bucket = batch_bucket
        self.pubsub_topic = pubsub_topic
        self.max_px = max_px
        self.jpeg_quality = jpeg_quality

        self._gcs = storage.Client(project=project)
        self._publisher = pubsub_v1.PublisherClient()
        self._topic_path = self._publisher.topic_path(project, pubsub_topic)

    @classmethod
    def from_env(cls) -> "Phi4Client":
        project = os.environ["PHI4_GCP_PROJECT"]
        bucket  = os.environ["PHI4_BATCH_BUCKET"]
        topic   = os.environ.get("PHI4_PUBSUB_TOPIC", "phi4-job-requests")
        return cls(project=project, batch_bucket=bucket, pubsub_topic=topic)

    # ── Image preparation ─────────────────────────────────────────────────────

    def _resize_image(self, path: Path) -> bytes:
        """Open, resize to max_px on longest side, encode as JPEG."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > self.max_px:
            scale = self.max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    # ── GCS upload ────────────────────────────────────────────────────────────

    def _upload_image(self, job_id: str, asset_id: str, image_bytes: bytes) -> str:
        """Upload resized image bytes to GCS. Returns gs:// URI."""
        blob_name = f"{job_id}/images/{asset_id}.jpg"
        bucket = self._gcs.bucket(self.batch_bucket)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type="image/jpeg")
        return f"gs://{self.batch_bucket}/{blob_name}"

    def _write_manifest(self, job_id: str, manifest: dict) -> None:
        blob_name = f"{job_id}/manifest.json"
        bucket = self._gcs.bucket(self.batch_bucket)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(manifest, indent=2), content_type="application/json")

    # ── Pub/Sub trigger ───────────────────────────────────────────────────────

    def _publish_job(self, job_id: str) -> None:
        message = json.dumps({
            "job_id": job_id,
            "batch_bucket": self.batch_bucket,
            "project_id": self.project,
        }).encode()
        future = self._publisher.publish(self._topic_path, message)
        future.result(timeout=30)
        log.info("Published job %s to %s", job_id, self._topic_path)

    # ── High-level API ────────────────────────────────────────────────────────

    def submit_batch(
        self,
        image_paths: list[Path],
        thinking_mode: str = "nothink",
        callback_url: Optional[str] = None,
        job_id: Optional[str] = None,
        on_progress: Optional[callable] = None,
    ) -> str:
        """
        Resize, upload, and queue a batch of images.

        Args:
            image_paths:   List of local file paths (JPEG, PNG, HEIC, etc.)
            thinking_mode: "nothink" (fast) or "think" (best accuracy)
            callback_url:  Optional HTTP endpoint to POST when job completes
            job_id:        Override auto-generated ID (useful for resuming)
            on_progress:   Callable(done, total) called after each upload

        Returns:
            job_id string
        """
        if not job_id:
            job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        log.info("Submitting batch job %s: %d images", job_id, len(image_paths))

        images_meta = []
        for i, path in enumerate(image_paths):
            path = Path(path)
            asset_id = path.stem  # filename without extension as asset ID

            try:
                image_bytes = self._resize_image(path)
                gcs_path = self._upload_image(job_id, asset_id, image_bytes)
            except Exception as e:
                log.warning("Skipping %s: %s", path, e)
                continue

            images_meta.append({
                "asset_id": asset_id,
                "gcs_path": gcs_path,
                "thinking_mode": thinking_mode,
                "original_path": str(path),
            })

            if on_progress:
                on_progress(i + 1, len(image_paths))

        if not images_meta:
            raise ValueError("No images could be processed")

        manifest = {
            "job_id": job_id,
            "images": images_meta,
            "result_bucket": self.batch_bucket,
            "result_prefix": f"{job_id}/results/",
            "image_prefix": f"{job_id}/images/",
            "thinking_mode": thinking_mode,
            "callback_url": callback_url,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_manifest(job_id, manifest)
        log.info("Manifest written for %d images", len(images_meta))

        self._publish_job(job_id)
        return job_id

    # ── Result polling ────────────────────────────────────────────────────────

    def get_progress(self, job_id: str) -> Optional[dict]:
        """Fetch the current progress JSON for a job, or None if not started."""
        try:
            bucket = self._gcs.bucket(self.batch_bucket)
            blob = bucket.blob(f"{job_id}/progress.json")
            return json.loads(blob.download_as_text())
        except Exception:
            return None

    def get_result(self, job_id: str, asset_id: str) -> Optional[dict]:
        """Fetch result JSON for one asset, or None if not ready."""
        try:
            bucket = self._gcs.bucket(self.batch_bucket)
            blob = bucket.blob(f"{job_id}/results/{asset_id}.json")
            return json.loads(blob.download_as_text())
        except Exception:
            return None

    def iter_results(self, job_id: str) -> Iterator[dict]:
        """Iterate over all result JSONs already written for a job."""
        bucket = self._gcs.bucket(self.batch_bucket)
        prefix = f"{job_id}/results/"
        for blob in bucket.list_blobs(prefix=prefix):
            try:
                yield json.loads(blob.download_as_text())
            except Exception as e:
                log.warning("Could not read result %s: %s", blob.name, e)

    def wait_for_results(
        self,
        job_id: str,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
        on_progress: Optional[callable] = None,
    ) -> list[dict]:
        """
        Block until the job completes and return all result dicts.

        Args:
            job_id:          Job to wait for
            timeout_s:       Give up after this many seconds
            poll_interval_s: How often to check GCS for progress
            on_progress:     Callable(done, total) called on each poll

        Returns:
            List of ExtractionResult dicts (one per image)
        """
        deadline = time.time() + timeout_s
        log.info("Waiting for job %s (timeout %ds, poll every %ds)…", job_id, timeout_s, poll_interval_s)

        while time.time() < deadline:
            prog = self.get_progress(job_id)

            if prog:
                done = len(prog.get("processed", [])) + len(prog.get("failed", []))
                total = prog.get("total", 0)
                log.info("Progress: %d/%d", done, total)
                if on_progress:
                    on_progress(done, total)

                if prog.get("completed"):
                    log.info("Job %s complete!", job_id)
                    return list(self.iter_results(job_id))
            else:
                log.info("Job %s: waiting for VM to start…", job_id)

            time.sleep(poll_interval_s)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout_s}s")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup_job(self, job_id: str) -> int:
        """Delete all GCS objects for a job. Returns count of deleted blobs."""
        bucket = self._gcs.bucket(self.batch_bucket)
        blobs = list(bucket.list_blobs(prefix=f"{job_id}/"))
        bucket.delete_blobs(blobs)
        log.info("Deleted %d objects for job %s", len(blobs), job_id)
        return len(blobs)
