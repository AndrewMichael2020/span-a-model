"""
dispatcher/main.py — Cloud Function (Gen 2, Python 3.11).

Triggered by a Pub/Sub message whenever media-organizer publishes a new batch job.
Idempotently starts the phi4-worker Spot VM if it isn't already running.

Pub/Sub message payload (JSON):
  {
    "job_id": "abc123",
    "batch_bucket": "fmo-phi4-batch",
    "project_id": "my-gcp-project"   // optional override
  }

Deploy:
  gcloud functions deploy phi4-dispatcher \\
    --gen2 --runtime python311 --region us-central1 \\
    --source dispatcher/ \\
    --entry-point dispatch \\
    --trigger-topic phi4-job-requests \\
    --service-account phi4-dispatcher@PROJECT.iam.gserviceaccount.com \\
    --set-env-vars GCP_PROJECT=PROJECT,GCP_ZONE=us-central1-a,VM_NAME=phi4-worker
"""

from __future__ import annotations

import base64
import json
import logging
import os

import functions_framework
from google.api_core.exceptions import NotFound
from google.cloud import compute_v1

log = logging.getLogger("phi4_dispatcher")
logging.basicConfig(level=logging.INFO)

# Configurable via environment variables
PROJECT_ID   = os.environ.get("GCP_PROJECT", "")
ZONE         = os.environ.get("GCP_ZONE", "us-central1-a")
VM_NAME      = os.environ.get("VM_NAME", "phi4-worker")

# States where the VM is effectively "working" — don't try to start again
ACTIVE_STATES = {"RUNNING", "STAGING", "STOPPING"}


@functions_framework.cloud_event
def dispatch(cloud_event) -> None:
    """Entry point: decode Pub/Sub message and start VM if needed."""
    try:
        raw = base64.b64decode(cloud_event.data["message"]["data"]).decode()
        payload = json.loads(raw)
        log.info("Received job request: %s", payload.get("job_id", "<no id>"))
    except Exception as e:
        log.error("Failed to decode message: %s", e)
        return

    project = payload.get("project_id") or PROJECT_ID
    if not project:
        log.error("No GCP project configured (set GCP_PROJECT env var)")
        return

    _ensure_vm_running(project, ZONE, VM_NAME)


def _ensure_vm_running(project: str, zone: str, vm_name: str) -> None:
    """Start the VM if it isn't already running. Idempotent."""
    client = compute_v1.InstancesClient()

    try:
        instance = client.get(project=project, zone=zone, instance=vm_name)
        status = instance.status
        log.info("VM %s is currently: %s", vm_name, status)

        if status in ACTIVE_STATES:
            log.info("VM already active (%s) — new job will be picked up by running runner", status)
            return

        if status in ("TERMINATED", "STOPPED"):
            log.info("Starting VM %s…", vm_name)
            op = client.start(project=project, zone=zone, instance=vm_name)
            log.info("Start operation: %s", op.name)
        else:
            log.warning("VM in unexpected state %s — taking no action", status)

    except NotFound:
        log.error(
            "VM '%s' not found in %s/%s. "
            "Run 'terraform apply' in infra/terraform/ first.",
            vm_name, project, zone,
        )
    except Exception as e:
        log.error("Error checking/starting VM: %s", e)
        raise  # Re-raise so Cloud Functions retries (Pub/Sub at-least-once)
