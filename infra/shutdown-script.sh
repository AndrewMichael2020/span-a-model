#!/bin/bash
# =============================================================================
# shutdown-script.sh — runs on the phi4-worker VM before shutdown/preemption.
#
# On spot VM preemption, GCP gives ~30 seconds before the VM is killed.
# This script signals the runner to checkpoint its current progress so
# that the next boot can resume from where it left off.
# =============================================================================
set -euo pipefail

log() { echo "[shutdown $(date +%H:%M:%S)] $*" | tee -a /var/log/phi4-shutdown.log; }

log "=== phi4-worker shutdown/preemption triggered ==="

# Signal the runner to stop gracefully after its current image
RUNNER_PID=$(pgrep -f "phi4_runner.py" || true)
if [ -n "$RUNNER_PID" ]; then
  log "Sending SIGTERM to runner (pid $RUNNER_PID)…"
  kill -TERM "$RUNNER_PID" 2>/dev/null || true
  # Give it up to 20 seconds to write the checkpoint
  for i in $(seq 1 20); do
    sleep 1
    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
      log "Runner exited cleanly after ${i}s"
      break
    fi
  done
else
  log "Runner not running (already finished or not started)"
fi

# Stop vLLM gracefully
VLLM_PID=$(pgrep -f "vllm.entrypoints" || true)
if [ -n "$VLLM_PID" ]; then
  log "Stopping vLLM (pid $VLLM_PID)…"
  kill -TERM "$VLLM_PID" 2>/dev/null || true
fi

log "Shutdown script complete — progress checkpointed to GCS"
