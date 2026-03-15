#!/bin/bash
# =============================================================================
# startup-script.sh — runs on the phi4-worker VM at every boot
#
# Steps:
#   1. Install system deps (CUDA drivers already present in Deep Learning VM)
#   2. Install Python packages (vLLM, runner deps)
#   3. Download / restore Phi-4-RV-15B AWQ model (cached on persistent disk)
#   4. Start vLLM OpenAI-compatible server (background)
#   5. Start phi4_runner.py (foreground, self-terminates VM when done)
#
# Environment (set via VM metadata or Terraform):
#   GCP_PROJECT       — GCP project ID
#   BATCH_BUCKET      — GCS bucket for batch jobs
#   MODEL_BUCKET      — GCS bucket for model cache (optional)
#   HF_MODEL_ID       — HuggingFace model ID (default: see below)
#   MODEL_DIR         — Local path for model (default: /mnt/data/phi4-rv-awq)
#   THINKING_MODE     — nothink | think (runner default, per-image overrides this)
#   RUNNER_REPO       — GCS path to runner code (gs://bucket/runner.tar.gz)
# =============================================================================
set -euo pipefail

log() { echo "[startup $(date +%H:%M:%S)] $*" | tee -a /var/log/phi4-startup.log; }

# ── Read metadata ─────────────────────────────────────────────────────────────
meta() {
  curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
    -H "Metadata-Flavor: Google" 2>/dev/null || echo ""
}

GCP_PROJECT=$(meta GCP_PROJECT)
BATCH_BUCKET=$(meta BATCH_BUCKET)
MODEL_BUCKET=$(meta MODEL_BUCKET)
HF_MODEL_ID=$(meta HF_MODEL_ID)
MODEL_DIR=$(meta MODEL_DIR)
RUNNER_REPO=$(meta RUNNER_REPO)

: "${GCP_PROJECT:?GCP_PROJECT metadata not set}"
: "${BATCH_BUCKET:?BATCH_BUCKET metadata not set}"
: "${HF_MODEL_ID:=microsoft/Phi-4-reasoning-vision-15B}"
: "${MODEL_DIR:=/mnt/data/phi4-rv-awq}"
: "${RUNNER_REPO:=}"

log "=== phi4-worker startup ==="
log "Project:      $GCP_PROJECT"
log "Batch bucket: $BATCH_BUCKET"
log "Model dir:    $MODEL_DIR"
log "HF model:     $HF_MODEL_ID"

# ── System packages ───────────────────────────────────────────────────────────
log "Installing system packages…"
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv screen aria2

# ── Python environment ────────────────────────────────────────────────────────
VENV=/opt/phi4-env
if [ ! -d "$VENV" ]; then
  log "Creating Python venv at $VENV…"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

log "Installing Python packages…"
pip install --upgrade pip -q
pip install -q \
  "vllm>=0.6" \
  "openai>=1.30" \
  "google-cloud-storage>=2.16" \
  "Pillow>=10.0" \
  "requests>=2.31" \
  "autoawq>=0.2"

# ── Runner code ───────────────────────────────────────────────────────────────
RUNNER_DIR=/opt/phi4-runner
mkdir -p "$RUNNER_DIR"

if [ -n "$RUNNER_REPO" ]; then
  log "Downloading runner from $RUNNER_REPO…"
  gsutil cp "$RUNNER_REPO" /tmp/runner.tar.gz
  tar -xzf /tmp/runner.tar.gz -C "$RUNNER_DIR"
else
  log "No RUNNER_REPO set — runner code must already be present at $RUNNER_DIR"
  # For first-time setup, copy scripts from this repo's GCS location
  # Alternatively, bake runner code into the VM disk image
fi

# ── Model: download or restore from cache ─────────────────────────────────────
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/config.json" ]; then
  log "Model already present at $MODEL_DIR, skipping download"
else
  log "Model not found. Checking GCS model cache…"

  MODEL_CACHED=false
  if [ -n "$MODEL_BUCKET" ]; then
    GCS_MODEL_PATH="gs://${MODEL_BUCKET}/phi4-rv-awq/"
    if gsutil -q stat "${GCS_MODEL_PATH}config.json" 2>/dev/null; then
      log "Restoring model from GCS cache $GCS_MODEL_PATH…"
      gsutil -m cp -r "${GCS_MODEL_PATH}*" "$MODEL_DIR/"
      MODEL_CACHED=true
    fi
  fi

  if [ "$MODEL_CACHED" = false ]; then
    log "Downloading base model from HuggingFace: $HF_MODEL_ID"
    log "(This is a one-time step; subsequent boots load from disk)"

    # Check if an AWQ community quant exists
    AWQ_MODEL_ID="${HF_MODEL_ID}-AWQ"  # e.g. bartowski/Phi-4-reasoning-vision-15B-AWQ
    pip install -q huggingface_hub

    python3 - <<PYEOF
from huggingface_hub import snapshot_download
import os

# Try AWQ variant first; fall back to base model + quantize
models_to_try = [
    os.environ.get("AWQ_MODEL_ID", "${HF_MODEL_ID}".replace("microsoft/", "bartowski/") + "-AWQ"),
    "${HF_MODEL_ID}",
]

downloaded = False
for model_id in models_to_try:
    try:
        print(f"Trying: {model_id}")
        snapshot_download(
            repo_id=model_id,
            local_dir="${MODEL_DIR}",
            ignore_patterns=["*.bin", "original/*"],
        )
        downloaded = True
        print(f"Downloaded: {model_id}")
        break
    except Exception as e:
        print(f"Failed {model_id}: {e}")

if not downloaded:
    raise RuntimeError("Could not download any model variant")
PYEOF

    # If we got the base model (not AWQ), quantize it
    if [ ! -f "$MODEL_DIR/quantize_config.json" ]; then
      log "Quantizing base model to AWQ 4-bit (one-time, ~30-60 min)…"
      python3 - <<PYEOF
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "${MODEL_DIR}"
save_path = "${MODEL_DIR}-quantized"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(save_path)
tokenizer.save_pretrained(save_path)
print(f"Quantized model saved to {save_path}")
PYEOF
      # Use quantized path from now on
      rm -rf "$MODEL_DIR"
      mv "${MODEL_DIR}-quantized" "$MODEL_DIR"
      log "Quantization complete"
    fi

    # Cache to GCS for future boots
    if [ -n "$MODEL_BUCKET" ]; then
      log "Caching model to GCS for future boots…"
      gsutil -m cp -r "$MODEL_DIR/" "gs://${MODEL_BUCKET}/phi4-rv-awq/"
    fi
  fi
fi

log "Model ready at $MODEL_DIR"

# ── Start vLLM ────────────────────────────────────────────────────────────────
log "Starting vLLM server…"

# Determine if model is AWQ quantized
QUANT_ARGS=""
if [ -f "$MODEL_DIR/quantize_config.json" ]; then
  QUANT_ARGS="--quantization awq"
  log "Using AWQ quantization"
fi

screen -dmS vllm bash -c "
  source $VENV/bin/activate
  python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_DIR \
    --served-model-name phi4-rv \
    --dtype bfloat16 \
    $QUANT_ARGS \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 \
    --port 8000 \
    --trust-remote-code \
    2>&1 | tee -a /var/log/phi4-vllm.log
"

log "vLLM starting in background (screen session: vllm)"
log "Monitor with: screen -r vllm   or   tail -f /var/log/phi4-vllm.log"

# ── Start runner ──────────────────────────────────────────────────────────────
log "Starting phi4_runner (will wait for vLLM to be healthy)…"

cd "$RUNNER_DIR"
source "$VENV/bin/activate"

python phi4_runner.py \
  --project "$GCP_PROJECT" \
  --batch-bucket "$BATCH_BUCKET" \
  --zone "$(meta zone | awk -F/ '{print $NF}')" \
  --instance-name "$(meta name)" \
  2>&1 | tee -a /var/log/phi4-runner.log

log "Runner exited — startup script complete"
