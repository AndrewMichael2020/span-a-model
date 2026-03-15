#!/bin/bash
# =============================================================================
# scripts/deploy.sh — one-shot deploy of the full span-a-model stack to GCP
#
# Usage:
#   ./scripts/deploy.sh --project my-project-id [--zone us-central1-a]
#
# Prerequisites:
#   - gcloud auth login && gcloud auth application-default login
#   - terraform >= 1.6 installed
#   - Billing enabled on the GCP project
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Args ──────────────────────────────────────────────────────────────────────
PROJECT=""
ZONE="us-central1-a"
REGION="us-central1"

usage() { echo "Usage: $0 --project PROJECT_ID [--zone ZONE]"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --zone)    ZONE="$2"; shift 2 ;;
    --region)  REGION="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$PROJECT" ]] && usage

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  span-a-model deploy"
echo "  Project: $PROJECT  |  Zone: $ZONE  |  Region: $REGION"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Terraform ─────────────────────────────────────────────────────────
echo "▶  Step 1/4: Terraform (GCS, Pub/Sub, VM, Cloud Function)…"

TF_DIR="$REPO_ROOT/infra/terraform"
cd "$TF_DIR"

if [ ! -f terraform.tfvars ]; then
  echo "   Creating terraform.tfvars from example…"
  sed "s/your-gcp-project-id/$PROJECT/" terraform.tfvars.example > terraform.tfvars
  sed -i "s/region            = \"us-central1\"/region            = \"$REGION\"/" terraform.tfvars
  sed -i "s/zone              = \"us-central1-a\"/zone              = \"$ZONE\"/" terraform.tfvars
fi

terraform init -upgrade -input=false
terraform apply -auto-approve -input=false \
  -var="project_id=$PROJECT" \
  -var="region=$REGION" \
  -var="zone=$ZONE"

# Capture outputs
BATCH_BUCKET=$(terraform output -raw batch_bucket)
MODEL_BUCKET=$(terraform output -raw model_bucket)
CLIENT_SA=$(terraform output -raw client_service_account)
VM_NAME=$(terraform output -raw vm_name)

cd "$REPO_ROOT"

# ── Step 2: Upload runner code ────────────────────────────────────────────────
echo ""
echo "▶  Step 2/4: Packaging and uploading runner code to GCS…"

tar -czf /tmp/phi4-runner.tar.gz -C runner .
gsutil cp /tmp/phi4-runner.tar.gz "gs://${MODEL_BUCKET}/runner.tar.gz"
echo "   Uploaded gs://${MODEL_BUCKET}/runner.tar.gz"

# Update VM metadata with runner location
gcloud compute instances add-metadata "$VM_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --metadata="RUNNER_REPO=gs://${MODEL_BUCKET}/runner.tar.gz"

# ── Step 3: Create client service account key ─────────────────────────────────
echo ""
echo "▶  Step 3/4: Creating client service account key…"

KEY_PATH="$REPO_ROOT/client/gcp-key.json"
if [ -f "$KEY_PATH" ]; then
  echo "   Key already exists at client/gcp-key.json — skipping"
else
  gcloud iam service-accounts keys create "$KEY_PATH" \
    --iam-account="$CLIENT_SA" \
    --project="$PROJECT"
  echo "   Key written to client/gcp-key.json"
fi

# ── Step 4: Write client .env ─────────────────────────────────────────────────
echo ""
echo "▶  Step 4/4: Writing client/.env…"

cat > "$REPO_ROOT/client/.env" <<EOF
PHI4_GCP_PROJECT=$PROJECT
PHI4_BATCH_BUCKET=$BATCH_BUCKET
PHI4_PUBSUB_TOPIC=phi4-job-requests
GOOGLE_APPLICATION_CREDENTIALS=$(realpath "$KEY_PATH")
EOF

echo "   client/.env written"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅  Deploy complete!"
echo ""
echo "  Test with a dry run:"
echo "    cd client"
echo "    pip install -r requirements.txt"
echo "    python ingest.py ~/Pictures --dry-run"
echo ""
echo "  Run a real batch:"
echo "    python ingest.py ~/Pictures --out ./results"
echo ""
echo "  Monitor VM startup:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT"
echo "    tail -f /var/log/phi4-startup.log"
echo "════════════════════════════════════════════════════════════"
echo ""
