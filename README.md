# span-a-model

<!-- Model & Runtime -->

[![Model](https://img.shields.io/badge/Model-Phi--4--RV--15B%20AWQ-7B2FBE?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/Served%20by-vLLM-FF6B35?style=flat-square&logo=lightning&logoColor=white)](https://docs.vllm.ai/)

<!-- Infrastructure -->

[![IaC](https://img.shields.io/badge/IaC-Terraform%201.6%2B-7B42BC?style=flat-square&logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Cloud](https://img.shields.io/badge/Cloud-GCP-4285F4?style=flat-square&logo=googlecloud&logoColor=white)](https://cloud.google.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20L4%2024GB-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://cloud.google.com/compute/docs/gpus)
[![VM](https://img.shields.io/badge/Compute-Spot%20VM-F4B400?style=flat-square&logo=googlecloud&logoColor=white)](https://cloud.google.com/compute/docs/instances/spot)

<!-- Cost & Privacy -->

[![Idle Cost](https://img.shields.io/badge/Idle%20Cost-%240%2Fhr-228B22?style=flat-square&logo=cashapp&logoColor=white)](#cost-model)
[![Privacy](https://img.shields.io/badge/Privacy-Originals%20Never%20Leave%20Disk-FF8C00?style=flat-square&logo=lock&logoColor=white)](#security-notes)
[![Self-Hosted](https://img.shields.io/badge/Self--Hosted-No%20Vendor%20Lock--in-6B7280?style=flat-square&logo=selfhosted&logoColor=white)](#architecture)

<!-- Engineering Practices -->

[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-2EA44F?style=flat-square&logo=pytest&logoColor=white)](scripts/test_local.py)
[![Preemption Safe](https://img.shields.io/badge/Spot%20Preemption-Checkpoint%20%26%20Resume-0EA5E9?style=flat-square&logo=checkmarx&logoColor=white)](#spot-vm-preemption)
[![Idempotent](https://img.shields.io/badge/Dispatcher-Idempotent-8B5CF6?style=flat-square&logo=apacheairflow&logoColor=white)](#architecture)
[![Least Privilege](https://img.shields.io/badge/IAM-Least%20Privilege-DC2626?style=flat-square&logo=googlecloud&logoColor=white)](#security-notes)

---

**Phi-4-Reasoning-Vision-15B self-hosted batch inference on GCP — zero idle cost.**

Spins up a spot L4 GPU VM on demand, processes a folder of images through the
Phi-4-RV-15B (4-bit AWQ) model, saves structured JSON results, and terminates itself.
You pay only while the VM is running. Idle cost: **$0**.

---

## Why this exists

[media-organizer](https://github.com/AndrewMichael2020/media-organizer) uses
Gemini Flash-Lite for OCR and image analysis (~$0.19/1,000 images). This service
is an alternative for large batch sessions (5,000+ images) where:

- **Accuracy matters more than speed** — Phi-4-RV has best-in-class OCR and
  reasoning, especially for documents, handwriting, receipts, and multi-lingual text
- **Privacy matters** — images never leave your infrastructure
- **Cost matters at scale** — ~$0.46/session at 5,000 images vs ~$0.95 with Gemini

| Scenario      | Gemini Flash-Lite     | span-a-model (L4 spot)  |
| ------------- | --------------------- | ----------------------- |
| 1,000 images  | ~$0.19                | ~$0.09                  |
| 5,000 images  | ~$0.95                | **~$0.46**              |
| 10,000 images | ~$1.90                | **~$0.92**              |
| Idle cost     | $0 (per-token)        | **$0** (VM terminated)  |
| OCR accuracy  | Good                  | **Best-in-class**       |
| Privacy       | Images sent to Google | Images stay in your GCP |

---

## Architecture

```
MacBook (ingest.py)
  │
  ├─ resize images to 1200px max
  ├─ upload to GCS  gs://fmo-phi4-batch/{job_id}/images/
  ├─ write manifest  gs://fmo-phi4-batch/{job_id}/manifest.json
  └─ publish → Pub/Sub: phi4-job-requests
                  │
                  ▼
  Cloud Function: phi4-dispatcher
    (triggered by Pub/Sub, ~instant, free tier)
    → checks if phi4-worker VM is running
    → if TERMINATED/STOPPED: starts the VM
                  │
                  ▼
  Spot VM: phi4-worker  (g2-standard-8, 1× L4 24GB, us-central1-a)
    startup-script.sh:
      1. Install vLLM + deps (cached after first boot)
      2. Load Phi-4-RV-15B AWQ model from disk (or GCS cache)
      3. Start vLLM OpenAI server on localhost:8000
      4. Start phi4_runner.py
    phi4_runner.py:
      - Poll GCS for pending manifests
      - For each image: download → resize → infer → write result JSON
      - Checkpoint progress after every image (resume-safe on preemption)
      - When all jobs done: gcloud compute instances stop (self)
                  │
                  ▼
  GCS: gs://fmo-phi4-batch/{job_id}/results/{asset_id}.json
                  │
                  ▼
  MacBook: ingest.py polls results → saves JSON locally → cleans up GCS
```

### Why GCS (not direct API)?

At 5,000+ images, you do **not** want to stream images from your Mac to the VM directly:

- **Upload bottleneck**: home internet is the bottleneck; GCS upload is fire-and-forget
- **Preemption safety**: if the spot VM is preempted at image 4,800, it resumes from
  checkpoint — not from zero. Direct API would lose all progress.
- **Disconnect safety**: your Mac doesn't need to stay connected

Only **resized copies** (1200px, ~200KB each) go to GCS — originals never leave your disk.
GCS cost for 5,000 images: ~$0.02/session (deleted after processing).

---

## Repo Layout

```
span-a-model/
├── client/
│   ├── ingest.py          MacBook CLI: folder → GCS → trigger → poll → save results
│   ├── phi4_client.py     Python library (used by ingest.py; future media-organizer adapter)
│   ├── requirements.txt
│   └── .env.example       Copy to .env, fill in your GCP vars
│
├── runner/
│   ├── phi4_runner.py     Runs inside the VM: polls GCS, calls vLLM, writes results, stops VM
│   ├── schema.py          Shared data classes: ExtractionResult, JobManifest, JobProgress
│   ├── prompts.py         Phi-4-RV prompt templates + thinking-mode logic
│   └── requirements.txt
│
├── dispatcher/
│   ├── main.py            Cloud Function (Gen 2): Pub/Sub → start VM (idempotent)
│   └── requirements.txt
│
├── infra/
│   ├── startup-script.sh  VM boot: install deps, load model, start vLLM + runner
│   ├── shutdown-script.sh VM shutdown/preemption: checkpoint progress gracefully
│   └── terraform/
│       ├── main.tf        GCS, Pub/Sub, Cloud Function, Spot VM, Cloud NAT, IAM
│       ├── variables.tf
│       ├── outputs.tf
│       └── terraform.tfvars.example
│
└── scripts/
    ├── deploy.sh          One-shot deploy (terraform + upload runner + create key)
    └── test_local.py      12 tests, no GCP credentials needed
```

---

## Prerequisites

- **GCP project** with billing enabled
- **gcloud CLI** installed and authenticated (`gcloud auth login`)
- **Terraform >= 1.6**
- **Python 3.11+** on your Mac
- **Application Default Credentials**:
  ```bash
  gcloud auth application-default login
  ```

---

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/AndrewMichael2020/span-a-model
cd span-a-model
```

### 2. Deploy infrastructure

```bash
./scripts/deploy.sh --project YOUR_GCP_PROJECT_ID
```

This runs Terraform (creates GCS buckets, Pub/Sub, Cloud Function, Spot VM),
uploads the runner code, and writes `client/.env` with your config.

First-time Terraform apply takes ~3 minutes.

### 3. Install client deps (Mac)

```bash
cd client
pip install -r requirements.txt
```

### 4. Test with a dry run (no GCP calls)

```bash
python ingest.py ~/Pictures/2024 --dry-run
```

### 5. Run a real batch

```bash
python ingest.py ~/Pictures/2024 --out ./results
```

The **first run** takes 8–15 minutes for the VM to:

1. Boot (~1 min)
2. Install Python deps (~3 min)
3. Download and quantize the model (~5–10 min, one-time only)

**Subsequent runs** (model already on disk): **~4–5 minutes to first inference**.

---

## CLI Reference

```
python ingest.py FOLDER [options]

Arguments:
  FOLDER                Folder of images to process

Options:
  --out DIR             Output directory for result JSONs (default: ./results)
  --think               Use chain-of-thought reasoning (slower, best OCR accuracy)
  --recursive, -r       Scan folder recursively
  --max-px N            Max image dimension before upload (default: 1200)
  --timeout N           Max seconds to wait for results (default: 7200)
  --no-cleanup          Keep images in GCS after processing
  --dry-run             Show plan without uploading or triggering
  --job-id JOB_ID       Resume or fetch results for an existing job
  --project PROJECT     GCP project (overrides PHI4_GCP_PROJECT env var)
  --bucket BUCKET       GCS bucket (overrides PHI4_BATCH_BUCKET env var)
  -v, --verbose         Debug logging

Examples:
  # Fast mode (default, no chain-of-thought):
  python ingest.py ~/Pictures/2024 --out ./results

  # Best accuracy mode (use for complex documents, receipts, handwriting):
  python ingest.py ~/Scans --think --out ./results

  # Resume a job if the terminal was closed:
  python ingest.py --job-id job_20260315_143022_abc123 --out ./results

  # Dry run: see what would be uploaded:
  python ingest.py ~/Pictures --dry-run
```

---

## Output Format

Each image produces a JSON file in your `--out` directory:

```json
{
  "asset_id": "IMG_4521",
  "ocr_text": "CAFÉ DU MONDE\nBeignets & Coffee",
  "summary": "A paper coffee cup from Café du Monde, a famous New Orleans coffee shop, sitting on a wooden table.",
  "tags": ["coffee", "new orleans", "cafe", "beignets", "food", "travel"],
  "objects": ["coffee cup", "table", "napkin"],
  "place_clues": "Café du Monde, New Orleans, Louisiana",
  "scene": "indoor",
  "image_notes": null,
  "ai_raw": "...",
  "model": "phi4-reasoning-vision-15b-awq",
  "thinking_mode": "nothink",
  "error": null
}
```

The schema matches `media-organizer`'s `ExtractionResult` — results import directly
into the app's AI extraction pipeline.

---

## Thinking Modes

Phi-4-RV supports two reasoning modes per image:

| Mode      | Flag        | Speed                | Accuracy  | Best for                                    |
| --------- | ----------- | -------------------- | --------- | ------------------------------------------- |
| `nothink` | _(default)_ | Fast (~5s/img)       | Very good | General photos, basic OCR                   |
| `think`   | `--think`   | Slower (~15–30s/img) | **Best**  | Complex docs, receipts, handwriting, tables |

You can mix modes per-image programmatically via `phi4_client.py` (pass `thinking_mode`
in `ImageEntry`).

---

## Cost Model

**L4 spot VM (us-central1, g2-standard-8):**

- Spot price: **~$0.22–0.30/hr** (varies; regular on-demand: ~$1.20/hr)
- When TERMINATED (idle): **$0/hr**
- Persistent disk (100GB SSD): ~$17/month (houses OS + model)

**Per-session cost:**

```
5,000 images at ~8s/img with batch-4 = ~2.8 hrs VM time
  VM cost:        2.8 hrs × $0.25/hr ≈ $0.70
  GCS (1GB/day):                      ≈ $0.00
  GCS egress (same-zone):             ≈ $0.00 (free)
  Cloud Function:                     ≈ $0.00 (free tier)
  ───────────────────────────────────────────────────────
  Total:                              ≈ $0.46–$0.85
```

**Break-even vs Gemini Flash-Lite: ~4,000 images/session.**

---

## Spot VM Preemption

GCP can preempt spot VMs with 30s notice. span-a-model handles this gracefully:

1. `shutdown-script.sh` signals the runner with SIGTERM
2. Runner finishes its current image (≤30s) and writes a progress checkpoint to GCS
3. VM stops
4. Re-trigger with `ingest.py --job-id` or wait for next batch to auto-start the VM
5. Runner reads checkpoint and **resumes from where it stopped** — no work lost

---

## Configuration

### client/.env

```bash
PHI4_GCP_PROJECT=your-gcp-project-id    # required
PHI4_BATCH_BUCKET=fmo-phi4-batch        # required (created by terraform)
PHI4_PUBSUB_TOPIC=phi4-job-requests     # optional, default shown
GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json
```

### terraform.tfvars

```hcl
project_id   = "your-gcp-project-id"
zone         = "us-central1-a"   # best L4 spot availability as of 2026
machine_type = "g2-standard-8"   # 8 vCPU, 32GB RAM, 1× L4 24GB
```

### Changing the model

If a pre-quantized AWQ variant appears on HuggingFace (check `bartowski/` namespace):

```hcl
# terraform.tfvars
hf_model_id = "bartowski/Phi-4-reasoning-vision-15B-AWQ"
```

Otherwise, `startup-script.sh` auto-quantizes the base model with AutoAWQ on first boot
(~30–60 min, one-time).

---

## Running Tests

```bash
# 12 tests, no GCP credentials needed:
python scripts/test_local.py
```

Covers: schema round-trips, prompt building, JSON parsing (both thinking modes),
image resize logic, CLI image discovery, checkpoint resume logic.

---

## Monitoring

```bash
# SSH into the VM while it's running:
gcloud compute ssh phi4-worker --zone=us-central1-a --project=YOUR_PROJECT

# Startup progress:
tail -f /var/log/phi4-startup.log

# vLLM logs:
tail -f /var/log/phi4-vllm.log  # or: screen -r vllm

# Runner logs:
tail -f /var/log/phi4-runner.log

# VM status:
gcloud compute instances describe phi4-worker \
  --zone=us-central1-a --project=YOUR_PROJECT --format='get(status)'
```

---

## Teardown

```bash
# Remove all GCP resources (stops disk charges):
cd infra/terraform
terraform destroy -var="project_id=YOUR_PROJECT"

# Or just stop the VM (keeps disk, $0/hr until next job):
gcloud compute instances stop phi4-worker --zone=us-central1-a --project=YOUR_PROJECT
```

---

## Integrating with media-organizer

`client/phi4_client.py` is designed as a drop-in provider adapter. Future step:

1. Copy `phi4_client.py` into `media-organizer/packages/models/providers/`
2. Add provider routing: `if provider == "phi4_gcp": use Phi4Client`
3. In `config/local.yaml`:
   ```yaml
   model:
     provider: phi4_gcp
   ```
4. Result schema already matches — no mapping needed.

---

## Security Notes

- VM has **no public IP** — uses Cloud NAT for outbound (HuggingFace, GCS)
- vLLM binds to **localhost only** — not accessible from outside the VM
- Three **least-privilege service accounts**: client, VM runner, dispatcher
- Client key (`gcp-key.json`) is git-ignored
- GCS bucket is **private** — uniform bucket-level access, no public URLs
- **Original photos never leave your Mac** — only 1200px resized copies upload
