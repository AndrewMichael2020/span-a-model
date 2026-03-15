terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Enable APIs ───────────────────────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "storage.googleapis.com",
    "pubsub.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "eventarc.googleapis.com",
    "logging.googleapis.com",
  ])
  service            = each.key
  disable_on_destroy = false
}

# ── GCS Buckets ───────────────────────────────────────────────────────────────

resource "google_storage_bucket" "batch" {
  name          = var.batch_bucket_name
  location      = var.region
  force_destroy = false

  # Auto-delete job objects after 7 days (safety net in case client cleanup fails)
  lifecycle_rule {
    condition { age = 7 }
    action { type = "Delete" }
  }

  uniform_bucket_level_access = true
  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "models" {
  name          = var.model_bucket_name
  location      = var.region
  force_destroy = false

  # Model files are large but rarely change — standard storage is fine
  uniform_bucket_level_access = true
  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "dispatcher_source" {
  name          = "${var.project_id}-phi4-dispatcher-src"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
  depends_on = [google_project_service.apis]
}

# ── Service Accounts ──────────────────────────────────────────────────────────

resource "google_service_account" "vm_runner" {
  account_id   = "phi4-vm-runner"
  display_name = "Phi4 Worker VM service account"
}

resource "google_service_account" "dispatcher" {
  account_id   = "phi4-dispatcher"
  display_name = "Phi4 Dispatcher Cloud Function service account"
}

resource "google_service_account" "client" {
  account_id   = "phi4-client"
  display_name = "Phi4 Client (MacBook) service account"
}

# ── IAM: VM runner needs GCS read/write + ability to stop itself ──────────────

resource "google_storage_bucket_iam_member" "vm_batch_rw" {
  bucket = google_storage_bucket.batch.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vm_runner.email}"
}

resource "google_storage_bucket_iam_member" "vm_models_r" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.vm_runner.email}"
}

# VM needs to stop itself (compute.instances.stop on itself)
resource "google_project_iam_member" "vm_compute_stop" {
  project = var.project_id
  role    = "roles/compute.instanceAdmin.v1"
  member  = "serviceAccount:${google_service_account.vm_runner.email}"
}

# ── IAM: Dispatcher needs to start the VM ─────────────────────────────────────

resource "google_project_iam_member" "dispatcher_compute" {
  project = var.project_id
  role    = "roles/compute.instanceAdmin.v1"
  member  = "serviceAccount:${google_service_account.dispatcher.email}"
}

# ── IAM: Client (MacBook) needs GCS upload + Pub/Sub publish ─────────────────

resource "google_storage_bucket_iam_member" "client_batch_rw" {
  bucket = google_storage_bucket.batch.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.client.email}"
}

resource "google_pubsub_topic_iam_member" "client_publisher" {
  topic  = google_pubsub_topic.job_requests.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.client.email}"
}

# ── Pub/Sub ───────────────────────────────────────────────────────────────────

resource "google_pubsub_topic" "job_requests" {
  name       = var.pubsub_topic_name
  depends_on = [google_project_service.apis]
}

# ── Cloud Function (Gen 2): Dispatcher ───────────────────────────────────────

data "archive_file" "dispatcher_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../dispatcher"
  output_path = "${path.module}/../../dispatcher.zip"
}

resource "google_storage_bucket_object" "dispatcher_zip" {
  name   = "dispatcher-${data.archive_file.dispatcher_zip.output_md5}.zip"
  bucket = google_storage_bucket.dispatcher_source.name
  source = data.archive_file.dispatcher_zip.output_path
}

resource "google_cloudfunctions2_function" "dispatcher" {
  name     = "phi4-dispatcher"
  location = var.region

  build_config {
    runtime     = "python311"
    entry_point = "dispatch"
    source {
      storage_source {
        bucket = google_storage_bucket.dispatcher_source.name
        object = google_storage_bucket_object.dispatcher_zip.name
      }
    }
  }

  service_config {
    min_instance_count    = 0
    max_instance_count    = 3
    available_memory      = "256M"
    timeout_seconds       = 60
    service_account_email = google_service_account.dispatcher.email

    environment_variables = {
      GCP_PROJECT = var.project_id
      GCP_ZONE    = var.zone
      VM_NAME     = var.vm_name
    }
  }

  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.job_requests.id
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  depends_on = [google_project_service.apis]
}

# Allow Eventarc to invoke the Cloud Function
resource "google_project_iam_member" "eventarc_sa" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.dispatcher.email}"
}

# ── Spot VM Instance ──────────────────────────────────────────────────────────

data "local_file" "startup_script" {
  filename = "${path.module}/../startup-script.sh"
}

data "local_file" "shutdown_script" {
  filename = "${path.module}/../shutdown-script.sh"
}

resource "google_compute_instance" "phi4_worker" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  # Spot VM: preemptible, no automatic restart, stops (not terminates) on preemption
  scheduling {
    preemptible                 = true
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
    provisioning_model          = "SPOT"
    instance_termination_action = "STOP"  # STOP keeps disk; DELETE would be cheaper but loses model cache
  }

  boot_disk {
    initialize_params {
      # Deep Learning VM with CUDA 12 pre-installed (PyTorch, CUDA drivers)
      image = "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu-debian-11"
      size  = var.vm_disk_size_gb
      type  = "pd-ssd"
    }
  }

  # L4 GPU
  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  network_interface {
    network = "default"
    # No external IP needed — use Cloud NAT for outbound (HuggingFace, GCS)
    # If your VPC doesn't have Cloud NAT, add access_config {} here for a temporary public IP
  }

  service_account {
    email  = google_service_account.vm_runner.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    startup-script  = data.local_file.startup_script.content
    shutdown-script = data.local_file.shutdown_script.content

    # Runner config (read by startup-script.sh via metadata API)
    GCP_PROJECT  = var.project_id
    BATCH_BUCKET = var.batch_bucket_name
    MODEL_BUCKET = var.model_bucket_name
    HF_MODEL_ID  = var.hf_model_id
    RUNNER_REPO  = var.runner_gcs_path
    MODEL_DIR    = "/mnt/data/phi4-rv-awq"
  }

  # Install GPU drivers on first boot (Deep Learning VM handles this automatically)
  metadata_startup_script = null  # we use metadata.startup-script instead

  labels = {
    role    = "phi4-inference"
    managed = "terraform"
  }

  # VM is created in TERMINATED state — the dispatcher starts it when a job arrives
  # To keep it terminated after terraform apply, we don't set desired_status.
  # First time: it will boot once to set up the disk, then stop itself.

  depends_on = [
    google_project_service.apis,
    google_storage_bucket.batch,
    google_storage_bucket.models,
  ]
}

# ── Cloud NAT (outbound internet for VM without public IP) ────────────────────

resource "google_compute_router" "default" {
  name    = "phi4-router"
  region  = var.region
  network = "default"
}

resource "google_compute_router_nat" "default" {
  name                               = "phi4-nat"
  router                             = google_compute_router.default.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
