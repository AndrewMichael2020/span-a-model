variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for the Spot VM (L4 availability: us-central1-a is best)"
  type        = string
  default     = "us-central1-a"
}

variable "batch_bucket_name" {
  description = "GCS bucket for batch job manifests, images, and results"
  type        = string
  default     = "fmo-phi4-batch"
}

variable "model_bucket_name" {
  description = "GCS bucket for persistent model cache (avoids re-downloading on each boot)"
  type        = string
  default     = "fmo-phi4-models"
}

variable "pubsub_topic_name" {
  description = "Pub/Sub topic that triggers the dispatcher Cloud Function"
  type        = string
  default     = "phi4-job-requests"
}

variable "vm_name" {
  description = "Name of the Spot VM instance"
  type        = string
  default     = "phi4-worker"
}

variable "machine_type" {
  description = "VM machine type. g2-standard-8 = 8 vCPU, 32GB RAM, 1x L4 24GB"
  type        = string
  default     = "g2-standard-8"
}

variable "vm_disk_size_gb" {
  description = "Boot disk size in GB (needs ~80GB: OS + vLLM + 10GB model)"
  type        = number
  default     = 100
}

variable "hf_model_id" {
  description = "HuggingFace model ID. Use an AWQ variant if available."
  type        = string
  # Community AWQ quant if available; falls back to base model + auto-quantize
  default     = "microsoft/Phi-4-reasoning-vision-15B"
}

variable "runner_gcs_path" {
  description = "GCS path to runner code tarball (gs://bucket/runner.tar.gz). Leave empty to bake into disk."
  type        = string
  default     = ""
}
