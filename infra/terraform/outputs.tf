output "batch_bucket" {
  description = "GCS bucket for batch jobs"
  value       = google_storage_bucket.batch.name
}

output "model_bucket" {
  description = "GCS bucket for model cache"
  value       = google_storage_bucket.models.name
}

output "pubsub_topic" {
  description = "Pub/Sub topic name"
  value       = google_pubsub_topic.job_requests.name
}

output "vm_name" {
  description = "Spot VM instance name"
  value       = google_compute_instance.phi4_worker.name
}

output "vm_zone" {
  description = "Spot VM zone"
  value       = google_compute_instance.phi4_worker.zone
}

output "dispatcher_function" {
  description = "Cloud Function name"
  value       = google_cloudfunctions2_function.dispatcher.name
}

output "client_service_account" {
  description = "Service account email for the MacBook client"
  value       = google_service_account.client.email
}

output "vm_service_account" {
  description = "Service account email for the VM runner"
  value       = google_service_account.vm_runner.email
}

output "setup_instructions" {
  description = "Next steps after terraform apply"
  value       = <<-EOT
    ✅ Infrastructure deployed!

    Next steps:
    1. Create a key for the client service account:
       gcloud iam service-accounts keys create client/gcp-key.json \
         --iam-account=${google_service_account.client.email}

    2. Copy the runner code to GCS (so the VM can download it):
       tar -czf /tmp/runner.tar.gz -C runner .
       gsutil cp /tmp/runner.tar.gz gs://${google_storage_bucket.models.name}/runner.tar.gz
       gcloud compute instances add-metadata ${google_compute_instance.phi4_worker.name} \
         --zone=${google_compute_instance.phi4_worker.zone} \
         --metadata=RUNNER_REPO=gs://${google_storage_bucket.models.name}/runner.tar.gz

    3. Test the client:
       cd client
       cp .env.example .env
       # Edit .env with your project and bucket names
       export GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json
       python ingest.py /path/to/test/images --out ./results --dry-run

    4. Run a real batch:
       python ingest.py /path/to/photos --out ./results
  EOT
}
