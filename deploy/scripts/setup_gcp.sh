#!/usr/bin/env bash
# One-time GCP setup: buckets, Pub/Sub notification + subscription with
# dead-letter, BigQuery table, GKE L4 Spot node pool, Workload Identity.
# Review every command before running; nothing here is idempotent-safe for
# production without checking existing resources first.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT}"
REGION="${REGION:-us-central1}"
CLUSTER="${CLUSTER:-physical-ai}"
RAW_BUCKET="${RAW_BUCKET:-project-raw}"
OUT_BUCKET="${OUT_BUCKET:-project-clean}"
MODELS_BUCKET="${MODELS_BUCKET:-project-models}"
TOPIC="${TOPIC:-raw-video-finalize}"
SUB="${SUB:-raw-video-finalize-sub}"
DATASET="${DATASET:-physical_ai}"
GSA="pap-worker@${PROJECT}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT}"

gcloud pubsub topics create "${TOPIC}"
gcloud pubsub topics create "${TOPIC}-dlq"
gcloud storage buckets notifications create "gs://${RAW_BUCKET}" \
  --topic="${TOPIC}" --event-types=OBJECT_FINALIZE --payload-format=json
# Long sessions: max ack deadline; the client library extends leases up to
# CloudConfig.max_lease_seconds.
gcloud pubsub subscriptions create "${SUB}" --topic="${TOPIC}" \
  --ack-deadline=600 --message-retention-duration=7d \
  --dead-letter-topic="${TOPIC}-dlq" --max-delivery-attempts=5 \
  --min-retry-delay=60s --max-retry-delay=600s

bq --location="${REGION}" mk --dataset "${PROJECT}:${DATASET}" || true
bq mk --table --time_partitioning_field=event_time --clustering_fields=session_id,reason \
  "${PROJECT}:${DATASET}.frame_audit_log" deploy/bigquery/frame_audit_log.json

# L4 Spot node pool (g2-standard-8 = 1x L4, 8 vCPU, 32 GB), scales to zero.
gcloud container node-pools create l4-spot \
  --cluster="${CLUSTER}" --region="${REGION}" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1,gpu-driver-version=latest \
  --spot --enable-autoscaling --min-nodes=0 --max-nodes=4 --num-nodes=0 \
  --node-taints=cloud.google.com/gke-spot=true:NoSchedule \
  --workload-metadata=GKE_METADATA \
  --ephemeral-storage-local-ssd=count=1

# Workload Identity: KSA pap/pap-worker -> GSA.
gcloud iam service-accounts create pap-worker --display-name="Physical AI pipeline worker" || true
for role in roles/pubsub.subscriber roles/bigquery.dataEditor roles/monitoring.viewer; do
  gcloud projects add-iam-policy-binding "${PROJECT}" --member="serviceAccount:${GSA}" --role="${role}" --condition=None
done
gcloud storage buckets add-iam-policy-binding "gs://${RAW_BUCKET}" --member="serviceAccount:${GSA}" --role=roles/storage.objectViewer
gcloud storage buckets add-iam-policy-binding "gs://${MODELS_BUCKET}" --member="serviceAccount:${GSA}" --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding "gs://${OUT_BUCKET}" --member="serviceAccount:${GSA}" --role=roles/storage.objectAdmin
gcloud iam service-accounts add-iam-policy-binding "${GSA}" \
  --role=roles/iam.workloadIdentityUser --member="serviceAccount:${PROJECT}.svc.id.goog[pap/pap-worker]"
