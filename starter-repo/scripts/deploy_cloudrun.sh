#!/usr/bin/env bash
# Monk Technologies - deploy Project 1 (Research Assistant) to Google Cloud Run.
# Day 4, last hour. No Terraform.

set -euo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32mok\033[0m %s\n" "$*"; }

# Load .env if present
if [[ -f .env ]]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs -0 2>/dev/null || true)
fi

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project)}"
REGION="${GCP_LOCATION:-asia-south1}"
SERVICE="${SERVICE:-monk-research-assistant}"

bold "Monk Technologies - deploy $SERVICE to Cloud Run"
echo "  project=$PROJECT  region=$REGION"
echo

# Optional: push secrets to Secret Manager on first deploy
push_secret() {
    local name="$1" value="$2"
    if [[ -z "$value" ]]; then return 0; fi
    if gcloud secrets describe "$name" --project "$PROJECT" >/dev/null 2>&1; then
        printf "%s" "$value" | gcloud secrets versions add "$name" --data-file=- --project "$PROJECT" >/dev/null
    else
        gcloud secrets create "$name" --replication-policy=automatic --project "$PROJECT" >/dev/null
        printf "%s" "$value" | gcloud secrets versions add "$name" --data-file=- --project "$PROJECT" >/dev/null
    fi
    ok "secret $name"
}

bold "1. Push secrets to Secret Manager"
push_secret monk-tavily          "${TAVILY_API_KEY:-}"
push_secret monk-langsmith       "${LANGSMITH_API_KEY:-}"
push_secret monk-postgres-dsn    "${POSTGRES_DSN:-}"
echo

bold "2. Deploy"
gcloud run deploy "$SERVICE" \
    --source . \
    --project "$PROJECT" \
    --region "$REGION" \
    --allow-unauthenticated \
    --set-env-vars "MONK_MODEL=${MONK_MODEL:-google_vertexai:gemini-2.5-pro},MONK_EMBEDDINGS=${MONK_EMBEDDINGS:-google_vertexai:text-embedding-005},LANGSMITH_PROJECT=$SERVICE,LANGSMITH_TRACING=true,GCP_PROJECT=$PROJECT,GCP_LOCATION=$REGION" \
    --set-secrets "TAVILY_API_KEY=monk-tavily:latest,LANGSMITH_API_KEY=monk-langsmith:latest,POSTGRES_DSN=monk-postgres-dsn:latest" \
    --memory 1Gi \
    --cpu 1 \
    --timeout 600 \
    --concurrency 4
echo

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format='value(status.url)')
bold "Deployed."
echo "  $URL"
