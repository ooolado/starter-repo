#!/usr/bin/env bash
# Monk Technologies - deploy research assistant to Cloud Run.
# Idempotent: safe to re-run. Handles API enablement, secrets, IAM, deploy.

set -euo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
skip() { printf "  \033[33m⊘\033[0m %s\n" "$*"; }

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
    printf "  \033[31mERR\033[0m No default GCP project. Run: gcloud config set project YOUR_PROJECT_ID\n" >&2
    exit 1
fi

SERVICE="${SERVICE:-orlando-research-assistant}"
REGION="${REGION:-asia-south1}"
CLOUDSQL_INSTANCE="${CLOUDSQL_INSTANCE:-monk-postgres}"
CONNECTION_NAME="${PROJECT}:${REGION}:${CLOUDSQL_INSTANCE}"

# ── Load .env (safe line-by-line parser) ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" || ! "$line" == *=* ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        export "$key=$value"
    done < "$ENV_FILE"
fi

bold "Monk Technologies - Cloud Run deploy"
echo "  project=$PROJECT  service=$SERVICE  region=$REGION"
echo "  cloudsql=$CONNECTION_NAME"
echo

# ── Step 1: Enable required APIs ─────────────────────────────────────────────
bold "Step 1: Enable APIs"
APIS=(
    run.googleapis.com
    cloudbuild.googleapis.com
    artifactregistry.googleapis.com
    secretmanager.googleapis.com
    sqladmin.googleapis.com
    aiplatform.googleapis.com
)
gcloud services enable "${APIS[@]}" --project="$PROJECT" --quiet
ok "APIs enabled: ${APIS[*]}"
echo

# ── Step 2: Create secrets ────────────────────────────────────────────────────
bold "Step 2: Secrets"

push_secret() {
    local name="$1" value="$2"
    if [[ -z "$value" ]]; then
        printf "  \033[31mERR\033[0m Secret %s has empty value — set it in .env\n" "$name" >&2
        exit 1
    fi
    if gcloud secrets describe "$name" --project="$PROJECT" &>/dev/null; then
        skip "secret $name already exists"
    else
        printf "%s" "$value" | gcloud secrets create "$name" \
            --replication-policy=automatic \
            --data-file=- \
            --project="$PROJECT" \
            --quiet
        ok "created secret $name"
    fi
}

push_secret "monk-postgres-dsn" "${POSTGRES_DSN:-}"
push_secret "monk-tavily"       "${TAVILY_API_KEY:-}"
push_secret "monk-langsmith"    "${LANGSMITH_API_KEY:-}"
echo

# ── Step 3: IAM roles ────────────────────────────────────────────────────────
bold "Step 3: IAM roles"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

grant_secret_access() {
    local secret="$1"
    gcloud secrets add-iam-policy-binding "$secret" \
        --member="serviceAccount:$SA" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$PROJECT" &>/dev/null || true
}

grant_secret_access "monk-postgres-dsn"
grant_secret_access "monk-tavily"
grant_secret_access "monk-langsmith"
ok "secretmanager.secretAccessor on all secrets"

for role in roles/storage.objectViewer roles/cloudbuild.builds.builder roles/artifactregistry.writer roles/aiplatform.user; do
    gcloud projects add-iam-policy-binding "$PROJECT" \
        --member="serviceAccount:$SA" \
        --role="$role" &>/dev/null || true
done
ok "project roles: objectViewer, builds.builder, artifactregistry.writer, aiplatform.user"
echo

# ── Step 4: Deploy ────────────────────────────────────────────────────────────
bold "Step 4: Deploy to Cloud Run"

gcloud run deploy "$SERVICE" \
    --source . \
    --project="$PROJECT" \
    --region="$REGION" \
    --add-cloudsql-instances="$CONNECTION_NAME" \
    --set-env-vars="MONK_MODEL=google_vertexai:gemini-2.5-pro,MONK_EMBEDDINGS=google_vertexai:text-embedding-005,LANGSMITH_PROJECT=$SERVICE,LANGSMITH_TRACING=true,GCP_PROJECT=$PROJECT,GCP_LOCATION=us-central1" \
    --set-secrets="POSTGRES_DSN=monk-postgres-dsn:latest,TAVILY_API_KEY=monk-tavily:latest,LANGSMITH_API_KEY=monk-langsmith:latest" \
    --memory=1Gi \
    --cpu=1 \
    --timeout=600 \
    --concurrency=4 \
    --allow-unauthenticated \
    --quiet

ok "deployed $SERVICE"
echo

# ── Print URL ─────────────────────────────────────────────────────────────────
URL="$(gcloud run services describe "$SERVICE" \
    --project="$PROJECT" \
    --region="$REGION" \
    --format='value(status.url)')"

bold "Done"
echo "  Service URL: $URL"
echo "  Connection:  $CONNECTION_NAME"
