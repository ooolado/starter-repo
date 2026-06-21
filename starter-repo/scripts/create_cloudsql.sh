#!/usr/bin/env bash
# Monk Technologies - provision Cloud SQL PostgreSQL + pgvector for Project 1.
# Idempotent: safe to re-run. Supports --delete to tear down.

set -euo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
skip() { printf "  \033[33m⊘\033[0m %s\n" "$*"; }
err()  { printf "  \033[31mERR\033[0m %s\n" "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_SQL="$SCRIPT_DIR/postgres-init.sql"

CLOUDSQL_INSTANCE="${CLOUDSQL_INSTANCE:-monk-postgres}"
REGION="${REGION:-asia-south1}"
DB_NAME="${DB_NAME:-monk}"
DB_PASS="${DB_PASS:-$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)}"

DELETE=false
for arg in "$@"; do
    case "$arg" in
        --delete) DELETE=true ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [--delete]

Provision a minimal Cloud SQL PostgreSQL 15 instance with pgvector.

Environment overrides:
  CLOUDSQL_INSTANCE  (default: monk-postgres)
  REGION             (default: asia-south1)
  DB_NAME            (default: monk)
  DB_PASS            (default: random)

  --delete           Tear down the instance and exit.
EOF
            exit 0
            ;;
    esac
done

if ! command -v gcloud >/dev/null 2>&1; then
    err "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
    err "No default GCP project. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

CONNECTION_NAME="${PROJECT}:${REGION}:${CLOUDSQL_INSTANCE}"

if $DELETE; then
    bold "Deleting Cloud SQL instance: $CLOUDSQL_INSTANCE"
    if gcloud sql instances describe "$CLOUDSQL_INSTANCE" --project="$PROJECT" >/dev/null 2>&1; then
        gcloud sql instances delete "$CLOUDSQL_INSTANCE" --project="$PROJECT" --quiet
        ok "deleted $CLOUDSQL_INSTANCE"
    else
        skip "instance $CLOUDSQL_INSTANCE does not exist"
    fi
    exit 0
fi

bold "Monk Technologies - Cloud SQL setup"
echo "  project=$PROJECT  instance=$CLOUDSQL_INSTANCE  region=$REGION  db=$DB_NAME"
echo

# ── Step 0: Enable Cloud SQL Admin API ───────────────────────────────────────
bold "Step 0: Cloud SQL Admin API"
if gcloud services list --enabled --project="$PROJECT" --format='value(name)' \
    | grep -q 'sqladmin.googleapis.com'; then
    skip "sqladmin.googleapis.com already enabled"
else
    gcloud services enable sqladmin.googleapis.com --project="$PROJECT"
    ok "sqladmin.googleapis.com enabled"
fi
echo

# ── Step 1: Create instance ──────────────────────────────────────────────────
bold "Step 1: Cloud SQL instance"
if gcloud sql instances describe "$CLOUDSQL_INSTANCE" --project="$PROJECT" >/dev/null 2>&1; then
    skip "instance $CLOUDSQL_INSTANCE already exists"
else
    gcloud sql instances create "$CLOUDSQL_INSTANCE" \
        --project="$PROJECT" \
        --database-version=POSTGRES_15 \
        --tier=db-f1-micro \
        --region="$REGION" \
        --root-password="$DB_PASS" \
        --storage-auto-increase \
        --quiet
    ok "created $CLOUDSQL_INSTANCE (db-f1-micro, Postgres 15)"
fi

gcloud sql users set-password postgres \
    --instance="$CLOUDSQL_INSTANCE" \
    --project="$PROJECT" \
    --password="$DB_PASS" \
    --quiet
ok "postgres password set"

echo "  waiting for instance to become RUNNABLE..."
while true; do
    STATE="$(gcloud sql instances describe "$CLOUDSQL_INSTANCE" --project="$PROJECT" \
        --format='value(state)' 2>/dev/null || echo "UNKNOWN")"
    [[ "$STATE" == "RUNNABLE" ]] && break
    sleep 5
done
ok "instance RUNNABLE"
echo

# ── Step 2: Create database ──────────────────────────────────────────────────
bold "Step 2: Database"
if gcloud sql databases list --instance="$CLOUDSQL_INSTANCE" --project="$PROJECT" \
    --format='value(name)' | grep -qx "$DB_NAME"; then
    skip "database $DB_NAME already exists"
else
    gcloud sql databases create "$DB_NAME" \
        --instance="$CLOUDSQL_INSTANCE" \
        --project="$PROJECT" \
        --quiet
    ok "created database $DB_NAME"
fi
echo

# ── Step 3: Run postgres-init.sql ─────────────────────────────────────────────
bold "Step 3: pgvector + schema (postgres-init.sql)"

MY_IP="$(curl -4 -s --max-time 10 ifconfig.me || true)"
if [[ -z "$MY_IP" ]]; then
    err "Could not determine public IPv4 (curl -4 ifconfig.me)"
    exit 1
fi

mapfile -t EXISTING_NETS < <(
    gcloud sql instances describe "$CLOUDSQL_INSTANCE" --project="$PROJECT" \
        --format='value(settings.ipConfiguration.authorizedNetworks.value)' 2>/dev/null \
        | grep -v '^$' || true
)
SAVED_NETWORKS=""
if ((${#EXISTING_NETS[@]})); then
    SAVED_NETWORKS="$(IFS=,; echo "${EXISTING_NETS[*]}")"
fi

if [[ -n "$SAVED_NETWORKS" ]]; then
    PATCH_NETS="${SAVED_NETWORKS},${MY_IP}/32"
else
    PATCH_NETS="${MY_IP}/32"
fi

gcloud sql instances patch "$CLOUDSQL_INSTANCE" --project="$PROJECT" \
    --authorized-networks="$PATCH_NETS" --quiet
ok "temporarily authorised $MY_IP/32"

restore_networks() {
    if [[ -n "$SAVED_NETWORKS" ]]; then
        gcloud sql instances patch "$CLOUDSQL_INSTANCE" --project="$PROJECT" \
            --authorized-networks="$SAVED_NETWORKS" --quiet
    else
        gcloud sql instances patch "$CLOUDSQL_INSTANCE" --project="$PROJECT" \
            --clear-authorized-networks --quiet
    fi
}
trap restore_networks EXIT

PUBLIC_IP="$(gcloud sql instances describe "$CLOUDSQL_INSTANCE" --project="$PROJECT" \
    --format='value(ipAddresses[0].ipAddress)')"
if [[ -z "$PUBLIC_IP" ]]; then
    err "No public IP on instance — cannot run init SQL remotely"
    exit 1
fi

echo "  waiting for network patch to propagate..."
sleep 10

export PGPASSWORD="$DB_PASS"
if command -v psql >/dev/null 2>&1; then
    psql -h "$PUBLIC_IP" -U postgres -d "$DB_NAME" \
        -v ON_ERROR_STOP=1 -f "$INIT_SQL"
    ok "postgres-init.sql applied via psql"
else
    gcloud sql connect "$CLOUDSQL_INSTANCE" \
        --user=postgres \
        --database="$DB_NAME" \
        --project="$PROJECT" \
        --quiet < "$INIT_SQL"
    ok "postgres-init.sql applied via gcloud sql connect"
fi

restore_networks
trap - EXIT
skip "removed temporary IP authorisation ($MY_IP/32)"
echo

# ── Output DSNs ───────────────────────────────────────────────────────────────
CLOUD_RUN_DSN="postgresql://postgres:${DB_PASS}@/${DB_NAME}?host=/cloudsql/${CONNECTION_NAME}"
DIRECT_DSN="postgresql://postgres:${DB_PASS}@${PUBLIC_IP}:5432/${DB_NAME}"

bold "Done — add to .env / Secret Manager"
echo
echo "  Cloud Run DSN (unix socket):"
echo "    POSTGRES_DSN=${CLOUD_RUN_DSN}"
echo
echo "  Instance connection name (--add-cloudsql-instances):"
echo "    ${CONNECTION_NAME}"
echo
echo "  Direct-connect DSN (local debugging):"
echo "    ${DIRECT_DSN}"
echo
echo "  DB password (postgres user):"
echo "    ${DB_PASS}"
