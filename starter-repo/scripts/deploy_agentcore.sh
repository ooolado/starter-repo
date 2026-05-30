#!/usr/bin/env bash
# Monk Technologies - deploy Project 2 (Ticket Triage) to AWS Bedrock AgentCore.
# Day 7. Requires bedrock-agentcore-starter-toolkit installed.

set -euo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$*"; }

# Load .env if present
if [[ -f .env ]]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs -0 2>/dev/null || true)
fi

NAME="${AGENT_NAME:-monk-ticket-triage}"
ENTRYPOINT="${ENTRYPOINT:-deploy/agentcore_entrypoint.py}"
REGION="${AWS_REGION:-us-east-1}"

bold "Monk Technologies - deploy $NAME to AWS Bedrock AgentCore"
echo "  region=$REGION  entrypoint=$ENTRYPOINT"
echo

if ! command -v agentcore >/dev/null 2>&1; then
    echo "agentcore CLI not found. Installing..."
    uv pip install bedrock-agentcore-starter-toolkit
fi

bold "1. Configure"
agentcore configure \
    --name "$NAME" \
    --entrypoint "$ENTRYPOINT" \
    --runtime python3.11 \
    --memory 1024 \
    --timeout 600 \
    --region "$REGION"

bold "2. Launch"
agentcore launch --name "$NAME"

bold "Done."
echo "Tail logs with: agentcore logs $NAME --follow"
