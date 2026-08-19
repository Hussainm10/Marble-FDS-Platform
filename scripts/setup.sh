#!/usr/bin/env bash
# setup.sh — Master orchestrator for Marble FDS platform provisioning.
#
# Provisions the generic platform:
#   - 12 data-model tables
#   - 12 inter-table links
#   - 12 empty custom lists
#   - 15 scenario shells (names + triggers + thresholds, no rules)
#
# To add rules, run a jurisdiction pack afterwards:
#   python scripts/rules_library/<pack_name>.py
#
# Prerequisites:
#   - Docker Compose services running: docker compose up -d
#   - CHECKMARBLE_API_KEY set in the environment
#
# Usage:
#   ./scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo " Marble FDS — Platform Setup"
echo "=============================================="

python3 "${SCRIPT_DIR}/setup.py"

echo ""
echo "✓ Platform provisioned. Next (optional):"
echo "  python scripts/rules_library/<pack_name>.py"
