#!/bin/bash
# Marble FDS - One-Click Startup Script
# Architect: Hussain Mansoor | AI/ML Engineer
# Usage: ./start.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              MARBLE FDS - FRAUD DETECTION SYSTEM             ║"
echo "║          Architect: Hussain Mansoor | AI/ML Engineer         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "[1/6] Creating .env from template..."
    cp .env.example .env
else
    echo "[1/6] .env already exists"
fi

# Start Docker containers
echo "[2/6] Starting Docker containers..."
docker compose up -d

# Wait for services
echo "[3/6] Waiting for services to initialize..."
echo "      (This may take 30-60 seconds on first run)"
sleep 20

# Wait for API
echo "[4/6] Checking API health..."
for i in {1..30}; do
    if curl -s http://localhost:8080/liveness > /dev/null 2>&1; then
        echo "      ✓ API is ready!"
        break
    fi
    echo "      Waiting... ($i/30)"
    sleep 2
done

# Run platform setup (creates 12 tables, 15 scenarios)
echo "[5/6] Setting up platform (12 tables, 15 scenario shells)..."
if command -v python3 &> /dev/null; then
    python3 scripts/setup.py 2>/dev/null || python3 scripts/setup.py
elif command -v python &> /dev/null; then
    python scripts/setup.py 2>/dev/null || python scripts/setup.py
else
    echo "      ⚠ Python not found. Run manually: python scripts/setup.py"
fi

# Publish Pakistan rules
echo "[6/6] Publishing Pakistan jurisdiction pack (87 rules)..."
if command -v python3 &> /dev/null; then
    ENABLE_ISLAMIC_SCENARIOS=true python3 scripts/rules_library/pakistan.py 2>/dev/null || \
    ENABLE_ISLAMIC_SCENARIOS=true python3 scripts/rules_library/pakistan.py
elif command -v python &> /dev/null; then
    ENABLE_ISLAMIC_SCENARIOS=true python scripts/rules_library/pakistan.py 2>/dev/null || \
    ENABLE_ISLAMIC_SCENARIOS=true python scripts/rules_library/pakistan.py
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     SETUP COMPLETE!                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Open in your browser:                                       ║"
echo "║  ─────────────────────                                       ║"
echo "║  Rule Engine UI:  http://localhost:3000                      ║"
echo "║  ML Bridge API:   http://localhost:8000/docs                 ║"
echo "║                                                              ║"
echo "║  Login credentials:                                          ║"
echo "║  ──────────────────                                          ║"
echo "║  Email:     admin@marble-fds.com                             ║"
echo "║  Password:  Admin123!                                        ║"
echo "║                                                              ║"
echo "║  Jurisdiction: Pakistan (87 rules)                           ║"
echo "║                                                              ║"
echo "║  ────────────────────────────────────────────────────────    ║"
echo "║  Architect: Hussain Mansoor | AI/ML Engineer                 ║"
echo "║  github.com/Hussainm10                                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
