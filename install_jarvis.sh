#!/bin/bash
# install_jarvis.sh — Jarvis Phase 1 Installer (Git Bash / WSL / Linux / Mac)
# Run: bash install_jarvis.sh

GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
CYAN='\033[96m'
BOLD='\033[1m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║  🜏 JARVIS PHASE 1 INSTALLER (bash)          ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""

# --- CHECK PYTHON ---
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python not found.${RESET}"
    echo "Download from: https://www.python.org/downloads/"
    exit 1
fi

# Use python3 if python not found
PYTHON=python
if ! command -v python &> /dev/null; then
    PYTHON=python3
fi

$PYTHON --version
echo -e "${GREEN}[OK] Python found.${RESET}"
echo ""

# --- UPGRADE PIP ---
echo "[1/8] Upgrading pip..."
$PYTHON -m pip install --upgrade pip --quiet
echo -e "${GREEN}[OK] pip upgraded.${RESET}"
echo ""

# --- CORE AI ---
echo "[2/8] Installing core AI stack..."
pip install openai fastapi uvicorn python-multipart --quiet
echo -e "${GREEN}[OK] Core AI stack installed.${RESET}"
echo ""

# --- SEMANTIC MEMORY ---
echo "[3/8] Installing semantic memory (sentence-transformers ~90MB first run)..."
pip install sentence-transformers --quiet
echo -e "${GREEN}[OK] Semantic memory ready.${RESET}"
echo ""

# --- VOICE ---
echo "[4/8] Installing voice stack..."
pip install pyttsx3 openai-whisper numpy --quiet
# pyaudio can be tricky — try, don't fail if it errors
pip install pyaudio --quiet 2>/dev/null || echo -e "${YELLOW}[WARN] pyaudio failed — voice input may not work. Try: pip install pipwin && pipwin install pyaudio${RESET}"
echo -e "${GREEN}[OK] Voice stack installed.${RESET}"
echo ""

# --- WEB INTELLIGENCE ---
echo "[5/8] Installing web intelligence tools..."
pip install playwright duckduckgo-search beautifulsoup4 requests --quiet
$PYTHON -m playwright install chromium --quiet 2>/dev/null || echo -e "${YELLOW}[WARN] Playwright browser install failed — run: python -m playwright install chromium${RESET}"
echo -e "${GREEN}[OK] Web intelligence ready.${RESET}"
echo ""

# --- SCHEDULER ---
echo "[6/8] Installing APScheduler..."
pip install apscheduler --quiet
echo -e "${GREEN}[OK] Scheduler installed.${RESET}"
echo ""

# --- TESTING ---
echo "[7/8] Installing pytest..."
pip install pytest pytest-asyncio --quiet
echo -e "${GREEN}[OK] Test engine ready.${RESET}"
echo ""

# --- VERIFY ---
echo "[8/8] Running verification..."
$PYTHON verify_install.py
echo ""

echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║  Phase 1 Complete.                           ║${RESET}"
echo -e "${CYAN}${BOLD}║  Next: set env vars, then run Jarvis.        ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Set your API keys (paste these, replace values):"
echo '     export DEEPSEEK_API_KEY=sk-71b52b116f3c432d8e7bfeeec42edf4c'
echo '     export GITHUB_TOKEN=ghp_yournewtoken'
echo ""
echo "  To make permanent (Git Bash), add to ~/.bashrc:"
echo '     echo export DEEPSEEK_API_KEY=sk-... >> ~/.bashrc'
echo '     echo export GITHUB_TOKEN=ghp_... >> ~/.bashrc'
echo '     source ~/.bashrc'
echo ""
echo "  2. Launch Jarvis:"
echo "     python paul_core.py          # Mode 1 - CLI Chat"
echo "     python jarvis_dashboard.py   # Mode 10 - Dashboard"
echo "     python jarvis_voice.py       # Mode 2 - Voice"
echo ""
