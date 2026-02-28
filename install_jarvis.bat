@echo off
title JARVIS PHASE 1 INSTALLER
color 0A
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║  🜏 JARVIS PHASE 1 INSTALLER                    ║
echo  ║  Installs ALL dependencies in the right order  ║
echo  ║  Run this ONCE after cloning the repo          ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: --- CHECK PYTHON ---
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Download from: https://www.python.org/downloads/
    echo  Make sure to check ADD TO PATH during install.
    pause
    exit /b 1
)
python --version
echo  [OK] Python found.
echo.

:: --- UPGRADE PIP ---
echo  [1/8] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  [OK] pip upgraded.
echo.

:: --- CORE AI + API ---
echo  [2/8] Installing core AI stack (openai, fastapi, uvicorn)...
pip install openai fastapi uvicorn python-multipart --quiet
echo  [OK] Core AI stack installed.
echo.

:: --- SEMANTIC MEMORY (most important) ---
echo  [3/8] Installing semantic memory engine...
echo        (sentence-transformers ~90MB first run, downloads once)
pip install sentence-transformers --quiet
echo  [OK] Semantic memory ready. Jarvis will remember HOW you code.
echo.

:: --- VOICE ---
echo  [4/8] Installing voice stack (Whisper STT + pyttsx3 TTS)...
pip install pyttsx3 openai-whisper pyaudio numpy --quiet
echo  [OK] Voice stack installed.
echo.

:: --- WEB INTELLIGENCE (Phase 9 prep) ---
echo  [5/8] Installing web intelligence tools (Playwright, search)...
pip install playwright duckduckgo-search beautifulsoup4 requests --quiet
python -m playwright install chromium --quiet
echo  [OK] Web intelligence ready. Jarvis can browse + search.
echo.

:: --- SCHEDULER ---
echo  [6/8] Installing task scheduler (apscheduler)...
pip install apscheduler --quiet
echo  [OK] Scheduler installed. Cron automations ready.
echo.

:: --- TESTING ---
echo  [7/8] Installing test engine (pytest)...
pip install pytest pytest-asyncio --quiet
echo  [OK] Test engine ready.
echo.

:: --- VERIFY EVERYTHING ---
echo  [8/8] Running verification...
python verify_install.py
echo.

echo  ╔══════════════════════════════════════════════╗
echo  ║  Phase 1 Complete.                              ║
echo  ║  Next: set your env vars below, then run        ║
echo  ║  boot_paulk.bat to launch Jarvis.              ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  NEXT STEPS:
echo  1. Set your API key (if not done already):
echo     setx DEEPSEEK_API_KEY sk-71b52b116f3c432d8e7bfeeec42edf4c
echo.
echo  2. Set your GitHub token:
echo     setx GITHUB_TOKEN ghp_yournewtoken
echo.
echo  3. RESTART this terminal window (so env vars load)
echo.
echo  4. Run: boot_paulk.bat
echo.
pause
