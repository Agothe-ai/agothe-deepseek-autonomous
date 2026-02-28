@echo off
title JARVIS v5.0 — Paulk AI
color 0A
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║  🜏 JARVIS v5.0 — PAULK AI               ║
echo  ║  The best coder AI. Yours.               ║
echo  ╚══════════════════════════════════════════╝
echo.

python --version >nul 2>&1
if errorlevel 1 ( echo Python not found. & pause & exit /b 1 )

echo  Installing core...
pip install openai fastapi uvicorn --quiet --disable-pip-version-check

if "%DEEPSEEK_API_KEY%"=="" set DEEPSEEK_API_KEY=sk-71b52b116f3c432d8e7bfeeec42edf4c

echo.
echo  [1] CLI Chat           — text terminal
echo  [2] Voice PTT          — press Enter to speak
echo  [3] Voice Wake Word    — say Hey Jarvis anytime
echo  [4] Web Dashboard      — browser UI localhost:8000
echo  [5] Coder Engine       — Planner+Executor+Verifier
echo  [6] Self-Heal Daemon   — auto-patch broken files
echo  [7] GitHub Watcher     — live review every push
echo  [8] GitHub + Voice     — push = spoken review in 30s
echo.
set /p MODE="Mode (1-8, Enter=1): "

if "%MODE%"=="2" (
    pip install pyttsx3 openai-whisper pyaudio numpy --quiet
    python jarvis_voice.py
) else if "%MODE%"=="3" (
    pip install pyttsx3 openai-whisper pyaudio numpy --quiet
    python jarvis_voice.py --wake
) else if "%MODE%"=="4" (
    start http://localhost:8000
    python -m uvicorn jarvis_api:app --host 0.0.0.0 --port 8000
) else if "%MODE%"=="5" (
    pip install pytest --quiet
    python jarvis_evolve.py
) else if "%MODE%"=="6" (
    python jarvis_self_heal.py
) else if "%MODE%"=="7" (
    python jarvis_github_watcher.py
) else if "%MODE%"=="8" (
    pip install pyttsx3 openai-whisper pyaudio numpy --quiet
    set JARVIS_VOICE=1
    python jarvis_github_watcher.py
) else (
    python paul_core.py
)

pause
