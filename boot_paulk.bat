@echo off
title JARVIS v4.0 — Paulk AI Assistant
color 0A
echo.
echo  ╔══════════════════════════════════════╗
echo  ║  🜏 JARVIS v4.0 — PAULK AI            ║
echo  ║  Status: ONLINE                     ║
echo  ╚══════════════════════════════════════╝
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from python.org
    pause & exit /b 1
)

echo  Installing core dependencies...
pip install openai fastapi uvicorn --quiet --disable-pip-version-check

if "%DEEPSEEK_API_KEY%"=="" set DEEPSEEK_API_KEY=sk-71b52b116f3c432d8e7bfeeec42edf4c

echo.
echo  Choose mode:
echo  [1] CLI Chat         — text terminal
echo  [2] Voice PTT        — press Enter to speak
echo  [3] Voice Wake Word  — say 'Hey Jarvis' anytime
echo  [4] Web Dashboard    — browser UI at localhost:8000
echo  [5] Coder Engine     — Planner+Executor+Verifier
echo  [6] Self-Heal Daemon — auto-patch broken files
echo.
set /p MODE="Mode (1-6, Enter=1): "

if "%MODE%"=="2" (
    echo  Installing voice deps...
    pip install pyttsx3 openai-whisper pyaudio numpy --quiet
    python jarvis_voice.py
) else if "%MODE%"=="3" (
    echo  Installing voice deps...
    pip install pyttsx3 openai-whisper pyaudio numpy --quiet
    python jarvis_voice.py --wake
) else if "%MODE%"=="4" (
    echo  Open: http://localhost:8000
    python -m uvicorn jarvis_api:app --host 0.0.0.0 --port 8000
) else if "%MODE%"=="5" (
    pip install pytest --quiet
    python jarvis_evolve.py
) else if "%MODE%"=="6" (
    python jarvis_self_heal.py
) else (
    python paul_core.py
)

echo.
pause
