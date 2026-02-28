@echo off
title JARVIS EVOLUTION ENGINE v3.0
color 0A
echo.
echo  ════════════════════════════════════════
echo  🜏 JARVIS WORLD-CLASS CODER ENGINE
echo  Architecture: Claude Code + Cursor + Devin
echo  ════════════════════════════════════════
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    pause
    exit /b 1
)

echo  Installing dependencies...
pip install openai pytest --quiet --disable-pip-version-check
echo  OK
echo.

if "%DEEPSEEK_API_KEY%"=="" set DEEPSEEK_API_KEY=sk-71b52b116f3c432d8e7bfeeec42edf4c

echo  Choose mode:
echo  [1] World-Class Coder    — Planner/Executor/Verifier loop
echo  [2] Self-Heal Daemon     — continuous file watcher + auto-patch
echo  [3] Jarvis v2 (full)     — standard Paul assistant
echo.
set /p MODE="Mode (1/2/3, Enter=1): "

if "%MODE%"=="2" (
    echo  Starting self-heal daemon...
    python jarvis_self_heal.py
) else if "%MODE%"=="3" (
    python paul_core.py
) else (
    echo  Starting World-Class Coder Engine...
    python jarvis_evolve.py
)

pause
