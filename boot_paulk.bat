@echo off
title JARVIS v2.0 — Paulk AI Assistant
color 0A
echo.
echo  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
echo  🜏 JARVIS v2.0 BOOTING
echo  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

:: Install deps
echo  Installing dependencies...
pip install openai fastapi uvicorn --quiet --disable-pip-version-check
echo  Dependencies OK
echo.

:: API Key
if "%DEEPSEEK_API_KEY%"=="" (
    set DEEPSEEK_API_KEY=sk-71b52b116f3c432d8e7bfeeec42edf4c
)

echo  Choose mode:
echo  [1] CLI  — terminal chat (default)
echo  [2] WEB  — browser dashboard at http://localhost:8000
echo.
set /p MODE="Mode (1 or 2, Enter=1): "

if "%MODE%"=="2" (
    echo.
    echo  Starting web dashboard...
    echo  Open: http://localhost:8000
    echo.
    python -m uvicorn jarvis_api:app --host 0.0.0.0 --port 8000 --reload
) else (
    echo.
    python paul_core.py
)

echo.
pause
