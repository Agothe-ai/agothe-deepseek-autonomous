@echo off
title JARVIS — Paulk AI Assistant
color 0A
echo.
echo  ^🜏 JARVIS BOOTING...
echo  ================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

:: Install deps silently if missing
echo  Checking dependencies...
pip install openai --quiet --disable-pip-version-check

echo  Dependencies OK
echo.
echo  Starting Jarvis...
echo  ================================
echo.

:: Set API key if not already in environment
if "%DEEPSEEK_API_KEY%"=="" (
    set DEEPSEEK_API_KEY=sk-71b52b116f3c432d8e7bfeeec42edf4c
)

:: Launch Paul core
python paul_core.py

echo.
pause
