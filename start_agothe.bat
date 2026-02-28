@echo off
title Agothe Core Launcher

:: ── FastAPI Backend ──────────────────────────────────────
start "Agothe API" cmd /k "cd /d C:\Users\gtsgo\agothe_core\apps\api && C:\Users\gtsgo\AppData\Local\Microsoft\WindowsApps\python.exe -m uvicorn main:app --reload --port 8000"

:: wait 4 seconds for API to initialize
timeout /t 4 /nobreak > nul

:: ── Next.js Frontend ─────────────────────────────────────
start "Agothe Web" cmd /k "cd /d C:\Users\gtsgo\agothe_core\apps\web && npm run dev"

echo.
echo  Agothe Core is starting...
echo  API  --  http://localhost:8000
echo  Web  --  http://localhost:3000
