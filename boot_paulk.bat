@echo off
cd C:\Users\gtsgo\agothe_core\apps\api
start "Agothe OS API" python -m uvicorn main:app --reload --port 8000
echo Paulk API booted on http://localhost:8000
