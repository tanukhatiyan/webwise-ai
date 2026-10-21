@echo off
title WebWise AI Backend Server
echo =========================================================
echo   Starting WebWise AI Backend
echo =========================================================
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" run.py
) else (
    python run.py
)

pause
