@echo off
title CardioGuard AI - Heart Disease Prediction Web System
echo ========================================================
echo Starting CardioGuard AI Web Server at http://localhost:5000
echo ========================================================
echo.
cd /d "%~dp0website"
start "" http://localhost:5000
py server.py
pause
