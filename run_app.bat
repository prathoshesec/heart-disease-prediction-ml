@echo off
title CardioGuard AI - Heart Disease Prediction System
echo ========================================================
echo Starting CardioGuard AI Web Application...
echo ========================================================
echo.
cd /d "%~dp0"
py -m streamlit run app.py
pause
