@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
title DLSS 5 Visual Enhancer
cls
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"
set "GRADIO_ANALYTICS_ENABLED=False"
if not defined GRADIO_SERVER_PORT set "GRADIO_SERVER_PORT=7865"
set "GRADIO_TEMP_DIR=%~dp0temp\gradio"
if not exist "%GRADIO_TEMP_DIR%" mkdir "%GRADIO_TEMP_DIR%"
"%~dp0bin\python-3.13.15-embed-amd64\python.exe" "%~dp0app.py"
if errorlevel 1 pause
