@echo off
title Couple Decision Support System
cd /d "%~dp0"

:: Activate virtual environment if present, then launch Streamlit
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Launching Decision Support System Dashboard...
streamlit run Home.py --server.headless=false --browser.gatherUsageStats=false

pause