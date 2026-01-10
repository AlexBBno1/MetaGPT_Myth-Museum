@echo off
title Myth Museum Video Generator
echo.
echo ============================================
echo   Myth Museum Video Generator
echo   Streamlit Web Interface
echo ============================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing Streamlit...
    pip install streamlit
    echo.
)

echo Starting Streamlit server...
echo.
echo Open your browser to: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Use python -m streamlit instead of direct streamlit command
REM This works even if Scripts folder is not in PATH
python -m streamlit run streamlit_app.py --server.headless false

pause
