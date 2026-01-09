@echo off
chcp 65001 > nul
title Myth Museum - 一鍵安裝

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       Myth Museum - 一鍵安裝程式                         ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

:: Check Python
echo [1/6] 檢查 Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     ❌ Python 未安裝！請先安裝 Python 3.10+
    echo     下載: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo     ✓ Python %PYVER%

:: Check pip
echo [2/6] 檢查 pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     ❌ pip 未安裝！
    pause
    exit /b 1
)
echo     ✓ pip 已安裝

:: Upgrade pip
echo [3/6] 更新 pip...
python -m pip install --upgrade pip -q
echo     ✓ pip 已更新

:: Install project dependencies
echo [4/6] 安裝專案依賴...
pip install -r requirements.txt -q 2>nul
if %errorlevel% neq 0 (
    echo     使用 pyproject.toml 安裝...
    pip install -e . -q
)
echo     ✓ 核心依賴已安裝

:: Install Jupyter and widgets
echo [5/6] 安裝 Jupyter Notebook...
pip install jupyter notebook ipywidgets matplotlib pillow -q
echo     ✓ Jupyter 已安裝

:: Check FFmpeg
echo [6/6] 檢查 FFmpeg...
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo     ⚠ FFmpeg 未安裝！
    echo.
    echo     嘗試使用 winget 自動安裝...
    winget install FFmpeg -e --silent >nul 2>&1
    if %errorlevel% neq 0 (
        echo     ⚠ 自動安裝失敗，請手動安裝:
        echo     1. 下載: https://www.gyan.dev/ffmpeg/builds/
        echo     2. 解壓縮並加入系統 PATH
        set FFMPEG_MISSING=1
    ) else (
        echo     ✓ FFmpeg 已透過 winget 安裝
    )
) else (
    echo     ✓ FFmpeg 已安裝
)

echo.
echo ═══════════════════════════════════════════════════════════
echo   安裝完成！
echo ═══════════════════════════════════════════════════════════
echo.
echo   已安裝套件:
echo   ├─ 核心: feedparser, httpx, pydantic, openai, rich, typer
echo   ├─ API: fastapi, uvicorn
echo   ├─ TTS: edge-tts, pydub
echo   ├─ AI: google-cloud-aiplatform, vertexai, google-generativeai
echo   ├─ Jupyter: notebook, ipywidgets
echo   └─ 圖像: Pillow, matplotlib
echo.
if defined FFMPEG_MISSING (
    echo   ⚠ 注意: FFmpeg 需要手動安裝才能生成影片
    echo.
)
echo   下一步:
echo   • 雙擊「啟動互動介面.bat」開始使用
echo   • 或執行「generate_video.bat」快速生成影片
echo.
pause
