@echo off
chcp 65001 > nul
title Myth Museum - Interactive Video Generator

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       Myth Museum - 互動式影片生成器                     ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/2] 檢查環境...
python -c "import jupyter" >nul 2>&1
if %errorlevel% neq 0 (
    echo     ❌ Jupyter 未安裝！
    echo     正在自動安裝...
    pip install jupyter ipywidgets matplotlib pillow -q
)
echo     ✓ Jupyter 已就緒

echo [2/2] 啟動 Jupyter Notebook...
echo.
echo ═══════════════════════════════════════════════════════════
echo   瀏覽器將自動開啟，請依序執行每個 Cell 生成影片
echo   按 Ctrl+C 關閉伺服器
echo ═══════════════════════════════════════════════════════════
echo.

python -m notebook notebooks/myth_generator.ipynb
