@echo off
chcp 65001 > nul
title Myth Museum - Quick Video Generator

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       Myth Museum - 快速影片生成器                       ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

if "%~1"=="" (
    set /p TOPIC="請輸入主題 (例如: Da Vinci Mona Lisa myth): "
) else (
    set "TOPIC=%~1"
)

echo.
echo 主題: %TOPIC%
echo.

set /p QUALITY="選擇品質 [high/standard/fallback] (預設 high): "
if "%QUALITY%"=="" set QUALITY=high

echo.
echo ═══════════════════════════════════════════════════════════
echo   開始生成影片...
echo   品質: %QUALITY%
echo ═══════════════════════════════════════════════════════════
echo.

python -m pipeline.generate_short generate "%TOPIC%" --quality %QUALITY%

echo.
echo ═══════════════════════════════════════════════════════════
echo   完成！影片位於 outputs/shorts/ 資料夾
echo ═══════════════════════════════════════════════════════════
echo.
pause
