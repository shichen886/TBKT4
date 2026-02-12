@echo off
chcp 65001 >nul
echo 正在启动知识追踪系统...

REM 获取脚本所在目录
set "APP_DIR=%~dp0"

REM 设置PaddleOCR缓存目录到应用目录，避免权限问题
set "PADDLEX_HOME=%APP_DIR%paddlex_cache"
set "PADDLE_HOME=%APP_DIR%paddle_cache"
set "MODELSCOPE_CACHE=%APP_DIR%modelscope_cache"
set "HUGGINGFACE_HUB_CACHE=%APP_DIR%huggingface_cache"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"

REM 创建缓存目录
if not exist "%PADDLEX_HOME%" mkdir "%PADDLEX_HOME%"
if not exist "%PADDLE_HOME%" mkdir "%PADDLE_HOME%"
if not exist "%MODELSCOPE_CACHE%" mkdir "%MODELSCOPE_CACHE%"
if not exist "%HUGGINGFACE_HUB_CACHE%" mkdir "%HUGGINGFACE_HUB_CACHE%"

echo 缓存目录已设置:
echo   PADDLEX_HOME=%PADDLEX_HOME%
echo   PADDLE_HOME=%PADDLE_HOME%
echo   MODELSCOPE_CACHE=%MODELSCOPE_CACHE%
echo.

echo 正在启动Streamlit服务器...
echo.

REM 启动Streamlit
C:/Users/32880/miniconda3/envs/emnist-gpu/python.exe -m streamlit run app.py --browser.gatherUsageStats false

pause
