@echo off
echo ========================================
echo   Knowledge Tracing System - One Click Start
echo ========================================
echo.
echo Starting system, please wait...
echo.

cd /d "%~dp0"

REM Check Python environment
C:/Users/32880/miniconda3/envs/emnist-gpu/python.exe --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python environment not found
    pause
    exit /b 1
)

REM Start Streamlit
echo [1/2] Starting server...
start /min cmd /c "C:/Users/32880/miniconda3/envs/emnist-gpu/python.exe -m streamlit run app.py --browser.gatherUsageStats false"

REM Wait for server to start
timeout /t 3 /nobreak >nul

REM Auto open browser
echo [2/2] Opening browser...
start http://localhost:8501

echo.
echo ========================================
echo   System Started Successfully!
echo ========================================
echo.
echo Access URLs:
echo   Local: http://localhost:8501
echo   Network: http://192.168.1.8:8501
echo.
echo Tips:
echo   - Keep this window open
echo   - Closing this window will stop the system
echo   - Other devices can use the network URL
echo.
echo ========================================
pause
