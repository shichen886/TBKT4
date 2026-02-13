@echo off
echo ========================================
echo   Knowledge Tracing System - One Click Start
echo ========================================
echo.
echo Starting system, please wait...
echo.

cd /d "%~dp0"

REM Check Python environment
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python environment not found
    pause
    exit /b 1
)

REM Start Django development server
echo [1/2] Starting Django server...
start /min cmd /c "python manage.py runserver"

REM Wait for server to start
timeout /t 5 /nobreak >nul

REM Auto open browser
echo [2/2] Opening browser...
start http://127.0.0.1:8000/

echo.
echo ========================================
echo   System Started Successfully!
echo ========================================
echo.
echo Access URLs:
echo   Local: http://127.0.0.1:8000/
echo.
echo Tips:
echo   - Keep this window open
   - Closing this window will NOT stop the server
   - Server runs in a separate window
   - Press CTRL+C in the server window to stop it
echo.
echo ========================================
pause