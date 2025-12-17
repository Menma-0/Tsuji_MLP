@echo off
echo ================================================================================
echo Onoma2DSP Web UI - Server Startup Script
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/3] Checking Python dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing FastAPI...
    pip install fastapi uvicorn[standard] python-multipart
)

echo [2/3] Starting Backend API Server...
echo.
echo Backend will run on: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.

start "Onoma2DSP Backend API" cmd /k "python api_server.py"

timeout /t 3 >nul

echo [3/3] Starting Frontend Development Server...
echo.
echo Frontend will run on: http://localhost:3000
echo.

cd frontend
start "Onoma2DSP Frontend" cmd /k "npm run dev"

echo.
echo ================================================================================
echo SERVERS STARTED SUCCESSFULLY!
echo ================================================================================
echo.
echo Open your browser and navigate to:
echo   http://localhost:3000
echo.
echo To stop the servers, close both terminal windows or press Ctrl+C
echo.
pause
