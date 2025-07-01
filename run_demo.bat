@echo off
echo ========================================
echo  PDF Question Answering System - Demo
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed or not in PATH
    pause
    exit /b 1
)

REM Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
) else (
    echo WARNING: requirements.txt not found
    echo.
)

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "vector_db" mkdir vector_db
if not exist "demo_data\sample_pdfs" mkdir demo_data\sample_pdfs

echo Starting PDF Question Answering System...
echo.
echo The application will be available at:
echo   http://localhost:8000
echo.
echo To stop the application, press Ctrl+C
echo.

REM Start the application
python main.py

REM If we get here, the application has stopped
echo.
echo Application stopped.
pause