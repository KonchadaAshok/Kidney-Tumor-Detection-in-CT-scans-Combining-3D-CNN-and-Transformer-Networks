@echo off
REM Quick start script for Kidney Disease Classification Flask App
REM Windows batch file

echo.
echo ============================================================
echo 🏥 Kidney Disease Classification - Flask Web App
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

echo.
echo ============================================================
echo Installing/Updating dependencies...
echo ============================================================
echo.

REM Install requirements
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully
echo.

REM Check if model exists
if not exist "hybrid_model_best.pth" (
    echo ❌ Error: Model file 'hybrid_model_best.pth' not found!
    echo Please ensure the model file is in the current directory
    pause
    exit /b 1
)

echo ✅ Model file found
echo.

echo ============================================================
echo 🚀 Starting Flask Web Server...
echo ============================================================
echo.
echo 📍 Open your browser to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

REM Start Flask app
python app.py

pause
