@echo off
echo ========================================
echo YOPO Tracker - Windows Docker Setup
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo Step 1: Download and install VcXsrv (X Server for Windows)
echo Please download from: https://sourceforge.net/projects/vcxsrv/
echo.
echo After installation, run XLaunch with these settings:
echo   - Display number: 0
echo   - Start no client
echo   - Disable access control (IMPORTANT!)
echo.
set /p READY="Press Enter when VcXsrv is running..."

echo.
echo Step 2: Building Docker image...
docker build -t yopo-tracker .

echo.
echo Step 3: Running YOPO Tracker...
docker run --rm -it ^
    -e DISPLAY=host.docker.internal:0.0 ^
    -v "%cd%":/app ^
    yopo-tracker

echo.
echo ========================================
echo Program finished!
echo GIF files should be in the gifs/ folder
echo ========================================
pause
