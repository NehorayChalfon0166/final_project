@echo off
REM One-click launcher for Windows. Download just THIS file and double-click it.
REM It downloads the analyzer script (if missing), builds an isolated Python
REM environment, and runs the analyzer inside it.

cd /d "%~dp0"

set "SCRIPT=wallet_analyzer.py"
set "SCRIPT_URL=https://raw.githubusercontent.com/NehorayChalfon0166/final_project/chore/cleanup-and-consolidate/standalone/wallet_analyzer.py"
set "VENV=.venv"

REM --- 1. Download the analyzer script if it isn't next to this launcher ---
if not exist "%SCRIPT%" (
    echo Downloading %SCRIPT% ...
    curl -fsSL -o "%SCRIPT%" "%SCRIPT_URL%" 2>nul
)
if not exist "%SCRIPT%" (
    powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri '%SCRIPT_URL%' -OutFile '%SCRIPT%' } catch { exit 1 }"
)
if not exist "%SCRIPT%" (
    echo.
    echo Could not download the analyzer script. Check your internet connection.
    pause
    exit /b 1
)

REM --- 2. Create an isolated environment the first time ---
if not exist "%VENV%\Scripts\python.exe" (
    echo Creating isolated environment (first run only) ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo.
        echo Could not create the environment. Make sure Python 3.9+ is installed
        echo and on your PATH ^(https://www.python.org/downloads/^).
        pause
        exit /b 1
    )
)

REM --- 3. Run it (loops until you press Ctrl+C; installs libraries as needed) ---
"%VENV%\Scripts\python.exe" "%SCRIPT%" %*
