@echo off
REM One-click launcher for Windows. Double-click this file.
REM Creates an isolated Python environment the first time, then runs the
REM wallet analyzer inside it (so nothing touches your system Python).

cd /d "%~dp0"

set "VENV=.venv"

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

REM The script itself installs any missing libraries into this venv.
"%VENV%\Scripts\python.exe" wallet_analyzer.py %*

echo.
pause
