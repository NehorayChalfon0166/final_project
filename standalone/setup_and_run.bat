@echo off
REM ============================================================
REM Bitcoin Wallet Risk Analyzer — Windows One-Click Installer
REM ============================================================
REM First run: Creates venv and installs dependencies (~2 min)
REM Subsequent runs: Skips install and runs directly
REM
REM Usage:
REM   setup_and_run.bat
REM   setup_and_run.bat 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
REM   setup_and_run.bat --no-charts bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
REM ============================================================

setlocal EnableDelayedExpansion

set VENV_DIR=.analyzer_env
set PYTHON_CMD=python

REM Check if Python is available
%PYTHON_CMD% --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo  ERROR: Python is not installed or not in PATH.
    echo.
    echo  Please install Python 3.10+ from:
    echo    https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: Check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Check Python version >= 3.10
for /f "tokens=2 delims= " %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if %MAJOR% lss 3 (
    echo  ERROR: Python 3.10+ required. Found: %PYVER%
    pause
    exit /b 1
)
if %MAJOR% equ 3 if %MINOR% lss 10 (
    echo  ERROR: Python 3.10+ required. Found: %PYVER%
    pause
    exit /b 1
)

echo  Python %PYVER% detected.

REM Create venv if it doesn't exist
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo  First run — setting up environment...
    echo  [1/3] Creating virtual environment...
    %PYTHON_CMD% -m venv %VENV_DIR%
    if %ERRORLEVEL% neq 0 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo  [2/3] Upgrading pip...
    call %VENV_DIR%\Scripts\activate.bat
    python -m pip install --upgrade pip --quiet

    echo  [3/3] Installing dependencies (this may take a few minutes)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    pip install torch-geometric --quiet
    pip install requests matplotlib numpy --quiet

    if %ERRORLEVEL% neq 0 (
        echo  ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )

    echo.
    echo  Setup complete!
    echo.
) else (
    call %VENV_DIR%\Scripts\activate.bat
)

REM Run the analyzer, passing through all arguments
python wallet_analyzer.py %*

REM Keep window open if double-clicked
if "%~1"=="" (
    echo.
    pause
)

endlocal
