@echo off
REM Windows batch script wrapper for generate_certs.py
REM This makes it easier to run the certificate generation utility on Windows

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

REM Try to find Python (try python3, python, then py launcher)
set "PYTHON_CMD="
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python3"
    goto :found_python
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    goto :found_python
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py"
    goto :found_python
)

echo.
echo ERROR: Could not find Python interpreter.
echo Please ensure Python 3 is installed and available in your PATH.
echo.
echo You can also run the script directly:
echo   python "%SCRIPT_DIR%generate_certs.py" %*
exit /b 1

:found_python
REM Check if cryptography is installed - use a more robust check
%PYTHON_CMD% -c "try: import cryptography; exit(0); except: exit(1)" >nul 2>&1
if %errorlevel% equ 0 (
    REM Cryptography is installed, skip pip check
    goto :run_script
)

REM Cryptography is not installed, check for pip
echo.
echo ⚠ Python cryptography library is not installed.
echo It's recommended as a fallback option for certificate generation.
echo.

REM Check for pip - try multiple methods
set "PIP_CMD="
set "PIP_FOUND=0"

REM Try python -m pip first (most reliable)
%PYTHON_CMD% -m pip --version >nul 2>&1
if not errorlevel 1 (
    set "PIP_CMD=%PYTHON_CMD% -m pip"
    set "PIP_FOUND=1"
    goto :check_pip_done
)

REM Reset errorlevel by running a successful command
ver >nul

REM Try pip3
pip3 --version >nul 2>&1
if not errorlevel 1 (
    set "PIP_CMD=pip3"
    set "PIP_FOUND=1"
    goto :check_pip_done
)

REM Reset errorlevel
ver >nul

REM Try pip
pip --version >nul 2>&1
if not errorlevel 1 (
    set "PIP_CMD=pip"
    set "PIP_FOUND=1"
    goto :check_pip_done
)

:check_pip_done
if %PIP_FOUND% equ 0 (
    echo pip is not available. Please install cryptography manually:
    echo   %PYTHON_CMD% -m pip install cryptography
    echo   or
    echo   pip install cryptography
    echo   or
    echo   pip3 install cryptography
    echo.
    goto :run_script
)

REM Pip found, offer to install
echo The Python cryptography library is needed for certificate generation.
echo It provides a fallback option if OpenSSL is not available.
set /p INSTALL_CRYPTO="Would you like to install the cryptography library now using %PIP_CMD%? (y/n): "
if /i "%INSTALL_CRYPTO%"=="y" (
    echo.
    echo Installing cryptography library using %PIP_CMD%...
    %PIP_CMD% install cryptography
    if %errorlevel% equ 0 (
        echo ✓ Successfully installed cryptography library
    ) else (
        echo ✗ Failed to install cryptography library
        echo You can install it manually later with: %PIP_CMD% install cryptography
    )
) else (
    echo Skipping installation. You can install it later with: %PIP_CMD% install cryptography
)
echo.

:run_script
REM Run the Python script with all passed arguments
%PYTHON_CMD% "%SCRIPT_DIR%generate_certs.py" %*
exit /b %errorlevel%

