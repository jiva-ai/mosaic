# PowerShell script wrapper for generate_certs.py
# This makes it easier to run the certificate generation utility on Windows

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Build the path to the Python script
$PythonScript = Join-Path $ScriptDir "generate_certs.py"

# Try to find Python (try python3, python, then py launcher)
$PythonCmd = $null
$PythonCommands = @("python3", "python", "py")

foreach ($cmd in $PythonCommands) {
    try {
        $null = Get-Command $cmd -ErrorAction Stop
        $PythonCmd = $cmd
        break
    } catch {
        continue
    }
}

if ($null -eq $PythonCmd) {
    Write-Host ""
    Write-Host "ERROR: Could not find Python interpreter." -ForegroundColor Red
    Write-Host "Please ensure Python 3 is installed and available in your PATH." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can also run the script directly:" -ForegroundColor Yellow
    Write-Host "  python `"$PythonScript`" $args" -ForegroundColor Cyan
    exit 1
}

# Function to check and offer to install dependencies
function Check-AndInstallDependencies {
    param($PythonCommand)
    
    # Check if cryptography is installed
    $cryptoCheck = & $PythonCommand -c "import cryptography" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "⚠ Python cryptography library is not installed." -ForegroundColor Yellow
        Write-Host "It's recommended as a fallback option for certificate generation." -ForegroundColor Yellow
        Write-Host ""
        
        # Check for pip
        $PipCmd = $null
        $PipCommands = @("pip3", "pip")
        
        foreach ($pip in $PipCommands) {
            try {
                $null = Get-Command $pip -ErrorAction Stop
                $PipCmd = $pip
                break
            } catch {
                continue
            }
        }
        
        # Also try python -m pip
        if ($null -eq $PipCmd) {
            try {
                $null = & $PythonCommand -m pip --version 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $PipCmd = "$PythonCommand -m pip"
                }
            } catch {
                # Ignore
            }
        }
        
        if ($null -ne $PipCmd) {
            Write-Host ""
            Write-Host "The Python cryptography library is needed for certificate generation." -ForegroundColor Yellow
            $response = Read-Host "Would you like to install the cryptography library now using $PipCmd? (y/n)"
            if ($response -match '^[Yy]') {
                Write-Host ""
                Write-Host "Installing cryptography library..." -ForegroundColor Cyan
                & $PipCmd install cryptography
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✓ Successfully installed cryptography library" -ForegroundColor Green
                } else {
                    Write-Host "✗ Failed to install cryptography library" -ForegroundColor Red
                    Write-Host "You can install it manually later with: $PipCmd install cryptography" -ForegroundColor Yellow
                }
            } else {
                Write-Host "Skipping installation. You can install it later with: $PipCmd install cryptography" -ForegroundColor Yellow
            }
        } else {
            Write-Host "pip is not available. Please install cryptography manually:" -ForegroundColor Yellow
            Write-Host "  pip install cryptography" -ForegroundColor Cyan
            Write-Host "  or" -ForegroundColor Yellow
            Write-Host "  pip3 install cryptography" -ForegroundColor Cyan
        }
        Write-Host ""
    }
}

# Check and offer to install dependencies interactively
Check-AndInstallDependencies -PythonCommand $PythonCmd

# Run the Python script with all passed arguments
& $PythonCmd $PythonScript $args
exit $LASTEXITCODE

