#!/bin/bash
# Shell wrapper for generate_certs.py
# This makes it easier to run the certificate generation utility

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to check and offer to install dependencies
check_and_install_dependencies() {
    local python_cmd=$1
    
    # Check if cryptography is installed
    if ! "$python_cmd" -c "import cryptography" 2>/dev/null; then
        echo ""
        echo "⚠ Python cryptography library is not installed."
        echo "It's recommended as a fallback option for certificate generation."
        echo ""
        
        # Check for pip
        local pip_cmd=""
        if command -v pip3 >/dev/null 2>&1; then
            pip_cmd="pip3"
        elif command -v pip >/dev/null 2>&1; then
            pip_cmd="pip"
        fi
        
        if [ -n "$pip_cmd" ]; then
            echo ""
            echo "The Python cryptography library is needed for certificate generation."
            echo "Would you like to install the cryptography library now using $pip_cmd? (y/n): "
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                echo ""
                echo "Installing cryptography library..."
                if "$pip_cmd" install cryptography; then
                    echo "✓ Successfully installed cryptography library"
                else
                    echo "✗ Failed to install cryptography library"
                    echo "You can install it manually later with: $pip_cmd install cryptography"
                fi
            else
                echo "Skipping installation. You can install it later with: $pip_cmd install cryptography"
            fi
        else
            echo "pip is not available. Please install cryptography manually:"
            echo "  pip install cryptography"
            echo "  or"
            echo "  pip3 install cryptography"
        fi
        echo ""
    fi
}

# Try to find Python (try python3 first, then python)
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo ""
    echo "ERROR: Could not find Python interpreter."
    echo "Please ensure Python 3 is installed and available in your PATH."
    echo ""
    echo "You can also run the script directly:"
    echo "  python3 \"${SCRIPT_DIR}/generate_certs.py\" $*"
    echo "  or"
    echo "  python \"${SCRIPT_DIR}/generate_certs.py\" $*"
    exit 1
fi

# Check and offer to install dependencies interactively
check_and_install_dependencies "$PYTHON_CMD"

# Run the Python script with all passed arguments
"$PYTHON_CMD" "${SCRIPT_DIR}/generate_certs.py" "$@"
exit $?

