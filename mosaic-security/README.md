# MOSAIC Security Utilities

This directory contains utilities for managing SSL/TLS certificates required for secure communication in MOSAIC.

## Certificate Generation Utility

The `generate_certs.py` script creates the SSL components required for secure communication:
- **CA certificate** (`ca.crt`) - Certificate Authority certificate
- **Server certificate** (`server.crt`) - Server certificate signed by the CA
- **Server private key** (`server.key`) - Private key for the server certificate

### Features

- **Multi-platform support**: Works on Windows, Linux, and macOS
- **Multiple fallback methods**: Automatically tries different tools in order:
  1. OpenSSL (command-line tool) - Most common and recommended
  2. Python cryptography library - Pure Python fallback
  3. Java keytool - Basic fallback option
- **Comprehensive error handling**: Provides detailed instructions if all methods fail
- **Configurable**: Customizable hostname, validity period, key size, and output directory

### Quick Start

#### Using Python directly:
```bash
# Generate certificates with default settings (localhost, 365 days validity)
python generate_certs.py

# Generate certificates for a specific hostname
python generate_certs.py --hostname my-server.example.com

# Generate certificates in a custom directory
python generate_certs.py --output-dir /etc/mosaic/certs

# Generate certificates with custom validity period and key size
python generate_certs.py --validity-days 730 --key-size 4096
```

#### Using wrapper scripts:

**Linux/macOS/WSL:**
```bash
./generate_certs.sh
./generate_certs.sh --hostname my-server.example.com
```

**Windows (Command Prompt):**
```cmd
generate_certs.bat
generate_certs.bat --hostname my-server.example.com
```

**Windows (PowerShell):**
```powershell
.\generate_certs.ps1
.\generate_certs.ps1 --hostname my-server.example.com
```

### Command-Line Options

```
--output-dir DIR     Directory to output certificate files (default: certs)
--hostname HOSTNAME  Hostname/CN for the server certificate (default: localhost)
--validity-days DAYS Certificate validity period in days (default: 365)
--key-size BITS      RSA key size in bits: 1024, 2048, or 4096 (default: 2048)
```

### Usage Examples

#### Basic Usage
```bash
python generate_certs.py
```
This creates:
- `certs/ca.crt` - CA certificate
- `certs/server.crt` - Server certificate
- `certs/server.key` - Server private key

#### Production Setup
```bash
python generate_certs.py \
  --hostname mosaic-server.example.com \
  --output-dir /etc/mosaic/certs \
  --validity-days 730 \
  --key-size 4096
```

#### Development Setup
```bash
python generate_certs.py \
  --hostname localhost \
  --output-dir ./dev-certs \
  --validity-days 90
```

### Integration with MOSAIC Configuration

After generating certificates, configure them in your MOSAIC config file:

```json
{
  "server_crt": "/path/to/certs/server.crt",
  "server_key": "/path/to/certs/server.key",
  "ca_crt": "/path/to/certs/ca.crt"
}
```

### Installation Requirements

The utility will automatically detect and use available tools. For best results, install one of:

#### Option 1: OpenSSL (Recommended)
- **Windows**: Download from [Win32OpenSSL](https://slproweb.com/products/Win32OpenSSL.html) or use `choco install openssl`
- **macOS**: `brew install openssl`
- **Linux**: Usually pre-installed. If not: `sudo apt-get install openssl` (Ubuntu/Debian) or `sudo dnf install openssl` (Fedora/RHEL)

#### Option 2: Python cryptography library
```bash
pip install cryptography
# or
pip3 install cryptography
```

#### Option 3: Java keytool
- Comes with Java JDK/JRE
- Install Java from [Oracle](https://www.oracle.com/java/) or use package manager

### Troubleshooting

#### No tools available
If the script reports that no certificate generation methods are available, it will automatically print comprehensive installation instructions for your operating system.

#### Permission errors
If you get permission errors when writing to the output directory:
- On Linux/macOS: Use `sudo` or change directory permissions
- On Windows: Run as Administrator or choose a directory you have write access to

#### Certificate validation errors
If your applications report certificate validation errors:
- Ensure the CA certificate is trusted by your application
- Check that the hostname in the certificate matches the hostname used to connect
- Verify certificate expiration dates

### Security Notes

⚠️ **Important Security Considerations:**

1. **Private Key Security**: 
   - Keep `server.key` secure and never share it
   - Use appropriate file permissions (e.g., `chmod 600 server.key` on Linux/macOS)
   - Do not commit private keys to version control

2. **Certificate Validity**:
   - Monitor certificate expiration dates
   - Set up renewal procedures before certificates expire
   - For production, consider using shorter validity periods for better security

3. **Key Size**:
   - 2048 bits is recommended for most use cases
   - 4096 bits provides stronger security but may impact performance
   - 1024 bits is deprecated and should not be used

4. **Production Use**:
   - For production environments, consider using certificates from a trusted CA
   - Self-signed certificates are suitable for development and internal networks
   - Ensure proper certificate chain validation in your applications

### File Permissions (Linux/macOS)

After generation, set appropriate permissions:

```bash
chmod 600 server.key    # Private key - read/write for owner only
chmod 644 server.crt    # Server certificate - readable by all
chmod 644 ca.crt        # CA certificate - readable by all
```

### Additional Resources

- [OpenSSL Documentation](https://www.openssl.org/docs/)
- [Python cryptography Documentation](https://cryptography.io/)
- [Java keytool Documentation](https://docs.oracle.com/javase/tutorial/security/toolsign/rstep2.html)

