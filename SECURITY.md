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

# Generate certificates for multi-node setup (recommended)
# Include all node IPs/hostnames so the same certificate works on all nodes
python generate_certs.py --hostname 192.168.1.10 --additional-hostnames 192.168.1.11 192.168.1.12 192.168.1.13

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
# For multi-node setup:
./generate_certs.sh --hostname 192.168.1.10 --additional-hostnames 192.168.1.11 192.168.1.12
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
--output-dir DIR              Directory to output certificate files (default: certs)
--hostname HOSTNAME           Primary hostname/CN for the server certificate (default: localhost)
--additional-hostnames HOSTS  Additional hostnames or IP addresses to include in Subject 
                              Alternative Name (SAN). Useful for multi-node setups where the 
                              same certificate is used on multiple nodes. Can specify multiple.
--validity-days DAYS          Certificate validity period in days (default: 365)
--key-size BITS               RSA key size in bits: 1024, 2048, or 4096 (default: 2048)
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

#### Production Setup (Multi-Node)
```bash
# For multi-node deployment, include all node IPs/hostnames
python generate_certs.py \
  --hostname 192.168.1.10 \
  --additional-hostnames 192.168.1.11 192.168.1.12 192.168.1.13 \
  --output-dir /etc/mosaic/certs \
  --validity-days 730 \
  --key-size 4096
```

#### Production Setup (Single Node)
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
- For multi-node setups: Ensure all node IPs/hostnames are included in the certificate SAN
- Verify certificate expiration dates

### Multi-Node Certificate Setup

For multi-node MOSAIC deployments, you can use the same certificate files on all nodes. This simplifies certificate management while maintaining security through CA signature verification.

**How it works:**
1. Generate certificates with all node IPs/hostnames included in the Subject Alternative Name (SAN) extension
2. Copy the same three certificate files (`ca.crt`, `server.crt`, `server.key`) to all nodes
3. MOSAIC automatically disables strict hostname verification for shared certificates while still verifying the certificate chain (CA signature)

**Example for 3-node network:**
```bash
# Generate certificates with all node IPs
python generate_certs.py \
  --hostname 192.168.1.10 \
  --additional-hostnames 192.168.1.11 192.168.1.12

# Copy the same files to all nodes
scp certs/* user@node1:/home/user/mosaic/certs/
scp certs/* user@node2:/home/user/mosaic/certs/
scp certs/* user@node3:/home/user/mosaic/certs/
```

**Security considerations:**
- The certificate chain (CA signature) is still verified, providing authentication
- Hostname verification is disabled to allow shared certificates across nodes
- All nodes must have the same CA certificate to trust each other
- Private key (`server.key`) must be kept secure on all nodes

### SSL Behavior in MOSAIC

⚠️ **Critical Security Behavior**: 

MOSAIC's communication system (beacon) has a **graceful degradation** behavior for SSL certificates:

- **If SSL certificate validation fails** (files missing, invalid, or cannot be loaded), MOSAIC will:
  - Log warning messages about the SSL validation failure
  - **Still create sockets and start the system**
  - **Run in unencrypted mode** (no SSL/TLS protection)
  - Continue normal operation without encryption

This behavior allows the system to start even if certificates are misconfigured, but **this means your communication will be unencrypted**. 

**What to check:**
- Verify certificate file paths in your configuration are correct
- Ensure certificate files exist and are readable
- Check that certificate files are valid (not corrupted)
- Review MOSAIC startup logs for SSL validation warnings
- In production, always verify SSL is enabled by checking logs for "SSL certificates validated successfully, SSL enabled"

**Why this matters:**
- Unencrypted communication exposes all data transmitted between nodes
- Model weights, training data, and inference results may be transmitted in plain text
- Always ensure SSL certificates are properly configured before deploying to production

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
   - **Always verify SSL is enabled** by checking startup logs for successful SSL validation messages

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

