#!/usr/bin/env python3
"""
MOSAIC SSL Certificate Generation Utility

This utility creates the SSL components required for secure communication:
- CA (Certificate Authority) certificate
- Server certificate
- Server private key

It attempts multiple methods in order of preference:
1. OpenSSL (command-line tool)
2. Python cryptography library
3. Java keytool (for basic certificates)
4. Comprehensive instructions if all methods fail
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
from typing import Optional, Tuple, List


class CertificateGenerator:
    """Handles SSL certificate generation with multiple fallback methods."""
    
    def __init__(self, output_dir: str = "certs", hostname: str = "localhost", 
                 validity_days: int = 365, key_size: int = 2048,
                 ca_key_name: Optional[str] = None, ca_crt_name: Optional[str] = None,
                 server_key_name: Optional[str] = None, server_crt_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.hostname = hostname
        self.validity_days = validity_days
        self.key_size = key_size
        
        # Set default filenames if not provided
        self.ca_key_name = ca_key_name or "ca.key"
        self.ca_crt_name = ca_crt_name or "ca.crt"
        self.server_key_name = server_key_name or "server.key"
        self.server_crt_name = server_crt_name or "server.crt"
        
        # Paths will be set after ensuring extensions
        self.ca_key_path = None
        self.ca_crt_path = None
        self.server_key_path = None
        self.server_crt_path = None
        self.server_csr_path = None
        
        # Ensure extensions and set paths
        self._ensure_extensions_and_set_paths()
        
    def _ensure_extension(self, filename: str, expected_extensions: List[str]) -> str:
        """Ensure filename has one of the expected extensions, add if missing."""
        filename_lower = filename.lower()
        has_extension = any(filename_lower.endswith(ext.lower()) for ext in expected_extensions)
        
        if not has_extension:
            # If filename has any extension (like .txt), strip it first
            if '.' in filename:
                # Find the last dot and remove everything after it
                filename = filename.rsplit('.', 1)[0]
            # Add the first (preferred) extension
            filename = filename + expected_extensions[0]
        
        return filename
    
    def _ensure_extensions_and_set_paths(self) -> None:
        """Ensure all filenames have appropriate extensions and set paths."""
        # Ensure extensions
        self.ca_key_name = self._ensure_extension(self.ca_key_name, [".key", ".pem"])
        self.ca_crt_name = self._ensure_extension(self.ca_crt_name, [".crt", ".pem", ".cer"])
        self.server_key_name = self._ensure_extension(self.server_key_name, [".key", ".pem"])
        self.server_crt_name = self._ensure_extension(self.server_crt_name, [".crt", ".pem", ".cer"])
        
        # Set paths
        self.ca_key_path = self.output_dir / self.ca_key_name
        self.ca_crt_path = self.output_dir / self.ca_crt_name
        self.server_key_path = self.output_dir / self.server_key_name
        self.server_crt_path = self.output_dir / self.server_crt_name
        # CSR is temporary, use a derived name from server cert
        csr_name = Path(self.server_crt_name).stem + ".csr"
        self.server_csr_path = self.output_dir / csr_name
    
    def prompt_for_filenames(self) -> None:
        """Interactively prompt user for preferred filenames."""
        print("\n" + "=" * 50)
        print("Certificate File Names")
        print("=" * 50)
        print("You can customize the filenames for the generated certificates.")
        print("Press Enter to use the default names shown in brackets.")
        print("=" * 50)
        
        # CA Key
        default_ca_key = self.ca_key_name
        prompt = f"\nCA Private Key filename [{default_ca_key}]: "
        try:
            user_input = input(prompt).strip()
            if user_input:
                self.ca_key_name = user_input
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default filename.")
        
        # CA Certificate
        default_ca_crt = self.ca_crt_name
        prompt = f"CA Certificate filename [{default_ca_crt}]: "
        try:
            user_input = input(prompt).strip()
            if user_input:
                self.ca_crt_name = user_input
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default filename.")
        
        # Server Key
        default_server_key = self.server_key_name
        prompt = f"Server Private Key filename [{default_server_key}]: "
        try:
            user_input = input(prompt).strip()
            if user_input:
                self.server_key_name = user_input
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default filename.")
        
        # Server Certificate
        default_server_crt = self.server_crt_name
        prompt = f"Server Certificate filename [{default_server_crt}]: "
        try:
            user_input = input(prompt).strip()
            if user_input:
                self.server_crt_name = user_input
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default filename.")
        
        # Re-ensure extensions and update paths
        self._ensure_extensions_and_set_paths()
        
        # Show final filenames
        print("\n" + "=" * 50)
        print("Final File Names (with extensions ensured):")
        print("=" * 50)
        print(f"  CA Private Key:     {self.ca_key_name}")
        print(f"  CA Certificate:     {self.ca_crt_name}")
        print(f"  Server Private Key:  {self.server_key_name}")
        print(f"  Server Certificate:  {self.server_crt_name}")
        print("=" * 50)
    
    def check_existing_files(self) -> bool:
        """Check for existing files and prompt user for overwrite confirmation."""
        existing_files = []
        
        # Check which files already exist
        if self.ca_key_path.exists():
            existing_files.append(("CA Private Key", self.ca_key_path))
        if self.ca_crt_path.exists():
            existing_files.append(("CA Certificate", self.ca_crt_path))
        if self.server_key_path.exists():
            existing_files.append(("Server Private Key", self.server_key_path))
        if self.server_crt_path.exists():
            existing_files.append(("Server Certificate", self.server_crt_path))
        
        if not existing_files:
            return True  # No existing files, proceed
        
        # Show existing files
        print("\n" + "=" * 50)
        print("⚠ Warning: The following files already exist:")
        print("=" * 50)
        for file_desc, file_path in existing_files:
            file_size = file_path.stat().st_size
            print(f"  {file_desc}: {file_path.name} ({file_size} bytes)")
        print("=" * 50)
        
        # Prompt for confirmation
        print("\nThese files will be overwritten if you continue.")
        try:
            response = input("Do you want to overwrite these files? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("✓ Proceeding with overwrite...")
                return True
            else:
                print("✗ Operation cancelled. Existing files will not be modified.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n\nOperation cancelled by user.")
            return False
    
    def ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir.absolute()}")
    
    def check_openssl(self) -> bool:
        """Check if OpenSSL is available."""
        try:
            result = subprocess.run(
                ["openssl", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ OpenSSL found: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return False
    
    def check_pip(self) -> Optional[str]:
        """Check if pip is available and return the command."""
        for pip_cmd in ["pip3", "pip"]:
            try:
                result = subprocess.run(
                    [pip_cmd, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return pip_cmd
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        return None
    
    def install_cryptography_interactive(self) -> bool:
        """Interactively install the cryptography library."""
        pip_cmd = self.check_pip()
        if not pip_cmd:
            print("\n✗ pip is not available. Cannot install cryptography library automatically.")
            print("Please install pip first, or install cryptography manually:")
            print("  pip install cryptography")
            print("  or")
            print("  pip3 install cryptography")
            return False
        
        print(f"\n⚠ Python cryptography library is not installed.")
        print(f"It's needed as a fallback if OpenSSL is not available.")
        print(f"\nWould you like to install it now using {pip_cmd}? (y/n): ", end="")
        
        try:
            response = input().strip().lower()
            if response in ['y', 'yes']:
                print(f"\nInstalling cryptography library using {pip_cmd}...")
                result = subprocess.run(
                    [pip_cmd, "install", "cryptography>=41.0.0"],
                    timeout=300  # 5 minutes should be enough
                )
                if result.returncode == 0:
                    print("✓ Successfully installed cryptography library")
                    return True
                else:
                    print("✗ Failed to install cryptography library")
                    print(f"Try installing manually: {pip_cmd} install cryptography")
                    return False
            else:
                print("Skipping cryptography installation. You can install it later with:")
                print(f"  {pip_cmd} install cryptography")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n\nInstallation cancelled by user.")
            return False
        except Exception as e:
            print(f"\n✗ Error during installation: {e}")
            return False
    
    def check_python_cryptography(self) -> bool:
        """Check if Python cryptography library is available."""
        try:
            import cryptography
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            print(f"✓ Python cryptography library found: {cryptography.__version__}")
            return True
        except ImportError:
            return False
    
    def check_java_keytool(self) -> bool:
        """Check if Java keytool is available."""
        try:
            result = subprocess.run(
                ["keytool", "-help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Try to get Java version
                java_result = subprocess.run(
                    ["java", "-version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                print("✓ Java keytool found")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return False
    
    def generate_with_openssl(self) -> bool:
        """Generate certificates using OpenSSL."""
        print("\n=== Attempting to generate certificates with OpenSSL ===")
        
        try:
            # Generate CA private key
            print("Generating CA private key...")
            subprocess.run(
                ["openssl", "genrsa", "-out", str(self.ca_key_path), str(self.key_size)],
                check=True,
                capture_output=True
            )
            
            # Generate CA certificate
            print("Generating CA certificate...")
            ca_subj = f"/C=US/ST=State/L=City/O=MOSAIC/CN=MOSAIC-CA"
            subprocess.run(
                [
                    "openssl", "req", "-new", "-x509", "-days", str(self.validity_days),
                    "-key", str(self.ca_key_path), "-out", str(self.ca_crt_path),
                    "-subj", ca_subj
                ],
                check=True,
                capture_output=True
            )
            
            # Generate server private key
            print("Generating server private key...")
            subprocess.run(
                ["openssl", "genrsa", "-out", str(self.server_key_path), str(self.key_size)],
                check=True,
                capture_output=True
            )
            
            # Generate server certificate signing request
            print("Generating server certificate signing request...")
            server_subj = f"/C=US/ST=State/L=City/O=MOSAIC/CN={self.hostname}"
            subprocess.run(
                [
                    "openssl", "req", "-new", "-key", str(self.server_key_path),
                    "-out", str(self.server_csr_path), "-subj", server_subj
                ],
                check=True,
                capture_output=True
            )
            
            # Generate server certificate signed by CA
            print("Generating server certificate signed by CA...")
            subprocess.run(
                [
                    "openssl", "x509", "-req", "-days", str(self.validity_days),
                    "-in", str(self.server_csr_path), "-CA", str(self.ca_crt_path),
                    "-CAkey", str(self.ca_key_path), "-CAcreateserial",
                    "-out", str(self.server_crt_path)
                ],
                check=True,
                capture_output=True
            )
            
            # Clean up CSR file
            if self.server_csr_path.exists():
                self.server_csr_path.unlink()
            
            print("✓ Successfully generated certificates with OpenSSL")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ OpenSSL generation failed: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error with OpenSSL: {e}")
            return False
    
    def generate_with_python_cryptography(self) -> bool:
        """Generate certificates using Python cryptography library."""
        print("\n=== Attempting to generate certificates with Python cryptography ===")
        
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.x509.oid import NameOID
            from datetime import datetime, timedelta, timezone
            import ipaddress
            
            # Generate CA private key
            print("Generating CA private key...")
            ca_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size
            )
            
            # Generate CA certificate
            print("Generating CA certificate...")
            ca_subject = ca_issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MOSAIC"),
                x509.NameAttribute(NameOID.COMMON_NAME, "MOSAIC-CA"),
            ])
            
            ca_cert = x509.CertificateBuilder().subject_name(
                ca_subject
            ).issuer_name(
                ca_issuer
            ).public_key(
                ca_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.now(timezone.utc)
            ).not_valid_after(
                datetime.now(timezone.utc) + timedelta(days=self.validity_days)
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            ).sign(ca_key, hashes.SHA256())
            
            # Write CA key
            with open(self.ca_key_path, "wb") as f:
                f.write(ca_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Write CA certificate
            with open(self.ca_crt_path, "wb") as f:
                f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
            
            # Generate server private key
            print("Generating server private key...")
            server_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size
            )
            
            # Generate server certificate
            print("Generating server certificate signed by CA...")
            server_subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MOSAIC"),
                x509.NameAttribute(NameOID.COMMON_NAME, self.hostname),
            ])
            
            server_cert = x509.CertificateBuilder().subject_name(
                server_subject
            ).issuer_name(
                ca_subject
            ).public_key(
                server_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.now(timezone.utc)
            ).not_valid_after(
                datetime.now(timezone.utc) + timedelta(days=self.validity_days)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(self.hostname),
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]),
                critical=False,
            ).sign(ca_key, hashes.SHA256())
            
            # Write server key
            with open(self.server_key_path, "wb") as f:
                f.write(server_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Write server certificate
            with open(self.server_crt_path, "wb") as f:
                f.write(server_cert.public_bytes(serialization.Encoding.PEM))
            
            print("✓ Successfully generated certificates with Python cryptography")
            return True
            
        except ImportError:
            print("✗ Python cryptography library not available")
            return False
        except Exception as e:
            print(f"✗ Python cryptography generation failed: {e}")
            return False
    
    def generate_with_java_keytool(self) -> bool:
        """Generate certificates using Java keytool (limited functionality)."""
        print("\n=== Attempting to generate certificates with Java keytool ===")
        print("⚠ Note: Java keytool has limited functionality. This is a basic fallback.")
        
        try:
            # Create a keystore for the server
            keystore_path = self.output_dir / "server.p12"
            print("Generating server keystore...")
            subprocess.run(
                [
                    "keytool", "-genkeypair",
                    "-alias", "server",
                    "-keyalg", "RSA",
                    "-keysize", str(self.key_size),
                    "-validity", str(self.validity_days),
                    "-keystore", str(keystore_path),
                    "-storetype", "PKCS12",
                    "-storepass", "changeit",
                    "-keypass", "changeit",
                    "-dname", f"CN={self.hostname}, OU=MOSAIC, O=MOSAIC, L=City, ST=State, C=US"
                ],
                check=True,
                input="yes\n",
                text=True,
                capture_output=True
            )
            
            # Export server certificate
            print("Exporting server certificate...")
            subprocess.run(
                [
                    "keytool", "-exportcert",
                    "-alias", "server",
                    "-keystore", str(keystore_path),
                    "-storepass", "changeit",
                    "-file", str(self.server_crt_path)
                ],
                check=True,
                capture_output=True
            )
            
            # Note: Java keytool doesn't easily create separate CA and server certs
            # For simplicity, we'll use the same cert as CA
            print("⚠ Using server certificate as CA certificate (keytool limitation)")
            import shutil
            shutil.copy(self.server_crt_path, self.ca_crt_path)
            
            # Extract private key (this is tricky with keytool, so we'll note it)
            print("⚠ Note: Private key is in PKCS12 keystore. Extracting...")
            # We can't easily extract the key with keytool alone, so we'll provide instructions
            
            print("✓ Generated keystore with Java keytool")
            print(f"  Server certificate: {self.server_crt_path}")
            print(f"  CA certificate: {self.ca_crt_path}")
            print(f"  Keystore (contains private key): {keystore_path}")
            print("  ⚠ You may need to extract the private key separately or use the keystore directly")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Java keytool generation failed: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error with Java keytool: {e}")
            return False
    
    def print_instructions(self) -> None:
        """Print comprehensive instructions for manual certificate generation."""
        print("\n" + "="*70)
        print("INSTRUCTIONS FOR MANUAL CERTIFICATE GENERATION")
        print("="*70)
        
        system = platform.system().lower()
        
        print(f"\nDetected OS: {platform.system()} ({platform.release()})")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Hostname: {self.hostname}")
        print(f"Validity: {self.validity_days} days")
        print(f"Key size: {self.key_size} bits")
        
        print("\n--- OPTION 1: Install OpenSSL (Recommended) ---")
        if system == "windows":
            print("""
Windows:
  1. Download OpenSSL from: https://slproweb.com/products/Win32OpenSSL.html
     OR use Chocolatey: choco install openssl
     OR use WSL (Windows Subsystem for Linux) and install openssl there
  
  2. After installation, add OpenSSL to your PATH or use full path
  
  3. Run these commands:
     cd """ + str(self.output_dir.absolute()) + """
     
     # Generate CA key
     openssl genrsa -out ca.key """ + str(self.key_size) + """
     
     # Generate CA certificate
     openssl req -new -x509 -days """ + str(self.validity_days) + """ -key ca.key -out ca.crt -subj "/C=US/ST=State/L=City/O=MOSAIC/CN=MOSAIC-CA"
     
     # Generate server key
     openssl genrsa -out server.key """ + str(self.key_size) + """
     
     # Generate server CSR
     openssl req -new -key server.key -out server.csr -subj "/C=US/ST=State/L=City/O=MOSAIC/CN=""" + self.hostname + """"
     
     # Generate server certificate
     openssl x509 -req -days """ + str(self.validity_days) + """ -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt
""")
        elif system == "darwin":  # macOS
            print("""
macOS:
  1. Install OpenSSL via Homebrew:
     brew install openssl
  
  2. Run these commands:
     cd """ + str(self.output_dir.absolute()) + """
     
     # Generate CA key
     openssl genrsa -out ca.key """ + str(self.key_size) + """
     
     # Generate CA certificate
     openssl req -new -x509 -days """ + str(self.validity_days) + """ -key ca.key -out ca.crt -subj "/C=US/ST=State/L=City/O=MOSAIC/CN=MOSAIC-CA"
     
     # Generate server key
     openssl genrsa -out server.key """ + str(self.key_size) + """
     
     # Generate server CSR
     openssl req -new -key server.key -out server.csr -subj "/C=US/ST=State/L=City/O=MOSAIC/CN=""" + self.hostname + """"
     
     # Generate server certificate
     openssl x509 -req -days """ + str(self.validity_days) + """ -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt
""")
        else:  # Linux
            print("""
Linux:
  1. Install OpenSSL (usually pre-installed, but if not):
     Ubuntu/Debian: sudo apt-get install openssl
     Fedora/RHEL: sudo dnf install openssl
     Arch: sudo pacman -S openssl
  
  2. Run these commands:
     cd """ + str(self.output_dir.absolute()) + """
     
     # Generate CA key
     openssl genrsa -out ca.key """ + str(self.key_size) + """
     
     # Generate CA certificate
     openssl req -new -x509 -days """ + str(self.validity_days) + """ -key ca.key -out ca.crt -subj "/C=US/ST=State/L=City/O=MOSAIC/CN=MOSAIC-CA"
     
     # Generate server key
     openssl genrsa -out server.key """ + str(self.key_size) + """
     
     # Generate server CSR
     openssl req -new -key server.key -out server.csr -subj "/C=US/ST=State/L=City/O=MOSAIC/CN=""" + self.hostname + """"
     
     # Generate server certificate
     openssl x509 -req -days """ + str(self.validity_days) + """ -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt
""")
        
        print("\n--- OPTION 2: Install Python cryptography library ---")
        print("""
  1. Install the cryptography library:
     pip install cryptography
     OR
     pip3 install cryptography
  
  2. Re-run this script - it will automatically use the Python cryptography library
""")
        
        print("\n--- OPTION 3: Use Java keytool (Limited) ---")
        print("""
  1. Ensure Java is installed (keytool comes with Java)
  
  2. Run this script again - it will attempt to use keytool as a fallback
  
  Note: keytool has limitations and may not produce the exact format needed
""")
        
        print("\n--- OPTION 4: Use Online Certificate Generators (Not Recommended for Production) ---")
        print("""
  For development/testing only:
  - Use online tools like: https://www.selfsignedcertificate.com/
  - Or use mkcert: https://github.com/FiloSottile/mkcert
  
  ⚠ WARNING: Do not use online tools for production certificates!
""")
        
        print("\n--- Required Files ---")
        print(f"""
After generation, you should have these files in {self.output_dir.absolute()}:
  - ca.crt      (Certificate Authority certificate)
  - server.crt  (Server certificate)
  - server.key  (Server private key)
  
These paths should be configured in your MOSAIC config file:
  - ca_crt: "{self.ca_crt_path.absolute()}"
  - server_crt: "{self.server_crt_path.absolute()}"
  - server_key: "{self.server_key_path.absolute()}"
""")
        
        print("="*70)
    
    def verify_certificates(self) -> bool:
        """Verify that all required certificate files exist."""
        required_files = [
            self.ca_crt_path,
            self.server_crt_path,
            self.server_key_path
        ]
        
        all_exist = all(f.exists() for f in required_files)
        
        if all_exist:
            print("\n✓ All required certificate files generated successfully:")
            for f in required_files:
                size = f.stat().st_size
                print(f"  - {f.name} ({size} bytes)")
            return True
        else:
            print("\n✗ Some certificate files are missing:")
            for f in required_files:
                status = "✓" if f.exists() else "✗"
                print(f"  {status} {f.name}")
            return False
    
    def generate(self, interactive_filenames: bool = True) -> bool:
        """Main method to generate certificates using available methods."""
        self.ensure_output_dir()
        
        print("MOSAIC SSL Certificate Generation Utility")
        print("=" * 50)
        print(f"Hostname: {self.hostname}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Validity: {self.validity_days} days")
        print(f"Key size: {self.key_size} bits")
        print("=" * 50)
        
        # Prompt for filenames if interactive mode
        if interactive_filenames:
            try:
                self.prompt_for_filenames()
            except (EOFError, KeyboardInterrupt):
                print("\n\nOperation cancelled by user.")
                return False
        
        # Check for existing files and prompt for overwrite confirmation
        if not self.check_existing_files():
            return False
        
        # Check available methods
        print("\nChecking for available certificate generation tools...")
        methods = []
        has_openssl = self.check_openssl()
        has_cryptography = self.check_python_cryptography()
        has_keytool = self.check_java_keytool()
        
        if has_openssl:
            methods.append(("OpenSSL", self.generate_with_openssl))
        if has_cryptography:
            methods.append(("Python cryptography", self.generate_with_python_cryptography))
        if has_keytool:
            methods.append(("Java keytool", self.generate_with_java_keytool))
        
        # If no methods available, offer to install cryptography
        if not methods:
            print("\n⚠ No certificate generation methods are currently available!")
            print("\nAvailable options:")
            print("  1. OpenSSL (command-line tool) - Recommended")
            print("  2. Python cryptography library - Can be installed automatically")
            print("  3. Java keytool - Requires Java installation")
            
            # Offer to install cryptography if pip is available
            if not has_cryptography:
                pip_cmd = self.check_pip()
                if pip_cmd:
                    print(f"\n💡 Tip: The Python cryptography library can be installed automatically.")
                    if self.install_cryptography_interactive():
                        # Re-check after installation
                        if self.check_python_cryptography():
                            methods.append(("Python cryptography", self.generate_with_python_cryptography))
            
            # If still no methods, show instructions
            if not methods:
                print("\n✗ No certificate generation methods available!")
                self.print_instructions()
                return False
        elif not has_cryptography and not has_openssl:
            # If we only have keytool, suggest installing cryptography as a better option
            pip_cmd = self.check_pip()
            if pip_cmd:
                print(f"\n💡 Tip: Installing Python cryptography library would provide a better fallback option.")
                if self.install_cryptography_interactive():
                    if self.check_python_cryptography():
                        methods.insert(0, ("Python cryptography", self.generate_with_python_cryptography))
        
        # Try each method in order
        for method_name, method_func in methods:
            if method_func():
                if self.verify_certificates():
                    print(f"\n✓ Successfully generated certificates using {method_name}")
                    print(f"\nCertificate files are ready in: {self.output_dir.absolute()}")
                    return True
                else:
                    print(f"\n⚠ {method_name} completed but verification failed")
                    continue
        
        # If all methods failed
        print("\n✗ All certificate generation methods failed!")
        self.print_instructions()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SSL certificates for MOSAIC secure communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate certificates with default settings
  python generate_certs.py
  
  # Generate certificates for a specific hostname
  python generate_certs.py --hostname my-server.example.com
  
  # Generate certificates in a custom directory
  python generate_certs.py --output-dir /etc/mosaic/certs
  
  # Generate certificates with custom validity period
  python generate_certs.py --validity-days 730
        """
    )
    
    parser.add_argument(
        "--output-dir",
        default="certs",
        help="Directory to output certificate files (default: certs)"
    )
    parser.add_argument(
        "--hostname",
        default="localhost",
        help="Hostname/CN for the server certificate (default: localhost)"
    )
    parser.add_argument(
        "--validity-days",
        type=int,
        default=365,
        help="Certificate validity period in days (default: 365)"
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=2048,
        choices=[1024, 2048, 4096],
        help="RSA key size in bits (default: 2048)"
    )
    parser.add_argument(
        "--ca-key-name",
        default=None,
        help="CA private key filename (default: ca.key, extension will be added if missing)"
    )
    parser.add_argument(
        "--ca-crt-name",
        default=None,
        help="CA certificate filename (default: ca.crt, extension will be added if missing)"
    )
    parser.add_argument(
        "--server-key-name",
        default=None,
        help="Server private key filename (default: server.key, extension will be added if missing)"
    )
    parser.add_argument(
        "--server-crt-name",
        default=None,
        help="Server certificate filename (default: server.crt, extension will be added if missing)"
    )
    parser.add_argument(
        "--no-interactive-filenames",
        action="store_true",
        help="Skip interactive filename prompts (use defaults or command-line arguments)"
    )
    
    args = parser.parse_args()
    
    generator = CertificateGenerator(
        output_dir=args.output_dir,
        hostname=args.hostname,
        validity_days=args.validity_days,
        key_size=args.key_size,
        ca_key_name=args.ca_key_name,
        ca_crt_name=args.ca_crt_name,
        server_key_name=args.server_key_name,
        server_crt_name=args.server_crt_name
    )
    
    success = generator.generate(interactive_filenames=not args.no_interactive_filenames)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

