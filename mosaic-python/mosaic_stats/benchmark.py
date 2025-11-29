"""Machine benchmarking module for Mosaic."""

import json
import logging
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from mosaic_config.config import MosaicConfig

try:
    import numpy as np
except ImportError:
    np = None

try:
    import psutil
except ImportError:
    psutil = None

# GPU libraries - try multiple for wide coverage
GPU_LIBRARIES = {
    "nvidia": None,
    "amd": None,
    "intel": None,
    "habana": None,
}

try:
    import pynvml

    GPU_LIBRARIES["nvidia"] = pynvml
except ImportError:
    pass

try:
    import GPUtil

    if GPU_LIBRARIES["nvidia"] is None:
        GPU_LIBRARIES["nvidia"] = GPUtil
except ImportError:
    pass

# Try PyTorch for GPU benchmarking (widely available)
try:
    import torch

    if torch.cuda.is_available():
        GPU_LIBRARIES["nvidia"] = "pytorch"
except ImportError:
    pass

# Try Habana Gaudi (AI accelerators)
try:
    # Check if hl-smi is available (Habana System Management Interface)
    result = subprocess.run(
        ["hl-smi", "--version"], capture_output=True, timeout=1, check=False
    )
    if result.returncode == 0:
        GPU_LIBRARIES["habana"] = "hl-smi"
except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
    pass

# Try PyTorch Habana backend
if GPU_LIBRARIES["habana"] is None:
    try:
        import torch

        if hasattr(torch, "habana") and torch.habana.is_available():
            GPU_LIBRARIES["habana"] = "pytorch"
    except (ImportError, AttributeError):
        pass

logger = logging.getLogger(__name__)

# Cache for performance backup data
_performance_backup_cache: Optional[List[Dict[str, Any]]] = None


def _load_performance_backup() -> List[Dict[str, Any]]:
    """
    Load performance backup data from performance_backup.json.

    Returns:
        List of performance entries, or empty list if file not found or invalid
    """
    global _performance_backup_cache

    # Return cached data if available
    if _performance_backup_cache is not None:
        return _performance_backup_cache

    try:
        # Try to find performance_backup.json in the same directory as this module
        module_dir = Path(__file__).parent
        backup_file = module_dir / "performance_backup.json"

        if not backup_file.exists():
            logger.debug("performance_backup.json not found")
            _performance_backup_cache = []
            return []

        with open(backup_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.warning("performance_backup.json has invalid format")
            _performance_backup_cache = []
            return []

        _performance_backup_cache = data
        return data
    except Exception as e:
        logger.warning(f"Failed to load performance_backup.json: {e}")
        _performance_backup_cache = []
        return []


def _lookup_gflops_from_backup(name: str) -> Optional[float]:
    """
    Look up gflops value from performance backup by matching device name.

    Args:
        name: Device name (CPU or GPU) to look up

    Returns:
        Gflops value if found, None otherwise
    """
    if not name:
        return None

    # Lowercase the name for matching
    name_lower = name.lower().strip()

    backup_data = _load_performance_backup()

    for entry in backup_data:
        identifiers = entry.get("gpu_identifier", [])
        if not isinstance(identifiers, list):
            continue

        # Check if any identifier matches (case-insensitive)
        for identifier in identifiers:
            if isinstance(identifier, str) and identifier.lower().strip() == name_lower:
                gflops = entry.get("gflops")
                if isinstance(gflops, (int, float)):
                    logger.info(f"Found gflops backup for '{name}': {gflops}")
                    return float(gflops)

        # Also check for partial matches (name contains identifier or vice versa)
        for identifier in identifiers:
            if isinstance(identifier, str):
                identifier_lower = identifier.lower().strip()
                if identifier_lower in name_lower or name_lower in identifier_lower:
                    gflops = entry.get("gflops")
                    if isinstance(gflops, (int, float)):
                        logger.info(f"Found gflops backup for '{name}' (partial match '{identifier}'): {gflops}")
                        return float(gflops)

    return None


def _get_hostname() -> str:
    """Get the hostname of the machine."""
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _benchmark_disk_speed(benchmark_data_location: Optional[Path]) -> Dict[str, Any]:
    """
    Benchmark disk I/O speed for the disk hosting benchmark_data_location.

    Args:
        benchmark_data_location: Path to benchmark data location

    Returns:
        Dictionary with disk benchmark results
    """
    if benchmark_data_location is None:
        return {"error": "No benchmark data location specified"}

    try:
        # Ensure directory exists
        benchmark_data_location.mkdir(parents=True, exist_ok=True)

        # Use a small test file for speed (1MB)
        test_file_size = 1024 * 1024  # 1 MB
        test_file = benchmark_data_location / ".benchmark_test.tmp"

        # Test write speed
        test_data = b"0" * test_file_size
        start_time = time.time_ns() // 1_000_000
        with open(test_file, "wb") as f:
            f.write(test_data)
        write_time = (time.time_ns() // 1_000_000) - start_time
        write_speed_mbps = (test_file_size / (1024 * 1024)) / write_time if write_time > 0 else 0

        # Sync/flush to ensure data is written
        if hasattr(os, "sync"):
            os.sync()

        # Test read speed
        start_time = time.time_ns() // 1_000_000
        with open(test_file, "rb") as f:
            _ = f.read()
        read_time = (time.time_ns() // 1_000_000) - start_time
        read_speed_mbps = (test_file_size / (1024 * 1024)) / read_time if read_time > 0 else 0

        # Cleanup
        try:
            test_file.unlink()
        except Exception:
            pass

        return {
            "write_speed_mbps": round(write_speed_mbps, 2),
            "read_speed_mbps": round(read_speed_mbps, 2),
            "test_size_bytes": test_file_size,
        }
    except Exception as e:
        logger.warning(f"Disk benchmark failed: {e}")
        return {"error": str(e)}


def _benchmark_cpu_flops() -> Dict[str, Any]:
    """
    Benchmark CPU floating point operations per second (FLOPS).

    Returns:
        Dictionary with CPU FLOPS benchmark results
    """
    try:
        if np is not None:
            # Use NumPy for faster computation
            size = 1000
            iterations = 100

            # Matrix multiplication benchmark
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)

            start_time = time.time_ns() // 1_000_000
            for _ in range(iterations):
                _ = np.dot(a, b)
            elapsed = (time.time_ns() // 1_000_000) - start_time

            # FLOPS = 2 * size^3 * iterations / time
            # (2 because multiply-add counts as 2 operations)
            flops = (2 * size**3 * iterations) / elapsed if elapsed > 0 else 0
            gflops = flops / 1e9

            result = {
                "gflops": round(gflops, 2),
                "test_size": size,
                "iterations": iterations,
            }
        else:
            # Fallback to pure Python (slower but works without numpy)
            size = 100
            iterations = 10

            # Simple floating point operations
            start_time = time.time_ns() // 1_000_000
            result = 0.0
            for _ in range(iterations):
                for i in range(size):
                    for j in range(size):
                        result += i * j * 0.5
            elapsed = (time.time_ns() // 1_000_000) - start_time

            # Rough estimate: size^2 * iterations operations
            ops = size * size * iterations
            flops = ops / elapsed if elapsed > 0 else 0
            gflops = flops / 1e9

            result = {
                "gflops": round(gflops, 4),  # Lower precision for pure Python
                "test_size": size,
                "iterations": iterations,
                "note": "pure_python",
            }

        # If gflops is None, try backup lookup
        if result.get("gflops") is None:
            try:
                # Get CPU name
                cpu_name = None
                if psutil is not None:
                    try:
                        cpu_name = platform.processor()
                        if not cpu_name or cpu_name == "":
                            # Try alternative method
                            cpu_name = platform.machine()
                    except Exception:
                        pass

                if not cpu_name:
                    # Fallback to platform info
                    try:
                        cpu_name = platform.processor() or platform.machine()
                    except Exception:
                        cpu_name = "unknown"

                backup_gflops = _lookup_gflops_from_backup(cpu_name)
                if backup_gflops is not None:
                    result["gflops"] = backup_gflops
                    result["note"] = result.get("note", "") + "_backup" if result.get("note") else "backup"
                    result["cpu_name"] = cpu_name
            except Exception as e:
                logger.debug(f"Failed to lookup CPU gflops from backup: {e}")

        return result
    except Exception as e:
        logger.warning(f"CPU benchmark failed: {e}")
        # Try backup lookup even on error
        try:
            cpu_name = platform.processor() or platform.machine() or "unknown"
            backup_gflops = _lookup_gflops_from_backup(cpu_name)
            if backup_gflops is not None:
                return {
                    "gflops": backup_gflops,
                    "note": "backup",
                    "cpu_name": cpu_name,
                    "error": str(e),
                }
        except Exception:
            pass
        return {"error": str(e)}


def _benchmark_gpu_flops() -> List[Dict[str, Any]]:
    """
    Benchmark GPU floating point operations per second (FLOPS).

    Returns:
        List of dictionaries with GPU benchmark results (one per GPU)
    """
    gpu_results = []

    # Try PyTorch first (most common and reliable)
    if GPU_LIBRARIES["nvidia"] == "pytorch":
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    try:
                        device = torch.device(f"cuda:{i}")
                        torch.cuda.set_device(i)

                        # Small matrix multiplication for speed
                        size = 500
                        iterations = 50

                        a = torch.randn(size, size, device=device, dtype=torch.float32)
                        b = torch.randn(size, size, device=device, dtype=torch.float32)

                        # Warmup
                        for _ in range(5):
                            _ = torch.matmul(a, b)
                        torch.cuda.synchronize()

                        # Benchmark
                        start_time = time.time_ns() // 1_000_000
                        for _ in range(iterations):
                            _ = torch.matmul(a, b)
                        torch.cuda.synchronize()
                        elapsed = (time.time_ns() // 1_000_000) - start_time

                        # FLOPS = 2 * size^3 * iterations / time
                        flops = (2 * size**3 * iterations) / elapsed if elapsed > 0 else 0
                        gflops = flops / 1e9

                        gpu_name = torch.cuda.get_device_name(i)

                        gpu_result = {
                            "gpu_id": i,
                            "gpu_name": gpu_name,
                            "gflops": round(gflops, 2),
                            "test_size": size,
                            "iterations": iterations,
                            "type": "nvidia",
                        }

                        # If gflops is None or 0, try backup lookup
                        if gpu_result.get("gflops") is None or gpu_result.get("gflops") == 0:
                            backup_gflops = _lookup_gflops_from_backup(gpu_name)
                            if backup_gflops is not None:
                                gpu_result["gflops"] = backup_gflops
                                gpu_result["note"] = "backup"

                        gpu_results.append(gpu_result)
                    except Exception as e:
                        logger.warning(f"GPU {i} benchmark failed: {e}")
                        # Try to get GPU name and lookup backup even on error
                        try:
                            import torch
                            gpu_name = torch.cuda.get_device_name(i)
                            backup_gflops = _lookup_gflops_from_backup(gpu_name)
                            if backup_gflops is not None:
                                gpu_results.append({
                                    "gpu_id": i,
                                    "gpu_name": gpu_name,
                                    "gflops": backup_gflops,
                                    "note": "backup",
                                    "type": "nvidia",
                                    "error": str(e),
                                })
                            else:
                                gpu_results.append({"gpu_id": i, "error": str(e), "type": "nvidia"})
                        except Exception:
                            gpu_results.append({"gpu_id": i, "error": str(e), "type": "nvidia"})
        except Exception as e:
            logger.warning(f"PyTorch GPU benchmark failed: {e}")

    # Try pynvml for basic GPU info (if PyTorch not available)
    elif GPU_LIBRARIES["nvidia"] is not None and hasattr(GPU_LIBRARIES["nvidia"], "nvmlInit"):
        try:
            pynvml = GPU_LIBRARIES["nvidia"]
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

                    # For pynvml, we can't easily run compute benchmarks
                    # So we'll just report the GPU exists and use a placeholder
                    # In practice, PyTorch or CUDA would be needed for actual FLOPS
                    # Try backup lookup for gflops
                    backup_gflops = _lookup_gflops_from_backup(name)
                    gpu_result = {
                        "gpu_id": i,
                        "gpu_name": name,
                        "gflops": backup_gflops if backup_gflops is not None else None,
                        "note": "pynvml_no_compute" + ("_backup" if backup_gflops is not None else ""),
                        "type": "nvidia",
                    }
                    gpu_results.append(gpu_result)
                except Exception as e:
                    logger.warning(f"GPU {i} info failed: {e}")
        except Exception as e:
            logger.warning(f"pynvml GPU benchmark failed: {e}")

    # Try AMD ROCm
    if not gpu_results:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                # Parse output to count GPUs and get names if available
                lines = result.stdout.strip().split("\n")
                gpu_count = sum(1 for line in lines if "GPU" in line or "Card" in line)
                for i in range(gpu_count):
                    # Try to extract GPU name from output
                    gpu_name = None
                    for line in lines:
                        if f"GPU {i}" in line or f"Card {i}" in line:
                            # Try to extract name from line
                            parts = line.split()
                            if len(parts) > 2:
                                gpu_name = " ".join(parts[2:])  # Everything after "GPU X" or "Card X"

                    backup_gflops = _lookup_gflops_from_backup(gpu_name) if gpu_name else None
                    gpu_result = {
                        "gpu_id": i,
                        "gflops": backup_gflops if backup_gflops is not None else None,
                        "note": "amd_rocm_no_compute" + ("_backup" if backup_gflops is not None else ""),
                        "type": "amd",
                    }
                    if gpu_name:
                        gpu_result["gpu_name"] = gpu_name
                    gpu_results.append(gpu_result)
        except Exception:
            pass

    # Try Habana Gaudi
    if not gpu_results:
        try:
            # Try hl-smi first
            result = subprocess.run(
                ["hl-smi", "-L"],  # List devices
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                # Parse output to count devices
                lines = result.stdout.strip().split("\n")
                device_count = sum(1 for line in lines if "HL" in line or "Gaudi" in line or "Device" in line)
                for i in range(device_count):
                    gpu_results.append(
                        {
                            "gpu_id": i,
                            "gflops": None,
                            "note": "habana_no_compute",
                            "type": "habana",
                        }
                    )
        except Exception:
            pass

        # Try PyTorch Habana backend if hl-smi didn't work
        if not gpu_results:
            try:
                import torch

                if hasattr(torch, "habana") and torch.habana.is_available():
                    device_count = torch.habana.device_count()
                    for i in range(device_count):
                        gpu_results.append(
                            {
                                "gpu_id": i,
                                "gflops": None,
                                "note": "habana_pytorch_no_compute",
                                "type": "habana",
                            }
                        )
            except (ImportError, AttributeError):
                pass

    return gpu_results


def _benchmark_ram_speed() -> Dict[str, Any]:
    """
    Benchmark RAM memory bandwidth.

    Returns:
        Dictionary with RAM benchmark results
    """
    try:
        if np is not None:
            # Use NumPy for memory operations
            size = 10 * 1024 * 1024  # 10 MB
            iterations = 100

            # Allocate arrays
            a = np.random.rand(size).astype(np.float32)
            b = np.zeros_like(a)

            # Memory copy benchmark
            start_time = time.time_ns() // 1_000_000
            for _ in range(iterations):
                b[:] = a[:]
            elapsed = (time.time_ns() // 1_000_000) - start_time

            # Bandwidth = (size * sizeof(float32) * iterations) / time
            bytes_transferred = size * 4 * iterations  # float32 = 4 bytes
            bandwidth_gbps = (bytes_transferred / (1024**3)) / elapsed if elapsed > 0 else 0

            return {
                "bandwidth_gbps": round(bandwidth_gbps, 2),
                "test_size_mb": size / (1024 * 1024),
                "iterations": iterations,
            }
        else:
            # Fallback to pure Python
            size = 1024 * 1024  # 1 MB
            iterations = 10

            a = [0.0] * size
            b = [0.0] * size

            start_time = time.time_ns() // 1_000_000
            for _ in range(iterations):
                b[:] = a[:]
            elapsed = (time.time_ns() // 1_000_000) - start_time

            bytes_transferred = size * 8 * iterations  # Python float = 8 bytes
            bandwidth_gbps = (bytes_transferred / (1024**3)) / elapsed if elapsed > 0 else 0

            return {
                "bandwidth_gbps": round(bandwidth_gbps, 4),
                "test_size_mb": size / (1024 * 1024),
                "iterations": iterations,
                "note": "pure_python",
            }
    except Exception as e:
        logger.warning(f"RAM benchmark failed: {e}")
        return {"error": str(e)}


def run_benchmarks(benchmark_data_location: str) -> Dict[str, Any]:
    """
    Run all benchmark tests and return results.

    Args:
        benchmark_data_location: Path to benchmark data location

    Returns:
        Dictionary with all benchmark results including timestamp
    """
    benchmark_data_path = Path(benchmark_data_location) if benchmark_data_location else None

    results = {
        "timestamp_ms": int(time.time_ns() // 1_000_000),
        "host": _get_hostname(),
        "disk": _benchmark_disk_speed(benchmark_data_path) if benchmark_data_path else {"error": "No location"},
        "cpu": _benchmark_cpu_flops(),
        "gpus": _benchmark_gpu_flops(),
        "ram": _benchmark_ram_speed(),
    }

    return results


def save_benchmarks(benchmark_data_location: str, results: Optional[Dict[str, Any]] = None) -> Path:
    """
    Save benchmark results to file.

    Args:
        benchmark_data_location: Path to benchmark data location
        results: Optional benchmark results (if None, runs benchmarks first)

    Returns:
        Path to the saved benchmark file
    """
    if results is None:
        results = run_benchmarks(benchmark_data_location)

    benchmark_data_path = Path(benchmark_data_location)
    benchmark_data_path.mkdir(parents=True, exist_ok=True)

    hostname = results.get("host", _get_hostname())
    benchmark_file = benchmark_data_path / f"{hostname}_core_benchmark.json"

    # Write atomically
    temp_file = benchmark_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    temp_file.replace(benchmark_file)

    logger.info(f"Benchmark results saved to {benchmark_file}")
    return benchmark_file


def benchmarks_captured(benchmark_data_location: str) -> bool:
    """
    Check whether benchmark results exist for the current host.

    Args:
        benchmark_data_location: Path to benchmark data location

    Returns:
        True if benchmark file exists, False otherwise
    """
    if not benchmark_data_location:
        return False

    try:
        benchmark_data_path = Path(benchmark_data_location)
        hostname = _get_hostname()
        benchmark_file = benchmark_data_path / f"{hostname}_core_benchmark.json"
        return benchmark_file.exists()
    except Exception as e:
        logger.warning(f"Error checking benchmark file: {e}")
        return False


def load_benchmarks(benchmark_data_location: str) -> Optional[Dict[str, Any]]:
    """
    Load saved benchmark results if they exist.

    Args:
        benchmark_data_location: Path to benchmark data location

    Returns:
        Dictionary containing benchmark results if file exists, None otherwise
    """
    if not benchmark_data_location:
        return None

    try:
        benchmark_data_path = Path(benchmark_data_location)
        hostname = _get_hostname()
        benchmark_file = benchmark_data_path / f"{hostname}_core_benchmark.json"

        if not benchmark_file.exists():
            return None

        with open(benchmark_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            return results
    except Exception as e:
        logger.warning(f"Error loading benchmark file: {e}")
        return None


def get_saved_benchmarks(config: "MosaicConfig") -> Optional[Dict[str, Any]]:
    """
    Load saved benchmark results using MosaicConfig to determine location.

    Args:
        config: MosaicConfig instance with benchmark_data_location attribute

    Returns:
        Dictionary containing benchmark results if file exists, None otherwise
    """
    try:
        benchmark_data_location = config.benchmark_data_location
    except AttributeError:
        logger.warning("Config object does not have benchmark_data_location attribute")
        return None

    return load_benchmarks(benchmark_data_location)

