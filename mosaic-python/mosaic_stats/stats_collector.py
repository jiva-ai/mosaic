"""Low-impact machine statistics collector for Mosaic."""

import copy
import json
import logging
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from mosaic_config.config import MosaicConfig

try:
    from mosaic_stats.benchmark import benchmarks_captured, save_benchmarks
except ImportError:
    # Handle case where benchmark module might not be available
    benchmarks_captured = None
    save_benchmarks = None

try:
    import psutil
except ImportError:
    psutil = None

# GPU detection - try multiple libraries for wide coverage
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

# Try AMD ROCm
try:
    # Check if rocm-smi is available
    result = subprocess.run(
        ["rocm-smi", "--version"], capture_output=True, timeout=1, check=False
    )
    if result.returncode == 0:
        GPU_LIBRARIES["amd"] = "rocm-smi"
except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
    pass

# Try Intel GPU
try:
    # Check if intel_gpu_top or similar tools are available
    result = subprocess.run(
        ["intel_gpu_top", "--version"], capture_output=True, timeout=1, check=False
    )
    if result.returncode == 0:
        GPU_LIBRARIES["intel"] = "intel_gpu_top"
except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
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


class StatsCollector:
    """Collects machine statistics with minimal resource impact."""

    def __init__(self, config: "MosaicConfig"):
        """
        Initialize the stats collector.

        Args:
            config: MosaicConfig instance with heartbeat_report_length and
                    benchmark_data_location
        """
        if psutil is None:
            raise ImportError("psutil is required for stats collection")

        self.config = config
        self.heartbeat_report_length = config.heartbeat_report_length
        self.benchmark_data_location = (
            Path(config.benchmark_data_location) if config.benchmark_data_location else None
        )

        # Rolling window data structure
        self._data_lock = threading.Lock()
        self._stats_data: deque = deque()

        # Thread control
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # GPU initialization
        self._gpu_handles: List[Any] = []
        self._gpu_type: Optional[str] = None
        self._init_gpus()

        # Disk partition detection for data_folder
        self._target_partition: Optional[Any] = None
        self._data_folder_path: Optional[Path] = None
        self._init_data_folder_partition()

        # File write tracking
        self._last_file_write = 0.0
        self._file_write_interval = 60.0  # Write every 60 seconds

    def _init_gpus(self) -> None:
        """Initialize GPU detection libraries."""
        # Try NVIDIA first (most common)
        if GPU_LIBRARIES["nvidia"] is not None:
            try:
                if hasattr(GPU_LIBRARIES["nvidia"], "nvmlInit"):
                    # pynvml
                    GPU_LIBRARIES["nvidia"].nvmlInit()
                    device_count = GPU_LIBRARIES["nvidia"].nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = GPU_LIBRARIES["nvidia"].nvmlDeviceGetHandleByIndex(i)
                        self._gpu_handles.append(("nvidia", handle))
                    self._gpu_type = "nvidia"
                    logger.info(f"Initialized {device_count} NVIDIA GPU(s)")
                elif hasattr(GPU_LIBRARIES["nvidia"], "getGPUs"):
                    # GPUtil
                    gpus = GPU_LIBRARIES["nvidia"].getGPUs()
                    for gpu in gpus:
                        self._gpu_handles.append(("nvidia", gpu))
                    self._gpu_type = "nvidia"
                    logger.info(f"Initialized {len(gpus)} NVIDIA GPU(s) via GPUtil")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA GPUs: {e}")

        # Try AMD ROCm
        if not self._gpu_handles and GPU_LIBRARIES["amd"] == "rocm-smi":
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showid", "--showproductname"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if result.returncode == 0:
                    # Count GPUs by parsing output
                    lines = result.stdout.strip().split("\n")
                    gpu_count = sum(1 for line in lines if "GPU" in line or "Card" in line)
                    for i in range(gpu_count):
                        self._gpu_handles.append(("amd", i))
                    self._gpu_type = "amd"
                    logger.info(f"Initialized {gpu_count} AMD GPU(s)")
            except Exception as e:
                logger.warning(f"Failed to initialize AMD GPUs: {e}")

        # Try Intel GPU
        if not self._gpu_handles and GPU_LIBRARIES["intel"] == "intel_gpu_top":
            try:
                # Intel GPU detection is more complex, try basic check
                result = subprocess.run(
                    ["intel_gpu_top", "-l"],
                    capture_output=True,
                    timeout=2,
                    check=False,
                )
                if result.returncode == 0:
                    self._gpu_handles.append(("intel", 0))
                    self._gpu_type = "intel"
                    logger.info("Initialized Intel GPU")
            except Exception as e:
                logger.warning(f"Failed to initialize Intel GPU: {e}")

        # Try Habana Gaudi
        if not self._gpu_handles and GPU_LIBRARIES["habana"] is not None:
            try:
                if GPU_LIBRARIES["habana"] == "hl-smi":
                    # Use hl-smi to detect Habana devices
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
                            self._gpu_handles.append(("habana", i))
                        self._gpu_type = "habana"
                        logger.info(f"Initialized {device_count} Habana device(s)")
                elif GPU_LIBRARIES["habana"] == "pytorch":
                    # Use PyTorch Habana backend
                    import torch

                    if torch.habana.is_available():
                        device_count = torch.habana.device_count()
                        for i in range(device_count):
                            self._gpu_handles.append(("habana", i))
                        self._gpu_type = "habana"
                        logger.info(f"Initialized {device_count} Habana device(s) via PyTorch")
            except Exception as e:
                logger.warning(f"Failed to initialize Habana devices: {e}")

    def _init_data_folder_partition(self) -> None:
        """Initialize the target partition for data_location."""
        try:
            data_folder = self.config.data_location if self.config.data_location else None

            if not data_folder:
                return

            # Resolve to absolute path to find the actual partition
            self._data_folder_path = Path(data_folder).resolve()

            # Find which partition this path is on
            partitions = psutil.disk_partitions()
            for partition in partitions:
                try:
                    partition_path = Path(partition.mountpoint).resolve()
                    # Check if data_folder_path is within this partition
                    try:
                        self._data_folder_path.relative_to(partition_path)
                        self._target_partition = partition
                        break
                    except ValueError:
                        # Path is not within this partition
                        continue
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Failed to initialize data folder partition: {e}")

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {e}")
            return 0.0

    def _get_ram_usage(self) -> float:
        """Get current RAM usage percentage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            logger.warning(f"Failed to get RAM usage: {e}")
            return 0.0

    def _get_disk_free_space(self) -> Dict[str, Any]:
        """Get disk free space for the partition containing data_folder."""
        try:
            disk_info = {}

            # If no data_folder was set, return empty dict
            if not self._data_folder_path:
                return {}

            # If we found a partition during init, use it
            if self._target_partition:
                try:
                    usage = psutil.disk_usage(self._target_partition.mountpoint)
                    disk_info[self._target_partition.mountpoint] = {
                        "total_bytes": usage.total,
                        "free_bytes": usage.free,
                        "used_bytes": usage.used,
                        "percent_used": usage.percent,
                    }
                except PermissionError:
                    # Skip if we can't access
                    pass
            elif self._data_folder_path:
                # If we couldn't find the partition, try to get usage directly from the path
                try:
                    usage = psutil.disk_usage(str(self._data_folder_path))
                    # Use the path itself as the key
                    disk_info[str(self._data_folder_path)] = {
                        "total_bytes": usage.total,
                        "free_bytes": usage.free,
                        "used_bytes": usage.used,
                        "percent_used": usage.percent,
                    }
                except Exception:
                    pass

            return disk_info
        except Exception as e:
            logger.warning(f"Failed to get disk usage: {e}")
            return {}

    def _get_gpu_usage(self) -> List[Dict[str, Any]]:
        """Get GPU usage for all detected GPUs."""
        gpu_stats = []
        for gpu_type, handle in self._gpu_handles:
            try:
                if gpu_type == "nvidia":
                    if hasattr(handle, "load"):
                        # GPUtil GPU object
                        gpu_stats.append({"gpu_id": handle.id, "utilization_percent": handle.load * 100})
                    elif GPU_LIBRARIES["nvidia"] is not None:
                        # pynvml handle
                        if hasattr(GPU_LIBRARIES["nvidia"], "nvmlDeviceGetUtilizationRates"):
                            util = GPU_LIBRARIES["nvidia"].nvmlDeviceGetUtilizationRates(handle)
                            gpu_stats.append({"gpu_id": len(gpu_stats), "utilization_percent": util.gpu})
                        elif hasattr(handle, "getUtilization"):
                            util = handle.getUtilization()
                            gpu_stats.append({"gpu_id": len(gpu_stats), "utilization_percent": util.gpu})
                elif gpu_type == "amd":
                    # Use rocm-smi to get GPU utilization
                    result = subprocess.run(
                        ["rocm-smi", "-d", str(handle), "--showuse"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        check=False,
                    )
                    if result.returncode == 0:
                        # Parse output to extract GPU utilization
                        for line in result.stdout.split("\n"):
                            if "GPU use" in line or "%" in line:
                                try:
                                    # Extract percentage value
                                    parts = line.split()
                                    for part in parts:
                                        if "%" in part:
                                            util = float(part.replace("%", ""))
                                            gpu_stats.append({"gpu_id": handle, "utilization_percent": util})
                                            break
                                except (ValueError, IndexError):
                                    pass
                elif gpu_type == "intel":
                    # Intel GPU stats are more complex, return 0 for now
                    # Could be enhanced with intel_gpu_top parsing
                    gpu_stats.append({"gpu_id": 0, "utilization_percent": 0.0})
                elif gpu_type == "habana":
                    # Use hl-smi to get Habana device utilization
                    utilization_found = False
                    if GPU_LIBRARIES["habana"] == "hl-smi":
                        result = subprocess.run(
                            ["hl-smi", "-d", str(handle), "-Q", "utilization"],
                            capture_output=True,
                            text=True,
                            timeout=2,
                            check=False,
                        )
                        if result.returncode == 0:
                            # Parse output to extract utilization
                            for line in result.stdout.split("\n"):
                                if "Utilization" in line or "%" in line:
                                    try:
                                        # Extract percentage value
                                        parts = line.split()
                                        for part in parts:
                                            if "%" in part:
                                                util = float(part.replace("%", ""))
                                                gpu_stats.append({"gpu_id": handle, "utilization_percent": util})
                                                utilization_found = True
                                                break
                                    except (ValueError, IndexError):
                                        pass
                    # If hl-smi fails or no utilization found, try PyTorch
                    if not utilization_found and GPU_LIBRARIES["habana"] == "pytorch":
                        try:
                            import torch

                            if torch.habana.is_available():
                                # PyTorch Habana doesn't have direct utilization API
                                # Return 0 as placeholder
                                gpu_stats.append({"gpu_id": handle, "utilization_percent": 0.0})
                                utilization_found = True
                        except Exception:
                            pass
                    # If still no result, add placeholder
                    if not utilization_found:
                        gpu_stats.append({"gpu_id": handle, "utilization_percent": 0.0})
            except Exception as e:
                logger.warning(f"Failed to get GPU {handle} usage: {e}")
        return gpu_stats

    def _collect_stats(self) -> Dict[str, Any]:
        """Collect all current statistics."""
        timestamp_ms = time.time_ns() // 1_000_000

        stats = {
            "timestamp_ms": timestamp_ms,
            "cpu_percent": self._get_cpu_usage(),
            "ram_percent": self._get_ram_usage(),
            "disk_free_space": self._get_disk_free_space(),
            "gpus": self._get_gpu_usage(),
        }

        return stats

    def _cleanup_old_data(self) -> None:
        """Remove data points older than heartbeat_report_length."""
        if not self._stats_data:
            return

        current_time_ms = time.time_ns() // 1_000_000
        cutoff_time_ms = current_time_ms - (self.heartbeat_report_length * 1000)

        with self._data_lock:
            # Remove old entries from the left (oldest)
            while self._stats_data and self._stats_data[0]["timestamp_ms"] < cutoff_time_ms:
                self._stats_data.popleft()

    def _write_stats_file(self) -> None:
        """Write current rolling window data to JSON file."""
        if not self.benchmark_data_location:
            return

        try:
            # Ensure directory exists
            self.benchmark_data_location.mkdir(parents=True, exist_ok=True)

            stats_file = self.benchmark_data_location / "rolling_stats.json"

            with self._data_lock:
                # Convert deque to list for JSON serialization
                data_to_write = list(self._stats_data)

            # Write atomically using a temporary file
            temp_file = stats_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data_to_write, f, indent=2)

            # Atomic rename
            temp_file.replace(stats_file)

            logger.debug(f"Wrote {len(data_to_write)} data points to {stats_file}")

        except Exception as e:
            logger.error(f"Failed to write stats file: {e}")

    def _collection_loop(self) -> None:
        """Main collection loop running in background thread."""
        sample_interval = 15.0  # Sample every 15 seconds
        last_sample_time = 0.0

        while not self._stop_event.is_set():
            current_time = time.time()

            # Sample every sample_interval seconds
            if current_time - last_sample_time >= sample_interval:
                stats = self._collect_stats()

                with self._data_lock:
                    self._stats_data.append(stats)

                self._cleanup_old_data()
                last_sample_time = current_time

            # Write to file approximately every minute
            if current_time - self._last_file_write >= self._file_write_interval:
                self._write_stats_file()
                self._last_file_write = current_time

            # Sleep briefly to avoid busy-waiting
            time.sleep(0.1)

        # Final write on shutdown
        self._write_stats_file()

    def start(self) -> None:
        """Start the stats collection thread."""
        if self._running:
            logger.warning("Stats collector is already running")
            return

        if not self.benchmark_data_location:
            logger.warning("benchmark_data_location not set, stats will not be persisted")
        else:
            # Run benchmarks if run_benchmark_at_startup is True OR benchmarks not captured
            if benchmarks_captured is not None and save_benchmarks is not None:
                should_run_benchmarks = (
                    self.config.run_benchmark_at_startup
                    or not benchmarks_captured(str(self.benchmark_data_location))
                )

                if should_run_benchmarks:
                    logger.info("Running startup benchmarks...")
                    try:
                        save_benchmarks(str(self.benchmark_data_location))
                        logger.info("Startup benchmarks completed")
                    except Exception as e:
                        logger.error(f"Failed to run startup benchmarks: {e}")
            else:
                logger.warning("Benchmark module not available, skipping startup benchmarks")

        self._stop_event.clear()
        self._running = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Stats collector started")

    def stop(self) -> None:
        """Stop the stats collection thread."""
        if not self._running:
            return

        self._stop_event.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        self._running = False
        logger.info("Stats collector stopped")

    def get_current_stats(self) -> List[Dict[str, Any]]:
        """
        Get a copy of the current rolling window stats.

        Returns:
            List of stat dictionaries in the rolling window
        """
        with self._data_lock:
            return copy.deepcopy(list(self._stats_data))

    def get_last_stats_json(self) -> str:
        """
        Get the last known stats snapshot as JSON string.

        Returns:
            JSON string of the last stats snapshot, or empty JSON object "{}" if
            stats_data is empty or an error occurs
        """
        try:
            with self._data_lock:
                if not self._stats_data:
                    return "{}"
                
                # Get the last item (most recent stats)
                last_stats = self._stats_data[-1]
                
                # Convert to JSON
                return json.dumps(last_stats, indent=None)
        except Exception as e:
            logger.warning(f"Failed to get last stats as JSON: {e}")
            return "{}"

