"""Unit tests for mosaic_stats.benchmark module."""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mosaic_stats.benchmark import (
    _benchmark_cpu_flops,
    _benchmark_disk_speed,
    _benchmark_gpu_flops,
    _benchmark_network_capacity,
    _benchmark_ram_speed,
    _get_hostname,
    benchmarks_captured,
    load_benchmarks,
    run_benchmarks,
    save_benchmarks,
)


class TestGetHostname:
    """Test cases for _get_hostname function."""

    def test_get_hostname_success(self):
        """Test getting hostname successfully."""
        with patch("socket.gethostname", return_value="test-host"):
            result = _get_hostname()
            assert result == "test-host"

    def test_get_hostname_error(self):
        """Test hostname fallback on error."""
        with patch("socket.gethostname", side_effect=Exception("Network error")):
            result = _get_hostname()
            assert result == "unknown"


class TestBenchmarkDiskSpeed:
    """Test cases for _benchmark_disk_speed function."""

    def test_benchmark_disk_speed_no_location(self):
        """Test disk benchmark with None location."""
        result = _benchmark_disk_speed(None)
        assert "error" in result
        assert result["error"] == "No benchmark data location specified"

    def test_benchmark_disk_speed_success(self, tmp_path):
        """Test successful disk benchmark."""
        # Mock time.time_ns() to ensure measurable time differences
        # Calls: write_start, write_end, read_start, read_end
        # Use a callable to handle logging calls as well
        call_count = [0]
        original_time_ns = time.time_ns

        def time_ns_side_effect(*args, **kwargs):
            call_count[0] += 1
            # Write: 0.0 -> 0.1 (0.1 seconds for 1MB write)
            # Read: 0.2 -> 0.25 (0.05 seconds for 1MB read)
            # Convert seconds to nanoseconds: 0.1s = 100_000_000 ns
            # For logging calls, return a normal time value
            if call_count[0] == 1:
                return 0  # write start (0 nanoseconds)
            elif call_count[0] == 2:
                return 100_000_000  # write end (0.1 seconds = 100ms = 100_000_000 ns)
            elif call_count[0] == 3:
                return 200_000_000  # read start (0.2 seconds = 200ms = 200_000_000 ns)
            elif call_count[0] == 4:
                return 250_000_000  # read end (0.25 seconds = 250ms = 250_000_000 ns)
            # For logging and other calls, return a normal time value
            return original_time_ns()

        with patch("time.time_ns", side_effect=time_ns_side_effect):
            result = _benchmark_disk_speed(tmp_path)

        assert "write_speed_mbps" in result
        assert "read_speed_mbps" in result
        assert "test_size_bytes" in result
        assert result["test_size_bytes"] == 512 * 1024
        assert result["write_speed_mbps"] > 0
        assert result["read_speed_mbps"] > 0

        # Verify test file was cleaned up
        test_file = tmp_path / ".benchmark_test.tmp"
        assert not test_file.exists()

    def test_benchmark_disk_speed_creates_directory(self, tmp_path):
        """Test that disk benchmark creates directory if needed."""
        new_dir = tmp_path / "new" / "benchmark" / "dir"
        result = _benchmark_disk_speed(new_dir)

        assert new_dir.exists()
        assert "write_speed_mbps" in result

    def test_benchmark_disk_speed_error_handling(self, tmp_path):
        """Test disk benchmark error handling."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            result = _benchmark_disk_speed(tmp_path)
            assert "error" in result


class TestBenchmarkCpuFlops:
    """Test cases for _benchmark_cpu_flops function."""

    def test_benchmark_cpu_flops_with_numpy(self):
        """Test CPU benchmark with NumPy available."""
        mock_np = MagicMock()
        mock_np.random.rand.return_value = MagicMock(astype=MagicMock(return_value=MagicMock()))
        mock_np.dot.return_value = MagicMock()

        with patch("mosaic_stats.benchmark.np", mock_np), patch("time.time_ns") as mock_time_ns:
            # Mock time progression: 1 second elapsed = 1_000_000_000 nanoseconds
            mock_time_ns.side_effect = [0, 1_000_000_000]  # 1 second elapsed

            result = _benchmark_cpu_flops()

            assert "gflops" in result
            assert "test_size" in result
            assert "iterations" in result
            assert result["test_size"] == 200
            assert result["iterations"] == 10

    def test_benchmark_cpu_flops_without_numpy(self):
        """Test CPU benchmark without NumPy (pure Python fallback)."""
        with patch("mosaic_stats.benchmark.np", None), patch("time.time_ns") as mock_time_ns:
            # Mock time progression: 0.1 second elapsed = 100_000_000 nanoseconds
            mock_time_ns.side_effect = [0, 100_000_000]  # 0.1 second elapsed

            result = _benchmark_cpu_flops()

            assert "gflops" in result
            assert "test_size" in result
            assert "iterations" in result
            assert "note" in result
            assert result["note"] == "pure_python"
            assert result["test_size"] == 50
            assert result["iterations"] == 5

    def test_benchmark_cpu_flops_error_handling(self):
        """Test CPU benchmark error handling."""
        # Make time.time_ns() raise on first call (benchmark code), but work for logging
        call_count = [0]
        original_time_ns = time.time_ns

        def time_ns_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Time error")
            return original_time_ns()

        with patch("mosaic_stats.benchmark.np", None), patch(
            "time.time_ns", side_effect=time_ns_side_effect
        ):
            result = _benchmark_cpu_flops()
            assert "error" in result


class TestBenchmarkGpuFlops:
    """Test cases for _benchmark_gpu_flops function."""

    def test_benchmark_gpu_flops_pytorch(self):
        """Test GPU benchmark with PyTorch."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.device.return_value = MagicMock()
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.randn.return_value = MagicMock()
        mock_torch.matmul.return_value = MagicMock()
        mock_torch.cuda.synchronize = MagicMock()
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        # Ensure habana is not available to prevent false detection
        mock_habana = MagicMock()
        mock_habana.is_available.return_value = False
        mock_torch.habana = mock_habana

        # Patch sys.modules to intercept the torch import inside the function
        # Use a callable for time.time_ns() to handle multiple calls (including logging)
        call_count = [0]
        original_time_ns = time.time_ns

        def time_ns_side_effect(*args, **kwargs):
            call_count[0] += 1
            # First call: start_time, second call: end_time (0.5 seconds later)
            # 0.5 seconds = 500ms = 500_000_000 nanoseconds
            if call_count[0] == 1:
                return 0  # start_time (0 nanoseconds)
            elif call_count[0] == 2:
                return 500_000_000  # end_time (0.5 seconds = 500ms = 500_000_000 ns)
            # For logging and other calls, return a normal time value
            return original_time_ns()

        with patch("mosaic_stats.benchmark.GPU_LIBRARIES", {"nvidia": "pytorch", "amd": None, "intel": None, "habana": None}), patch.dict(
            sys.modules, {"torch": mock_torch}
        ), patch("time.time_ns", side_effect=time_ns_side_effect):
            result = _benchmark_gpu_flops()

            assert len(result) == 2
            assert result[0]["gpu_id"] == 0
            assert result[0]["gpu_name"] == "Test GPU"
            assert "gflops" in result[0]
            assert result[0]["type"] == "nvidia"

    def test_benchmark_gpu_flops_pynvml(self):
        """Test GPU benchmark with pynvml."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA Test GPU"

        with patch(
            "mosaic_stats.benchmark.GPU_LIBRARIES", {"nvidia": mock_pynvml, "amd": None, "intel": None, "habana": None}
        ):
            result = _benchmark_gpu_flops()

            assert len(result) == 1
            assert result[0]["gpu_id"] == 0
            assert result[0]["gpu_name"] == "NVIDIA Test GPU"
            assert result[0]["gflops"] is None
            assert result[0]["note"] == "pynvml_no_compute"
            assert result[0]["type"] == "nvidia"

    def test_benchmark_gpu_flops_amd_rocm(self):
        """Test GPU benchmark with AMD ROCm."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "GPU 0\nCard 1\nGPU 2"

        with patch(
            "mosaic_stats.benchmark.GPU_LIBRARIES", {"nvidia": None, "amd": "rocm-smi", "intel": None, "habana": None}
        ), patch("subprocess.run", return_value=mock_result):
            result = _benchmark_gpu_flops()

            assert len(result) == 3  # 3 GPUs detected
            assert result[0]["gpu_id"] == 0
            assert result[0]["gflops"] is None
            assert result[0]["note"] == "amd_rocm_no_compute"
            assert result[0]["type"] == "amd"

    def test_benchmark_gpu_flops_no_gpus(self):
        """Test GPU benchmark with no GPUs available."""
        with patch("mosaic_stats.benchmark.GPU_LIBRARIES", {"nvidia": None, "amd": None, "intel": None, "habana": None}):
            result = _benchmark_gpu_flops()
            assert result == []

    def test_benchmark_gpu_flops_habana_hl_smi(self):
        """Test GPU benchmark with Habana hl-smi."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "HL-225 Device 0\nHL-225 Device 1\nGaudi Device 2"

        with patch(
            "mosaic_stats.benchmark.GPU_LIBRARIES", {"nvidia": None, "amd": None, "intel": None, "habana": "hl-smi"}
        ), patch("subprocess.run", return_value=mock_result):
            result = _benchmark_gpu_flops()

            assert len(result) == 3  # 3 Habana devices detected
            assert result[0]["gpu_id"] == 0
            assert result[0]["gflops"] is None
            assert result[0]["note"] == "habana_no_compute"
            assert result[0]["type"] == "habana"

    def test_benchmark_gpu_flops_habana_pytorch(self):
        """Test GPU benchmark with Habana PyTorch backend."""
        mock_torch = MagicMock()
        mock_torch.habana.is_available.return_value = True
        mock_torch.habana.device_count.return_value = 2

        with patch(
            "mosaic_stats.benchmark.GPU_LIBRARIES", {"nvidia": None, "amd": None, "intel": None, "habana": "pytorch"}
        ), patch.dict(sys.modules, {"torch": mock_torch}):
            result = _benchmark_gpu_flops()

            assert len(result) == 2
            assert result[0]["gpu_id"] == 0
            assert result[0]["gflops"] is None
            assert result[0]["note"] == "habana_pytorch_no_compute"
            assert result[0]["type"] == "habana"

    def test_benchmark_gpu_flops_pytorch_error(self):
        """Test GPU benchmark error handling with PyTorch."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.side_effect = Exception("CUDA error")
        # Ensure habana attribute check fails - use spec_set to prevent auto-creation
        # or explicitly set habana to None and make is_available return False
        mock_habana = MagicMock()
        mock_habana.is_available.return_value = False
        mock_torch.habana = mock_habana

        # Patch sys.modules to intercept the torch import inside the function
        with patch("mosaic_stats.benchmark.GPU_LIBRARIES", {"nvidia": "pytorch", "amd": None, "intel": None, "habana": None}), patch.dict(
            sys.modules, {"torch": mock_torch}
        ):
            result = _benchmark_gpu_flops()
            assert result == []

    def test_benchmark_gpu_flops_pytorch_gpu_error(self):
        """Test GPU benchmark with individual GPU error."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.device.side_effect = [MagicMock(), Exception("GPU 1 error")]
        # Ensure habana is not available to prevent false detection
        mock_habana = MagicMock()
        mock_habana.is_available.return_value = False
        mock_torch.habana = mock_habana

        # Patch sys.modules to intercept the torch import inside the function
        # Use a callable for time.time_ns() to handle multiple calls (including logging)
        call_count = [0]
        original_time_ns = time.time_ns

        def time_ns_side_effect(*args, **kwargs):
            call_count[0] += 1
            # First call: start_time, second call: end_time (0.5 seconds later)
            # 0.5 seconds = 500ms = 500_000_000 nanoseconds
            if call_count[0] == 1:
                return 0  # start_time (0 nanoseconds)
            elif call_count[0] == 2:
                return 500_000_000  # end_time (0.5 seconds = 500ms = 500_000_000 ns)
            # For logging and other calls, return a normal time value
            return original_time_ns()

        with patch("mosaic_stats.benchmark.GPU_LIBRARIES", {"nvidia": "pytorch", "amd": None, "intel": None, "habana": None}), patch.dict(
            sys.modules, {"torch": mock_torch}
        ), patch("time.time_ns", side_effect=time_ns_side_effect):
            result = _benchmark_gpu_flops()

            # Should have one successful result and one error
            assert len(result) >= 1


class TestBenchmarkRamSpeed:
    """Test cases for _benchmark_ram_speed function."""

    def test_benchmark_ram_speed_with_numpy(self):
        """Test RAM benchmark with NumPy available."""
        mock_np = MagicMock()
        mock_array = MagicMock()
        mock_np.random.rand.return_value = mock_array
        mock_np.zeros_like.return_value = mock_array

        with patch("mosaic_stats.benchmark.np", mock_np), patch("time.time_ns") as mock_time_ns:
            # Mock time progression: 1 second elapsed = 1_000_000_000 nanoseconds
            mock_time_ns.side_effect = [0, 1_000_000_000]  # 1 second elapsed

            result = _benchmark_ram_speed()

            assert "bandwidth_gbps" in result
            assert "test_size_mb" in result
            assert "iterations" in result
            assert result["test_size_mb"] == 1.0
            assert result["iterations"] == 20

    def test_benchmark_ram_speed_without_numpy(self):
        """Test RAM benchmark without NumPy (pure Python fallback)."""
        with patch("mosaic_stats.benchmark.np", None), patch("time.time_ns") as mock_time_ns:
            # Mock time progression: 0.1 second elapsed = 100_000_000 nanoseconds
            mock_time_ns.side_effect = [0, 100_000_000]  # 0.1 second elapsed

            result = _benchmark_ram_speed()

            assert "bandwidth_gbps" in result
            assert "test_size_mb" in result
            assert "iterations" in result
            assert "note" in result
            assert result["note"] == "pure_python"
            assert result["test_size_mb"] == 0.5  # 512 KB = 0.5 MB
            assert result["iterations"] == 5

    def test_benchmark_ram_speed_error_handling(self):
        """Test RAM benchmark error handling."""
        # Make time.time_ns() raise on first call (benchmark code), but work for logging
        call_count = [0]
        original_time_ns = time.time_ns

        def time_ns_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Time error")
            return original_time_ns()

        with patch("mosaic_stats.benchmark.np", None), patch(
            "time.time_ns", side_effect=time_ns_side_effect
        ):
            result = _benchmark_ram_speed()
            assert "error" in result


class TestBenchmarkNetworkCapacity:
    """Test cases for _benchmark_network_capacity function."""

    def test_benchmark_network_capacity_success(self):
        """Test network capacity benchmark with valid psutil data."""
        mock_stats = MagicMock()
        mock_stats.isup = True
        mock_stats.speed = 1000

        mock_stats2 = MagicMock()
        mock_stats2.isup = True
        mock_stats2.speed = 500

        mock_net_if_stats = {
            "eth0": mock_stats,
            "wlan0": mock_stats2,
            "lo": MagicMock(isup=True, speed=0),  # Loopback with 0 speed
        }

        with patch("mosaic_stats.benchmark.psutil") as mock_psutil:
            mock_psutil.net_if_stats.return_value = mock_net_if_stats
            result = _benchmark_network_capacity()
            assert result == 1500  # 1000 + 500

    def test_benchmark_network_capacity_only_up_interfaces(self):
        """Test that only up interfaces are counted."""
        mock_stats = MagicMock()
        mock_stats.isup = True
        mock_stats.speed = 1000

        mock_stats2 = MagicMock()
        mock_stats2.isup = False  # Down interface
        mock_stats2.speed = 500

        mock_net_if_stats = {
            "eth0": mock_stats,
            "eth1": mock_stats2,
        }

        with patch("mosaic_stats.benchmark.psutil") as mock_psutil:
            mock_psutil.net_if_stats.return_value = mock_net_if_stats
            result = _benchmark_network_capacity()
            assert result == 1000  # Only eth0 counted

    def test_benchmark_network_capacity_zero_speed(self):
        """Test that interfaces with zero speed are not counted."""
        mock_stats = MagicMock()
        mock_stats.isup = True
        mock_stats.speed = 0

        mock_net_if_stats = {
            "eth0": mock_stats,
        }

        with patch("mosaic_stats.benchmark.psutil") as mock_psutil:
            mock_psutil.net_if_stats.return_value = mock_net_if_stats
            result = _benchmark_network_capacity()
            assert result == 0

    def test_benchmark_network_capacity_no_psutil(self):
        """Test network capacity benchmark when psutil is not available."""
        with patch("mosaic_stats.benchmark.psutil", None):
            result = _benchmark_network_capacity()
            assert result == 0

    def test_benchmark_network_capacity_error_handling(self):
        """Test network capacity benchmark error handling."""
        with patch("mosaic_stats.benchmark.psutil") as mock_psutil:
            mock_psutil.net_if_stats.side_effect = Exception("Network error")
            result = _benchmark_network_capacity()
            assert result == 0

    def test_benchmark_network_capacity_empty_interfaces(self):
        """Test network capacity benchmark with no interfaces."""
        with patch("mosaic_stats.benchmark.psutil") as mock_psutil:
            mock_psutil.net_if_stats.return_value = {}
            result = _benchmark_network_capacity()
            assert result == 0


class TestRunBenchmarks:
    """Test cases for run_benchmarks function."""

    def test_run_benchmarks_success(self, tmp_path):
        """Test running all benchmarks successfully."""
        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"), patch(
            "mosaic_stats.benchmark._benchmark_disk_speed"
        ) as mock_disk, patch("mosaic_stats.benchmark._benchmark_cpu_flops") as mock_cpu, patch(
            "mosaic_stats.benchmark._benchmark_gpu_flops"
        ) as mock_gpu, patch("mosaic_stats.benchmark._benchmark_ram_speed") as mock_ram, patch(
            "mosaic_stats.benchmark._benchmark_network_capacity"
        ) as mock_network, patch("time.time_ns", return_value=1_000_000_000_000):
            mock_disk.return_value = {"write_speed_mbps": 100.0, "read_speed_mbps": 150.0}
            mock_cpu.return_value = {"gflops": 50.0}
            mock_gpu.return_value = [{"gpu_id": 0, "gflops": 1000.0}]
            mock_ram.return_value = {"bandwidth_gbps": 20.0}
            mock_network.return_value = 1000

            result = run_benchmarks(str(tmp_path))

            assert "timestamp_ms" in result
            assert result["timestamp_ms"] == 1000000
            assert result["host"] == "test-host"
            assert "disk" in result
            assert "cpu" in result
            assert "gpus" in result
            assert "ram" in result
            assert "network_capacity" in result
            assert result["network_capacity"] == 1000

    def test_run_benchmarks_no_location(self):
        """Test running benchmarks with no location."""
        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"), patch(
            "mosaic_stats.benchmark._benchmark_cpu_flops"
        ) as mock_cpu, patch("mosaic_stats.benchmark._benchmark_gpu_flops") as mock_gpu, patch(
            "mosaic_stats.benchmark._benchmark_ram_speed"
        ) as mock_ram, patch("mosaic_stats.benchmark._benchmark_network_capacity") as mock_network, patch(
            "time.time_ns", return_value=1_000_000_000_000
        ):
            mock_cpu.return_value = {"gflops": 50.0}
            mock_gpu.return_value = []
            mock_ram.return_value = {"bandwidth_gbps": 20.0}
            mock_network.return_value = 1000

            result = run_benchmarks("")

            assert "disk" in result
            assert "error" in result["disk"]
            assert "network_capacity" in result
            assert result["network_capacity"] == 1000


class TestSaveBenchmarks:
    """Test cases for save_benchmarks function."""

    def test_save_benchmarks_with_results(self, tmp_path):
        """Test saving benchmarks with provided results."""
        test_results = {
            "timestamp_ms": 1000000,
            "host": "test-host",
            "disk": {"write_speed_mbps": 100.0},
            "cpu": {"gflops": 50.0},
            "gpus": [],
            "ram": {"bandwidth_gbps": 20.0},
        }

        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            result_path = save_benchmarks(str(tmp_path), test_results)

            assert result_path.exists()
            assert result_path.name == "test-host_core_benchmark.json"

            # Verify file contents
            with open(result_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                assert saved_data == test_results

    def test_save_benchmarks_runs_benchmarks(self, tmp_path):
        """Test that save_benchmarks runs benchmarks if results not provided."""
        with patch("mosaic_stats.benchmark.run_benchmarks") as mock_run, patch(
            "mosaic_stats.benchmark._get_hostname", return_value="test-host"
        ):
            mock_results = {
                "timestamp_ms": 1000000,
                "host": "test-host",
                "disk": {},
                "cpu": {},
                "gpus": [],
                "ram": {},
            }
            mock_run.return_value = mock_results

            result_path = save_benchmarks(str(tmp_path))

            mock_run.assert_called_once_with(str(tmp_path))
            assert result_path.exists()

    def test_save_benchmarks_creates_directory(self, tmp_path):
        """Test that save_benchmarks creates directory if needed."""
        new_dir = tmp_path / "new" / "benchmark" / "dir"
        test_results = {
            "timestamp_ms": 1000000,
            "host": "test-host",
            "disk": {},
            "cpu": {},
            "gpus": [],
            "ram": {},
        }

        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            result_path = save_benchmarks(str(new_dir), test_results)

            assert new_dir.exists()
            assert result_path.exists()

    def test_save_benchmarks_atomic_write(self, tmp_path):
        """Test that save_benchmarks uses atomic file writes."""
        test_results = {
            "timestamp_ms": 1000000,
            "host": "test-host",
            "disk": {},
            "cpu": {},
            "gpus": [],
            "ram": {},
        }

        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            result_path = save_benchmarks(str(tmp_path), test_results)

            # Verify temp file doesn't exist
            temp_file = result_path.with_suffix(".tmp")
            assert not temp_file.exists()
            assert result_path.exists()


class TestBenchmarksCaptured:
    """Test cases for benchmarks_captured function."""

    def test_benchmarks_captured_exists(self, tmp_path):
        """Test benchmarks_captured when file exists."""
        hostname = socket.gethostname()
        benchmark_file = tmp_path / f"{hostname}_core_benchmark.json"
        benchmark_file.write_text('{"test": "data"}', encoding="utf-8")

        result = benchmarks_captured(str(tmp_path))
        assert result is True

    def test_benchmarks_captured_not_exists(self, tmp_path):
        """Test benchmarks_captured when file doesn't exist."""
        result = benchmarks_captured(str(tmp_path))
        assert result is False

    def test_benchmarks_captured_empty_location(self):
        """Test benchmarks_captured with empty location."""
        result = benchmarks_captured("")
        assert result is False

    def test_benchmarks_captured_error_handling(self):
        """Test benchmarks_captured error handling."""
        with patch("pathlib.Path.exists", side_effect=Exception("Path error")):
            result = benchmarks_captured("/some/path")
            assert result is False

    def test_benchmarks_captured_uses_hostname(self, tmp_path):
        """Test that benchmarks_captured uses correct hostname."""
        with patch("mosaic_stats.benchmark._get_hostname", return_value="custom-host"):
            # File doesn't exist
            result = benchmarks_captured(str(tmp_path))
            assert result is False

            # Create file with custom hostname
            benchmark_file = tmp_path / "custom-host_core_benchmark.json"
            benchmark_file.write_text('{"test": "data"}', encoding="utf-8")

            result = benchmarks_captured(str(tmp_path))
            assert result is True


class TestIntegration:
    """Integration tests for benchmark module."""

    def test_full_benchmark_cycle(self, tmp_path):
        """Test a full cycle: run, save, and check benchmarks."""
        # Run benchmarks
        results = run_benchmarks(str(tmp_path))

        assert "timestamp_ms" in results
        assert "host" in results
        assert "disk" in results
        assert "cpu" in results
        assert "gpus" in results
        assert "ram" in results
        assert "network_capacity" in results

        # Save benchmarks
        saved_path = save_benchmarks(str(tmp_path), results)
        assert saved_path.exists()

        # Check if captured
        assert benchmarks_captured(str(tmp_path)) is True

        # Verify file contents
        with open(saved_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["host"] == results["host"]
            assert saved_data["timestamp_ms"] == results["timestamp_ms"]
            assert "network_capacity" in saved_data

    def test_benchmark_file_format(self, tmp_path):
        """Test that benchmark file has correct format."""
        test_results = {
            "timestamp_ms": 1234567890,
            "host": "test-host",
            "disk": {"write_speed_mbps": 100.0, "read_speed_mbps": 150.0, "test_size_bytes": 524288},
            "cpu": {"gflops": 50.0, "test_size": 200, "iterations": 10},
            "gpus": [{"gpu_id": 0, "gflops": 1000.0, "type": "nvidia"}],
            "ram": {"bandwidth_gbps": 20.0, "test_size_mb": 1.0, "iterations": 20},
            "network_capacity": 1000,
        }

        saved_path = save_benchmarks(str(tmp_path), test_results)

        with open(saved_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        # Verify all expected keys are present
        assert "timestamp_ms" in saved_data
        assert "host" in saved_data
        assert "disk" in saved_data
        assert "cpu" in saved_data
        assert "gpus" in saved_data
        assert "ram" in saved_data
        assert "network_capacity" in saved_data
        assert saved_data["network_capacity"] == 1000

        # Verify nested structure
        assert "write_speed_mbps" in saved_data["disk"]
        assert "gflops" in saved_data["cpu"]
        assert isinstance(saved_data["gpus"], list)


class TestLoadBenchmarks:
    """Test cases for load_benchmarks function."""

    def test_load_benchmarks_file_exists(self, tmp_path):
        """Test loading benchmarks when file exists."""
        # Create test data
        test_data = {
            "timestamp_ms": 1234567890123,
            "host": "test-host",
            "disk": {"write_speed_mbps": 150.5, "read_speed_mbps": 200.3},
            "cpu": {"gflops": 75.8},
            "gpus": [{"gpu_id": 0, "gflops": 1200.5, "type": "nvidia"}],
            "ram": {"bandwidth_gbps": 25.5},
            "network_capacity": 1000,
        }

        # Save the benchmark file
        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            save_benchmarks(str(tmp_path), test_data)

        # Load the benchmarks
        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            result = load_benchmarks(str(tmp_path))

        # Verify the loaded data matches the saved data
        assert result is not None
        assert result == test_data
        assert result["timestamp_ms"] == 1234567890123
        assert result["host"] == "test-host"
        assert result["disk"]["write_speed_mbps"] == 150.5
        assert result["cpu"]["gflops"] == 75.8
        assert len(result["gpus"]) == 1
        assert result["gpus"][0]["gflops"] == 1200.5

    def test_load_benchmarks_file_not_exists(self, tmp_path):
        """Test loading benchmarks when file doesn't exist."""
        # Use a non-existent location
        non_existent_path = tmp_path / "nonexistent" / "benchmarks"

        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            result = load_benchmarks(str(non_existent_path))

        # Should return None when file doesn't exist
        assert result is None

    def test_load_benchmarks_empty_location(self):
        """Test loading benchmarks with empty location."""
        result = load_benchmarks("")
        assert result is None

    def test_load_benchmarks_invalid_json(self, tmp_path):
        """Test loading benchmarks with invalid JSON file."""
        # Create a file with invalid JSON
        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            benchmark_file = tmp_path / "test-host_core_benchmark.json"
            with open(benchmark_file, "w", encoding="utf-8") as f:
                f.write("invalid json content {")

            # Should handle the error gracefully and return None
            result = load_benchmarks(str(tmp_path))
            assert result is None

    def test_load_benchmarks_permission_error(self, tmp_path):
        """Test loading benchmarks with permission error."""
        # Create the file first
        test_data = {
            "timestamp_ms": 1234567890123,
            "host": "test-host",
            "disk": {},
            "cpu": {},
            "gpus": [],
            "ram": {},
        }

        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            save_benchmarks(str(tmp_path), test_data)

        # Mock open to raise PermissionError
        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"), patch(
            "builtins.open", side_effect=PermissionError("Permission denied")
        ):
            result = load_benchmarks(str(tmp_path))
            # Should return None on error
            assert result is None

    def test_load_benchmarks_from_dummy_file(self, tmp_path):
        """Test loading benchmarks from the dummy data file in tests folder."""
        # Get the tests directory
        tests_dir = Path(__file__).parent
        dummy_file = tests_dir / "dummy_benchmark_data.json"

        # Copy dummy file to the expected location in tmp_path
        import shutil

        hostname = "test-host"
        benchmark_file = tmp_path / f"{hostname}_core_benchmark.json"
        shutil.copy(dummy_file, benchmark_file)

        # Load the benchmarks
        with patch("mosaic_stats.benchmark._get_hostname", return_value=hostname):
            result = load_benchmarks(str(tmp_path))

        # Verify the loaded data matches the dummy file
        assert result is not None
        assert result["timestamp_ms"] == 1234567890123
        assert result["host"] == "test-host"
        assert result["disk"]["write_speed_mbps"] == 150.5
        assert result["cpu"]["gflops"] == 75.8
        assert len(result["gpus"]) == 2
        assert result["gpus"][0]["gflops"] == 1200.5
        assert result["gpus"][1]["gflops"] == 1100.2
        assert result["ram"]["bandwidth_gbps"] == 25.5
        assert result["network_capacity"] == 1000

    def test_load_benchmarks_different_hostname(self, tmp_path):
        """Test that load_benchmarks uses correct hostname for file lookup."""
        # Save benchmark with one hostname
        test_data = {
            "timestamp_ms": 1234567890123,
            "host": "host-a",
            "disk": {},
            "cpu": {},
            "gpus": [],
            "ram": {},
        }

        with patch("mosaic_stats.benchmark._get_hostname", return_value="host-a"):
            save_benchmarks(str(tmp_path), test_data)

        # Try to load with different hostname
        with patch("mosaic_stats.benchmark._get_hostname", return_value="host-b"):
            result = load_benchmarks(str(tmp_path))
            # Should return None because file doesn't exist for host-b
            assert result is None

        # Load with correct hostname
        with patch("mosaic_stats.benchmark._get_hostname", return_value="host-a"):
            result = load_benchmarks(str(tmp_path))
            # Should return the data
            assert result is not None
            assert result["host"] == "host-a"

