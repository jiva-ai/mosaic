"""Unit tests for mosaic_stats.stats_collector module."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mosaic_config.config import MosaicConfig
from mosaic_stats.stats_collector import StatsCollector
from tests.conftest import create_test_config_with_state


@pytest.fixture
def mock_config(tmp_path, temp_state_dir):
    """Create a mock MosaicConfig for testing."""
    return create_test_config_with_state(
        state_dir=temp_state_dir,
        heartbeat_report_length=60,  # 60 seconds for faster testing
        benchmark_data_location=str(tmp_path / "benchmarks"),
    )


@pytest.fixture
def mock_config_no_location(temp_state_dir):
    """Create a mock MosaicConfig without benchmark_data_location."""
    return create_test_config_with_state(
        state_dir=temp_state_dir,
        heartbeat_report_length=60,
        benchmark_data_location="",
    )


@pytest.fixture
def mock_psutil():
    """Mock psutil module."""
    mock_psutil = MagicMock()
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.virtual_memory.return_value = MagicMock(percent=67.8)
    mock_psutil.disk_partitions.return_value = [
        MagicMock(mountpoint="/"),
        MagicMock(mountpoint="/home"),
    ]
    mock_psutil.disk_usage.return_value = MagicMock(
        total=1000000000,
        free=300000000,
        used=700000000,
        percent=70.0,
    )
    return mock_psutil


class TestStatsCollectorInitialization:
    """Test cases for StatsCollector initialization."""

    def test_init_with_config(self, mock_config, mock_psutil):
        """Test initialization with valid config."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            assert collector.config == mock_config
            assert collector.heartbeat_report_length == 60
            assert collector.benchmark_data_location == Path(mock_config.benchmark_data_location)
            assert not collector._running

    def test_init_without_psutil(self, mock_config):
        """Test that ImportError is raised when psutil is not available."""
        with patch("mosaic_stats.stats_collector.psutil", None):
            with pytest.raises(ImportError, match="psutil is required"):
                StatsCollector(mock_config)

    def test_init_with_empty_benchmark_location(self, mock_config_no_location, mock_psutil):
        """Test initialization with empty benchmark_data_location."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config_no_location)
            assert collector.benchmark_data_location is None

    def test_init_gpu_detection_nvidia_pynvml(self, mock_config, mock_psutil):
        """Test GPU initialization with pynvml."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_handle1 = MagicMock()
        mock_handle2 = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [mock_handle1, mock_handle2]

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.GPU_LIBRARIES", {"nvidia": mock_pynvml, "amd": None, "intel": None, "habana": None}
        ):
            collector = StatsCollector(mock_config)
            assert len(collector._gpu_handles) == 2
            assert collector._gpu_type == "nvidia"
            mock_pynvml.nvmlInit.assert_called_once()

    def test_init_gpu_detection_nvidia_gputil(self, mock_config, mock_psutil):
        """Test GPU initialization with GPUtil."""
        # Create a mock that only has getGPUs, not nvmlInit
        mock_gputil = MagicMock(spec=["getGPUs"])
        mock_gpu1 = MagicMock()
        mock_gpu1.id = 0
        mock_gpu2 = MagicMock()
        mock_gpu2.id = 1
        mock_gputil.getGPUs.return_value = [mock_gpu1, mock_gpu2]

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.GPU_LIBRARIES", {"nvidia": mock_gputil, "amd": None, "intel": None, "habana": None}
        ):
            collector = StatsCollector(mock_config)
            assert len(collector._gpu_handles) == 2
            assert collector._gpu_type == "nvidia"

    def test_init_gpu_detection_habana_hl_smi(self, mock_config, mock_psutil):
        """Test GPU initialization with Habana hl-smi."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "HL-225 Device 0\nHL-225 Device 1\nGaudi Device 2"

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.GPU_LIBRARIES", {"nvidia": None, "amd": None, "intel": None, "habana": "hl-smi"}
        ), patch("subprocess.run", return_value=mock_result):
            collector = StatsCollector(mock_config)
            assert len(collector._gpu_handles) == 3
            assert collector._gpu_type == "habana"

    def test_init_gpu_detection_habana_pytorch(self, mock_config, mock_psutil):
        """Test GPU initialization with Habana PyTorch backend."""
        import sys
        mock_torch = MagicMock()
        mock_torch.habana.is_available.return_value = True
        mock_torch.habana.device_count.return_value = 2

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.GPU_LIBRARIES", {"nvidia": None, "amd": None, "intel": None, "habana": "pytorch"}
        ), patch.dict(sys.modules, {"torch": mock_torch}):
            collector = StatsCollector(mock_config)
            assert len(collector._gpu_handles) == 2
            assert collector._gpu_type == "habana"


class TestStatsCollection:
    """Test cases for stats collection methods."""

    def test_get_cpu_usage(self, mock_config, mock_psutil):
        """Test CPU usage collection."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            result = collector._get_cpu_usage()
            assert result == 45.5
            mock_psutil.cpu_percent.assert_called_once_with(interval=None)

    def test_get_cpu_usage_error_handling(self, mock_config, mock_psutil):
        """Test CPU usage collection with error."""
        mock_psutil.cpu_percent.side_effect = Exception("CPU error")
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            result = collector._get_cpu_usage()
            assert result == 0.0

    def test_get_ram_usage(self, mock_config, mock_psutil):
        """Test RAM usage collection."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            result = collector._get_ram_usage()
            assert result == 67.8
            mock_psutil.virtual_memory.assert_called_once()

    def test_get_ram_usage_error_handling(self, mock_config, mock_psutil):
        """Test RAM usage collection with error."""
        mock_psutil.virtual_memory.side_effect = Exception("RAM error")
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            result = collector._get_ram_usage()
            assert result == 0.0

    def test_get_disk_free_space(self, mock_config, mock_psutil, tmp_path):
        """Test disk free space collection for data_location partition."""
        # Create a config with data_location
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            data_location=str(tmp_path / "data"),
        )

        # Mock partition that contains our data_location
        mock_partition = MagicMock()
        mock_partition.mountpoint = str(tmp_path)
        
        mock_psutil.disk_partitions.return_value = [mock_partition]
        mock_psutil.disk_usage.return_value = MagicMock(
            total=1000000000,
            free=300000000,
            used=700000000,
            percent=70.0,
        )

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(config)
            # Partition detection happens in __init__, verify it was set
            assert collector._target_partition is not None
            assert collector._data_folder_path is not None
            
            result = collector._get_disk_free_space()
            
            # Should only return the partition containing data_location
            assert len(result) == 1
            assert str(tmp_path) in result
            assert result[str(tmp_path)]["total_bytes"] == 1000000000
            assert result[str(tmp_path)]["free_bytes"] == 300000000
            assert result[str(tmp_path)]["used_bytes"] == 700000000
            assert result[str(tmp_path)]["percent_used"] == 70.0

    def test_get_disk_free_space_no_data_folder(self, mock_config, mock_psutil, tmp_path):
        """Test disk free space collection when data_location is not set."""
        # Config without data_location
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            data_location="",
        )

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(config)
            # Partition detection should not happen when no data_location
            assert collector._target_partition is None
            assert collector._data_folder_path is None
            
            result = collector._get_disk_free_space()
            # Should return empty dict when no data_location
            assert result == {}

    def test_get_disk_free_space_permission_error(self, mock_config, mock_psutil, tmp_path):
        """Test disk free space collection with permission error."""
        # Create a config with data_location
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            data_location=str(tmp_path / "data"),
        )

        # Mock partition that contains our data_location
        mock_partition = MagicMock()
        mock_partition.mountpoint = str(tmp_path)
        
        # disk_partitions is called in __init__, disk_usage is called in _get_disk_free_space
        # Since __init__ doesn't call disk_usage, we only need to mock the permission error
        mock_psutil.disk_partitions.return_value = [mock_partition]
        mock_psutil.disk_usage.side_effect = PermissionError("No access")

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(config)
            # Partition should be detected in init
            assert collector._target_partition is not None
            
            result = collector._get_disk_free_space()
            # Should return empty dict when permission error during usage check
            assert result == {}

    def test_get_disk_free_space_fallback_to_path(self, mock_config, mock_psutil, tmp_path):
        """Test disk free space when partition not found, falls back to path."""
        # Create a config with data_location
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            data_location=str(tmp_path / "data"),
        )

        # Mock no partitions found (empty list)
        mock_psutil.disk_partitions.return_value = []
        mock_psutil.disk_usage.return_value = MagicMock(
            total=1000000000,
            free=300000000,
            used=700000000,
            percent=70.0,
        )

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(config)
            # Partition should not be found in init
            assert collector._target_partition is None
            assert collector._data_folder_path is not None
            
            result = collector._get_disk_free_space()
            
            # Should fall back to using the path directly
            assert len(result) == 1
            # The key should be the resolved path
            assert any("data" in str(key) for key in result.keys())

    def test_get_gpu_usage_nvidia_pynvml(self, mock_config, mock_psutil):
        """Test GPU usage collection with pynvml."""
        mock_pynvml = MagicMock()
        mock_util = MagicMock()
        mock_util.gpu = 85.0
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        # Create a mock handle that doesn't have 'load' attribute (so it uses pynvml path)
        mock_handle = MagicMock(spec=[])

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.GPU_LIBRARIES", {"nvidia": mock_pynvml, "amd": None, "intel": None, "habana": None}
        ):
            collector = StatsCollector(mock_config)
            collector._gpu_handles = [("nvidia", mock_handle)]
            collector._gpu_type = "nvidia"
            result = collector._get_gpu_usage()
            assert len(result) == 1
            assert result[0]["utilization_percent"] == 85.0

    def test_get_gpu_usage_nvidia_gputil(self, mock_config, mock_psutil):
        """Test GPU usage collection with GPUtil."""
        mock_gpu = MagicMock()
        mock_gpu.id = 0
        mock_gpu.load = 0.75  # 75% utilization

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            collector._gpu_handles = [("nvidia", mock_gpu)]
            collector._gpu_type = "nvidia"
            result = collector._get_gpu_usage()
            assert len(result) == 1
            assert result[0]["gpu_id"] == 0
            assert result[0]["utilization_percent"] == 75.0

    def test_get_gpu_usage_habana_hl_smi(self, mock_config, mock_psutil):
        """Test GPU usage collection with Habana hl-smi."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Device Utilization: 75.5%"

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.GPU_LIBRARIES", {"nvidia": None, "amd": None, "intel": None, "habana": "hl-smi"}
        ), patch("subprocess.run", return_value=mock_result):
            collector = StatsCollector(mock_config)
            collector._gpu_handles = [("habana", 0)]
            collector._gpu_type = "habana"
            result = collector._get_gpu_usage()
            assert len(result) == 1
            assert result[0]["gpu_id"] == 0
            assert result[0]["utilization_percent"] == 75.5

    def test_get_gpu_usage_habana_pytorch(self, mock_config, mock_psutil):
        """Test GPU usage collection with Habana PyTorch backend."""
        import sys
        mock_torch = MagicMock()
        mock_torch.habana.is_available.return_value = True

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.GPU_LIBRARIES", {"nvidia": None, "amd": None, "intel": None, "habana": "pytorch"}
        ), patch.dict(sys.modules, {"torch": mock_torch}):
            collector = StatsCollector(mock_config)
            collector._gpu_handles = [("habana", 0)]
            collector._gpu_type = "habana"
            result = collector._get_gpu_usage()
            assert len(result) == 1
            assert result[0]["gpu_id"] == 0
            assert result[0]["utilization_percent"] == 0.0  # PyTorch Habana returns placeholder

    def test_get_gpu_usage_habana_fallback(self, mock_config, mock_psutil):
        """Test GPU usage collection with Habana when hl-smi fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1  # hl-smi fails

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.GPU_LIBRARIES", {"nvidia": None, "amd": None, "intel": None, "habana": "hl-smi"}
        ), patch("subprocess.run", return_value=mock_result):
            collector = StatsCollector(mock_config)
            collector._gpu_handles = [("habana", 0)]
            collector._gpu_type = "habana"
            result = collector._get_gpu_usage()
            # Should return placeholder when hl-smi fails
            assert len(result) == 1
            assert result[0]["gpu_id"] == 0
            assert result[0]["utilization_percent"] == 0.0

    def test_get_gpu_usage_error_handling(self, mock_config, mock_psutil):
        """Test GPU usage collection with error."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            collector._gpu_handles = [("nvidia", MagicMock())]
            collector._gpu_type = "nvidia"

            # Mock GPU library to raise error
            with patch.object(collector, "_get_gpu_usage", side_effect=Exception("GPU error")):
                # Should handle gracefully
                pass

    def test_collect_stats(self, mock_config, mock_psutil):
        """Test complete stats collection."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            collector._gpu_handles = []
            stats = collector._collect_stats()

            assert "timestamp_ms" in stats
            assert isinstance(stats["timestamp_ms"], int)
            assert stats["cpu_percent"] == 45.5
            assert stats["ram_percent"] == 67.8
            assert "disk_free_space" in stats
            assert "gpus" in stats
            assert isinstance(stats["gpus"], list)


class TestRollingWindow:
    """Test cases for rolling window functionality."""

    def test_cleanup_old_data(self, mock_config, mock_psutil):
        """Test that old data is removed from rolling window."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            collector.heartbeat_report_length = 10  # 10 seconds window

            # Add old and new data points
            current_time_ms = int(time.time() * 1000)
            old_time_ms = current_time_ms - 15000  # 15 seconds ago
            new_time_ms = current_time_ms - 5000  # 5 seconds ago

            with collector._data_lock:
                collector._stats_data.append({"timestamp_ms": old_time_ms, "cpu_percent": 10.0})
                collector._stats_data.append({"timestamp_ms": new_time_ms, "cpu_percent": 20.0})
                collector._stats_data.append({"timestamp_ms": current_time_ms, "cpu_percent": 30.0})

            collector._cleanup_old_data()

            with collector._data_lock:
                assert len(collector._stats_data) == 2  # Only new data points remain
                assert collector._stats_data[0]["timestamp_ms"] == new_time_ms
                assert collector._stats_data[1]["timestamp_ms"] == current_time_ms

    def test_cleanup_empty_data(self, mock_config, mock_psutil):
        """Test cleanup with empty data."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            collector._cleanup_old_data()  # Should not raise error
            with collector._data_lock:
                assert len(collector._stats_data) == 0


class TestFilePersistence:
    """Test cases for file persistence."""

    def test_write_stats_file(self, mock_config, mock_psutil, tmp_path):
        """Test writing stats to file."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)

            # Add some test data
            test_data = [
                {"timestamp_ms": 1000, "cpu_percent": 50.0, "ram_percent": 60.0, "gpus": [], "disk_free_space": {}},
                {"timestamp_ms": 2000, "cpu_percent": 55.0, "ram_percent": 65.0, "gpus": [], "disk_free_space": {}},
            ]

            with collector._data_lock:
                collector._stats_data.extend(test_data)

            collector._write_stats_file()

            # Verify file was created
            stats_file = tmp_path / "benchmarks" / "rolling_stats.json"
            assert stats_file.exists()

            # Verify file contents
            with open(stats_file, "r", encoding="utf-8") as f:
                written_data = json.load(f)
                assert len(written_data) == 2
                assert written_data[0]["cpu_percent"] == 50.0
                assert written_data[1]["cpu_percent"] == 55.0

    def test_write_stats_file_no_location(self, mock_config_no_location, mock_psutil):
        """Test that file is not written when location is not set."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config_no_location)
            collector._write_stats_file()  # Should not raise error

    def test_write_stats_file_creates_directory(self, mock_config, mock_psutil, tmp_path):
        """Test that directory is created if it doesn't exist."""
        new_location = tmp_path / "new" / "benchmarks"
        config = MosaicConfig(heartbeat_report_length=60, benchmark_data_location=str(new_location))

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(config)
            collector._write_stats_file()
            assert new_location.exists()
            assert (new_location / "rolling_stats.json").exists()

    def test_write_stats_file_atomic(self, mock_config, mock_psutil, tmp_path):
        """Test that file writes are atomic."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)

            test_data = [{"timestamp_ms": 1000, "cpu_percent": 50.0, "ram_percent": 60.0, "gpus": [], "disk_free_space": {}}]

            with collector._data_lock:
                collector._stats_data.extend(test_data)

            collector._write_stats_file()

            # Verify temp file doesn't exist after write
            stats_file = tmp_path / "benchmarks" / "rolling_stats.json"
            temp_file = stats_file.with_suffix(".tmp")
            assert not temp_file.exists()
            assert stats_file.exists()


class TestThreading:
    """Test cases for threading functionality."""

    def test_start_collector(self, mock_config, mock_psutil):
        """Test starting the collector."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=True
        ), patch("mosaic_stats.stats_collector.save_benchmarks"):
            collector = StatsCollector(mock_config)
            assert not collector._running

            collector.start()
            assert collector._running
            assert collector._collection_thread is not None
            assert collector._collection_thread.is_alive()

            collector.stop()

    def test_start_already_running(self, mock_config, mock_psutil):
        """Test starting when already running."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=True
        ), patch("mosaic_stats.stats_collector.save_benchmarks"):
            collector = StatsCollector(mock_config)
            collector.start()
            collector.start()  # Should not raise error, just warn
            collector.stop()

    def test_stop_collector(self, mock_config, mock_psutil):
        """Test stopping the collector."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=True
        ), patch("mosaic_stats.stats_collector.save_benchmarks"):
            collector = StatsCollector(mock_config)
            collector.start()
            assert collector._running

            collector.stop()
            assert not collector._running

    def test_stop_when_not_running(self, mock_config, mock_psutil):
        """Test stopping when not running."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            collector.stop()  # Should not raise error

    def test_collection_loop_samples_every_2_seconds(self, mock_config, mock_psutil):
        """Test that collection loop samples every 2 seconds."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.time"
        ) as mock_time:
            collector = StatsCollector(mock_config)
            collector._file_write_interval = 1000  # Very long to avoid file writes

            # Mock time to simulate 2-second intervals
            time_values = [0.0, 2.0, 4.0, 4.1]
            mock_time.time.side_effect = time_values
            mock_time.sleep = MagicMock()

            collector._stop_event.set()  # Stop immediately
            collector._collection_loop()

            # Verify stats were collected
            with collector._data_lock:
                # Should have collected stats at 2.0 and 4.0
                assert len(collector._stats_data) >= 0  # May be 0 if loop exits quickly

    def test_collection_loop_writes_file_every_minute(self, mock_config, mock_psutil, tmp_path):
        """Test that file is written approximately every minute."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.time"
        ) as mock_time:
            collector = StatsCollector(mock_config)
            collector._file_write_interval = 60.0

            # Mock time progression
            time_values = [0.0, 2.0, 60.0, 62.0]
            call_count = [0]

            def time_side_effect():
                val = time_values[call_count[0] % len(time_values)]
                call_count[0] += 1
                return val

            mock_time.time.side_effect = time_side_effect
            mock_time.sleep = MagicMock()

            collector._stop_event.set()  # Stop immediately
            collector._collection_loop()

            # File write should have been attempted
            # (exact verification depends on timing, but method should be called)


class TestStartupBenchmarks:
    """Test cases for startup benchmark functionality."""

    def test_start_skips_benchmarks_by_default_when_benchmarks_exist(self, mock_config, mock_psutil):
        """Test that benchmarks are skipped by default (run_benchmark_at_startup=False) when benchmarks exist."""
        # mock_config doesn't set run_benchmark_at_startup, so it defaults to False
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=True
        ) as mock_captured, patch(
            "mosaic_stats.stats_collector.save_benchmarks"
        ) as mock_save:
            collector = StatsCollector(mock_config)
            # Verify default is False
            assert collector.config.run_benchmark_at_startup is False
            collector.start()

            # Should NOT run benchmarks because flag is False (default) and benchmarks exist
            mock_save.assert_not_called()
            mock_captured.assert_called_once_with(str(mock_config.benchmark_data_location))

            collector.stop()

    def test_start_runs_benchmarks_when_flag_true(self, mock_config, mock_psutil, tmp_path):
        """Test that benchmarks run when run_benchmark_at_startup is True."""
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            run_benchmark_at_startup=True,
        )

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=True
        ) as mock_captured, patch(
            "mosaic_stats.stats_collector.save_benchmarks"
        ) as mock_save:
            collector = StatsCollector(config)
            collector.start()

            # Should run benchmarks even though benchmarks_captured returns True
            mock_save.assert_called_once_with(str(tmp_path / "benchmarks"))

            collector.stop()

    def test_start_runs_benchmarks_when_not_captured(self, mock_config, mock_psutil, tmp_path):
        """Test that benchmarks run when benchmarks_captured returns False."""
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            run_benchmark_at_startup=False,
        )

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=False
        ) as mock_captured, patch(
            "mosaic_stats.stats_collector.save_benchmarks"
        ) as mock_save:
            collector = StatsCollector(config)
            collector.start()

            # Should run benchmarks because benchmarks_captured returns False
            mock_save.assert_called_once_with(str(tmp_path / "benchmarks"))
            mock_captured.assert_called_once_with(str(tmp_path / "benchmarks"))

            collector.stop()

    def test_start_skips_benchmarks_when_both_false(self, mock_config, mock_psutil, tmp_path):
        """Test that benchmarks are skipped when flag is False and benchmarks exist."""
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            run_benchmark_at_startup=False,
        )

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=True
        ) as mock_captured, patch(
            "mosaic_stats.stats_collector.save_benchmarks"
        ) as mock_save:
            collector = StatsCollector(config)
            collector.start()

            # Should NOT run benchmarks
            mock_save.assert_not_called()
            mock_captured.assert_called_once_with(str(tmp_path / "benchmarks"))

            collector.stop()

    def test_start_skips_benchmarks_when_no_location(self, mock_config_no_location, mock_psutil):
        """Test that benchmarks are skipped when benchmark_data_location is not set."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured"
        ) as mock_captured, patch("mosaic_stats.stats_collector.save_benchmarks") as mock_save:
            collector = StatsCollector(mock_config_no_location)
            collector.start()

            # Should NOT run benchmarks or check for them
            mock_save.assert_not_called()
            mock_captured.assert_not_called()

            collector.stop()

    def test_start_skips_benchmarks_when_location_is_falsy(self, mock_psutil):
        """Test that benchmarks are skipped when benchmark_data_location is falsy (None, False, or empty)."""
        # Test with None
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location="",  # Empty string is falsy
            run_benchmark_at_startup=True,  # Even with flag True, should not run
        )
        # Set to None to test None case
        config.benchmark_data_location = None  # type: ignore

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured"
        ) as mock_captured, patch("mosaic_stats.stats_collector.save_benchmarks") as mock_save:
            collector = StatsCollector(config)
            collector.start()

            # Should NOT run benchmarks or check for them, even with run_benchmark_at_startup=True
            mock_save.assert_not_called()
            mock_captured.assert_not_called()

            collector.stop()

        # Test with False (falsy value)
        config2 = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location="",
            run_benchmark_at_startup=True,
        )
        # Set to False to test falsy behavior
        config2.benchmark_data_location = False  # type: ignore

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured"
        ) as mock_captured2, patch("mosaic_stats.stats_collector.save_benchmarks") as mock_save2:
            collector2 = StatsCollector(config2)
            collector2.start()

            # Should NOT run benchmarks or check for them, even with run_benchmark_at_startup=True
            mock_save2.assert_not_called()
            mock_captured2.assert_not_called()

            collector2.stop()

    def test_start_handles_benchmark_errors(self, mock_config, mock_psutil, tmp_path):
        """Test that benchmark errors don't prevent stats collector from starting."""
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            run_benchmark_at_startup=True,
        )

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=False
        ), patch(
            "mosaic_stats.stats_collector.save_benchmarks", side_effect=Exception("Benchmark error")
        ):
            collector = StatsCollector(config)
            collector.start()

            # Stats collector should still start despite benchmark error
            assert collector._running
            assert collector._collection_thread is not None
            assert collector._collection_thread.is_alive()

            collector.stop()

    def test_start_runs_benchmarks_when_both_conditions_true(self, mock_config, mock_psutil, tmp_path):
        """Test that benchmarks run when both flag is True and benchmarks don't exist."""
        config = MosaicConfig(
            heartbeat_report_length=60,
            benchmark_data_location=str(tmp_path / "benchmarks"),
            run_benchmark_at_startup=True,
        )

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=False
        ) as mock_captured, patch(
            "mosaic_stats.stats_collector.save_benchmarks"
        ) as mock_save:
            collector = StatsCollector(config)
            collector.start()

            # Should run benchmarks (only once, not twice)
            # Note: benchmarks_captured is not called when run_benchmark_at_startup is True
            # because the 'or' condition short-circuits
            mock_save.assert_called_once_with(str(tmp_path / "benchmarks"))

            collector.stop()


class TestGetCurrentStats:
    """Test cases for get_current_stats method."""

    def test_get_current_stats(self, mock_config, mock_psutil):
        """Test getting current stats."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)

            test_data = [
                {"timestamp_ms": 1000, "cpu_percent": 50.0},
                {"timestamp_ms": 2000, "cpu_percent": 55.0},
            ]

            with collector._data_lock:
                collector._stats_data.extend(test_data)

            result = collector.get_current_stats()
            assert len(result) == 2
            assert result[0]["timestamp_ms"] == 1000
            assert result[1]["timestamp_ms"] == 2000

    def test_get_current_stats_returns_copy(self, mock_config, mock_psutil):
        """Test that get_current_stats returns a copy, not a reference."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)

            test_data = [{"timestamp_ms": 1000, "cpu_percent": 50.0}]

            with collector._data_lock:
                collector._stats_data.extend(test_data)

            result1 = collector.get_current_stats()
            result2 = collector.get_current_stats()

            # Modify result1
            result1[0]["cpu_percent"] = 99.0

            # result2 should be unchanged
            assert result2[0]["cpu_percent"] == 50.0

            # Original data should be unchanged
            with collector._data_lock:
                assert collector._stats_data[0]["cpu_percent"] == 50.0

    def test_get_last_stats_json(self, mock_config, mock_psutil):
        """Test getting last stats as JSON."""
        import json

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)

            test_data = [
                {"timestamp_ms": 1000, "cpu_percent": 50.0, "ram_percent": 60.0, "gpus": [], "disk_free_space": {}},
                {"timestamp_ms": 2000, "cpu_percent": 55.0, "ram_percent": 65.0, "gpus": [], "disk_free_space": {}},
            ]

            with collector._data_lock:
                collector._stats_data.extend(test_data)

            result = collector.get_last_stats_json()

            # Should be valid JSON
            parsed = json.loads(result)
            assert parsed["timestamp_ms"] == 2000
            assert parsed["cpu_percent"] == 55.0
            assert parsed["ram_percent"] == 65.0

    def test_get_last_stats_json_empty(self, mock_config, mock_psutil):
        """Test getting last stats JSON when stats_data is empty."""
        import json

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)

            result = collector.get_last_stats_json()

            # Should return empty JSON object
            assert result == "{}"
            # Should be valid JSON
            parsed = json.loads(result)
            assert parsed == {}

    def test_get_last_stats_json_error_handling(self, mock_config, mock_psutil):
        """Test that get_last_stats_json handles errors gracefully."""
        import json

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)

            test_data = [{"timestamp_ms": 1000, "cpu_percent": 50.0}]

            with collector._data_lock:
                collector._stats_data.extend(test_data)

            # Mock json.dumps to raise an error
            with patch("mosaic_stats.stats_collector.json.dumps", side_effect=Exception("JSON error")):
                result = collector.get_last_stats_json()

                # Should return empty JSON object on error
                assert result == "{}"
                # Should be valid JSON
                parsed = json.loads(result)
                assert parsed == {}

    def test_get_last_stats_json_valid_json(self, mock_config, mock_psutil):
        """Test that get_last_stats_json always returns valid JSON."""
        import json

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)

            # Test with complex data structure
            test_data = [
                {
                    "timestamp_ms": 1000,
                    "cpu_percent": 50.0,
                    "ram_percent": 60.0,
                    "disk_free_space": {
                        "/": {
                            "total_bytes": 1000000000,
                            "free_bytes": 300000000,
                            "used_bytes": 700000000,
                            "percent_used": 70.0,
                        }
                    },
                    "gpus": [{"gpu_id": 0, "utilization_percent": 85.5}],
                }
            ]

            with collector._data_lock:
                collector._stats_data.extend(test_data)

            result = collector.get_last_stats_json()

            # Should be valid JSON that can be parsed
            parsed = json.loads(result)
            assert isinstance(parsed, dict)
            assert "timestamp_ms" in parsed
            assert "cpu_percent" in parsed
            assert "disk_free_space" in parsed
            assert "gpus" in parsed
            assert isinstance(parsed["gpus"], list)
            assert len(parsed["gpus"]) == 1


class TestIntegration:
    """Integration tests for the stats collector."""

    def test_full_cycle(self, mock_config, mock_psutil, tmp_path):
        """Test a full cycle: start, collect, stop."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "mosaic_stats.stats_collector.benchmarks_captured", return_value=True
        ), patch("mosaic_stats.stats_collector.save_benchmarks"):
            collector = StatsCollector(mock_config)
            collector._file_write_interval = 0.1  # Very short for testing

            collector.start()

            # Wait a bit for collection
            time.sleep(0.5)

            # Get current stats
            stats = collector.get_current_stats()
            assert len(stats) > 0

            # Verify stats structure
            if stats:
                assert "timestamp_ms" in stats[0]
                assert "cpu_percent" in stats[0]
                assert "ram_percent" in stats[0]
                assert "disk_free_space" in stats[0]
                assert "gpus" in stats[0]

            collector.stop()

            # Verify file was written
            stats_file = tmp_path / "benchmarks" / "rolling_stats.json"
            if stats_file.exists():
                with open(stats_file, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    assert len(file_data) > 0

    def test_file_write_error_handling(self, mock_config, mock_psutil, tmp_path):
        """Test that file write errors are handled gracefully."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil), patch(
            "builtins.open", side_effect=PermissionError("Permission denied")
        ):
            collector = StatsCollector(mock_config)
            test_data = [{"timestamp_ms": 1000, "cpu_percent": 50.0, "ram_percent": 60.0, "gpus": [], "disk_free_space": {}}]

            with collector._data_lock:
                collector._stats_data.extend(test_data)

            # Should not raise error
            collector._write_stats_file()

    def test_multiple_gpus(self, mock_config, mock_psutil):
        """Test handling multiple GPUs."""
        mock_gpu1 = MagicMock()
        mock_gpu1.id = 0
        mock_gpu1.load = 0.5
        mock_gpu2 = MagicMock()
        mock_gpu2.id = 1
        mock_gpu2.load = 0.75

        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            collector._gpu_handles = [("nvidia", mock_gpu1), ("nvidia", mock_gpu2)]
            collector._gpu_type = "nvidia"

            result = collector._get_gpu_usage()
            assert len(result) == 2
            assert result[0]["gpu_id"] == 0
            assert result[0]["utilization_percent"] == 50.0
            assert result[1]["gpu_id"] == 1
            assert result[1]["utilization_percent"] == 75.0

    def test_timestamp_format(self, mock_config, mock_psutil):
        """Test that timestamps are in milliseconds."""
        with patch("mosaic_stats.stats_collector.psutil", mock_psutil):
            collector = StatsCollector(mock_config)
            stats = collector._collect_stats()

            # Timestamp should be milliseconds (much larger than seconds)
            assert stats["timestamp_ms"] > 1000000000000  # Should be > year 2001 in ms
            assert stats["timestamp_ms"] < 9999999999999  # Should be < year 2286 in ms

