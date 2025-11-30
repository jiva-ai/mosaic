"""Unit tests for mosaic_config.config module."""

import json
import os
import platform
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mosaic_config.config import MosaicConfig, Peer, read_config, read_json_config


class TestReadJsonConfig:
    """Test cases for read_json_config function."""

    def test_read_json_config_from_command_line(self, tmp_path, monkeypatch):
        """Test reading config from --config command-line argument."""
        config_file = tmp_path / "test_config.json"
        config_data = {"key1": "value1", "key2": 42}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Mock sys.argv to include --config
        test_args = ["script.py", "--config", str(config_file)]
        with patch.object(sys, "argv", test_args):
            result = read_json_config()
            assert result == config_data

    def test_read_json_config_from_env_var(self, tmp_path, monkeypatch):
        """Test reading config from MOSAIC_CONFIG environment variable."""
        config_file = tmp_path / "env_config.json"
        config_data = {"env_key": "env_value", "number": 123}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Set environment variable
        monkeypatch.setenv("MOSAIC_CONFIG", str(config_file))
        # Mock sys.argv to not have --config
        with patch.object(sys, "argv", ["script.py"]):
            result = read_json_config()
            assert result == config_data

    def test_read_json_config_from_cwd(self, tmp_path, monkeypatch):
        """Test reading config from mosaic.config in current working directory."""
        config_file = tmp_path / "mosaic.config"
        config_data = {"cwd_key": "cwd_value", "nested": {"a": 1, "b": 2}}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        # Unset environment variable if it exists
        monkeypatch.delenv("MOSAIC_CONFIG", raising=False)
        # Mock sys.argv to not have --config
        with patch.object(sys, "argv", ["script.py"]):
            result = read_json_config()
            assert result == config_data

    def test_priority_command_line_over_env(self, tmp_path, monkeypatch):
        """Test that command-line argument takes priority over environment variable."""
        cli_config = tmp_path / "cli_config.json"
        env_config = tmp_path / "env_config.json"
        cli_data = {"source": "command_line"}
        env_data = {"source": "environment"}

        cli_config.write_text(json.dumps(cli_data), encoding="utf-8")
        env_config.write_text(json.dumps(env_data), encoding="utf-8")

        monkeypatch.setenv("MOSAIC_CONFIG", str(env_config))
        test_args = ["script.py", "--config", str(cli_config)]
        with patch.object(sys, "argv", test_args):
            result = read_json_config()
            assert result == cli_data
            assert result["source"] == "command_line"

    def test_priority_env_over_cwd(self, tmp_path, monkeypatch):
        """Test that environment variable takes priority over CWD file."""
        env_config = tmp_path / "env_config.json"
        cwd_config = tmp_path / "mosaic.config"
        env_data = {"source": "environment"}
        cwd_data = {"source": "cwd"}

        env_config.write_text(json.dumps(env_data), encoding="utf-8")
        cwd_config.write_text(json.dumps(cwd_data), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MOSAIC_CONFIG", str(env_config))
        with patch.object(sys, "argv", ["script.py"]):
            result = read_json_config()
            assert result == env_data
            assert result["source"] == "environment"

    def test_file_not_found_command_line(self, monkeypatch):
        """Test FileNotFoundError when --config file doesn't exist."""
        test_args = ["script.py", "--config", "/nonexistent/file.json"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                read_json_config()

    def test_file_not_found_env_var(self, monkeypatch):
        """Test FileNotFoundError when MOSAIC_CONFIG file doesn't exist."""
        monkeypatch.setenv("MOSAIC_CONFIG", "/nonexistent/file.json")
        with patch.object(sys, "argv", ["script.py"]):
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                read_json_config()

    def test_no_config_found(self, tmp_path, monkeypatch):
        """Test FileNotFoundError when no config is found anywhere."""
        # Ensure no config exists
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MOSAIC_CONFIG", raising=False)
        with patch.object(sys, "argv", ["script.py"]):
            with pytest.raises(FileNotFoundError, match="No configuration file found"):
                read_json_config()

    def test_invalid_json(self, tmp_path, monkeypatch):
        """Test JSONDecodeError when config file contains invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }", encoding="utf-8")

        test_args = ["script.py", "--config", str(config_file)]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(json.JSONDecodeError):
                read_json_config()

    def test_config_is_directory_not_file(self, tmp_path, monkeypatch):
        """Test ValueError when config path points to a directory."""
        config_dir = tmp_path / "config_dir"
        config_dir.mkdir()

        test_args = ["script.py", "--config", str(config_dir)]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(ValueError, match="Configuration path is not a file"):
                read_json_config()

    def test_complex_json_structure(self, tmp_path, monkeypatch):
        """Test reading complex nested JSON structure."""
        config_file = tmp_path / "complex.json"
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {"user": "admin", "password": "secret"},
            },
            "features": ["feature1", "feature2", "feature3"],
            "settings": {"timeout": 30, "retries": 3},
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        test_args = ["script.py", "--config", str(config_file)]
        with patch.object(sys, "argv", test_args):
            result = read_json_config()
            assert result == config_data
            assert result["database"]["host"] == "localhost"
            assert result["features"][0] == "feature1"

    def test_path_expansion_home_directory_cli(self, tmp_path, monkeypatch):
        """Test that ~ (home directory) is expanded in command-line argument."""
        # Create a config file in a subdirectory
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_file = config_dir / "test.json"
        config_data = {"expanded": "home"}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Mock home directory to point to tmp_path
        # On Windows, os.path.expanduser() uses USERPROFILE, not HOME
        monkeypatch.setenv("HOME", str(tmp_path))
        if platform.system() == "Windows":
            monkeypatch.setenv("USERPROFILE", str(tmp_path))
        # Use ~/configs/test.json as the path
        test_args = ["script.py", "--config", "~/configs/test.json"]
        with patch.object(sys, "argv", test_args):
            result = read_json_config()
            assert result == config_data

    def test_path_expansion_home_directory_env(self, tmp_path, monkeypatch):
        """Test that ~ (home directory) is expanded in environment variable."""
        # Create a config file in a subdirectory
        config_dir = tmp_path / "my_configs"
        config_dir.mkdir()
        config_file = config_dir / "app.json"
        config_data = {"expanded": "env_home"}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Mock home directory to point to tmp_path
        # On Windows, os.path.expanduser() uses USERPROFILE, not HOME
        monkeypatch.setenv("HOME", str(tmp_path))
        if platform.system() == "Windows":
            monkeypatch.setenv("USERPROFILE", str(tmp_path))
        # Set environment variable with ~ expansion
        monkeypatch.setenv("MOSAIC_CONFIG", "~/my_configs/app.json")
        with patch.object(sys, "argv", ["script.py"]):
            result = read_json_config()
            assert result == config_data

    def test_path_expansion_env_var_cli(self, tmp_path, monkeypatch):
        """Test that environment variables are expanded in command-line argument."""
        config_file = tmp_path / "config.json"
        config_data = {"expanded": "env_var"}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Set a test environment variable
        monkeypatch.setenv("TEST_CONFIG_DIR", str(tmp_path))
        # Use $TEST_CONFIG_DIR/config.json as the path
        test_args = ["script.py", "--config", "$TEST_CONFIG_DIR/config.json"]
        with patch.object(sys, "argv", test_args):
            result = read_json_config()
            assert result == config_data

    def test_path_expansion_env_var_env(self, tmp_path, monkeypatch):
        """Test that environment variables are expanded in environment variable."""
        config_file = tmp_path / "settings.json"
        config_data = {"expanded": "env_var_in_env"}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Set test environment variables
        monkeypatch.setenv("APP_DIR", str(tmp_path))
        monkeypatch.setenv("MOSAIC_CONFIG", "${APP_DIR}/settings.json")
        with patch.object(sys, "argv", ["script.py"]):
            result = read_json_config()
            assert result == config_data

    def test_path_expansion_combined(self, tmp_path, monkeypatch):
        """Test that both ~ and environment variables can be combined."""
        config_dir = tmp_path / "app" / "configs"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "combined.json"
        config_data = {"expanded": "combined"}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Set environment variable pointing to subdirectory
        # On Windows, os.path.expanduser() uses USERPROFILE, not HOME
        monkeypatch.setenv("CONFIG_SUBDIR", "app/configs")
        monkeypatch.setenv("HOME", str(tmp_path))
        if platform.system() == "Windows":
            monkeypatch.setenv("USERPROFILE", str(tmp_path))
        # Use ~/$CONFIG_SUBDIR/combined.json
        test_args = ["script.py", "--config", "~/$CONFIG_SUBDIR/combined.json"]
        with patch.object(sys, "argv", test_args):
            result = read_json_config()
            assert result == config_data


class TestReadConfig:
    """Test cases for read_config function that returns MosaicConfig."""

    def test_read_config_defaults(self, tmp_path, monkeypatch):
        """Test that read_config returns defaults when config file is empty."""
        config_file = tmp_path / "empty_config.json"
        config_file.write_text("{}", encoding="utf-8")

        test_args = ["script.py", "--config", str(config_file)]
        with patch.object(sys, "argv", test_args):
            config = read_config()
            assert isinstance(config, MosaicConfig)
            assert config.host == "localhost"
            assert config.heartbeat_port == 5000
            assert config.comms_port == 5001
            assert config.peers == []
            assert config.heartbeat_frequency == 5
            assert config.heartbeat_tolerance == 15
            assert config.heartbeat_report_length == 300
            assert config.heartbeat_wait_timeout == 2
            assert config.stats_request_timeout == 30
            assert config.server_crt == ""
            assert config.server_key == ""
            assert config.ca_crt == ""
            assert config.benchmark_data_location == ""
            assert config.run_benchmark_at_startup is False
            assert config.data_location == ""
            assert config.plans_location == "plans"
            assert config.models_location == "models"
            assert config.state_location == ""

    def test_read_config_with_all_fields(self, tmp_path, monkeypatch):
        """Test reading a complete configuration file."""
        config_file = tmp_path / "full_config.json"
        config_data = {
            "host": "192.168.1.100",
            "heartbeat_port": 6000,
            "comms_port": 6001,
            "peers": [
                {"host": "192.168.1.101", "comms_port": 6001, "heartbeat_port": 6000},
                {"host": "192.168.1.102", "comms_port": 6001, "heartbeat_port": 6000},
            ],
            "heartbeat_frequency": 10,
            "heartbeat_tolerance": 30,
            "heartbeat_report_length": 600,
            "heartbeat_wait_timeout": 3,
            "stats_request_timeout": 45,
            "server_crt": "/path/to/server.crt",
            "server_key": "/path/to/server.key",
            "ca_crt": "/path/to/ca.crt",
            "benchmark_data_location": "/data/benchmarks",
            "run_benchmark_at_startup": True,
            "data_location": "/data/mosaic",
            "plans_location": "/data/plans",
            "models_location": "/data/models",
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        test_args = ["script.py", "--config", str(config_file)]
        with patch.object(sys, "argv", test_args):
            config = read_config()
            assert isinstance(config, MosaicConfig)
            assert config.host == "192.168.1.100"
            assert config.heartbeat_port == 6000
            assert config.comms_port == 6001
            assert len(config.peers) == 2
            assert config.peers[0].host == "192.168.1.101"
            assert config.peers[0].comms_port == 6001
            assert config.peers[0].heartbeat_port == 6000
            assert config.peers[1].host == "192.168.1.102"
            assert config.peers[1].comms_port == 6001
            assert config.peers[1].heartbeat_port == 6000
            assert config.heartbeat_frequency == 10
            assert config.heartbeat_tolerance == 30
            assert config.heartbeat_report_length == 600
            assert config.heartbeat_wait_timeout == 3
            assert config.stats_request_timeout == 45
            assert config.server_crt == "/path/to/server.crt"
            assert config.server_key == "/path/to/server.key"
            assert config.ca_crt == "/path/to/ca.crt"
            assert config.benchmark_data_location == "/data/benchmarks"
            assert config.run_benchmark_at_startup is True
            assert config.data_location == "/data/mosaic"
            assert config.plans_location == "/data/plans"
            assert config.models_location == "/data/models"

    def test_read_config_partial_override(self, tmp_path, monkeypatch):
        """Test that only specified fields override defaults."""
        config_file = tmp_path / "partial_config.json"
        config_data = {
            "host": "10.0.0.1",
            "heartbeat_port": 7000,
            "peers": [{"host": "10.0.0.2", "comms_port": 7001, "heartbeat_port": 7000}],
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        test_args = ["script.py", "--config", str(config_file)]
        with patch.object(sys, "argv", test_args):
            config = read_config()
            assert config.host == "10.0.0.1"
            assert config.heartbeat_port == 7000
            assert config.comms_port == 5001  # Default value
            assert len(config.peers) == 1
            assert config.peers[0].host == "10.0.0.2"
            assert config.peers[0].comms_port == 7001
            assert config.peers[0].heartbeat_port == 7000
            assert config.heartbeat_frequency == 5  # Default value
            assert config.heartbeat_wait_timeout == 2  # Default value
            assert config.stats_request_timeout == 30  # Default value
            assert config.plans_location == "plans"  # Default value
            assert config.models_location == "models"  # Default value

    def test_read_config_with_config_path_parameter(self, tmp_path, monkeypatch):
        """Test that read_config accepts config_path parameter."""
        config_file = tmp_path / "param_config.json"
        config_data = {
            "host": "192.168.1.200",
            "heartbeat_port": 7000,
            "comms_port": 7001,
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        # Call read_config with config_path parameter
        config = read_config(config_path=str(config_file))
        assert isinstance(config, MosaicConfig)
        assert config.host == "192.168.1.200"
        assert config.heartbeat_port == 7000
        assert config.comms_port == 7001

    def test_read_config_config_path_takes_precedence(self, tmp_path, monkeypatch):
        """Test that config_path parameter takes precedence over other sources."""
        # Create two config files
        file1 = tmp_path / "config1.json"
        file1.write_text(json.dumps({"host": "file1", "heartbeat_port": 8000}), encoding="utf-8")
        
        file2 = tmp_path / "config2.json"
        file2.write_text(json.dumps({"host": "file2", "heartbeat_port": 9000}), encoding="utf-8")

        # Set environment variable pointing to file1
        monkeypatch.setenv("MOSAIC_CONFIG", str(file1))
        
        # But pass file2 as config_path parameter - should use file2
        config = read_config(config_path=str(file2))
        assert config.host == "file2"
        assert config.heartbeat_port == 9000

    def test_read_config_config_path_not_found(self, tmp_path):
        """Test that read_config raises FileNotFoundError for non-existent config_path."""
        non_existent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            read_config(config_path=str(non_existent))

    def test_read_config_nested_heartbeat(self, tmp_path, monkeypatch):
        """Test reading heartbeat config from nested structure."""
        config_file = tmp_path / "nested_heartbeat.json"
        config_data = {
            "host": "localhost",
            "heartbeat": {
                "frequency": 20,
                "tolerance": 45,
                "report_length": 900,
                "wait_timeout": 5,
            },
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        test_args = ["script.py", "--config", str(config_file)]
        with patch.object(sys, "argv", test_args):
            config = read_config()
            assert config.heartbeat_frequency == 20
            assert config.heartbeat_tolerance == 45
            assert config.heartbeat_report_length == 900
            assert config.heartbeat_wait_timeout == 5


