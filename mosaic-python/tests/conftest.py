"""Pytest configuration and shared fixtures for Mosaic tests."""

import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from mosaic_config.config import MosaicConfig
from mosaic_config.state_utils import StateIdentifiers


@pytest.fixture(autouse=True)
def clean_state_files(tmp_path):
    """
    Automatically clean up state files before and after each test.
    
    This ensures that state files from previous tests don't interfere
    with current tests. Cleans both the tests/temp_state directory
    and any temporary state directories created during tests.
    """
    # Get the tests directory
    tests_dir = Path(__file__).parent
    temp_state_dir = tests_dir / "temp_state"
    
    # Ensure temp_state directory exists
    temp_state_dir.mkdir(exist_ok=True)
    
    # Clean up before test
    if temp_state_dir.exists():
        for state_file in temp_state_dir.glob("*.pkl"):
            try:
                state_file.unlink()
            except Exception:
                pass
        for state_file in temp_state_dir.glob("*.pkl.tmp"):
            try:
                state_file.unlink()
            except Exception:
                pass
    
    yield
    
    # Clean up after test - both tests/temp_state and any temp dirs
    if temp_state_dir.exists():
        for state_file in temp_state_dir.glob("*.pkl"):
            try:
                state_file.unlink()
            except Exception:
                pass
        for state_file in temp_state_dir.glob("*.pkl.tmp"):
            try:
                state_file.unlink()
            except Exception:
                pass
    
    # Also clean up any state files in tmp_path (for tests using temp_state_dir fixture)
    if tmp_path.exists():
        for state_file in tmp_path.rglob("*.pkl"):
            try:
                state_file.unlink()
            except Exception:
                pass
        for state_file in tmp_path.rglob("*.pkl.tmp"):
            try:
                state_file.unlink()
            except Exception:
                pass


@pytest.fixture
def temp_state_dir(tmp_path) -> Path:
    """
    Create a temporary state directory for tests.
    
    Returns:
        Path to temporary state directory
    """
    state_dir = tmp_path / "temp_state"
    state_dir.mkdir(exist_ok=True)
    return state_dir


def create_test_config_with_state(*, state_dir: Path, **kwargs) -> MosaicConfig:
    """
    Create a MosaicConfig with state_location set to the provided directory.
    
    Args:
        state_dir: Directory path for state storage (keyword argument)
        **kwargs: Additional config parameters to override
        
    Returns:
        MosaicConfig instance with state_location set
    """
    config = MosaicConfig(**kwargs)
    config.state_location = str(state_dir)
    return config


@pytest.fixture
def mock_state_utils():
    """
    Mock state save/read functions to prevent actual file I/O in tests.
    
    Use this fixture when you want to test without actual state persistence.
    """
    with patch("mosaic_config.state_utils.save_state") as mock_save, patch(
        "mosaic_config.state_utils.read_state"
    ) as mock_read:
        # Default behavior: read_state returns None (no saved state)
        mock_read.return_value = None
        # save_state does nothing by default
        mock_save.return_value = None
        yield {"save": mock_save, "read": mock_read}


@pytest.fixture
def beacon_config_no_ssl(temp_state_dir):
    """
    Create a MosaicConfig for beacon testing without SSL.
    
    Uses port 5000 for heartbeat and 5001 for comms.
    """
    return create_test_config_with_state(
        state_dir=temp_state_dir,
        host="127.0.0.1",
        heartbeat_port=5000,
        comms_port=5001,
        heartbeat_frequency=2,
        heartbeat_tolerance=5,
        heartbeat_wait_timeout=2,
        stats_request_timeout=10,
        server_crt="",
        server_key="",
        ca_crt="",
        benchmark_data_location="",
    )


@pytest.fixture
def sender_config_no_ssl(temp_state_dir):
    """
    Create a MosaicConfig for sender beacon testing without SSL.
    
    Uses port 5002 for heartbeat and 5003 for comms.
    """
    return create_test_config_with_state(
        state_dir=temp_state_dir,
        host="127.0.0.1",
        heartbeat_port=5002,
        comms_port=5003,
        heartbeat_frequency=2,
        heartbeat_tolerance=5,
        heartbeat_wait_timeout=2,
        stats_request_timeout=10,
        server_crt="",
        server_key="",
        ca_crt="",
        benchmark_data_location="",
    )
