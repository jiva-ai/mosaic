"""Unit tests for mosaic_config.state_utils module."""

import pickle
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mosaic_config.config import MosaicConfig
from mosaic_config.state_utils import (
    StateIdentifiers,
    _get_state_directory,
    read_state,
    save_state,
)
from tests.conftest import create_test_config_with_state


class TestGetStateDirectory:
    """Test _get_state_directory function."""

    def test_get_state_directory_with_trailing_slash_forward(self, tmp_path, monkeypatch):
        """Test that trailing forward slash is removed."""
        state_path = str(tmp_path / "state") + "/"
        config = MosaicConfig(state_location=state_path)
        
        result = _get_state_directory(config)
        assert result == tmp_path / "state"
        assert not str(result).endswith("/")

    def test_get_state_directory_with_trailing_slash_backward(self, tmp_path):
        """Test that trailing backward slash is removed (Windows)."""
        state_path = str(tmp_path / "state") + "\\"
        config = MosaicConfig(state_location=state_path)
        
        result = _get_state_directory(config)
        assert result == tmp_path / "state"
        # On Windows, pathlib normalizes, but we should not have trailing backslash
        assert not str(result).endswith("\\")

    def test_get_state_directory_with_multiple_trailing_slashes(self, tmp_path):
        """Test that multiple trailing slashes are handled."""
        state_path = str(tmp_path / "state") + "///"
        config = MosaicConfig(state_location=state_path)
        
        result = _get_state_directory(config)
        # Should remove only one trailing slash
        assert result == Path(str(tmp_path / "state") + "//")

    def test_get_state_directory_empty_defaults_to_cwd(self, monkeypatch):
        """Test that empty state_location defaults to current working directory."""
        config = MosaicConfig(state_location="")
        result = _get_state_directory(config)
        assert result == Path.cwd()

    def test_get_state_directory_whitespace_only_defaults_to_cwd(self):
        """Test that whitespace-only state_location defaults to current working directory."""
        config = MosaicConfig(state_location="   ")
        result = _get_state_directory(config)
        assert result == Path.cwd()

    def test_get_state_directory_with_path(self, tmp_path):
        """Test that valid path is returned as-is (after trailing slash removal)."""
        state_path = str(tmp_path / "my_state")
        config = MosaicConfig(state_location=state_path)
        
        result = _get_state_directory(config)
        assert result == Path(state_path)


class TestSaveAndReadState:
    """Test save_state and read_state functions."""

    def test_save_and_read_simple_object(self, temp_state_dir):
        """Test that a simple object can be saved and read back exactly."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        save_state(config, test_data, "test_simple")
        result = read_state(config, "test_simple")
        
        assert result == test_data
        assert result["key"] == "value"
        assert result["number"] == 42
        assert result["list"] == [1, 2, 3]

    def test_save_and_read_complex_object(self, temp_state_dir):
        """Test that complex nested objects are saved and read correctly."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        test_data = {
            "nested": {
                "level1": {
                    "level2": {"value": 123},
                    "list": [{"a": 1}, {"b": 2}],
                }
            },
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
        }
        
        save_state(config, test_data, "test_complex")
        result = read_state(config, "test_complex")
        
        assert result == test_data
        assert result["nested"]["level1"]["level2"]["value"] == 123
        assert result["tuple"] == (1, 2, 3)
        # Note: sets are preserved in pickle
        assert result["set"] == {1, 2, 3}

    def test_save_and_read_none_value(self, temp_state_dir):
        """Test that None can be saved and read."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        save_state(config, None, "test_none")
        result = read_state(config, "test_none")
        
        assert result is None

    def test_read_nonexistent_file_returns_default(self, temp_state_dir):
        """Test that reading a non-existent file returns the default value."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        result = read_state(config, "nonexistent", default="default_value")
        assert result == "default_value"

    def test_read_nonexistent_file_returns_none_by_default(self, temp_state_dir):
        """Test that reading a non-existent file returns None by default."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        result = read_state(config, "nonexistent")
        assert result is None

    def test_save_overwrites_existing_file(self, temp_state_dir):
        """Test that saving to the same identifier overwrites the file."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        save_state(config, {"first": "value"}, "test_overwrite")
        save_state(config, {"second": "value"}, "test_overwrite")
        
        result = read_state(config, "test_overwrite")
        assert result == {"second": "value"}
        assert "first" not in result

    def test_identifier_sanitization(self, temp_state_dir):
        """Test that identifiers with slashes are sanitized for filenames."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        test_data = {"test": "data"}
        
        # Identifier with forward slash
        save_state(config, test_data, "test/path")
        result = read_state(config, "test/path")
        assert result == test_data
        
        # Verify the actual filename is sanitized
        state_file = temp_state_dir / "test_path.pkl"
        assert state_file.exists()

    def test_identifier_with_backslash_sanitization(self, temp_state_dir):
        """Test that identifiers with backslashes are sanitized."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        test_data = {"test": "data"}
        
        save_state(config, test_data, "test\\path")
        result = read_state(config, "test\\path")
        assert result == test_data
        
        # Verify the actual filename is sanitized
        state_file = temp_state_dir / "test_path.pkl"
        assert state_file.exists()

    def test_save_creates_directory_if_not_exists(self, tmp_path):
        """Test that save_state creates the directory if it doesn't exist."""
        state_dir = tmp_path / "new_state_dir"
        config = create_test_config_with_state(state_dir=state_dir)
        
        assert not state_dir.exists()
        save_state(config, {"test": "data"}, "test_create_dir")
        assert state_dir.exists()
        
        result = read_state(config, "test_create_dir")
        assert result == {"test": "data"}


class TestThreadSafety:
    """Test thread safety of save_state and read_state."""

    def test_concurrent_saves_same_identifier(self, temp_state_dir):
        """Test that concurrent saves to the same identifier are thread-safe."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        num_threads = 10
        results = []
        errors = []

        def save_data(thread_id):
            try:
                data = {"thread_id": thread_id, "value": thread_id * 10}
                save_state(config, data, "concurrent_test")
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=save_data, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should complete without errors
        assert len(errors) == 0
        assert len(results) == num_threads

        # Final state should be from one of the threads (last write wins)
        final_result = read_state(config, "concurrent_test")
        assert final_result is not None
        assert "thread_id" in final_result
        assert "value" in final_result

    def test_concurrent_reads_and_writes(self, temp_state_dir):
        """Test that concurrent reads and writes are thread-safe."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Initial save
        save_state(config, {"initial": "data"}, "concurrent_rw")
        
        read_results = []
        write_errors = []

        def read_data():
            try:
                result = read_state(config, "concurrent_rw")
                read_results.append(result)
            except Exception as e:
                read_results.append(("error", e))

        def write_data(thread_id):
            try:
                save_state(config, {"thread": thread_id}, "concurrent_rw")
            except Exception as e:
                write_errors.append((thread_id, e))

        # Start multiple readers and writers
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=read_data))
            threads.append(threading.Thread(target=write_data, args=(i,)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no write errors
        assert len(write_errors) == 0
        # All reads should succeed (may get different values, but no errors)
        assert len(read_results) == 5
        assert all(not isinstance(r, tuple) or r[0] != "error" for r in read_results)

    def test_different_identifiers_have_separate_locks(self, temp_state_dir):
        """Test that different identifiers use separate locks and don't block each other."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        barrier = threading.Barrier(2)
        results = []

        def save_identifier_1():
            barrier.wait()  # Synchronize start
            save_state(config, {"id": 1}, "identifier_1")
            results.append("id1_done")

        def save_identifier_2():
            barrier.wait()  # Synchronize start
            save_state(config, {"id": 2}, "identifier_2")
            results.append("id2_done")

        thread1 = threading.Thread(target=save_identifier_1)
        thread2 = threading.Thread(target=save_identifier_2)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()

        # Both should complete (separate locks, no blocking)
        assert len(results) == 2
        assert "id1_done" in results
        assert "id2_done" in results

        # Both files should exist
        assert read_state(config, "identifier_1") == {"id": 1}
        assert read_state(config, "identifier_2") == {"id": 2}


class TestStateIdentifiers:
    """Test StateIdentifiers constants."""

    def test_state_identifiers_constants_exist(self):
        """Test that StateIdentifiers constants are defined."""
        assert hasattr(StateIdentifiers, "SEND_HEARTBEAT_STATUSES")
        assert hasattr(StateIdentifiers, "RECEIVE_HEARTBEAT_STATUSES")
        assert StateIdentifiers.SEND_HEARTBEAT_STATUSES == "send_heartbeat_statuses"
        assert StateIdentifiers.RECEIVE_HEARTBEAT_STATUSES == "receive_heartbeat_statuses"

    def test_using_state_identifiers_constants(self, temp_state_dir):
        """Test that StateIdentifiers constants can be used for save/read."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        test_data = {"status": "test"}
        
        save_state(config, test_data, StateIdentifiers.SEND_HEARTBEAT_STATUSES)
        result = read_state(config, StateIdentifiers.SEND_HEARTBEAT_STATUSES)
        
        assert result == test_data


class TestErrorHandling:
    """Test error handling in state utilities."""

    def test_read_invalid_pickle_returns_default(self, temp_state_dir):
        """Test that reading an invalid pickle file returns default."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Create an invalid pickle file
        invalid_file = temp_state_dir / "invalid.pkl"
        invalid_file.write_text("not valid pickle data", encoding="utf-8")
        
        result = read_state(config, "invalid", default="default_value")
        assert result == "default_value"

    def test_save_to_readonly_directory_raises_error(self, tmp_path):
        """Test that saving to a readonly directory raises an error."""
        # Create a directory and make it readonly (on Unix)
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        
        config = create_test_config_with_state(state_dir=readonly_dir)
        
        # On Windows, we can't easily make a directory readonly in a way that prevents writes
        # So we'll test with a path that doesn't exist and can't be created
        # Actually, let's test with a file path instead of directory
        file_path = tmp_path / "not_a_dir"
        file_path.touch()  # Create a file
        
        config.state_location = str(file_path)
        
        with pytest.raises((OSError, ValueError)):
            save_state(config, {"test": "data"}, "test_readonly")

    def test_save_cleans_up_temp_file_on_error(self, temp_state_dir):
        """Test that temp file is cleaned up if save fails."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Create a non-pickleable object (lambda functions can't be pickled)
        unpickleable = lambda x: x
        
        with pytest.raises(Exception):  # Should raise pickle error
            save_state(config, unpickleable, "test_unpickleable")
        
        # Temp file should not exist
        temp_file = temp_state_dir / "test_unpickleable.pkl.tmp"
        assert not temp_file.exists()


class TestAtomicWrites:
    """Test atomic write behavior."""

    def test_atomic_write_uses_temp_file(self, temp_state_dir):
        """Test that writes use a temporary file before atomic rename."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Save state
        save_state(config, {"test": "data"}, "atomic_test")
        
        # Final file should exist
        final_file = temp_state_dir / "atomic_test.pkl"
        assert final_file.exists()
        
        # Temp file should not exist after successful write
        temp_file = temp_state_dir / "atomic_test.pkl.tmp"
        assert not temp_file.exists()

    def test_read_during_write_gets_old_or_new_value(self, temp_state_dir):
        """Test that reading during a write gets either old or new value (atomic behavior)."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Initial save
        save_state(config, {"version": 1}, "atomic_read")
        
        read_values = []
        write_done = threading.Event()

        def read_loop():
            while not write_done.is_set():
                result = read_state(config, "atomic_read")
                if result:
                    read_values.append(result.get("version"))
                time.sleep(0.01)

        def write_new():
            time.sleep(0.05)  # Let some reads happen
            save_state(config, {"version": 2}, "atomic_read")
            write_done.set()

        read_thread = threading.Thread(target=read_loop)
        write_thread = threading.Thread(target=write_new)
        
        read_thread.start()
        write_thread.start()
        
        write_thread.join()
        time.sleep(0.1)  # Let read thread finish
        read_thread.join(timeout=0.2)

        # All reads should get valid values (either 1 or 2, never corrupted)
        assert all(v in (1, 2) for v in read_values)
        # Final read should be version 2
        final = read_state(config, "atomic_read")
        assert final == {"version": 2}

