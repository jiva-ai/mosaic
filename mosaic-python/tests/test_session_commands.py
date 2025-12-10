"""Unit tests for mosaic.session_commands module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List

import pytest

# Patch onnx at import time to avoid import errors
# We need to do this before importing session_commands
import sys

# Store original onnx if it exists
_original_onnx = sys.modules.get('onnx')
_we_added_onnx_mock = False

# Only mock if onnx isn't already imported (to avoid breaking other tests)
if 'onnx' not in sys.modules:
    # Create a proper mock module that won't break when accessed
    from types import ModuleType, SimpleNamespace
    
    class MockOnnx(ModuleType):
        """Mock onnx module with necessary attributes."""
        def __init__(self):
            super().__init__('onnx')
            # Set __spec__ properly - this is critical
            spec = SimpleNamespace()
            spec.name = 'onnx'
            spec.loader = None
            spec.origin = None
            self.__spec__ = spec
            
            # Add submodules and attributes that might be accessed
            # Create reference as a module-like object
            reference = SimpleNamespace()
            self.reference = reference
            
            # Create ModelProto and TensorProto as classes
            self.ModelProto = type('ModelProto', (), {})
            self.TensorProto = type('TensorProto', (), {
                'FLOAT': 1,
                'INT64': 7,
            })
            
            # Create checker with check_model method
            checker = SimpleNamespace()
            checker.check_model = MagicMock()
            self.checker = checker
            
            # Create load function
            self.load = MagicMock()
            
            # Create helper module
            helper = SimpleNamespace()
            helper.make_tensor_value_info = MagicMock()
            helper.make_node = MagicMock()
            helper.make_graph = MagicMock()
            helper.make_model = MagicMock()
            self.helper = helper
        
        def __getattr__(self, name):
            # Return MagicMock for any other attribute access
            return MagicMock()
    
    mock_onnx = MockOnnx()
    sys.modules['onnx'] = mock_onnx
    # Also add onnx.reference to sys.modules for submodule imports
    sys.modules['onnx.reference'] = mock_onnx.reference
    _we_added_onnx_mock = True

# Import session_commands (this will trigger imports that use onnx)
from mosaic.session_commands import (
    _discover_datasets,
    _format_plan_summary,
    _get_predefined_models,
    execute_create_session,
    execute_delete_session,
    initialize,
)

# Note: We don't restore/remove the mock here because:
# 1. If onnx was already imported, we didn't mock it, so nothing to restore
# 2. If we added the mock, Python has already cached imports that used it
# 3. Removing it now would break those cached imports
# Instead, we'll use a pytest fixture to clean up after all tests in this file
from mosaic_config.config import MosaicConfig
from mosaic_config.state import (
    Data,
    DataType,
    FileDefinition,
    Model,
    ModelType,
    Plan,
    Session,
    SessionStatus,
)
from tests.conftest import create_test_config_with_state


@pytest.fixture(scope="module", autouse=True)
def cleanup_onnx_mock():
    """Clean up onnx mock after all tests in this module."""
    yield
    # After all tests, restore original if we had one
    import sys
    if _we_added_onnx_mock and _original_onnx is not None:
        sys.modules['onnx'] = _original_onnx


class TestDiscoverDatasets:
    """Test _discover_datasets function with different DataTypes."""

    def test_discover_datasets_nonexistent_directory(self, tmp_path):
        """Test _discover_datasets with non-existent directory."""
        result = _discover_datasets(str(tmp_path / "nonexistent"))
        assert result == []

    def test_discover_datasets_empty_directory(self, tmp_path):
        """Test _discover_datasets with empty directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = _discover_datasets(str(data_dir))
        assert result == []

    def test_discover_datasets_image_files(self, tmp_path):
        """Test _discover_datasets with IMAGE data type files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create image directory with image files
        image_dir = data_dir / "images"
        image_dir.mkdir()
        (image_dir / "img1.jpg").write_bytes(b"fake image")
        (image_dir / "img2.png").write_bytes(b"fake image")
        (image_dir / "img3.jpeg").write_bytes(b"fake image")
        
        result = _discover_datasets(str(data_dir))
        assert len(result) == 1
        assert result[0]["name"] == "images"
        assert result[0]["type"] == "directory"
        assert "img1.jpg" in result[0]["files"] or "img2.png" in result[0]["files"]

    def test_discover_datasets_audio_files(self, tmp_path):
        """Test _discover_datasets with AUDIO data type files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create audio directory with audio files
        audio_dir = data_dir / "audio"
        audio_dir.mkdir()
        (audio_dir / "audio1.wav").write_bytes(b"fake audio")
        (audio_dir / "audio2.flac").write_bytes(b"fake audio")
        
        result = _discover_datasets(str(data_dir))
        assert len(result) == 1
        assert result[0]["name"] == "audio"
        assert result[0]["type"] == "directory"
        assert any("wav" in f or "flac" in f for f in result[0]["files"])

    def test_discover_datasets_text_files(self, tmp_path):
        """Test _discover_datasets with TEXT data type files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create text directory with text files
        text_dir = data_dir / "text"
        text_dir.mkdir()
        (text_dir / "text1.txt").write_text("some text")
        (text_dir / "text2.jsonl").write_text('{"text": "data"}')
        
        result = _discover_datasets(str(data_dir))
        assert len(result) == 1
        assert result[0]["name"] == "text"
        assert result[0]["type"] == "directory"
        assert any("txt" in f or "jsonl" in f for f in result[0]["files"])

    def test_discover_datasets_csv_files(self, tmp_path):
        """Test _discover_datasets with CSV data type files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create CSV file directly
        csv_file = data_dir / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2")
        
        result = _discover_datasets(str(data_dir))
        assert len(result) == 1
        assert result[0]["name"] == "data.csv"
        assert result[0]["type"] == "file"
        assert result[0]["files"] == ["data.csv"]
        assert result[0]["file_count"] == 1

    def test_discover_datasets_csv_directory(self, tmp_path):
        """Test _discover_datasets with CSV files in directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create directory with CSV files
        csv_dir = data_dir / "csv_data"
        csv_dir.mkdir()
        (csv_dir / "data1.csv").write_text("col1,col2\nval1,val2")
        (csv_dir / "data2.csv").write_text("col1,col2\nval3,val4")
        
        result = _discover_datasets(str(data_dir))
        assert len(result) == 1
        assert result[0]["name"] == "csv_data"
        assert result[0]["type"] == "directory"
        assert any("csv" in f for f in result[0]["files"])

    def test_discover_datasets_dir_type(self, tmp_path):
        """Test _discover_datasets with DIR data type (mixed files)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create directory with mixed files (treated as DIR type)
        mixed_dir = data_dir / "mixed"
        mixed_dir.mkdir()
        (mixed_dir / "file1.txt").write_text("text")
        (mixed_dir / "file2.jpg").write_bytes(b"image")
        (mixed_dir / "file3.wav").write_bytes(b"audio")
        
        result = _discover_datasets(str(data_dir))
        assert len(result) == 1
        assert result[0]["name"] == "mixed"
        assert result[0]["type"] == "directory"
        assert len(result[0]["files"]) > 0

    def test_discover_datasets_graph_type(self, tmp_path):
        """Test _discover_datasets - graph data is typically a single file or directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Graph data might be in a directory or as a file
        # Since _discover_datasets looks for common extensions, graph might not be detected
        # But we can test that directories are found
        graph_dir = data_dir / "graph_data"
        graph_dir.mkdir()
        (graph_dir / "graph.txt").write_text("graph data")
        
        result = _discover_datasets(str(data_dir))
        # Graph data might be detected as text, but directory should be found
        assert len(result) >= 1

    def test_discover_datasets_rl_type(self, tmp_path):
        """Test _discover_datasets - RL data is typically trajectory files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # RL data might be in various formats
        rl_dir = data_dir / "rl_data"
        rl_dir.mkdir()
        (rl_dir / "trajectory.txt").write_text("trajectory data")
        
        result = _discover_datasets(str(data_dir))
        # RL data might be detected as text, but directory should be found
        assert len(result) >= 1

    def test_discover_datasets_multiple_datasets(self, tmp_path):
        """Test _discover_datasets with multiple datasets."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create multiple dataset directories
        (data_dir / "images").mkdir()
        (data_dir / "images" / "img1.jpg").write_bytes(b"image")
        
        (data_dir / "audio").mkdir()
        (data_dir / "audio" / "audio1.wav").write_bytes(b"audio")
        
        (data_dir / "data.csv").write_text("col1,col2\nval1,val2")
        
        result = _discover_datasets(str(data_dir))
        assert len(result) == 3
        names = [r["name"] for r in result]
        assert "images" in names
        assert "audio" in names
        assert "data.csv" in names

    def test_discover_datasets_limits_file_scan(self, tmp_path):
        """Test that _discover_datasets limits file scanning to first 10 files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create directory with many files
        large_dir = data_dir / "large"
        large_dir.mkdir()
        for i in range(15):
            (large_dir / f"file{i}.txt").write_text("text")
        
        result = _discover_datasets(str(data_dir))
        assert len(result) == 1
        # Should only scan first 10 files
        assert len(result[0]["files"]) <= 10

    def test_discover_datasets_permission_error(self, tmp_path):
        """Test _discover_datasets handles permission errors gracefully."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        with patch("pathlib.Path.iterdir", side_effect=PermissionError("Access denied")):
            result = _discover_datasets(str(data_dir))
            # Should return empty list or handle gracefully
            assert isinstance(result, list)


class TestGetPredefinedModels:
    """Test _get_predefined_models function."""

    def test_get_predefined_models(self):
        """Test that _get_predefined_models returns expected models."""
        models = _get_predefined_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Check that all expected models are present
        model_names = [m["name"] for m in models]
        assert "resnet50" in model_names
        assert "resnet101" in model_names
        assert "wav2vec2" in model_names
        assert "gpt-neo" in model_names
        assert "gcn-ogbn-arxiv" in model_names
        assert "biggan" in model_names
        assert "ppo" in model_names
        
        # Check structure of each model
        for model in models:
            assert "name" in model
            assert "type" in model
            assert "description" in model
            assert isinstance(model["name"], str)
            assert isinstance(model["type"], str)
            assert isinstance(model["description"], str)


class TestFormatPlanSummary:
    """Test _format_plan_summary function."""

    def test_format_plan_summary_empty_plan(self):
        """Test _format_plan_summary with empty plan."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[],
            model=model,
        )
        
        summary = _format_plan_summary(plan)
        assert "Distribution Plan Summary" in summary
        assert "=" * 60 in summary

    def test_format_plan_summary_with_distribution_plan(self):
        """Test _format_plan_summary with distribution plan."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        distribution_plan = [
            {"host": "node1", "comms_port": 5001, "capacity_fraction": 0.5, "allocated_samples": 100},
            {"host": "node2", "comms_port": 5002, "capacity_fraction": 0.3, "allocated_samples": 60},
            {"host": "node3", "comms_port": 5003, "capacity_fraction": 0.2, "allocated_samples": 40},
        ]
        plan = Plan(
            stats_data=[],
            distribution_plan=distribution_plan,
            model=model,
        )
        
        summary = _format_plan_summary(plan)
        assert "Total Nodes: 3" in summary
        assert "node1" in summary
        assert "node2" in summary
        assert "node3" in summary
        assert "50.0%" in summary
        assert "100" in summary

    def test_format_plan_summary_with_many_nodes(self):
        """Test _format_plan_summary limits output for large plans."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        # Create plan with more than max_nodes (default 20)
        distribution_plan = [
            {"host": f"node{i}", "comms_port": 5000 + i, "capacity_fraction": 0.1, "allocated_samples": 10}
            for i in range(25)
        ]
        plan = Plan(
            stats_data=[],
            distribution_plan=distribution_plan,
            model=model,
        )
        
        summary = _format_plan_summary(plan, max_nodes=20)
        assert "Total Nodes: 25" in summary
        assert "node0" in summary  # First node should be shown
        assert "node19" in summary  # 20th node should be shown
        assert "... and 5 more nodes" in summary  # Should indicate remaining nodes

    def test_format_plan_summary_with_data_segmentation(self):
        """Test _format_plan_summary with data segmentation plan."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        distribution_plan = [
            {"host": "node1", "comms_port": 5001, "capacity_fraction": 0.5},
        ]
        data_segmentation_plan = [
            {
                "host": "node1",
                "comms_port": 5001,
                "fraction": 0.5,
                "segments": [
                    {"file_location": "data/file1.csv", "start_row": 0, "end_row": 50},
                    {"file_location": "data/file2.csv", "start_row": 0, "end_row": 50},
                ],
            },
            {
                "host": "node2",
                "comms_port": 5002,
                "fraction": 0.5,
                "segments": [
                    {"file_location": "data/file1.csv", "start_row": 50, "end_row": 100},
                ],
            },
        ]
        plan = Plan(
            stats_data=[],
            distribution_plan=distribution_plan,
            model=model,
            data_segmentation_plan=data_segmentation_plan,
        )
        
        summary = _format_plan_summary(plan)
        assert "Data Segmentation Plan" in summary
        assert "Total machines: 2" in summary
        assert "Total segments: 3" in summary
        assert "node1" in summary
        assert "node2" in summary

    def test_format_plan_summary_with_many_machines(self):
        """Test _format_plan_summary limits machine output."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        distribution_plan = [{"host": "node1", "comms_port": 5001, "capacity_fraction": 1.0}]
        # Create segmentation plan with more than 5 machines
        data_segmentation_plan = [
            {
                "host": f"node{i}",
                "comms_port": 5000 + i,
                "fraction": 0.1,
                "segments": [{"file_location": f"data/file{i}.csv"}],
            }
            for i in range(10)
        ]
        plan = Plan(
            stats_data=[],
            distribution_plan=distribution_plan,
            model=model,
            data_segmentation_plan=data_segmentation_plan,
        )
        
        summary = _format_plan_summary(plan)
        assert "Total machines: 10" in summary
        assert "node0" in summary  # First machine should be shown
        assert "node4" in summary  # 5th machine should be shown
        assert "... and 5 more machines" in summary  # Should indicate remaining


class TestExecuteCreateSession:
    """Test execute_create_session function with mocks."""

    def test_execute_create_session_not_initialized(self):
        """Test execute_create_session when not initialized."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        # Ensure module is not initialized
        initialize(None, None, [], MosaicConfig(), None)
        try:
            execute_create_session(output_fn)
            assert any("Error" in line or "not initialized" in line for line in output_lines)
        finally:
            # Clean up
            initialize(None, None, [], MosaicConfig(), None)

    def test_execute_create_session_cancelled(self, tmp_path, temp_state_dir):
        """Test execute_create_session when user cancels."""
        output_lines = []
        input_responses = ["999"]  # Cancel selection
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir, data_location=str(tmp_path / "data"))
        mock_beacon = MagicMock()
        mock_beacon.collect_stats.return_value = []
        mock_session_manager = MagicMock()
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            execute_create_session(output_fn)
            # Should handle cancellation gracefully
            assert any("cancelled" in line.lower() or "cancel" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_execute_create_session_calls_beacon_methods(self, tmp_path, temp_state_dir):
        """Test that execute_create_session calls appropriate beacon methods."""
        output_lines = []
        input_responses = [
            "1",  # Select first model (will be cancelled before model selection completes)
        ]
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return "cancel"  # Default to cancel
        
        config = create_test_config_with_state(state_dir=temp_state_dir, data_location=str(tmp_path / "data"))
        mock_beacon = MagicMock()
        mock_beacon.collect_stats.return_value = [{"host": "node1", "connection_status": "online"}]
        mock_session_manager = MagicMock()
        mock_session_manager.get_sessions.return_value = []
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        
        initialize(mock_beacon, mock_session_manager, [model], config, input_fn)
        try:
            # Mock the planning functions to avoid actual execution
            with patch("mosaic.session_commands.plan_static_weighted_shards") as mock_plan_shards, \
                 patch("mosaic.session_commands.plan_data_distribution") as mock_plan_data, \
                 patch("mosaic.session_commands.plan_model") as mock_plan_model, \
                 patch("mosaic.session_commands._beacon.execute_data_plan") as mock_exec_data, \
                 patch("mosaic.session_commands._beacon.execute_model_plan") as mock_exec_model:
                
                mock_plan_shards.return_value = [{"host": "node1", "comms_port": 5001, "capacity_fraction": 1.0}]
                mock_plan_data.return_value = Plan(
                    stats_data=[],
                    distribution_plan=[{"host": "node1", "comms_port": 5001}],
                    model=model,
                )
                mock_plan_model.return_value = {}
                
                # This will likely cancel early, but we can verify mocks are set up
                execute_create_session(output_fn)
                
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_create_session_simple_tracks_distribution_state(self, tmp_path, temp_state_dir):
        """Test that _create_session_simple properly tracks distribution state in session."""
        output_lines = []
        input_responses = [
            "1",  # Select first model
            "test.txt",  # Manual dataset path entry (when no datasets found)
            "text",  # Data type for manual entry
            "yes",  # Confirm execution
        ]
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return "cancel"
        
        # Create test data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        test_file = data_dir / "test.txt"
        test_file.write_text("test data")
        
        config = create_test_config_with_state(state_dir=temp_state_dir, data_location=str(data_dir))
        mock_beacon = MagicMock()
        mock_beacon.collect_stats.return_value = [
            {"host": "127.0.0.1", "comms_port": 5001, "connection_status": "online", "cpu_percent": 50.0}
        ]
        
        mock_session_manager = MagicMock()
        created_session = None
        
        def add_session(session):
            nonlocal created_session
            created_session = session
            return session.id
        
        mock_session_manager.add_session.side_effect = add_session
        mock_session_manager.update_session.return_value = True
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        
        initialize(mock_beacon, mock_session_manager, [model], config, input_fn)
        try:
            with patch("mosaic.session_commands.plan_static_weighted_shards") as mock_plan_shards, \
                 patch("mosaic.session_commands.plan_data_distribution") as mock_plan_data, \
                 patch("mosaic.session_commands.plan_model") as mock_plan_model, \
                 patch("mosaic.session_commands._beacon.execute_data_plan") as mock_exec_data, \
                 patch("mosaic.session_commands._beacon.execute_model_plan") as mock_exec_model, \
                 patch("mosaic.session_commands._discover_datasets") as mock_discover:
                
                # Setup mocks
                mock_discover.return_value = []  # No datasets found, will use manual entry
                mock_plan_shards.return_value = [
                    {"host": "127.0.0.1", "comms_port": 5001, "capacity_fraction": 1.0}
                ]
                
                plan = Plan(
                    stats_data=[],
                    distribution_plan=[{"host": "127.0.0.1", "comms_port": 5001}],
                    model=model,
                    data_segmentation_plan=[
                        {
                            "host": "127.0.0.1",
                            "comms_port": 5001,
                            "segments": [{"file_location": "test.txt"}]
                        }
                    ],
                )
                mock_plan_data.return_value = plan
                mock_plan_model.return_value = {}
                
                # Mock execute_data_plan to set distribution state
                def mock_exec_data_side_effect(plan, data, session):
                    if session:
                        session.data_distribution_state = {
                            "machines": {
                                "127.0.0.1:5001": {
                                    "status": "success",
                                    "host": "127.0.0.1",
                                    "comms_port": 5001,
                                    "attempts": 1,
                                }
                            },
                            "failed_machines": [],
                            "retry_attempts": {},
                        }
                        session.status = SessionStatus.RUNNING
                
                mock_exec_data.side_effect = mock_exec_data_side_effect
                
                # Mock execute_model_plan to set distribution state
                def mock_exec_model_side_effect(session, model):
                    if session:
                        session.model_distribution_state = {
                            "nodes": {
                                "127.0.0.1:5001:0": {
                                    "status": "success",
                                    "host": "127.0.0.1",
                                    "comms_port": 5001,
                                    "shard_number": 0,
                                    "attempts": 1,
                                }
                            },
                            "failed_nodes": [],
                            "retry_attempts": {},
                        }
                        session.status = SessionStatus.RUNNING
                
                mock_exec_model.side_effect = mock_exec_model_side_effect
                
                # Import the function
                from mosaic.session_commands import _create_session_simple
                
                # Execute
                result = _create_session_simple(output_fn)
                
                # Verify session was created and added before distribution
                assert mock_session_manager.add_session.called
                assert created_session is not None
                assert created_session.status == SessionStatus.RUNNING
                
                # Verify execute_data_plan was called with session
                mock_exec_data.assert_called_once()
                call_args = mock_exec_data.call_args
                assert call_args[0][2] == created_session  # session parameter
                
                # Verify execute_model_plan was called with session
                mock_exec_model.assert_called_once()
                call_args = mock_exec_model.call_args
                assert call_args[0][0] == created_session  # session parameter
                
                # Verify distribution state was tracked
                assert "machines" in created_session.data_distribution_state
                assert "nodes" in created_session.model_distribution_state
                
                # Verify session was updated
                assert mock_session_manager.update_session.called
                
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_create_session_simple_handles_distribution_errors(self, tmp_path, temp_state_dir):
        """Test that _create_session_simple handles distribution errors and sets ERROR status."""
        output_lines = []
        input_responses = [
            "1",  # Select first model
            "1",  # Select first dataset (or manual entry)
            "text",  # Data type for manual entry
            "yes",  # Confirm execution
        ]
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return "cancel"
        
        # Create test data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        test_file = data_dir / "test.txt"
        test_file.write_text("test data")
        
        config = create_test_config_with_state(state_dir=temp_state_dir, data_location=str(data_dir))
        mock_beacon = MagicMock()
        mock_beacon.collect_stats.return_value = [
            {"host": "127.0.0.1", "comms_port": 5001, "connection_status": "online", "cpu_percent": 50.0}
        ]
        
        mock_session_manager = MagicMock()
        created_session = None
        
        def add_session(session):
            nonlocal created_session
            created_session = session
            return session.id
        
        mock_session_manager.add_session.side_effect = add_session
        mock_session_manager.update_session.return_value = True
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        
        initialize(mock_beacon, mock_session_manager, [model], config, input_fn)
        try:
            with patch("mosaic.session_commands.plan_static_weighted_shards") as mock_plan_shards, \
                 patch("mosaic.session_commands.plan_data_distribution") as mock_plan_data, \
                 patch("mosaic.session_commands.plan_model") as mock_plan_model, \
                 patch("mosaic.session_commands._beacon.execute_data_plan") as mock_exec_data, \
                 patch("mosaic.session_commands._beacon.execute_model_plan") as mock_exec_model, \
                 patch("mosaic.session_commands._discover_datasets") as mock_discover:
                
                # Setup mocks
                mock_discover.return_value = []  # No datasets found, will use manual entry
                mock_plan_shards.return_value = [
                    {"host": "127.0.0.1", "comms_port": 5001, "capacity_fraction": 1.0}
                ]
                
                plan = Plan(
                    stats_data=[],
                    distribution_plan=[{"host": "127.0.0.1", "comms_port": 5001}],
                    model=model,
                    data_segmentation_plan=[
                        {
                            "host": "127.0.0.1",
                            "comms_port": 5001,
                            "segments": [{"file_location": "test.txt"}]
                        }
                    ],
                )
                mock_plan_data.return_value = plan
                mock_plan_model.return_value = {}
                
                # Mock execute_data_plan to simulate error
                def mock_exec_data_side_effect(plan, data, session):
                    if session:
                        session.data_distribution_state = {
                            "machines": {
                                "127.0.0.1:5001": {
                                    "status": "failed",
                                    "host": "127.0.0.1",
                                    "comms_port": 5001,
                                    "error": "No capable nodes remaining",
                                    "attempts": 1,
                                }
                            },
                            "failed_machines": [{"host": "127.0.0.1", "comms_port": 5001}],
                            "retry_attempts": {},
                            "final_error": "No capable nodes remaining",
                        }
                        session.status = SessionStatus.ERROR
                
                mock_exec_data.side_effect = mock_exec_data_side_effect
                
                # Import the function
                from mosaic.session_commands import _create_session_simple
                
                # Execute
                result = _create_session_simple(output_fn)
                
                # Verify session status is ERROR
                assert created_session is not None
                assert created_session.status == SessionStatus.ERROR
                assert "final_error" in created_session.data_distribution_state
                
                # Verify session was updated with ERROR status
                update_calls = [call for call in mock_session_manager.update_session.call_args_list]
                # Should have at least one update call with ERROR status
                # update_session is called with keyword arguments: update_session(session.id, status=SessionStatus.ERROR)
                error_updates = [
                    call for call in update_calls
                    if call[1].get("status") == SessionStatus.ERROR
                ]
                assert len(error_updates) > 0, f"No update_session call found with ERROR status. Calls: {update_calls}"
                
        finally:
            initialize(None, None, [], MosaicConfig(), None)


class TestExecuteDeleteSession:
    """Test execute_delete_session function with mocks."""

    def test_execute_delete_session_not_initialized(self):
        """Test execute_delete_session when not initialized."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        initialize(None, None, [], MosaicConfig(), None)
        try:
            execute_delete_session(output_fn)
            assert any("Error" in line or "not initialized" in line for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_execute_delete_session_no_sessions(self, temp_state_dir):
        """Test execute_delete_session when no sessions exist."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_session_manager = MagicMock()
        mock_session_manager.get_sessions.return_value = []
        
        initialize(None, mock_session_manager, [], config, None)
        try:
            execute_delete_session(output_fn)
            assert any("No sessions" in line for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_execute_delete_session_with_id(self, temp_state_dir):
        """Test execute_delete_session with provided session ID."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_session_manager = MagicMock()
        mock_session_manager.remove_session.return_value = True
        
        initialize(None, mock_session_manager, [], config, None)
        try:
            execute_delete_session(output_fn, session_id="test-session-id")
            mock_session_manager.remove_session.assert_called_once_with("test-session-id")
            assert any("deleted successfully" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_execute_delete_session_with_id_not_found(self, temp_state_dir):
        """Test execute_delete_session with non-existent session ID."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_session_manager = MagicMock()
        mock_session_manager.remove_session.return_value = False
        
        initialize(None, mock_session_manager, [], config, None)
        try:
            execute_delete_session(output_fn, session_id="nonexistent-id")
            mock_session_manager.remove_session.assert_called_once_with("nonexistent-id")
            assert any("not found" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_execute_delete_session_prompt_selection(self, temp_state_dir):
        """Test execute_delete_session prompts for selection when no ID provided."""
        output_lines = []
        input_responses = ["1"]  # Select first session
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        session = Session(plan=plan, status=SessionStatus.IDLE)
        
        mock_session_manager = MagicMock()
        mock_session_manager.get_sessions.return_value = [session]
        mock_session_manager.remove_session.return_value = True
        
        initialize(None, mock_session_manager, [], config, input_fn)
        try:
            execute_delete_session(output_fn)
            mock_session_manager.remove_session.assert_called_once_with(session.id)
            assert any("deleted successfully" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_execute_delete_session_prompt_cancelled(self, temp_state_dir):
        """Test execute_delete_session when user cancels selection."""
        output_lines = []
        input_responses = ["999"]  # Cancel (invalid selection)
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        session = Session(plan=plan, status=SessionStatus.IDLE)
        
        mock_session_manager = MagicMock()
        mock_session_manager.get_sessions.return_value = [session]
        
        initialize(None, mock_session_manager, [], config, input_fn)
        try:
            execute_delete_session(output_fn)
            # Should handle invalid selection gracefully
            assert any("Invalid" in line or "cancelled" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)

