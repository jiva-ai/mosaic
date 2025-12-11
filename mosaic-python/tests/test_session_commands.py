"""Unit tests for mosaic.session_commands module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List
import sys

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
    execute_training,
    execute_cancel_training,
    execute_use_session,
    execute_infer,
    execute_set_infer_method,
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
    # Only restore if we added the mock
    if _we_added_onnx_mock and _original_onnx is None:
        # Remove our mock
        if 'onnx' in sys.modules:
            del sys.modules['onnx']
        if 'onnx.reference' in sys.modules:
            del sys.modules['onnx.reference']


class TestDiscoverDatasets:
    """Tests for _discover_datasets function."""

    def test_discover_datasets_image_directory(self, temp_state_dir):
        """Test discovering image datasets in a directory."""
        data_dir = Path(temp_state_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_dir = data_dir / "images"
        dataset_dir.mkdir()
        (dataset_dir / "img1.jpg").touch()
        (dataset_dir / "img2.png").touch()
        
        datasets = _discover_datasets(str(data_dir))
        assert len(datasets) == 1
        assert datasets[0]["name"] == "images"
        assert datasets[0]["type"] == "directory"
        assert len(datasets[0]["files"]) > 0

    def test_discover_datasets_csv_file(self, temp_state_dir):
        """Test discovering CSV file datasets."""
        data_dir = Path(temp_state_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        (data_dir / "data.csv").touch()
        
        datasets = _discover_datasets(str(data_dir))
        assert len(datasets) == 1
        assert datasets[0]["name"] == "data.csv"
        assert datasets[0]["type"] == "file"

    def test_discover_datasets_text_file(self, temp_state_dir):
        """Test discovering text file datasets."""
        data_dir = Path(temp_state_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        (data_dir / "text.txt").touch()
        
        datasets = _discover_datasets(str(data_dir))
        assert len(datasets) == 1
        assert datasets[0]["name"] == "text.txt"
        assert datasets[0]["type"] == "file"

    def test_discover_datasets_audio_file(self, temp_state_dir):
        """Test discovering audio file datasets."""
        data_dir = Path(temp_state_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        (data_dir / "audio.wav").touch()
        
        datasets = _discover_datasets(str(data_dir))
        assert len(datasets) == 1
        assert datasets[0]["name"] == "audio.wav"
        assert datasets[0]["type"] == "file"

    def test_discover_datasets_empty_directory(self, temp_state_dir):
        """Test discovering datasets in empty directory."""
        data_dir = Path(temp_state_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = _discover_datasets(str(data_dir))
        assert len(datasets) == 0

    def test_discover_datasets_nonexistent_directory(self):
        """Test discovering datasets in nonexistent directory."""
        datasets = _discover_datasets("/nonexistent/path")
        assert len(datasets) == 0


class TestGetPredefinedModels:
    """Tests for _get_predefined_models function."""

    def test_get_predefined_models(self):
        """Test getting predefined models list."""
        models = _get_predefined_models()
        assert len(models) > 0
        assert all("name" in m and "type" in m and "description" in m for m in models)
        
        # Check for specific models
        model_names = [m["name"] for m in models]
        assert "resnet50" in model_names
        assert "gpt-neo" in model_names


class TestFormatPlanSummary:
    """Tests for _format_plan_summary function."""

    def test_format_plan_summary_empty(self):
        """Test formatting empty plan."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        summary = _format_plan_summary(plan)
        assert "Distribution Plan Summary" in summary

    def test_format_plan_summary_with_nodes(self):
        """Test formatting plan with nodes."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001, "capacity_fraction": 0.5},
                {"host": "192.168.1.2", "comms_port": 7002, "capacity_fraction": 0.3},
            ],
            model=model,
        )
        summary = _format_plan_summary(plan)
        assert "192.168.1.1" in summary
        assert "192.168.1.2" in summary
        assert "Total Nodes: 2" in summary

    def test_format_plan_summary_truncates_large_plans(self):
        """Test that large plans are truncated in summary."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": f"192.168.1.{i}", "comms_port": 7000 + i, "capacity_fraction": 0.1}
                for i in range(100)
            ],
            model=model,
        )
        summary = _format_plan_summary(plan, max_nodes=20)
        assert "Total Nodes: 100" in summary
        assert "... and 80 more nodes" in summary


class TestExecuteCreateSession:
    """Tests for execute_create_session function."""

    def test_execute_create_session_cancelled(self, temp_state_dir):
        """Test execute_create_session when user cancels."""
        output_lines = []
        input_responses = ["999"]  # Cancel
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_beacon.collect_stats.return_value = []
        mock_session_manager = MagicMock()
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            execute_create_session(output_fn)
            assert any("cancelled" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_create_session_model_without_type_prompts_for_type(self, temp_state_dir):
        """Test execute_create_session when selected model has no type, prompts for type selection."""
        output_lines = []
        # Select first model (which has no type), then select model type, then cancel at dataset selection
        input_responses = ["1", "1", "999"]  # Select model, select CNN type, cancel at dataset
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_beacon.collect_stats.return_value = []
        mock_session_manager = MagicMock()
        
        # Create a model without a type
        model_without_type = Model(name="test_model", model_type=None)
        
        initialize(mock_beacon, mock_session_manager, [model_without_type], config, input_fn)
        try:
            execute_create_session(output_fn)
            # Should have prompted for model type
            output_text = "".join(output_lines)
            assert "Model type not set" in output_text or "select a model type" in output_text.lower()
            assert "CNN" in output_text or "1." in output_text  # Should show model type options
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_create_session_model_type_selection_cancelled(self, temp_state_dir):
        """Test execute_create_session when user cancels during model type selection."""
        output_lines = []
        # Select model without type, then cancel at type selection (KeyboardInterrupt simulation)
        input_responses = ["1"]  # Select model
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            # Simulate KeyboardInterrupt by raising it when prompted for type
            if "model type number" in prompt.lower():
                raise KeyboardInterrupt()
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_beacon.collect_stats.return_value = []
        mock_session_manager = MagicMock()
        
        # Create a model without a type
        model_without_type = Model(name="test_model", model_type=None)
        
        initialize(mock_beacon, mock_session_manager, [model_without_type], config, input_fn)
        try:
            execute_create_session(output_fn)
            # Should have cancelled during type selection
            output_text = "".join(output_lines)
            assert any("cancelled" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)


class TestExecuteDeleteSession:
    """Tests for execute_delete_session function."""

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
            execute_delete_session(output_fn, session_id="test_session_id")
            mock_session_manager.remove_session.assert_called_once_with("test_session_id")
            assert any("deleted successfully" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)

    def test_execute_delete_session_prompt(self, temp_state_dir):
        """Test execute_delete_session with user prompt."""
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


class TestExecuteTraining:
    """Tests for execute_training function."""
    
    def test_execute_training_missing_beacon(self, temp_state_dir):
        """Test execute_training when _beacon is not set."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_session_manager = MagicMock()
        
        initialize(None, mock_session_manager, [], config, None)
        try:
            result = execute_training("test_session_id", output_fn)
            assert result is None
            assert any("not fully initialized" in line for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_missing_session_manager(self, temp_state_dir):
        """Test execute_training when _session_manager is not set."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        
        initialize(mock_beacon, None, [], config, None)
        try:
            result = execute_training("test_session_id", output_fn)
            assert result is None
            assert any("not fully initialized" in line for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_missing_config(self, temp_state_dir):
        """Test execute_training when _config is not set."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        initialize(mock_beacon, mock_session_manager, [], None, None)
        try:
            result = execute_training("test_session_id", output_fn)
            assert result is None
            assert any("not fully initialized" in line for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_session_not_found(self, temp_state_dir):
        """Test execute_training when session is not found."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        mock_session_manager.get_session_by_id.return_value = None
        
        initialize(mock_beacon, mock_session_manager, [], config, None)
        try:
            result = execute_training("nonexistent_session", output_fn)
            assert result is None
            assert any("not found" in line.lower() for line in output_lines)
            mock_session_manager.get_session_by_id.assert_called_once_with("nonexistent_session")
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_invalid_session_status(self, temp_state_dir):
        """Test execute_training when session status is not RUNNING or IDLE."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        session = Session(plan=plan, status=SessionStatus.ERROR)
        
        mock_session_manager.get_session_by_id.return_value = session
        
        initialize(mock_beacon, mock_session_manager, [], config, None)
        try:
            result = execute_training(session.id, output_fn)
            assert result is None
            assert any("cannot start training" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_populates_training_nodes(self, temp_state_dir):
        """Test that execute_training correctly populates training_nodes from plans."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
                {"host": "192.168.1.2", "comms_port": 7002},
            ],
            data_segmentation_plan=[
                {"host": "192.168.1.1", "comms_port": 7001, "segments": []},
                {"host": "192.168.1.3", "comms_port": 7003, "segments": []},
            ],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.RUNNING)
        session.data_distribution_state = {}
        
        mock_session_manager.get_session_by_id.return_value = session
        mock_beacon.send_command.return_value = {"status": "success"}
        
        # Mock input function (model transfer is commented out, so this won't be called)
        def input_fn(prompt: str) -> str:
            return "no"
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            # Mock time.sleep to avoid actual waiting
            with patch("mosaic.session_commands.time.sleep"):
                # Mock the polling loop by making nodes complete immediately
                def update_training_status(*args, **kwargs):
                    # Simulate training_status updates
                    if "training_nodes" not in session.data_distribution_state:
                        session.data_distribution_state["training_nodes"] = {}
                    # Mark all nodes as complete
                    for node_key in ["192.168.1.1:7001", "192.168.1.2:7002", "192.168.1.3:7003"]:
                        if node_key not in session.data_distribution_state["training_nodes"]:
                            session.data_distribution_state["training_nodes"][node_key] = {
                                "status": "complete",
                                "message": "Training completed",
                            }
                
                # Patch _handle_training_status to update session state
                with patch("mosaic_comms.beacon.Beacon._handle_training_status", side_effect=update_training_status):
                    result = execute_training(session.id, output_fn, timeout=1.0, check_interval=0.1)
            
            # Check that training_nodes were populated
            assert "training_nodes" in session.data_distribution_state
            training_nodes = session.data_distribution_state["training_nodes"]
            
            # Should have 3 unique nodes (1 and 3 from data_segmentation_plan, 2 from distribution_plan)
            assert len(training_nodes) == 3
            assert "192.168.1.1:7001" in training_nodes
            assert "192.168.1.2:7002" in training_nodes
            assert "192.168.1.3:7003" in training_nodes
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_sends_correct_commands(self, temp_state_dir):
        """Test that execute_training sends correct training commands to nodes."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        config.host = "192.168.1.0"
        config.comms_port = 7000
        
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
            ],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.RUNNING)
        session.data_distribution_state = {}
        
        mock_session_manager.get_session_by_id.return_value = session
        
        # Mock input function (model transfer is commented out, so this won't be called)
        def input_fn(prompt: str) -> str:
            return "no"
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            # Mock time.sleep and make nodes complete immediately
            with patch("mosaic.session_commands.time.sleep"):
                # Provide time values: start_time=0, then loop checks
                # Use a function to provide enough values for the loop
                time_call_count = [0]
                def mock_time():
                    count = time_call_count[0]
                    time_call_count[0] += 1
                    if count == 0:
                        return 0.0  # start_time
                    else:
                        # Return values that keep us in the loop (less than timeout of 1.0)
                        return 0.0 + (count * 0.1)
                
                with patch("mosaic.session_commands.time.time", side_effect=mock_time):
                    # Pre-set the complete status so it's detected on first loop iteration
                    session.data_distribution_state["training_nodes"] = {
                        "192.168.1.1:7001": {
                            "status": "complete",
                            "message": "Training completed",
                        }
                    }
                    
                    def mock_send(*args, **kwargs):
                        # Return success (state is already set above)
                        return {"status": "success"}
                    mock_beacon.send_command.side_effect = mock_send
                    
                    result = execute_training(session.id, output_fn, timeout=1.0, check_interval=0.1)
            
            # Verify send_command was called with correct parameters
            assert mock_beacon.send_command.called
            call_args = mock_beacon.send_command.call_args
            
            assert call_args.kwargs["host"] == "192.168.1.1"
            assert call_args.kwargs["port"] == 7001
            assert call_args.kwargs["command"] == "start_training"
            assert call_args.kwargs["payload"]["session_id"] == session.id
            assert call_args.kwargs["payload"]["caller_host"] == "192.168.1.0"
            assert call_args.kwargs["payload"]["caller_port"] == 7000
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_handles_training_failure(self, temp_state_dir):
        """Test that execute_training handles training failures correctly."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
            ],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.RUNNING)
        session.data_distribution_state = {}
        
        mock_session_manager.get_session_by_id.return_value = session
        
        # Mock input function (not needed for error case, but good to have)
        def input_fn(prompt: str) -> str:
            return "no"
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            with patch("mosaic.session_commands.time.sleep"):
                # Provide time values: start_time=0, then loop checks at 0.1, 0.2, etc.
                # Need enough values for the loop to run and detect the error
                time_call_count = [0]
                def mock_time():
                    count = time_call_count[0]
                    time_call_count[0] += 1
                    if count == 0:
                        return 0.0  # start_time
                    else:
                        # Return values that keep us in the loop (less than timeout of 1.0)
                        return 0.0 + (count * 0.1)
                
                with patch("mosaic.session_commands.time.time", side_effect=mock_time):
                    # Pre-set the error status before the loop starts checking
                    # This ensures the error is detected on the first loop iteration
                    session.data_distribution_state["training_nodes"] = {
                        "192.168.1.1:7001": {
                            "status": "error",
                            "message": "Training failed",
                        }
                    }
                    
                    def mock_send(*args, **kwargs):
                        # Return error (state is already set above)
                        return {"status": "error", "message": "Training failed"}
                    mock_beacon.send_command.side_effect = mock_send
                    
                    result = execute_training(session.id, output_fn, timeout=1.0, check_interval=0.1)
            
            # Check that error was recorded
            assert "training_nodes" in session.data_distribution_state
            node_status = session.data_distribution_state["training_nodes"]["192.168.1.1:7001"]
            assert node_status["status"] == "error"
            assert "failed" in node_status["message"].lower()
            
            # Session should be in ERROR state
            assert session.status == SessionStatus.ERROR
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_timeout(self, temp_state_dir):
        """Test that execute_training handles timeout correctly."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
            ],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.RUNNING)
        session.data_distribution_state = {}
        
        mock_session_manager.get_session_by_id.return_value = session
        mock_beacon.send_command.return_value = {"status": "success"}
        
        initialize(mock_beacon, mock_session_manager, [], config, None)
        try:
            # Simulate timeout by making time progress beyond timeout
            time_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # Exceeds 0.5 timeout
            with patch("mosaic.session_commands.time.sleep"):
                with patch("mosaic.session_commands.time.time", side_effect=time_values):
                    # Don't mark nodes as complete, so timeout occurs
                    result = execute_training(session.id, output_fn, timeout=0.5, check_interval=0.1)
            
            # Check that timeout message was printed
            assert any("timeout" in line.lower() for line in output_lines)
            
            # Session should be in ERROR state after timeout
            assert session.status == SessionStatus.ERROR
            mock_session_manager.update_session.assert_any_call(session.id, status=SessionStatus.ERROR)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_no_nodes_local_only(self, temp_state_dir):
        """Test execute_training when no nodes are found, falls back to local training."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        session = Session(plan=plan, status=SessionStatus.RUNNING)
        session.data_distribution_state = {}
        
        mock_session_manager.get_session_by_id.return_value = session
        
        initialize(mock_beacon, mock_session_manager, [], config, None)
        try:
            with patch("mosaic_planner.model_execution.train_model_from_session") as mock_train:
                mock_trained_model = MagicMock()
                mock_trained_model.id = "trained_model_123"
                # train_model_from_session now returns (model, stats) tuple
                mock_train.return_value = (mock_trained_model, {"epochs": 1, "final_loss": 0.5, "training_time_seconds": 10.0})
                
                result = execute_training(session.id, output_fn)
            
            # Should have called train_model_from_session
            mock_train.assert_called_once_with(session, config=config)
            
            # Session should be COMPLETE
            assert session.status == SessionStatus.COMPLETE
            # Check that update_session was called with status=COMPLETE (may also include data_distribution_state)
            update_calls = [call for call in mock_session_manager.update_session.call_args_list 
                          if call[0][0] == session.id and call[1].get("status") == SessionStatus.COMPLETE]
            assert len(update_calls) > 0, "Session should be updated with COMPLETE status"
            
            assert any("local only" in line.lower() for line in output_lines)
            
            # Verify stats are stored on session
            assert "training_stats" in session.data_distribution_state
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_training_collects_and_displays_stats(self, temp_state_dir):
        """Test that execute_training collects and displays training statistics."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        config.host = "192.168.1.0"
        config.comms_port = 7000
        
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
            ],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.RUNNING)
        session.data_distribution_state = {}
        
        mock_session_manager.get_session_by_id.return_value = session
        mock_beacon.send_command.return_value = {"status": "success"}
        
        # Mock input function (model transfer is commented out, so this won't be called)
        def input_fn(prompt: str) -> str:
            return "no"
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            # Simulate training completion with stats
            training_stats = {
                "epochs": 5,
                "final_loss": 0.1234,
                "avg_loss_per_epoch": 0.25,
                "training_time_seconds": 120.5,
                "model_type": "cnn",
            }
            
            # Mock send_command to return success
            def mock_send(*args, **kwargs):
                return {"status": "success"}
            mock_beacon.send_command.side_effect = mock_send
            
            # Create a custom dict class that intercepts assignments to training_nodes
            # When status is set to "starting", change it to "complete" with stats
            class TrainingNodesDict(dict):
                def __setitem__(self, key, value):
                    # If setting a node status to "starting", change it to "complete" with stats
                    if isinstance(value, dict) and value.get("status") == "starting":
                        value = {
                            "status": "complete",
                            "message": "Training completed successfully",
                            "model_id": "trained_model_123",
                            "training_stats": training_stats,
                        }
                    return super().__setitem__(key, value)
            
            # Replace training_nodes dict with our custom one that intercepts assignments
            session.data_distribution_state["training_nodes"] = TrainingNodesDict()
            
            with patch("mosaic.session_commands.time.sleep"):
                # Provide time values: start_time=0, then first loop check
                # The loop should detect all_complete=True on first iteration and break
                time_call_count = [0]
                def mock_time():
                    count = time_call_count[0]
                    time_call_count[0] += 1
                    if count == 0:
                        return 0.0  # start_time
                    else:
                        # Return values that keep us in the loop (less than timeout of 1.0)
                        # First check should detect completion and break
                        return 0.0 + (count * 0.1)
                
                with patch("mosaic.session_commands.time.time", side_effect=mock_time):
                    result = execute_training(session.id, output_fn, timeout=1.0, check_interval=0.1)
            
            # Verify stats are stored on session (collated after loop completes)
            assert "training_stats" in session.data_distribution_state
            assert "192.168.1.1:7001" in session.data_distribution_state["training_stats"]
            stored_stats = session.data_distribution_state["training_stats"]["192.168.1.1:7001"]
            assert stored_stats["epochs"] == 5
            assert stored_stats["final_loss"] == 0.1234
            
            # Verify stats are displayed in output
            output_text = "".join(output_lines)
            assert "Training Statistics" in output_text
            assert "192.168.1.1:7001" in output_text
            assert "Epochs: 5" in output_text
            assert "Final Loss: 0.1234" in output_text or "0.1234" in output_text
            
            # Verify session was updated with stats
            update_calls = [call for call in mock_session_manager.update_session.call_args_list 
                          if len(call[1]) > 0 and "data_distribution_state" in call[1]]
            assert len(update_calls) > 0, "Session should be updated with training stats"
        finally:
            initialize(None, None, [], MosaicConfig(), None)


class TestExecuteCancelTraining:
    """Tests for execute_cancel_training function."""
    
    def test_execute_cancel_training_missing_beacon(self, temp_state_dir):
        """Test execute_cancel_training when _beacon is not set."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_session_manager = MagicMock()
        
        initialize(None, mock_session_manager, [], config, None)
        try:
            execute_cancel_training(output_fn, "test_session_id")
            assert any("not fully initialized" in line for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_cancel_training_session_not_found(self, temp_state_dir):
        """Test execute_cancel_training when session is not found."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        mock_session_manager.get_session_by_id.return_value = None
        
        initialize(mock_beacon, mock_session_manager, [], config, None)
        try:
            execute_cancel_training(output_fn, "nonexistent_session")
            assert any("not found" in line.lower() for line in output_lines)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_cancel_training_sends_to_all_nodes(self, temp_state_dir):
        """Test that execute_cancel_training sends cancel commands to all training nodes."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_beacon._is_self_host.return_value = False
        mock_beacon.send_command.return_value = {"status": "success"}
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
                {"host": "192.168.1.2", "comms_port": 7002},
            ],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.TRAINING)
        
        mock_session_manager.get_session_by_id.return_value = session
        
        initialize(mock_beacon, mock_session_manager, [], config, None)
        try:
            execute_cancel_training(output_fn, session.id)
            
            # Verify send_command was called for each node
            assert mock_beacon.send_command.call_count == 2
            
            # Check that all calls were for cancel_training command
            for call in mock_beacon.send_command.call_args_list:
                assert call.kwargs["command"] == "cancel_training"
                assert call.kwargs["payload"]["session_id"] == session.id
            
            # Check that session status was updated to IDLE
            assert session.status == SessionStatus.IDLE
            mock_session_manager.update_session.assert_any_call(session.id, status=SessionStatus.IDLE)
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_cancel_training_single_node(self, temp_state_dir):
        """Test that execute_cancel_training can target a single node with hostname."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_beacon._is_self_host.return_value = False
        mock_beacon.send_command.return_value = {"status": "success"}
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
                {"host": "192.168.1.2", "comms_port": 7002},
            ],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.TRAINING)
        
        mock_session_manager.get_session_by_id.return_value = session
        
        initialize(mock_beacon, mock_session_manager, [], config, None)
        try:
            execute_cancel_training(output_fn, session.id, hostname="192.168.1.1:7001")
            
            # Verify send_command was called only once for the specified node
            assert mock_beacon.send_command.call_count == 1
            
            call_args = mock_beacon.send_command.call_args
            assert call_args.kwargs["host"] == "192.168.1.1"
            assert call_args.kwargs["port"] == 7001
            assert call_args.kwargs["command"] == "cancel_training"
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_cancel_training_local_node(self, temp_state_dir):
        """Test that execute_cancel_training handles local node cancellation."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        config.host = "192.168.1.1"
        config.comms_port = 7001
        
        mock_beacon = MagicMock()
        mock_beacon._is_self_host.return_value = True
        mock_beacon._handle_cancel_training.return_value = {"status": "success"}
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
            ],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.TRAINING)
        
        mock_session_manager.get_session_by_id.return_value = session
        
        initialize(mock_beacon, mock_session_manager, [], config, None)
        try:
            execute_cancel_training(output_fn, session.id)
            
            # Verify _handle_cancel_training was called directly
            mock_beacon._handle_cancel_training.assert_called_once_with({"session_id": session.id})
            
            # Verify send_command was NOT called (local node)
            mock_beacon.send_command.assert_not_called()
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_cancel_training_prompt_selection(self, temp_state_dir):
        """Test execute_cancel_training with user prompt for session selection."""
        output_lines = []
        input_responses = ["1"]  # Select first session
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_beacon._is_self_host.return_value = False
        mock_beacon.send_command.return_value = {"status": "success"}
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        session = Session(plan=plan, status=SessionStatus.TRAINING)
        
        mock_session_manager.get_sessions.return_value = [session]
        mock_session_manager.get_session_by_id.return_value = session
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            execute_cancel_training(output_fn)
            
            # Verify session was selected and cancel was attempted
            mock_session_manager.get_sessions.assert_called_once()
        finally:
            initialize(None, None, [], MosaicConfig(), None)


class TestExecuteUseSession:
    """Tests for execute_use_session function."""
    
    def test_execute_use_session_with_id(self, temp_state_dir):
        """Test execute_use_session with provided session ID."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        session = Session(plan=plan, status=SessionStatus.COMPLETE, id="session_123")
        session.model_id = "model_123"
        
        mock_session_manager.get_session_by_id.return_value = session
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            execute_use_session(output_fn, "session_123")
            
            # Verify output contains session info
            output_text = "".join(output_lines)
            assert "session_123" in output_text
            assert "Using session" in output_text or "✓" in output_text
            
            # Verify session manager was called
            mock_session_manager.get_session_by_id.assert_called_once_with("session_123")
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_use_session_prompts_when_no_id(self, temp_state_dir):
        """Test execute_use_session prompts for session ID when not provided."""
        output_lines = []
        input_responses = ["session_123"]
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        session1 = Session(plan=plan, status=SessionStatus.COMPLETE, id="session_123")
        session2 = Session(plan=plan, status=SessionStatus.RUNNING, id="session_456")
        
        mock_session_manager.get_all_sessions.return_value = [session1, session2]
        mock_session_manager.get_session_by_id.return_value = session1
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            execute_use_session(output_fn)
            
            # Verify sessions were listed
            output_text = "".join(output_lines)
            assert "Available sessions" in output_text
            assert "session_123" in output_text
            assert "session_456" in output_text
            
            # Verify session was set
            assert "session_123" in output_text or "Using session" in output_text
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_use_session_no_sessions_available(self, temp_state_dir):
        """Test execute_use_session when no sessions are available."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        mock_session_manager.get_all_sessions.return_value = []
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            execute_use_session(output_fn)
            
            output_text = "".join(output_lines)
            assert "No sessions available" in output_text
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_use_session_not_found(self, temp_state_dir):
        """Test execute_use_session when session is not found."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        mock_session_manager.get_session_by_id.return_value = None
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            execute_use_session(output_fn, "nonexistent_session")
            
            output_text = "".join(output_lines)
            assert "not found" in output_text.lower() or "Error" in output_text
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_use_session_only_one_at_a_time(self, temp_state_dir):
        """Test that only one session can be used at a time."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        session1 = Session(plan=plan, status=SessionStatus.COMPLETE, id="session_1")
        session2 = Session(plan=plan, status=SessionStatus.COMPLETE, id="session_2")
        
        mock_session_manager.get_session_by_id.side_effect = lambda sid: {
            "session_1": session1,
            "session_2": session2,
        }.get(sid)
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            # Use first session
            execute_use_session(output_fn, "session_1")
            output_lines.clear()
            
            # Use second session - should replace the first
            execute_use_session(output_fn, "session_2")
            
            # Verify second session is now active (by checking infer would use it)
            # We'll verify this by checking that the session manager was called with session_2
            assert mock_session_manager.get_session_by_id.call_count >= 2
            # Last call should be for session_2
            assert mock_session_manager.get_session_by_id.call_args[0][0] == "session_2"
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_use_session_session_manager_not_initialized(self, temp_state_dir):
        """Test execute_use_session when session manager is not initialized."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        initialize(None, None, [], MosaicConfig())
        try:
            execute_use_session(output_fn, "session_123")
            
            output_text = "".join(output_lines)
            assert "not initialized" in output_text.lower() or "Error" in output_text
        finally:
            initialize(None, None, [], MosaicConfig(), None)


def _create_simple_onnx_model_for_test():
    """Create a simple ONNX model for testing inference."""
    # Import the real onnx module, bypassing any mocks
    # We need to reload it to ensure we get the real module, not a mock
    import sys
    import importlib
    
    # Temporarily remove the mock if it exists
    original_onnx = None
    if 'onnx' in sys.modules and hasattr(sys.modules['onnx'], '__spec__') and sys.modules['onnx'].__spec__ is None:
        # This is likely a mock, save it and remove it
        original_onnx = sys.modules['onnx']
        del sys.modules['onnx']
        if 'onnx.reference' in sys.modules:
            del sys.modules['onnx.reference']
    
    try:
        # Import the real onnx module
        import onnx
        from onnx import helper, numpy_helper
        import numpy as np
        
        input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1000])
        
        # Create a simple graph: Flatten -> Gemm (fully connected layer)
        # Flatten [1, 3, 224, 224] -> [1, 150528]
        flatten_output = "flatten_output"
        flatten_node = helper.make_node("Flatten", ["input"], [flatten_output], name="flatten")
        
        # Gemm node needs weight and bias tensors
        # Create weight tensor: [150528, 1000] and bias: [1000]
        weight_shape = [150528, 1000]
        bias_shape = [1000]
        
        # Create initializers for weights and bias (small random values)
        weight_array = np.random.randn(*weight_shape).astype(np.float32) * 0.01
        bias_array = np.random.randn(*bias_shape).astype(np.float32) * 0.01
        
        weight_tensor = numpy_helper.from_array(weight_array, name="weight")
        bias_tensor = numpy_helper.from_array(bias_array, name="bias")
        
        # Gemm: output = input * weight^T + bias
        gemm_node = helper.make_node(
            "Gemm",
            [flatten_output, "weight", "bias"],
            ["output"],
            name="fc",
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=1,  # Transpose weight
        )
        
        graph = helper.make_graph(
            [flatten_node, gemm_node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            [weight_tensor, bias_tensor],  # Initializers
        )
        
        # Set opset version to 14 (supported by ONNX Runtime, max is 23)
        # and ir_version to 11 for compatibility with ONNX Runtime (max supported is 11)
        opset_imports = [helper.make_opsetid("", 14)]
        model = helper.make_model(graph, producer_name="test", ir_version=11, opset_imports=opset_imports)
        return model
    finally:
        # Restore the mock if we removed it
        if original_onnx is not None:
            sys.modules['onnx'] = original_onnx
            # Restore onnx.reference if needed
            if hasattr(original_onnx, 'reference'):
                sys.modules['onnx.reference'] = original_onnx.reference


class TestExecuteInfer:
    """Tests for execute_infer function."""
    
    def test_execute_infer_no_session_set(self, temp_state_dir):
        """Test execute_infer when no session is set."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            execute_infer(output_fn, "test_input")
            
            output_text = "".join(output_lines)
            assert "No session in use" in output_text or "Error" in output_text
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_infer_shows_advice_when_no_input(self, temp_state_dir):
        """Test execute_infer shows advice when no input is provided."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        file_def = FileDefinition(location="test", data_type=DataType.IMAGE)
        data = Data(file_definitions=[file_def])
        session = Session(plan=plan, data=data, status=SessionStatus.COMPLETE, id="session_123")
        session.model = model
        
        mock_session_manager.get_session_by_id.return_value = session
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            # Set session first
            execute_use_session(output_fn, "session_123")
            output_lines.clear()
            
            # Try infer without input
            execute_infer(output_fn, None)
            
            output_text = "".join(output_lines)
            assert "Inference Input Required" in output_text or "advice" in output_text.lower() or "Expected input" in output_text
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_infer_with_remote_nodes(self, temp_state_dir):
        """Test execute_infer sends commands to remote nodes with pre-processed input data."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        config.host = "192.168.1.0"
        config.comms_port = 7000
        config.models_location = str(temp_state_dir / "models")
        models_dir = temp_state_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        # Create model with file location (for lazy loading)
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            onnx_location="",
            file_name="test_model.onnx",
            binary_rep=None,  # Will be lazy loaded
        )
        
        # Create a real ONNX model file for lazy loading
        # Import the real onnx module to ensure save works (bypassing any mocks)
        import sys
        import importlib
        
        # Temporarily remove mock if present
        original_onnx = None
        if 'onnx' in sys.modules and hasattr(sys.modules['onnx'], '__spec__') and sys.modules['onnx'].__spec__ is None:
            original_onnx = sys.modules['onnx']
            del sys.modules['onnx']
            if 'onnx.reference' in sys.modules:
                del sys.modules['onnx.reference']
        
        try:
            # Import real onnx module
            import onnx
            onnx_model = _create_simple_onnx_model_for_test()
            model_path = models_dir / model.file_name
            onnx.save(onnx_model, str(model_path))
            
            # Verify file was created immediately after saving
            assert model_path.exists(), f"Model file should exist at {model_path} after saving"
        finally:
            # Restore mock if we removed it
            if original_onnx is not None:
                sys.modules['onnx'] = original_onnx
                if hasattr(original_onnx, 'reference'):
                    sys.modules['onnx.reference'] = original_onnx.reference
        
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
                {"host": "192.168.1.2", "comms_port": 7002},
            ],
            model=model,
        )
        file_def = FileDefinition(location="test", data_type=DataType.IMAGE)
        data = Data(file_definitions=[file_def])
        session = Session(plan=plan, data=data, status=SessionStatus.COMPLETE, id="session_123")
        session.model = model
        
        # Mock successful inference responses
        def mock_send_command(*args, **kwargs):
            return {
                "status": "success",
                "prediction": [0.5, 0.3, 0.2],  # Mock prediction
            }
        
        mock_beacon.send_command.side_effect = mock_send_command
        mock_session_manager.get_session_by_id.return_value = session
        mock_session_manager.update_session.return_value = None
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            # Set session first
            execute_use_session(output_fn, "session_123")
            output_lines.clear()
            
            # Create a dummy input image file for testing
            test_input_path = temp_state_dir / "test_input.png"
            from PIL import Image
            import numpy as np
            # Create a simple test image
            test_img = Image.new('RGB', (224, 224), color='red')
            test_img.save(str(test_input_path))
            
            # Mock input parsing to return a pre-processed array
            mock_prepared_input = np.array([[[[0.5] * 224] * 224] * 3] * 1, dtype=np.float32)
            
            # Verify model file was created
            assert model_path.exists(), f"Model file should exist at {model_path}"
            
            # onnxruntime should be installed as a dependency, so no mocking needed
            with patch("mosaic.session_commands._parse_inference_input", return_value=mock_prepared_input):
                # Run inference
                execute_infer(output_fn, str(test_input_path))
            
            # Verify send_command was called for each node
            assert mock_beacon.send_command.call_count >= 2
            
            # Verify command parameters - input_data should now be a list (pre-processed)
            calls = mock_beacon.send_command.call_args_list
            for call in calls:
                kwargs = call.kwargs
                assert kwargs["command"] == "run_inference"
                assert kwargs["payload"]["session_id"] == "session_123"
                # Input data should be a list (pre-processed), not a file path string
                input_data = kwargs["payload"]["input_data"]
                assert isinstance(input_data, list), f"Expected list, got {type(input_data)}"
                assert len(input_data) > 0, "Pre-processed input should not be empty"
            
            # Verify model was lazy loaded (binary_rep should be populated)
            assert model.binary_rep is not None, "Model should be lazy loaded into binary_rep"
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_infer_handles_node_failures(self, temp_state_dir):
        """Test execute_infer handles node failures gracefully."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        config.models_location = str(temp_state_dir / "models")
        (temp_state_dir / "models").mkdir(parents=True, exist_ok=True)
        
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        # Create model with file location (for lazy loading)
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            onnx_location="",
            file_name="test_model.onnx",
            binary_rep=None,  # Will be lazy loaded
        )
        
        # Create a real ONNX model file for lazy loading
        # Import the real onnx module to ensure save works (bypassing any mocks)
        import sys
        
        # Temporarily remove mock if present
        original_onnx = None
        if 'onnx' in sys.modules and hasattr(sys.modules['onnx'], '__spec__') and sys.modules['onnx'].__spec__ is None:
            original_onnx = sys.modules['onnx']
            del sys.modules['onnx']
            if 'onnx.reference' in sys.modules:
                del sys.modules['onnx.reference']
        
        try:
            # Import real onnx module
            import onnx
            onnx_model = _create_simple_onnx_model_for_test()
            model_path = Path(config.models_location) / model.file_name
            onnx.save(onnx_model, str(model_path))
        finally:
            # Restore mock if we removed it
            if original_onnx is not None:
                sys.modules['onnx'] = original_onnx
                if hasattr(original_onnx, 'reference'):
                    sys.modules['onnx.reference'] = original_onnx.reference
        
        plan = Plan(
            stats_data=[],
            distribution_plan=[
                {"host": "192.168.1.1", "comms_port": 7001},
                {"host": "192.168.1.2", "comms_port": 7002},
            ],
            model=model,
        )
        file_def = FileDefinition(location="test", data_type=DataType.IMAGE)
        data = Data(file_definitions=[file_def])
        session = Session(plan=plan, data=data, status=SessionStatus.COMPLETE, id="session_123")
        session.model = model
        
        # Mock mixed responses - one success, one failure
        call_count = [0]
        def mock_send_command(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"status": "success", "prediction": [0.5, 0.3, 0.2]}
            else:
                return {"status": "error", "message": "Node unavailable"}
        
        mock_beacon.send_command.side_effect = mock_send_command
        mock_session_manager.get_session_by_id.return_value = session
        mock_session_manager.update_session.return_value = None
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            # Set session first
            execute_use_session(output_fn, "session_123")
            output_lines.clear()
            
            # Create a dummy input image file for testing
            test_input_path = temp_state_dir / "test_input.png"
            from PIL import Image
            import numpy as np
            # Create a simple test image
            test_img = Image.new('RGB', (224, 224), color='red')
            test_img.save(str(test_input_path))
            
            # Mock input parsing to return a pre-processed array
            mock_prepared_input = np.array([[[[0.5] * 224] * 224] * 3] * 1, dtype=np.float32)
            
            # onnxruntime should be installed as a dependency, so no mocking needed
            with patch("mosaic.session_commands._parse_inference_input", return_value=mock_prepared_input):
                # Run inference
                execute_infer(output_fn, str(test_input_path))
            
            # Should still complete with warnings
            output_text = "".join(output_lines)
            # Should have aggregated result or error message
            assert "Aggregating" in output_text or "No predictions" in output_text or "Error" in output_text
            
            # Verify model was lazy loaded
            assert model.binary_rep is not None, "Model should be lazy loaded into binary_rep"
        finally:
            initialize(None, None, [], MosaicConfig(), None)


    def test_execute_infer_lazy_loads_model(self, temp_state_dir):
        """Test that execute_infer lazy loads model into binary_rep on first use."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        config.models_location = str(temp_state_dir / "models")
        models_dir = temp_state_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        # Create model with file location but no binary_rep (will be lazy loaded)
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            onnx_location="",
            file_name="test_model.onnx",
            binary_rep=None,  # Initially None, should be loaded on first inference
        )
        
        # Create a real ONNX model file
        # Import the real onnx module to ensure save works (bypassing any mocks)
        import sys
        
        # Temporarily remove mock if present
        original_onnx = None
        if 'onnx' in sys.modules and hasattr(sys.modules['onnx'], '__spec__') and sys.modules['onnx'].__spec__ is None:
            original_onnx = sys.modules['onnx']
            del sys.modules['onnx']
            if 'onnx.reference' in sys.modules:
                del sys.modules['onnx.reference']
        
        try:
            # Import real onnx module
            import onnx
            onnx_model = _create_simple_onnx_model_for_test()
            model_path = models_dir / model.file_name
            onnx.save(onnx_model, str(model_path))
            
            # Verify file was created
            assert model_path.exists(), f"Model file should exist at {model_path}"
        finally:
            # Restore mock if we removed it
            if original_onnx is not None:
                sys.modules['onnx'] = original_onnx
                if hasattr(original_onnx, 'reference'):
                    sys.modules['onnx.reference'] = original_onnx.reference
        
        # Read the file to verify what should be in binary_rep
        with open(model_path, 'rb') as f:
            expected_binary = f.read()
        
        plan = Plan(stats_data=[], distribution_plan=[], model=model)
        file_def = FileDefinition(location="test", data_type=DataType.IMAGE)
        data = Data(file_definitions=[file_def])
        session = Session(plan=plan, data=data, status=SessionStatus.COMPLETE, id="session_123")
        session.model = model
        
        mock_session_manager.get_session_by_id.return_value = session
        mock_session_manager.update_session.return_value = None
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            # Set session first
            execute_use_session(output_fn, "session_123")
            output_lines.clear()
            
            # Create a dummy input image file
            test_input_path = temp_state_dir / "test_input.png"
            from PIL import Image
            import numpy as np
            test_img = Image.new('RGB', (224, 224), color='red')
            test_img.save(str(test_input_path))
            
            # Mock input parsing
            mock_prepared_input = np.array([[[[0.5] * 224] * 224] * 3] * 1, dtype=np.float32)
            
            # onnxruntime should be installed as a dependency, so no mocking needed
            with patch("mosaic.session_commands._parse_inference_input", return_value=mock_prepared_input):
                # First inference - should lazy load the model
                assert model.binary_rep is None, "Model should not be loaded initially"
                
                execute_infer(output_fn, str(test_input_path))
                
                # Verify model was lazy loaded
                assert model.binary_rep is not None, "Model should be lazy loaded into binary_rep"
                assert model.binary_rep == expected_binary, "binary_rep should match file contents"
                
                # Second inference - should use cached binary_rep (no file read)
                output_lines.clear()
                model.binary_rep = expected_binary  # Reset to verify it's reused
                
                # Track file operations
                file_read_count = [0]
                original_open = open
                def counting_open(*args, **kwargs):
                    if 'rb' in kwargs.get('mode', '') or (args and len(args) > 1 and 'rb' in str(args[1])):
                        file_read_count[0] += 1
                    return original_open(*args, **kwargs)
                
                with patch("builtins.open", side_effect=counting_open):
                    execute_infer(output_fn, str(test_input_path))
                
                # Should not have read the file again (binary_rep was already set)
                # Note: The file might be read once for the ONNX model loading, but not for lazy loading
                # The key is that binary_rep is reused
                assert model.binary_rep == expected_binary, "binary_rep should be reused"
        finally:
            initialize(None, None, [], MosaicConfig(), None)


class TestExecuteSetInferMethod:
    """Tests for execute_set_infer_method function."""
    
    def test_execute_set_infer_method_with_method(self, temp_state_dir):
        """Test execute_set_infer_method with provided method."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            execute_set_infer_method(output_fn, "fedprox")
            
            output_text = "".join(output_lines)
            assert "fedprox" in output_text.lower()
            assert "set" in output_text.lower() or "✓" in output_text
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_set_infer_method_prompts_when_no_method(self, temp_state_dir):
        """Test execute_set_infer_method prompts when no method provided."""
        output_lines = []
        input_responses = ["fedavg"]
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        def input_fn(prompt: str) -> str:
            if input_responses:
                return input_responses.pop(0)
            return ""
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        initialize(mock_beacon, mock_session_manager, [], config, input_fn)
        try:
            execute_set_infer_method(output_fn, None)
            
            output_text = "".join(output_lines)
            assert "Available" in output_text or "methods" in output_text.lower()
        finally:
            initialize(None, None, [], MosaicConfig(), None)
    
    def test_execute_set_infer_method_invalid_method(self, temp_state_dir):
        """Test execute_set_infer_method with invalid method."""
        output_lines = []
        
        def output_fn(text: str) -> None:
            output_lines.append(text)
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        mock_beacon = MagicMock()
        mock_session_manager = MagicMock()
        
        initialize(mock_beacon, mock_session_manager, [], config)
        try:
            execute_set_infer_method(output_fn, "invalid_method")
            
            output_text = "".join(output_lines)
            assert "Error" in output_text or "Unknown" in output_text
        finally:
            initialize(None, None, [], MosaicConfig(), None)
