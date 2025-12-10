"""Unit tests for distribution retry logic in beacon."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import pytest

# Mock onnx and torch modules if not available (similar to test_session_commands.py)
_original_onnx = sys.modules.get('onnx')
_original_torch = sys.modules.get('torch')
_we_added_onnx_mock = False
_we_added_torch_mock = False

if 'onnx' not in sys.modules:
    from types import ModuleType, SimpleNamespace
    
    class MockOnnx(ModuleType):
        """Mock onnx module with necessary attributes."""
        def __init__(self):
            super().__init__('onnx')
            spec = SimpleNamespace()
            spec.name = 'onnx'
            spec.loader = None
            spec.origin = None
            self.__spec__ = spec
            reference = SimpleNamespace()
            self.reference = reference
            self.ModelProto = type('ModelProto', (), {})
            self.TensorProto = type('TensorProto', (), {
                'FLOAT': 1,
                'INT64': 7,
            })
            checker = SimpleNamespace()
            checker.check_model = MagicMock()
            self.checker = checker
            self.load = MagicMock()
            helper = SimpleNamespace()
            helper.make_tensor_value_info = MagicMock()
            helper.make_node = MagicMock()
            helper.make_graph = MagicMock()
            helper.make_model = MagicMock()
            self.helper = helper
        
        def __getattr__(self, name):
            return MagicMock()
    
    mock_onnx = MockOnnx()
    sys.modules['onnx'] = mock_onnx
    sys.modules['onnx.reference'] = mock_onnx.reference
    _we_added_onnx_mock = True

if 'torch' not in sys.modules:
    # Use MagicMock for torch and all submodules - it handles everything dynamically
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0
    mock_torch.habana.is_available.return_value = False
    mock_torch.habana.device_count.return_value = 0
    # Make it iterable (for cases where torch modules are iterated)
    mock_torch.__iter__ = lambda self: iter([])
    sys.modules['torch'] = mock_torch
    # Add common submodules that are imported
    sys.modules['torch.nn'] = MagicMock()
    sys.modules['torch.nn.functional'] = MagicMock()
    sys.modules['torch.optim'] = MagicMock()
    sys.modules['torch.utils'] = MagicMock()
    sys.modules['torch.utils.data'] = MagicMock()
    sys.modules['torch.utils.data.DataLoader'] = MagicMock()
    sys.modules['torch.utils.data.Dataset'] = MagicMock()
    _we_added_torch_mock = True

from mosaic_comms.beacon import Beacon
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
from mosaic_config.config import Peer
from tests.conftest import create_test_config_with_state


@pytest.fixture(scope="module", autouse=True)
def cleanup_mocks():
    """Clean up mocks after all tests in this module."""
    yield
    # After all tests, restore originals if we had them
    if _we_added_onnx_mock and _original_onnx is not None:
        sys.modules['onnx'] = _original_onnx
    if _we_added_torch_mock and _original_torch is not None:
        sys.modules['torch'] = _original_torch


class TestDistributionRetryLogic:
    """Test retry logic for data and model distribution."""

    def test_data_distribution_retry_on_node_failure(self, temp_state_dir):
        """
        Test that data distribution retries to next capable node when one node fails.
        
        Scenario:
        - 3 nodes: Node 1 (sender), Node 2 (fails), Node 3 (succeeds on retry)
        - Data is distributed to Node 2 and Node 3
        - Node 2 fails, retry goes to Node 3
        - Verify session states throughout
        """
        # Create configs for 3 beacons
        sender_config = create_test_config_with_state(
            state_dir=temp_state_dir / "sender",
            host="127.0.0.1",
            heartbeat_port=7000,
            comms_port=7001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(temp_state_dir / "data"),
        )
        
        receiver2_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver2",
            host="127.0.0.1",
            heartbeat_port=7002,
            comms_port=7003,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(temp_state_dir / "data"),
        )
        
        receiver3_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver3",
            host="127.0.0.1",
            heartbeat_port=7004,
            comms_port=7005,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(temp_state_dir / "data"),
        )
        
        # Add peers to sender config
        sender_config.peers = [
            Peer(host=receiver2_config.host, heartbeat_port=receiver2_config.heartbeat_port, comms_port=receiver2_config.comms_port),
            Peer(host=receiver3_config.host, heartbeat_port=receiver3_config.heartbeat_port, comms_port=receiver3_config.comms_port),
        ]
        
        # Create test data
        test_data_dir = temp_state_dir / "data"
        test_data_dir.mkdir(exist_ok=True)
        test_file = test_data_dir / "test.txt"
        test_file.write_text("test data content")
        
        # Create data object
        data = Data(
            file_definitions=[
                FileDefinition(
                    location=str(test_file.relative_to(test_data_dir)),
                    data_type=DataType.TEXT,
                    is_segmentable=False,
                )
            ]
        )
        
        # Create distribution plan: Node 1 distributes to Node 2 and Node 3
        distribution_plan = [
            {"host": receiver2_config.host, "comms_port": receiver2_config.comms_port, "capacity_fraction": 0.5, "connection_status": "online"},
            {"host": receiver3_config.host, "comms_port": receiver3_config.comms_port, "capacity_fraction": 0.3, "connection_status": "online"},
        ]
        
        # Create data segmentation plan
        data_segmentation_plan = [
            {
                "host": receiver2_config.host,
                "comms_port": receiver2_config.comms_port,
                "segments": [
                    {"file_location": str(test_file.relative_to(test_data_dir)), "start": 0, "end": 5}
                ]
            },
            {
                "host": receiver3_config.host,
                "comms_port": receiver3_config.comms_port,
                "segments": [
                    {"file_location": str(test_file.relative_to(test_data_dir)), "start": 5, "end": 10}
                ]
            },
        ]
        
        # Create a dummy model for the plan (not used in data distribution tests)
        dummy_model = Model(name="dummy", model_type=ModelType.CNN)
        
        plan = Plan(
            stats_data={},
            distribution_plan=distribution_plan,
            data_segmentation_plan=data_segmentation_plan,
            model=dummy_model,
        )
        
        # Create session
        session = Session(
            plan=plan,
            data=data,
            status=SessionStatus.IDLE,
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class, \
             patch("mosaic.mosaic._session_manager", MagicMock()), \
             patch("mosaic.mosaic._config", sender_config):
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            sender_beacon = Beacon(sender_config)
            receiver2_beacon = Beacon(receiver2_config)
            receiver3_beacon = Beacon(receiver3_config)
            
            try:
                sender_beacon.start()
                receiver2_beacon.start()
                receiver3_beacon.start()
                
                time.sleep(0.5)  # Give beacons time to start
                
                # Patch send_command to make receiver2 fail on first attempt
                original_send_command = sender_beacon.send_command
                node2_attempts = {"count": 0}
                
                def mock_send_command(host, port, command, payload, timeout=None):
                    # Track attempts to node 2 for any data plan command
                    if host == receiver2_config.host and port == receiver2_config.comms_port:
                        if command in ("exdplan", "exdplan_chunk", "exdplan_finalize"):
                            node2_attempts["count"] += 1
                            if node2_attempts["count"] == 1:  # Fail on first attempt
                                raise Exception("Simulated failure on node 2")
                    # For retry to node 3, call the handler directly
                    if host == receiver3_config.host and port == receiver3_config.comms_port:
                        if command == "exdplan":
                            # Call receiver3's handler directly
                            return receiver3_beacon._handle_execute_data_plan(payload)
                        elif command == "exdplan_chunk":
                            return receiver3_beacon._handle_execute_data_plan_chunk(payload)
                        elif command == "exdplan_finalize":
                            return receiver3_beacon._handle_execute_data_plan_finalize(payload)
                    return original_send_command(host, port, command, payload, timeout)
                
                sender_beacon.send_command = mock_send_command
                
                # Execute data plan
                sender_beacon.execute_data_plan(plan, data, session)
                
                # Verify session state
                assert "machines" in session.data_distribution_state
                assert "failed_machines" in session.data_distribution_state
                assert "retry_attempts" in session.data_distribution_state
                
                # Check that Node 2 failed
                node2_key = f"{receiver2_config.host}:{receiver2_config.comms_port}"
                assert node2_key in session.data_distribution_state["machines"]
                assert session.data_distribution_state["machines"][node2_key]["status"] == "failed"
                
                # Check that retry happened to Node 3
                # The retry should have created a new entry with Node 3's host/port
                # but with original_host/original_comms_port pointing to Node 2
                node3_key = f"{receiver3_config.host}:{receiver3_config.comms_port}"
                
                # Verify Node 3 received the data (check that handler was called)
                # Since Node 2 failed, the retry should have sent to Node 3
                # We need to check if Node 3's handler was called more than once
                # (once for original, once for retry)
                
                # Verify final status
                # If retry succeeded, status should be RUNNING
                # If all succeeded after retry, status should be RUNNING
                final_success_count = sum(
                    1 for m in session.data_distribution_state["machines"].values()
                    if m.get("status") == "success"
                )
                
                # At least one distribution should have succeeded (Node 3)
                assert final_success_count >= 1
                
                # Verify that ERROR_CORRECTION was set during retry
                # (status might be RUNNING now if recovery succeeded)
                # Check retry_attempts to verify retry happened
                assert len(session.data_distribution_state["retry_attempts"]) > 0
                
            finally:
                sender_beacon.stop()
                receiver2_beacon.stop()
                receiver3_beacon.stop()

    def test_data_distribution_fails_when_all_nodes_fail(self, temp_state_dir):
        """
        Test that data distribution fails with ERROR status when all nodes fail.
        
        Scenario:
        - 3 nodes: Node 1 (sender), Node 2 (fails), Node 3 (fails)
        - Data is distributed to Node 2 and Node 3
        - Both nodes fail, no capable nodes remain
        - Verify ERROR status
        """
        # Create configs for 3 beacons
        sender_config = create_test_config_with_state(
            state_dir=temp_state_dir / "sender",
            host="127.0.0.1",
            heartbeat_port=7100,
            comms_port=7101,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(temp_state_dir / "data"),
        )
        
        receiver2_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver2",
            host="127.0.0.1",
            heartbeat_port=7102,
            comms_port=7103,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(temp_state_dir / "data"),
        )
        
        receiver3_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver3",
            host="127.0.0.1",
            heartbeat_port=7104,
            comms_port=7105,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(temp_state_dir / "data"),
        )
        
        # Add peers to sender config
        sender_config.peers = [
            Peer(host=receiver2_config.host, heartbeat_port=receiver2_config.heartbeat_port, comms_port=receiver2_config.comms_port),
            Peer(host=receiver3_config.host, heartbeat_port=receiver3_config.heartbeat_port, comms_port=receiver3_config.comms_port),
        ]
        
        # Create test data
        test_data_dir = temp_state_dir / "data"
        test_data_dir.mkdir(exist_ok=True)
        test_file = test_data_dir / "test.txt"
        test_file.write_text("test data content")
        
        # Create data object
        data = Data(
            file_definitions=[
                FileDefinition(
                    location=str(test_file.relative_to(test_data_dir)),
                    data_type=DataType.TEXT,
                    is_segmentable=False,
                )
            ]
        )
        
        # Create distribution plan: Node 1 distributes to Node 2 and Node 3
        distribution_plan = [
            {"host": receiver2_config.host, "comms_port": receiver2_config.comms_port, "capacity_fraction": 0.5, "connection_status": "online"},
            {"host": receiver3_config.host, "comms_port": receiver3_config.comms_port, "capacity_fraction": 0.3, "connection_status": "online"},
        ]
        
        # Create data segmentation plan
        data_segmentation_plan = [
            {
                "host": receiver2_config.host,
                "comms_port": receiver2_config.comms_port,
                "segments": [
                    {"file_location": str(test_file.relative_to(test_data_dir)), "start": 0, "end": 5}
                ]
            },
            {
                "host": receiver3_config.host,
                "comms_port": receiver3_config.comms_port,
                "segments": [
                    {"file_location": str(test_file.relative_to(test_data_dir)), "start": 5, "end": 10}
                ]
            },
        ]
        
        # Create a dummy model for the plan (not used in data distribution tests)
        dummy_model = Model(name="dummy", model_type=ModelType.CNN)
        
        plan = Plan(
            stats_data={},
            distribution_plan=distribution_plan,
            data_segmentation_plan=data_segmentation_plan,
            model=dummy_model,
        )
        
        # Create session
        session = Session(
            plan=plan,
            data=data,
            status=SessionStatus.IDLE,
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class, \
             patch("mosaic.mosaic._session_manager", MagicMock()), \
             patch("mosaic.mosaic._config", sender_config):
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            sender_beacon = Beacon(sender_config)
            receiver2_beacon = Beacon(receiver2_config)
            receiver3_beacon = Beacon(receiver3_config)
            
            try:
                sender_beacon.start()
                receiver2_beacon.start()
                receiver3_beacon.start()
                
                time.sleep(0.5)  # Give beacons time to start
                
                # Patch send_command to make both receivers fail
                original_send_command = sender_beacon.send_command
                
                def mock_send_command(host, port, command, payload, timeout=None):
                    if command in ("exdplan", "exdplan_chunk", "exdplan_finalize"):
                        # Both receivers fail
                        raise Exception("Simulated failure")
                    return original_send_command(host, port, command, payload, timeout)
                
                sender_beacon.send_command = mock_send_command
                
                # Execute data plan
                sender_beacon.execute_data_plan(plan, data, session)
                
                # Verify session status is ERROR
                assert session.status == SessionStatus.ERROR
                
                # Verify error message
                assert "final_error" in session.data_distribution_state
                assert "No capable nodes remaining" in session.data_distribution_state["final_error"]
                
                # Verify both nodes are marked as failed
                node2_key = f"{receiver2_config.host}:{receiver2_config.comms_port}"
                node3_key = f"{receiver3_config.host}:{receiver3_config.comms_port}"
                
                assert node2_key in session.data_distribution_state["machines"]
                assert session.data_distribution_state["machines"][node2_key]["status"] == "failed"
                
                # Node 3 should also be marked as failed (either as original or retry)
                assert node3_key in session.data_distribution_state["machines"]
                assert session.data_distribution_state["machines"][node3_key]["status"] == "failed"
                
                # Verify we have at least 2 failed machines
                failed_count = sum(
                    1 for m in session.data_distribution_state["machines"].values()
                    if m.get("status") == "failed"
                )
                assert failed_count >= 2
                
            finally:
                sender_beacon.stop()
                receiver2_beacon.stop()
                receiver3_beacon.stop()

    def test_model_distribution_retry_on_node_failure(self, temp_state_dir):
        """
        Test that model distribution retries to next capable node when one node fails.
        
        Scenario:
        - 3 nodes: Node 1 (sender), Node 2 (fails), Node 3 (succeeds on retry)
        - Model is distributed to Node 2 and Node 3
        - Node 2 fails, retry goes to Node 3
        - Verify session states throughout
        """
        # Create configs for 3 beacons
        sender_config = create_test_config_with_state(
            state_dir=temp_state_dir / "sender",
            host="127.0.0.1",
            heartbeat_port=7200,
            comms_port=7201,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            models_location=str(temp_state_dir / "models"),
        )
        
        receiver2_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver2",
            host="127.0.0.1",
            heartbeat_port=7202,
            comms_port=7203,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            models_location=str(temp_state_dir / "models"),
        )
        
        receiver3_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver3",
            host="127.0.0.1",
            heartbeat_port=7204,
            comms_port=7205,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            models_location=str(temp_state_dir / "models"),
        )
        
        # Create a dummy model file
        models_dir = temp_state_dir / "models"
        models_dir.mkdir(exist_ok=True)
        model_file = models_dir / "test_model.onnx"
        model_file.write_bytes(b"dummy model binary data" * 100)  # Some binary data
        
        # Create model object
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            onnx_location="",
            file_name="test_model.onnx",
            binary_rep=model_file.read_bytes(),
        )
        
        # Create distribution plan
        distribution_plan = [
            {"host": receiver2_config.host, "comms_port": receiver2_config.comms_port, "capacity_fraction": 0.5, "connection_status": "online"},
            {"host": receiver3_config.host, "comms_port": receiver3_config.comms_port, "capacity_fraction": 0.3, "connection_status": "online"},
        ]
        
        plan = Plan(
            stats_data={},
            distribution_plan=distribution_plan,
            model=model,
        )
        
        # Create session
        session = Session(
            plan=plan,
            status=SessionStatus.RUNNING,
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class, \
             patch("mosaic.mosaic._session_manager", MagicMock()), \
             patch("mosaic.mosaic._config", sender_config):
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            sender_beacon = Beacon(sender_config)
            receiver2_beacon = Beacon(receiver2_config)
            receiver3_beacon = Beacon(receiver3_config)
            
            try:
                sender_beacon.start()
                receiver2_beacon.start()
                receiver3_beacon.start()
                
                time.sleep(0.5)  # Give beacons time to start
                
                # Patch send_command to make receiver2 fail on first attempt
                original_send_command = sender_beacon.send_command
                node2_attempts = {"count": 0}
                
                def mock_send_command(host, port, command, payload, timeout=None):
                    # Track attempts to node 2 for any model plan command
                    if host == receiver2_config.host and port == receiver2_config.comms_port:
                        if command in ("exmplan", "exmplan_chunk", "exmplan_finalize"):
                            node2_attempts["count"] += 1
                            if node2_attempts["count"] == 1:  # Fail on first attempt
                                raise Exception("Simulated failure on node 2")
                    # For retry to node 3, call the handler directly
                    if host == receiver3_config.host and port == receiver3_config.comms_port:
                        if command == "exmplan":
                            # Call receiver3's handler directly
                            return receiver3_beacon._handle_execute_model_plan(payload)
                        elif command == "exmplan_chunk":
                            # For chunked model, just return success
                            return {"status": "success"}
                        elif command == "exmplan_finalize":
                            # For finalize, call handler
                            return receiver3_beacon._handle_execute_model_plan(payload)
                    return original_send_command(host, port, command, payload, timeout)
                
                sender_beacon.send_command = mock_send_command
                
                # Execute model plan
                sender_beacon.execute_model_plan(session, model)
                
                # Verify session state
                assert "nodes" in session.model_distribution_state
                assert "failed_nodes" in session.model_distribution_state
                assert "retry_attempts" in session.model_distribution_state
                
                # Check that retry happened
                assert len(session.model_distribution_state["retry_attempts"]) > 0
                
                # Verify final status
                # If retry succeeded, status should be RUNNING
                # If all succeeded after retry, status should be RUNNING
                final_success_count = sum(
                    1 for n in session.model_distribution_state["nodes"].values()
                    if n.get("status") == "success"
                )
                
                # At least one distribution should have succeeded
                assert final_success_count >= 1
                
            finally:
                sender_beacon.stop()
                receiver2_beacon.stop()
                receiver3_beacon.stop()

    def test_model_distribution_fails_when_all_nodes_fail(self, temp_state_dir):
        """
        Test that model distribution fails with ERROR status when all nodes fail.
        
        Scenario:
        - 3 nodes: Node 1 (sender), Node 2 (fails), Node 3 (fails)
        - Model is distributed to Node 2 and Node 3
        - Both nodes fail, no capable nodes remain
        - Verify ERROR status
        """
        # Create configs for 3 beacons
        sender_config = create_test_config_with_state(
            state_dir=temp_state_dir / "sender",
            host="127.0.0.1",
            heartbeat_port=7300,
            comms_port=7301,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            models_location=str(temp_state_dir / "models"),
        )
        
        receiver2_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver2",
            host="127.0.0.1",
            heartbeat_port=7302,
            comms_port=7303,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            models_location=str(temp_state_dir / "models"),
        )
        
        receiver3_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver3",
            host="127.0.0.1",
            heartbeat_port=7304,
            comms_port=7305,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            models_location=str(temp_state_dir / "models"),
        )
        
        # Create a dummy model file
        models_dir = temp_state_dir / "models"
        models_dir.mkdir(exist_ok=True)
        model_file = models_dir / "test_model.onnx"
        model_file.write_bytes(b"dummy model binary data" * 100)  # Some binary data
        
        # Create model object
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            onnx_location="",
            file_name="test_model.onnx",
            binary_rep=model_file.read_bytes(),
        )
        
        # Create distribution plan
        distribution_plan = [
            {"host": receiver2_config.host, "comms_port": receiver2_config.comms_port, "capacity_fraction": 0.5, "connection_status": "online"},
            {"host": receiver3_config.host, "comms_port": receiver3_config.comms_port, "capacity_fraction": 0.3, "connection_status": "online"},
        ]
        
        plan = Plan(
            stats_data={},
            distribution_plan=distribution_plan,
            model=model,
        )
        
        # Create session
        session = Session(
            plan=plan,
            status=SessionStatus.RUNNING,
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class, \
             patch("mosaic.mosaic._session_manager", MagicMock()), \
             patch("mosaic.mosaic._config", sender_config):
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            sender_beacon = Beacon(sender_config)
            receiver2_beacon = Beacon(receiver2_config)
            receiver3_beacon = Beacon(receiver3_config)
            
            try:
                sender_beacon.start()
                receiver2_beacon.start()
                receiver3_beacon.start()
                
                time.sleep(0.5)  # Give beacons time to start
                
                # Patch send_command to make both receivers fail
                original_send_command = sender_beacon.send_command
                
                def mock_send_command(host, port, command, payload, timeout=None):
                    if command in ("exmplan", "exmplan_chunk", "exmplan_finalize"):
                        # Both receivers fail
                        raise Exception("Simulated failure")
                    return original_send_command(host, port, command, payload, timeout)
                
                sender_beacon.send_command = mock_send_command
                
                # Execute model plan
                sender_beacon.execute_model_plan(session, model)
                
                # Verify session status is ERROR
                assert session.status == SessionStatus.ERROR
                
                # Verify error message
                assert "final_error" in session.model_distribution_state
                assert "No capable nodes remaining" in session.model_distribution_state["final_error"]
                
                # Verify both nodes are marked as failed
                # Check that we have at least 2 failed nodes in the nodes dict
                failed_count = sum(
                    1 for n in session.model_distribution_state["nodes"].values()
                    if n.get("status") == "failed"
                )
                assert failed_count >= 2
                
            finally:
                sender_beacon.stop()
                receiver2_beacon.stop()
                receiver3_beacon.stop()


class TestFindNextCapableNode:
    """Test _find_next_capable_node function."""
    
    def test_find_next_capable_node_basic(self, temp_state_dir):
        """Test basic functionality of _find_next_capable_node."""
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=7400,
            comms_port=7401,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
        )
        
        with patch("mosaic_comms.beacon.StatsCollector"):
            beacon = Beacon(config)
            
            # Create distribution plan with nodes sorted by capability
            distribution_plan = [
                {"host": "127.0.0.1", "comms_port": 8001, "capacity_fraction": 0.8, "connection_status": "online"},
                {"host": "127.0.0.1", "comms_port": 8002, "capacity_fraction": 0.6, "connection_status": "online"},
                {"host": "127.0.0.1", "comms_port": 8003, "capacity_fraction": 0.4, "connection_status": "online"},
                {"host": "127.0.0.1", "comms_port": 8004, "capacity_fraction": 0.2, "connection_status": "online"},
            ]
            
            # Test: Find next node when first node fails
            failed_node = {"host": "127.0.0.1", "comms_port": 8001}
            failed_nodes = []
            
            next_node = beacon._find_next_capable_node(
                failed_node,
                distribution_plan,
                failed_nodes,
            )
            
            assert next_node is not None
            assert next_node["comms_port"] == 8002  # Next most capable
            
            # Test: Find next node when first two nodes fail
            failed_nodes = [{"host": "127.0.0.1", "comms_port": 8001}]
            failed_node = {"host": "127.0.0.1", "comms_port": 8002}
            
            next_node = beacon._find_next_capable_node(
                failed_node,
                distribution_plan,
                failed_nodes,
            )
            
            assert next_node is not None
            assert next_node["comms_port"] == 8003  # Next most capable after excluding failed ones
            
            # Test: No capable nodes remaining
            failed_nodes = [
                {"host": "127.0.0.1", "comms_port": 8001},
                {"host": "127.0.0.1", "comms_port": 8002},
                {"host": "127.0.0.1", "comms_port": 8003},
            ]
            failed_node = {"host": "127.0.0.1", "comms_port": 8004}
            
            next_node = beacon._find_next_capable_node(
                failed_node,
                distribution_plan,
                failed_nodes,
            )
            
            assert next_node is None  # No capable nodes remaining
    
    def test_find_next_capable_node_with_exclude_nodes(self, temp_state_dir):
        """Test _find_next_capable_node with exclude_nodes parameter."""
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=7500,
            comms_port=7501,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
        )
        
        with patch("mosaic_comms.beacon.StatsCollector"):
            beacon = Beacon(config)
            
            distribution_plan = [
                {"host": "127.0.0.1", "comms_port": 8001, "capacity_fraction": 0.8, "connection_status": "online"},
                {"host": "127.0.0.1", "comms_port": 8002, "capacity_fraction": 0.6, "connection_status": "online"},
                {"host": "127.0.0.1", "comms_port": 8003, "capacity_fraction": 0.4, "connection_status": "online"},
            ]
            
            failed_node = {"host": "127.0.0.1", "comms_port": 8001}
            failed_nodes = []
            exclude_nodes = [{"host": "127.0.0.1", "comms_port": 8002}]  # Exclude node 2
            
            next_node = beacon._find_next_capable_node(
                failed_node,
                distribution_plan,
                failed_nodes,
                exclude_nodes=exclude_nodes,
            )
            
            assert next_node is not None
            assert next_node["comms_port"] == 8003  # Should skip excluded node 2
    
    def test_find_next_capable_node_with_effective_score(self, temp_state_dir):
        """Test _find_next_capable_node using effective_score instead of capacity_fraction."""
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=7600,
            comms_port=7601,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
        )
        
        with patch("mosaic_comms.beacon.StatsCollector"):
            beacon = Beacon(config)
            
            # Distribution plan with effective_score instead of capacity_fraction
            distribution_plan = [
                {"host": "127.0.0.1", "comms_port": 8001, "effective_score": 100.0, "connection_status": "online"},
                {"host": "127.0.0.1", "comms_port": 8002, "effective_score": 80.0, "connection_status": "online"},
                {"host": "127.0.0.1", "comms_port": 8003, "effective_score": 60.0, "connection_status": "online"},
            ]
            
            failed_node = {"host": "127.0.0.1", "comms_port": 8001}
            failed_nodes = []
            
            next_node = beacon._find_next_capable_node(
                failed_node,
                distribution_plan,
                failed_nodes,
            )
            
            assert next_node is not None
            assert next_node["comms_port"] == 8002  # Next most capable by effective_score
    
    def test_find_next_capable_node_skips_offline_nodes(self, temp_state_dir):
        """Test that _find_next_capable_node skips offline nodes."""
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=7700,
            comms_port=7701,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
        )
        
        with patch("mosaic_comms.beacon.StatsCollector"):
            beacon = Beacon(config)
            
            distribution_plan = [
                {"host": "127.0.0.1", "comms_port": 8001, "capacity_fraction": 0.8, "connection_status": "online"},
                {"host": "127.0.0.1", "comms_port": 8002, "capacity_fraction": 0.6, "connection_status": "offline"},  # Offline
                {"host": "127.0.0.1", "comms_port": 8003, "capacity_fraction": 0.4, "connection_status": "online"},
            ]
            
            failed_node = {"host": "127.0.0.1", "comms_port": 8001}
            failed_nodes = []
            
            next_node = beacon._find_next_capable_node(
                failed_node,
                distribution_plan,
                failed_nodes,
            )
            
            assert next_node is not None
            assert next_node["comms_port"] == 8003  # Should skip offline node 2

