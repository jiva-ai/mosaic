"""Unit tests for mosaic.mosaic module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from mosaic.mosaic import calculate_data_distribution
from mosaic_planner.state import Model, ModelType, Plan, Session, SessionStatus


def test_calculate_data_distribution_weighted_shard_default():
    """Test that calculate_data_distribution calls plan_static_weighted_shards when method is None."""
    mock_beacon = MagicMock()
    mock_stats_data = [
        {"host": "node1", "connection_status": "online"},
        {"host": "node2", "connection_status": "online"},
    ]
    mock_beacon.collect_stats.return_value = mock_stats_data
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            mock_plan_shards.return_value = [
                {"host": "node1", "allocated_samples": 1},
                {"host": "node2", "allocated_samples": 1},
            ]
            
            calculate_data_distribution(method=None)
            
            # Verify collect_stats was called
            mock_beacon.collect_stats.assert_called_once()
            
            # Verify plan_static_weighted_shards was called with correct args
            mock_plan_shards.assert_called_once_with(
                mock_stats_data,
                total_samples=len(mock_stats_data)
            )


def test_calculate_data_distribution_weighted_shard_explicit():
    """Test that calculate_data_distribution calls plan_static_weighted_shards when method is 'weighted_shard'."""
    mock_beacon = MagicMock()
    mock_stats_data = [
        {"host": "node1", "connection_status": "online"},
        {"host": "node2", "connection_status": "online"},
        {"host": "node3", "connection_status": "online"},
    ]
    mock_beacon.collect_stats.return_value = mock_stats_data
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            mock_plan_shards.return_value = [
                {"host": "node1", "allocated_samples": 1},
                {"host": "node2", "allocated_samples": 1},
                {"host": "node3", "allocated_samples": 1},
            ]
            
            calculate_data_distribution(method="weighted_shard")
            
            # Verify collect_stats was called
            mock_beacon.collect_stats.assert_called_once()
            
            # Verify plan_static_weighted_shards was called with correct args
            mock_plan_shards.assert_called_once_with(
                mock_stats_data,
                total_samples=len(mock_stats_data)
            )


def test_calculate_data_distribution_weighted_batches():
    """Test that calculate_data_distribution calls plan_dynamic_weighted_batches when method is 'weighted_batches'."""
    mock_beacon = MagicMock()
    mock_stats_data = [
        {"host": "node1", "connection_status": "online"},
        {"host": "node2", "connection_status": "online"},
        {"host": "node3", "connection_status": "online"},
        {"host": "node4", "connection_status": "online"},
    ]
    mock_beacon.collect_stats.return_value = mock_stats_data
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_dynamic_weighted_batches") as mock_plan_batches:
            mock_plan_batches.return_value = [
                {"host": "node1", "allocated_batches": 1},
                {"host": "node2", "allocated_batches": 1},
                {"host": "node3", "allocated_batches": 1},
                {"host": "node4", "allocated_batches": 1},
            ]
            
            calculate_data_distribution(method="weighted_batches")
            
            # Verify collect_stats was called
            mock_beacon.collect_stats.assert_called_once()
            
            # Verify plan_dynamic_weighted_batches was called with correct args
            mock_plan_batches.assert_called_once_with(
                mock_stats_data,
                total_batches=len(mock_stats_data)
            )


def test_calculate_data_distribution_no_stats_data():
    """Test that calculate_data_distribution handles empty stats data."""
    mock_beacon = MagicMock()
    mock_beacon.collect_stats.return_value = []
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            calculate_data_distribution(method=None)
            
            # Verify collect_stats was called
            mock_beacon.collect_stats.assert_called_once()
            
            # Verify planner function was not called (no stats data)
            mock_plan_shards.assert_not_called()


def test_calculate_data_distribution_invalid_method():
    """Test that calculate_data_distribution handles invalid method."""
    mock_beacon = MagicMock()
    mock_stats_data = [{"host": "node1", "connection_status": "online"}]
    mock_beacon.collect_stats.return_value = mock_stats_data
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            with patch("mosaic.mosaic.plan_dynamic_weighted_batches") as mock_plan_batches:
                calculate_data_distribution(method="invalid_method")
                
                # Verify collect_stats was called
                mock_beacon.collect_stats.assert_called_once()
                
                # Verify neither planner function was called
                mock_plan_shards.assert_not_called()
                mock_plan_batches.assert_not_called()


def test_calculate_data_distribution_no_beacon():
    """Test that calculate_data_distribution handles missing beacon."""
    with patch("mosaic.mosaic._beacon", None):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            calculate_data_distribution(method=None)
            
            # Verify planner function was not called
            mock_plan_shards.assert_not_called()


class TestMosaicCommandHandlers:
    """Test command handlers for sessions and plans."""

    def test_sessions_handler_registered_on_startup(self, temp_state_dir):
        """Test that sessions handler is registered when mosaic starts up."""
        import mosaic.mosaic as mosaic_module
        from mosaic_comms.beacon import Beacon
        from tests.conftest import create_test_config_with_state

        # Create a config
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5100,
            comms_port=5101,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create a real beacon instance
            beacon = Beacon(config)

            # Register handlers (simulating what main() does)
            beacon.register("sessions", mosaic_module._handle_sessions_command)
            beacon.register("plans", mosaic_module._handle_plans_command)

            # Verify handlers are registered by checking _command_handlers
            assert "sessions" in beacon._command_handlers, "sessions handler should be registered"
            assert "plans" in beacon._command_handlers, "plans handler should be registered"
            assert beacon._command_handlers["sessions"] == mosaic_module._handle_sessions_command
            assert beacon._command_handlers["plans"] == mosaic_module._handle_plans_command

    def test_sessions_handler_returns_sessions(self, temp_state_dir):
        """Test that sessions handler returns the list of sessions."""
        import mosaic.mosaic as mosaic_module

        # Create test sessions
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan1 = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        plan2 = Plan(
            stats_data=[{"host": "node2"}],
            distribution_plan=[{"host": "node2", "allocated_samples": 20}],
            model=model,
        )

        session1 = Session(plan=plan1, status=SessionStatus.RUNNING)
        session2 = Session(plan=plan2, status=SessionStatus.COMPLETE)

        # Set up global sessions list
        with patch("mosaic.mosaic._sessions", [session1, session2]):
            # Call the handler
            result = mosaic_module._handle_sessions_command({})

            # Verify result is a list
            assert isinstance(result, list), "Handler should return a list"
            assert len(result) == 2, f"Expected 2 sessions, got {len(result)}"

            # Verify each entry is a dictionary
            for entry in result:
                assert isinstance(entry, dict), "Each entry should be a dictionary"
                assert "plan" in entry, "Entry should have 'plan' field"
                assert "status" in entry, "Entry should have 'status' field"
                assert "time_started" in entry, "Entry should have 'time_started' field"

            # Verify specific values
            assert result[0]["status"] == "running"
            assert result[1]["status"] == "complete"
            assert result[0]["plan"]["model"]["name"] == "test_model"
            assert result[1]["plan"]["model"]["name"] == "test_model"
            
            # Verify IDs are present and unique
            assert "id" in result[0], "Session should have 'id' field"
            assert "id" in result[1], "Session should have 'id' field"
            assert result[0]["id"] != result[1]["id"], "Session IDs should be unique"
            assert "id" in result[0]["plan"], "Plan should have 'id' field"
            assert "id" in result[1]["plan"], "Plan should have 'id' field"
            assert result[0]["plan"]["id"] != result[1]["plan"]["id"], "Plan IDs should be unique"
            
            # Verify IDs match original objects
            assert result[0]["id"] == session1.id, "Session ID should persist"
            assert result[1]["id"] == session2.id, "Session ID should persist"
            assert result[0]["plan"]["id"] == plan1.id, "Plan ID should persist"
            assert result[1]["plan"]["id"] == plan2.id, "Plan ID should persist"

    def test_plans_handler_returns_plans(self, temp_state_dir):
        """Test that plans handler returns the list of plans."""
        import mosaic.mosaic as mosaic_module

        # Create test plans
        model1 = Model(name="model1", model_type=ModelType.CNN)
        model2 = Model(name="model2", model_type=ModelType.BERT)

        plan1 = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model1,
        )
        plan2 = Plan(
            stats_data=[{"host": "node2"}],
            distribution_plan=[{"host": "node2", "allocated_samples": 20}],
            model=model2,
        )

        # Set up global plans list
        with patch("mosaic.mosaic._plans", [plan1, plan2]):
            # Call the handler
            result = mosaic_module._handle_plans_command({})

            # Verify result is a list
            assert isinstance(result, list), "Handler should return a list"
            assert len(result) == 2, f"Expected 2 plans, got {len(result)}"

            # Verify each entry is a dictionary
            for entry in result:
                assert isinstance(entry, dict), "Each entry should be a dictionary"
                assert "stats_data" in entry, "Entry should have 'stats_data' field"
                assert "distribution_plan" in entry, "Entry should have 'distribution_plan' field"
                assert "model" in entry, "Entry should have 'model' field"

            # Verify specific values
            assert result[0]["model"]["name"] == "model1", f"Expected 'model1', got '{result[0]['model']['name']}'"
            assert result[1]["model"]["name"] == "model2", f"Expected 'model2', got '{result[1]['model']['name']}'"
            # Enums are converted to their values for JSON serialization
            assert result[0]["model"]["model_type"] == "cnn", f"Expected 'cnn', got '{result[0]['model']['model_type']}'"
            assert result[1]["model"]["model_type"] == "bert", f"Expected 'bert', got '{result[1]['model']['model_type']}'"
            
            # Verify IDs are present and unique
            assert "id" in result[0], "Plan should have 'id' field"
            assert "id" in result[1], "Plan should have 'id' field"
            assert result[0]["id"] != result[1]["id"], "Plan IDs should be unique"
            
            # Verify IDs match original objects
            assert result[0]["id"] == plan1.id, "Plan ID should persist"
            assert result[1]["id"] == plan2.id, "Plan ID should persist"

    def test_sessions_handler_returns_empty_list_when_no_sessions(self):
        """Test that sessions handler returns empty list when no sessions exist."""
        import mosaic.mosaic as mosaic_module
        # Set up empty sessions list
        with patch("mosaic.mosaic._sessions", []):
            result = mosaic_module._handle_sessions_command({})
            assert isinstance(result, list), "Handler should return a list"
            assert len(result) == 0, "Should return empty list when no sessions"

    def test_plans_handler_returns_empty_list_when_no_plans(self):
        """Test that plans handler returns empty list when no plans exist."""
        import mosaic.mosaic as mosaic_module
        # Set up empty plans list
        with patch("mosaic.mosaic._plans", []):
            result = mosaic_module._handle_plans_command({})
            assert isinstance(result, list), "Handler should return a list"
            assert len(result) == 0, "Should return empty list when no plans"

    def test_sessions_handler_via_send_command(self, beacon_config_no_ssl, sender_config_no_ssl, temp_state_dir):
        """Test that sessions handler works when called via send_command."""
        import mosaic.mosaic as mosaic_module
        from mosaic_comms.beacon import Beacon

        # Create test sessions
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan1 = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session1 = Session(plan=plan1, status=SessionStatus.RUNNING)

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create two beacons
            beacon1 = Beacon(beacon_config_no_ssl)
            beacon2 = Beacon(sender_config_no_ssl)

            # Register handlers on beacon2 (simulating what main() does)
            beacon2.register("sessions", mosaic_module._handle_sessions_command)

            # Set up global sessions list for beacon2
            with patch("mosaic.mosaic._sessions", [session1]):
                # Start both beacons
                beacon1.start()
                beacon2.start()

                try:
                    # Wait for listeners to start
                    time.sleep(1.0)

                    # Send sessions command from beacon1 to beacon2
                    response = beacon1.send_command(
                        host=sender_config_no_ssl.host,
                        port=sender_config_no_ssl.comms_port,
                        command="sessions",
                        payload={},
                    )

                    # Verify response
                    assert response is not None, "send_command should return a response"
                    assert isinstance(response, list), "Response should be a list"
                    assert len(response) == 1, f"Expected 1 session, got {len(response)}"
                    assert response[0]["status"] == "running"
                    assert response[0]["plan"]["model"]["name"] == "test_model"
                    
                    # Verify IDs are present and persist across beacons
                    assert "id" in response[0], "Session should have 'id' field"
                    assert response[0]["id"] == session1.id, "Session ID should persist across beacons"
                    assert "id" in response[0]["plan"], "Plan should have 'id' field"
                    assert response[0]["plan"]["id"] == plan1.id, "Plan ID should persist across beacons"
                finally:
                    beacon1.stop()
                    beacon2.stop()

    def test_plans_handler_via_send_command(self, beacon_config_no_ssl, sender_config_no_ssl, temp_state_dir):
        """Test that plans handler works when called via send_command."""
        import mosaic.mosaic as mosaic_module
        from mosaic_comms.beacon import Beacon

        # Create test plans
        model1 = Model(name="model1", model_type=ModelType.CNN)
        plan1 = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model1,
        )

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create two beacons
            beacon1 = Beacon(beacon_config_no_ssl)
            beacon2 = Beacon(sender_config_no_ssl)

            # Register handlers on beacon2 (simulating what main() does)
            beacon2.register("plans", mosaic_module._handle_plans_command)

            # Set up global plans list for beacon2
            with patch("mosaic.mosaic._plans", [plan1]):
                # Start both beacons
                beacon1.start()
                beacon2.start()

                try:
                    # Wait for listeners to start
                    time.sleep(1.0)

                    # Send plans command from beacon1 to beacon2
                    response = beacon1.send_command(
                        host=sender_config_no_ssl.host,
                        port=sender_config_no_ssl.comms_port,
                        command="plans",
                        payload={},
                    )

                    # Verify response
                    assert response is not None, "send_command should return a response"
                    assert isinstance(response, list), "Response should be a list"
                    assert len(response) == 1, f"Expected 1 plan, got {len(response)}"
                    assert response[0]["model"]["name"] == "model1"
                    assert response[0]["model"]["model_type"] == "cnn"
                    
                    # Verify IDs are present and persist across beacons
                    assert "id" in response[0], "Plan should have 'id' field"
                    assert response[0]["id"] == plan1.id, "Plan ID should persist across beacons"
                finally:
                    beacon1.stop()
                    beacon2.stop()


class TestPlanAndSessionIDs:
    """Test that Plan and Session objects have unique IDs."""

    def test_plan_has_unique_id(self):
        """Test that Plan objects have unique IDs when created."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan1 = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        plan2 = Plan(
            stats_data=[{"host": "node2"}],
            distribution_plan=[{"host": "node2", "allocated_samples": 20}],
            model=model,
        )
        
        # Verify IDs are set
        assert hasattr(plan1, 'id'), "Plan should have 'id' attribute"
        assert hasattr(plan2, 'id'), "Plan should have 'id' attribute"
        assert plan1.id is not None, "Plan ID should not be None"
        assert plan2.id is not None, "Plan ID should not be None"
        assert isinstance(plan1.id, str), "Plan ID should be a string"
        assert isinstance(plan2.id, str), "Plan ID should be a string"
        assert len(plan1.id) > 0, "Plan ID should not be empty"
        assert len(plan2.id) > 0, "Plan ID should not be empty"
        
        # Verify IDs are unique
        assert plan1.id != plan2.id, "Plan IDs should be unique"
        
        # Verify ID persists when plan is serialized/deserialized
        import pickle
        pickled = pickle.dumps(plan1)
        unpickled = pickle.loads(pickled)
        assert unpickled.id == plan1.id, "Plan ID should persist after serialization"

    def test_session_has_unique_id(self):
        """Test that Session objects have unique IDs when created."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session1 = Session(plan=plan, status=SessionStatus.RUNNING)
        session2 = Session(plan=plan, status=SessionStatus.COMPLETE)
        
        # Verify IDs are set
        assert hasattr(session1, 'id'), "Session should have 'id' attribute"
        assert hasattr(session2, 'id'), "Session should have 'id' attribute"
        assert session1.id is not None, "Session ID should not be None"
        assert session2.id is not None, "Session ID should not be None"
        assert isinstance(session1.id, str), "Session ID should be a string"
        assert isinstance(session2.id, str), "Session ID should be a string"
        assert len(session1.id) > 0, "Session ID should not be empty"
        assert len(session2.id) > 0, "Session ID should not be empty"
        
        # Verify IDs are unique
        assert session1.id != session2.id, "Session IDs should be unique"
        
        # Verify ID persists when session is serialized/deserialized
        import pickle
        pickled = pickle.dumps(session1)
        unpickled = pickle.loads(pickled)
        assert unpickled.id == session1.id, "Session ID should persist after serialization"
        assert unpickled.plan.id == plan.id, "Plan ID should persist within Session after serialization"

    def test_plan_id_can_be_set_explicitly(self):
        """Test that Plan ID can be set explicitly when creating."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        custom_id = "custom-plan-id-123"
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
            id=custom_id,
        )
        
        assert plan.id == custom_id, "Plan ID should be set to custom value"

    def test_session_id_can_be_set_explicitly(self):
        """Test that Session ID can be set explicitly when creating."""
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        custom_id = "custom-session-id-456"
        session = Session(plan=plan, status=SessionStatus.RUNNING, id=custom_id)
        
        assert session.id == custom_id, "Session ID should be set to custom value"

