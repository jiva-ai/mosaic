"""Unit tests for mosaic.mosaic module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from mosaic.mosaic import (
    _sanitize_filename,
    add_model,
    add_session,
    calculate_data_distribution,
    remove_model,
    remove_session,
)
from mosaic_config.state import Model, ModelType, Plan, Session, SessionStatus


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

            # Verify handlers are registered by checking _command_handlers
            assert "sessions" in beacon._command_handlers, "sessions handler should be registered"
            assert beacon._command_handlers["sessions"] == mosaic_module._handle_sessions_command

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

        session1 = Session(plan=plan1, status=SessionStatus.TRAINING)
        session2 = Session(plan=plan2, status=SessionStatus.COMPLETE)

        # Set up global session manager
        from mosaic_config.state_manager import SessionStateManager
        mock_session_manager = MagicMock()
        mock_session_manager.get_sessions.return_value = [session1, session2]
        with patch("mosaic.mosaic._session_manager", mock_session_manager):
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
            assert result[0]["status"] == "training"
            assert result[1]["status"] == "complete"
            # Model objects are not persisted - only model_id is saved
            assert "model_id" in result[0]["plan"], "Plan should have 'model_id' field"
            assert "model_id" in result[1]["plan"], "Plan should have 'model_id' field"
            assert result[0]["plan"]["model_id"] == model.id
            assert result[1]["plan"]["model_id"] == model.id
            
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

    def test_sessions_handler_returns_empty_list_when_no_sessions(self):
        """Test that sessions handler returns empty list when no sessions exist."""
        import mosaic.mosaic as mosaic_module
        # Set up empty session manager
        from mosaic_config.state_manager import SessionStateManager
        mock_session_manager = MagicMock()
        mock_session_manager.get_sessions.return_value = []
        with patch("mosaic.mosaic._session_manager", mock_session_manager):
            result = mosaic_module._handle_sessions_command({})
            assert isinstance(result, list), "Handler should return a list"
            assert len(result) == 0, "Should return empty list when no sessions"

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
        session1 = Session(plan=plan1, status=SessionStatus.TRAINING)

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create two beacons
            beacon1 = Beacon(beacon_config_no_ssl)
            beacon2 = Beacon(sender_config_no_ssl)

            # Register handlers on beacon2 (simulating what main() does)
            beacon2.register("sessions", mosaic_module._handle_sessions_command)

            # Set up global session manager for beacon2
            from mosaic_config.state_manager import SessionStateManager
            mock_session_manager = MagicMock()
            mock_session_manager.get_sessions.return_value = [session1]
            with patch("mosaic.mosaic._session_manager", mock_session_manager):
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
                    assert response[0]["status"] == "training"
                    # Model objects are not persisted - only model_id is saved
                    assert "model_id" in response[0]["plan"], "Plan should have 'model_id' field"
                    assert response[0]["plan"]["model_id"] == model.id
                    
                    # Verify IDs are present and persist across beacons
                    assert "id" in response[0], "Session should have 'id' field"
                    assert response[0]["id"] == session1.id, "Session ID should persist across beacons"
                    assert "id" in response[0]["plan"], "Plan should have 'id' field"
                    assert response[0]["plan"]["id"] == plan1.id, "Plan ID should persist across beacons"
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
        session1 = Session(plan=plan, status=SessionStatus.TRAINING)
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
        session = Session(plan=plan, status=SessionStatus.TRAINING, id=custom_id)
        
        assert session.id == custom_id, "Session ID should be set to custom value"


class TestAddRemoveSession:
    """Test add_session and remove_session functions."""

    def test_add_session_adds_to_list_and_persists(self, temp_state_dir):
        """Test that add_session adds a session to the list and persists state."""
        import mosaic.mosaic as mosaic_module
        from mosaic_config.config import MosaicConfig
        from mosaic_config.state_utils import read_state, StateIdentifiers

        # Create a test config with state location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)

        # Create a test session
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.TRAINING)

        # Set up global config and session manager
        from mosaic_config.state_manager import SessionStateManager
        session_manager = SessionStateManager(config)
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._session_manager", session_manager):
                # Add session
                add_session(session)

                # Verify session was added to the list
                sessions = session_manager.get_sessions()
                assert len(sessions) == 1
                assert sessions[0].id == session.id

                # Verify state was persisted
                loaded_sessions = read_state(config, StateIdentifiers.SESSIONS, default=None)
                assert isinstance(loaded_sessions, list)
                assert len(loaded_sessions) == 1
                assert loaded_sessions[0].id == session.id

    def test_remove_session_removes_from_list_and_persists(self, temp_state_dir):
        """Test that remove_session removes a session by ID and persists state."""
        import mosaic.mosaic as mosaic_module
        from mosaic_config.config import MosaicConfig
        from mosaic_config.state_utils import read_state, StateIdentifiers

        # Create a test config with state location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)

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
        session1 = Session(plan=plan1, status=SessionStatus.TRAINING)
        session2 = Session(plan=plan2, status=SessionStatus.COMPLETE)

        # Set up global config and session manager
        from mosaic_config.state_manager import SessionStateManager
        session_manager = SessionStateManager(config)
        session_manager.add_session(session1)
        session_manager.add_session(session2)
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._session_manager", session_manager):
                # Remove session1
                result = remove_session(session1.id)

                # Verify removal was successful
                assert result is True
                sessions = session_manager.get_sessions()
                assert len(sessions) == 1
                assert sessions[0].id == session2.id

                # Verify state was persisted
                loaded_sessions = read_state(config, StateIdentifiers.SESSIONS, default=None)
                assert isinstance(loaded_sessions, list)
                assert len(loaded_sessions) == 1
                assert loaded_sessions[0].id == session2.id

    def test_remove_session_returns_false_when_not_found(self, temp_state_dir):
        """Test that remove_session returns False when session ID is not found."""
        import mosaic.mosaic as mosaic_module
        from mosaic_config.config import MosaicConfig

        # Create a test config with state location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)

        # Create a test session
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.TRAINING)

        # Set up global config and session manager
        from mosaic_config.state_manager import SessionStateManager
        session_manager = SessionStateManager(config)
        session_manager.add_session(session)
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._session_manager", session_manager):
                # Try to remove non-existent session
                result = remove_session("non-existent-id")

                # Verify removal failed
                assert result is False
                sessions = session_manager.get_sessions()
                assert len(sessions) == 1
                assert sessions[0].id == session.id

    def test_add_multiple_sessions_persists_all(self, temp_state_dir):
        """Test that adding multiple sessions persists all of them."""
        import mosaic.mosaic as mosaic_module
        from mosaic_config.config import MosaicConfig
        from mosaic_config.state_utils import read_state, StateIdentifiers

        # Create a test config with state location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)

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
        session1 = Session(plan=plan1, status=SessionStatus.TRAINING)
        session2 = Session(plan=plan2, status=SessionStatus.IDLE)

        # Set up global config and session manager
        from mosaic_config.state_manager import SessionStateManager
        session_manager = SessionStateManager(config)
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._session_manager", session_manager):
                # Add both sessions
                add_session(session1)
                add_session(session2)

                # Verify both sessions were added
                sessions = session_manager.get_sessions()
                assert len(sessions) == 2
                session_ids = [s.id for s in sessions]
                assert session1.id in session_ids
                assert session2.id in session_ids

                # Verify state was persisted with both sessions
                loaded_sessions = read_state(config, StateIdentifiers.SESSIONS, default=None)
                assert isinstance(loaded_sessions, list)
                assert len(loaded_sessions) == 2
                loaded_ids = [s.id for s in loaded_sessions]
                assert session1.id in loaded_ids
                assert session2.id in loaded_ids


class TestSanitizeFilename:
    """Test _sanitize_filename function."""

    def test_sanitize_filename_valid_name(self):
        """Test that valid filenames are unchanged."""
        assert _sanitize_filename("model_name") == "model_name"
        assert _sanitize_filename("my_model") == "my_model"
        assert _sanitize_filename("model123") == "model123"

    def test_sanitize_filename_spaces(self):
        """Test that spaces are replaced with underscores."""
        assert _sanitize_filename("my model") == "my_model"
        assert _sanitize_filename("model with spaces") == "model_with_spaces"
        assert _sanitize_filename("  model  ") == "model"

    def test_sanitize_filename_symbols(self):
        """Test that symbols are replaced with underscores."""
        assert _sanitize_filename("model@name") == "model_name"
        assert _sanitize_filename("model#123") == "model_123"
        assert _sanitize_filename("model$test") == "model_test"
        assert _sanitize_filename("model%file") == "model_file"
        assert _sanitize_filename("model&name") == "model_name"
        assert _sanitize_filename("model*test") == "model_test"
        assert _sanitize_filename("model+name") == "model_name"
        assert _sanitize_filename("model=test") == "model_test"

    def test_sanitize_filename_invalid_chars(self):
        """Test that invalid filename characters are replaced."""
        assert _sanitize_filename("model/name") == "model_name"
        assert _sanitize_filename("model\\name") == "model_name"
        assert _sanitize_filename("model:name") == "model_name"
        assert _sanitize_filename("model<name") == "model_name"
        assert _sanitize_filename("model>name") == "model_name"
        assert _sanitize_filename("model|name") == "model_name"
        assert _sanitize_filename("model?name") == "model_name"
        assert _sanitize_filename("model\"name") == "model_name"

    def test_sanitize_filename_leading_trailing_dots(self):
        """Test that leading and trailing dots are removed."""
        assert _sanitize_filename(".model") == "model"
        assert _sanitize_filename("model.") == "model"
        assert _sanitize_filename(".model.") == "model"
        assert _sanitize_filename("..model..") == "model"

    def test_sanitize_filename_empty(self):
        """Test that empty or all-invalid names get a default."""
        assert _sanitize_filename("") == "unnamed"
        assert _sanitize_filename("...") == "unnamed"
        assert _sanitize_filename("   ") == "unnamed"

    def test_sanitize_filename_long_name(self):
        """Test that very long names are truncated."""
        long_name = "a" * 300
        result = _sanitize_filename(long_name)
        assert len(result) == 255
        assert result == "a" * 255


class TestAddRemoveModel:
    """Test add_model and remove_model functions."""

    def test_add_model_adds_to_list_and_persists(self, temp_state_dir):
        """Test that add_model adds a model to the list and persists state."""
        import mosaic.mosaic as mosaic_module
        from mosaic_config.config import MosaicConfig
        from mosaic_config.state_utils import read_state, StateIdentifiers

        # Create a test config with state and models location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)
        config.models_location = str(temp_state_dir / "models")

        # Create a test model without binary
        model = Model(name="test_model", model_type=ModelType.CNN)

        # Set up global config and empty models list
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._models", []):
                # Add model
                add_model(model)

                # Verify model was added to the list
                assert len(mosaic_module._models) == 1
                assert mosaic_module._models[0].name == model.name

                # Verify state was persisted
                loaded_models = read_state(config, StateIdentifiers.MODELS, default=None)
                assert isinstance(loaded_models, list)
                assert len(loaded_models) == 1
                assert loaded_models[0].name == model.name

    def test_add_model_saves_binary_to_disk(self, temp_state_dir):
        """Test that add_model saves ONNX binary to disk and clears it from model."""
        import mosaic.mosaic as mosaic_module
        from pathlib import Path
        from mosaic_config.config import MosaicConfig

        # Create a test config with models location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)
        config.models_location = str(temp_state_dir / "models")

        # Create test binary data
        test_binary = b"fake_onnx_model_data_12345"

        # Create a test model with binary
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            binary_rep=test_binary,
        )

        # Set up global config and empty models list
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._models", []):
                # Add model
                add_model(model)

                # Verify model was added
                assert len(mosaic_module._models) == 1
                assert mosaic_module._models[0].name == "test_model"

                # Verify binary was cleared from model
                assert mosaic_module._models[0].binary_rep is None

                # Verify file_name was set
                assert mosaic_module._models[0].file_name == "test_model"

                # Verify file was saved to disk
                models_dir = Path(config.models_location)
                saved_file = models_dir / "test_model"
                assert saved_file.exists(), "Model file should be saved to disk"

                # Verify file content matches original binary
                with open(saved_file, 'rb') as f:
                    saved_content = f.read()
                assert saved_content == test_binary, "Saved file content should match original binary"

    def test_add_model_saves_to_onnx_location(self, temp_state_dir):
        """Test that add_model saves to onnx_location subdirectory when specified."""
        import mosaic.mosaic as mosaic_module
        from pathlib import Path
        from mosaic_config.config import MosaicConfig

        # Create a test config with models location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)
        config.models_location = str(temp_state_dir / "models")

        # Create test binary data
        test_binary = b"fake_onnx_model_data"

        # Create a test model with binary and onnx_location
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            onnx_location="onnx_models",
            binary_rep=test_binary,
        )

        # Set up global config and empty models list
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._models", []):
                # Add model
                add_model(model)

                # Verify file was saved to onnx_location subdirectory
                models_dir = Path(config.models_location)
                saved_file = models_dir / "onnx_models" / "test_model"
                assert saved_file.exists(), "Model file should be saved to onnx_location subdirectory"

                # Verify file content
                with open(saved_file, 'rb') as f:
                    saved_content = f.read()
                assert saved_content == test_binary

    def test_add_model_sanitizes_filename(self, temp_state_dir):
        """Test that add_model sanitizes the filename when saving."""
        import mosaic.mosaic as mosaic_module
        from pathlib import Path
        from mosaic_config.config import MosaicConfig

        # Create a test config with models location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)
        config.models_location = str(temp_state_dir / "models")

        # Create test binary data
        test_binary = b"fake_onnx_model_data"

        # Create a test model with spaces and symbols in name
        model = Model(
            name="my model@test#123",
            model_type=ModelType.CNN,
            binary_rep=test_binary,
        )

        # Set up global config and empty models list
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._models", []):
                # Add model
                add_model(model)

                # Verify file_name was set to sanitized version
                assert mosaic_module._models[0].file_name == "my_model_test_123"

                # Verify file was saved with sanitized name
                models_dir = Path(config.models_location)
                saved_file = models_dir / "my_model_test_123"
                assert saved_file.exists(), "Model file should be saved with sanitized name"

    def test_remove_model_removes_from_list_and_persists(self, temp_state_dir):
        """Test that remove_model removes a model by name and persists state."""
        import mosaic.mosaic as mosaic_module
        from mosaic_config.config import MosaicConfig
        from mosaic_config.state_utils import read_state, StateIdentifiers

        # Create a test config with state location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)
        config.models_location = str(temp_state_dir / "models")

        # Create test models
        model1 = Model(name="model1", model_type=ModelType.CNN)
        model2 = Model(name="model2", model_type=ModelType.BERT)

        # Set up global config and models list
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._models", [model1, model2]):
                # Remove model1
                result = remove_model("model1")

                # Verify removal was successful
                assert result is True
                assert len(mosaic_module._models) == 1
                assert mosaic_module._models[0].name == "model2"

                # Verify state was persisted
                loaded_models = read_state(config, StateIdentifiers.MODELS, default=None)
                assert isinstance(loaded_models, list)
                assert len(loaded_models) == 1
                assert loaded_models[0].name == "model2"

    def test_remove_model_returns_false_when_not_found(self, temp_state_dir):
        """Test that remove_model returns False when model name is not found."""
        import mosaic.mosaic as mosaic_module
        from mosaic_config.config import MosaicConfig

        # Create a test config with state location
        config = MosaicConfig()
        config.state_location = str(temp_state_dir)
        config.models_location = str(temp_state_dir / "models")

        # Create a test model
        model = Model(name="test_model", model_type=ModelType.CNN)

        # Set up global config and models list
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._models", [model]):
                # Try to remove non-existent model
                result = remove_model("non-existent-model")

                # Verify removal failed
                assert result is False
                assert len(mosaic_module._models) == 1
                assert mosaic_module._models[0].name == "test_model"


class TestModelTransferBetweenBeacons:
    """Test model transfer between beacons."""

    def test_model_transfer_saves_binary_and_strips_from_object(self, beacon_config_no_ssl, sender_config_no_ssl, temp_state_dir):
        """Test that when a model is sent between beacons, binary is saved and stripped."""
        import pickle
        import time
        from pathlib import Path
        from mosaic_comms.beacon import Beacon
        from mosaic_config.config import MosaicConfig
        from mosaic_config.state_utils import StateIdentifiers, read_state

        # Set up models location for receiver
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        models_dir = test_data_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Create receiver config with models_location
        receiver_config = MosaicConfig(
            host=sender_config_no_ssl.host,
            heartbeat_port=sender_config_no_ssl.heartbeat_port,
            comms_port=sender_config_no_ssl.comms_port,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            models_location=str(models_dir),
            state_location=str(temp_state_dir / "receiver"),
        )

        # Create test binary data
        test_binary = b"fake_onnx_model_binary_data_123456789"

        # Create a test model with binary
        model = Model(
            name="transferred_model",
            model_type=ModelType.CNN,
            binary_rep=test_binary,
        )

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create beacons
            sender_beacon = Beacon(beacon_config_no_ssl)
            receiver_beacon = Beacon(receiver_config)

            # Register add_model handler on receiver
            from mosaic.mosaic import _handle_add_model_command
            receiver_beacon.register("add_model", _handle_add_model_command)

            # Set up global config for receiver
            with patch("mosaic.mosaic._config", receiver_config):
                with patch("mosaic.mosaic._models", []):
                    # Start beacons
                    sender_beacon.start()
                    receiver_beacon.start()

                    try:
                        # Wait for listeners to start
                        time.sleep(1.0)

                        # Serialize model and send it
                        serialized_model = pickle.dumps(model)
                        response = sender_beacon.send_command(
                            host=receiver_config.host,
                            port=receiver_config.comms_port,
                            command="add_model",
                            payload=serialized_model,
                        )

                        # Verify response
                        assert response is not None
                        assert response["status"] == "success"

                        # Verify model was added to receiver's list
                        from mosaic.mosaic import _models
                        assert len(_models) == 1
                        received_model = _models[0]

                        # Verify model name
                        assert received_model.name == "transferred_model"

                        # Verify binary was stripped from model object
                        assert received_model.binary_rep is None, "Binary should be stripped from model object"

                        # Verify file_name was set
                        assert received_model.file_name == "transferred_model"

                        # Verify file was written to correct location
                        saved_file = models_dir / "transferred_model"
                        assert saved_file.exists(), "Model file should be saved to models_location"

                        # Verify file content matches original binary
                        with open(saved_file, 'rb') as f:
                            saved_content = f.read()
                        assert saved_content == test_binary, "Saved file content should match original binary"

                    finally:
                        sender_beacon.stop()
                        receiver_beacon.stop()
                        # Clean up
                        if saved_file.exists():
                            saved_file.unlink()

    def test_model_transfer_with_special_characters_in_name(self, beacon_config_no_ssl, sender_config_no_ssl, temp_state_dir):
        """Test that model names with special characters are sanitized correctly."""
        import pickle
        import time
        from pathlib import Path
        from mosaic_comms.beacon import Beacon
        from mosaic_config.config import MosaicConfig

        # Set up models location for receiver
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        models_dir = test_data_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Create receiver config with models_location
        receiver_config = MosaicConfig(
            host=sender_config_no_ssl.host,
            heartbeat_port=sender_config_no_ssl.heartbeat_port,
            comms_port=sender_config_no_ssl.comms_port,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            models_location=str(models_dir),
            state_location=str(temp_state_dir / "receiver"),
        )

        # Create test binary data
        test_binary = b"fake_onnx_model_binary"

        # Create a test model with special characters in name
        model = Model(
            name="my model@test#123",
            model_type=ModelType.CNN,
            binary_rep=test_binary,
        )

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create beacons
            sender_beacon = Beacon(beacon_config_no_ssl)
            receiver_beacon = Beacon(receiver_config)

            # Register add_model handler on receiver
            from mosaic.mosaic import _handle_add_model_command
            receiver_beacon.register("add_model", _handle_add_model_command)

            # Set up global config for receiver
            with patch("mosaic.mosaic._config", receiver_config):
                with patch("mosaic.mosaic._models", []):
                    # Start beacons
                    sender_beacon.start()
                    receiver_beacon.start()

                    try:
                        # Wait for listeners to start
                        time.sleep(1.0)

                        # Serialize model and send it
                        serialized_model = pickle.dumps(model)
                        response = sender_beacon.send_command(
                            host=receiver_config.host,
                            port=receiver_config.comms_port,
                            command="add_model",
                            payload=serialized_model,
                        )

                        # Verify response
                        assert response is not None
                        assert response["status"] == "success"

                        # Verify model was added
                        from mosaic.mosaic import _models
                        assert len(_models) == 1
                        received_model = _models[0]

                        # Verify original name is preserved
                        assert received_model.name == "my model@test#123"

                        # Verify file_name is sanitized
                        assert received_model.file_name == "my_model_test_123"

                        # Verify file was written with sanitized name
                        saved_file = models_dir / "my_model_test_123"
                        assert saved_file.exists(), "Model file should be saved with sanitized name"

                        # Verify file content
                        with open(saved_file, 'rb') as f:
                            saved_content = f.read()
                        assert saved_content == test_binary

                    finally:
                        sender_beacon.stop()
                        receiver_beacon.stop()
                        # Clean up
                        if saved_file.exists():
                            saved_file.unlink()

    def test_model_transfer_with_onnx_location(self, beacon_config_no_ssl, sender_config_no_ssl, temp_state_dir):
        """Test that model with onnx_location saves to correct subdirectory."""
        import pickle
        import time
        from pathlib import Path
        from mosaic_comms.beacon import Beacon
        from mosaic_config.config import MosaicConfig

        # Set up models location for receiver
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        models_dir = test_data_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Create receiver config with models_location
        receiver_config = MosaicConfig(
            host=sender_config_no_ssl.host,
            heartbeat_port=sender_config_no_ssl.heartbeat_port,
            comms_port=sender_config_no_ssl.comms_port,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            models_location=str(models_dir),
            state_location=str(temp_state_dir / "receiver"),
        )

        # Create test binary data
        test_binary = b"fake_onnx_model_binary"

        # Create a test model with onnx_location
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            onnx_location="onnx_models",
            binary_rep=test_binary,
        )

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create beacons
            sender_beacon = Beacon(beacon_config_no_ssl)
            receiver_beacon = Beacon(receiver_config)

            # Register add_model handler on receiver
            from mosaic.mosaic import _handle_add_model_command
            receiver_beacon.register("add_model", _handle_add_model_command)

            # Set up global config for receiver
            with patch("mosaic.mosaic._config", receiver_config):
                with patch("mosaic.mosaic._models", []):
                    # Start beacons
                    sender_beacon.start()
                    receiver_beacon.start()

                    try:
                        # Wait for listeners to start
                        time.sleep(1.0)

                        # Serialize model and send it
                        serialized_model = pickle.dumps(model)
                        response = sender_beacon.send_command(
                            host=receiver_config.host,
                            port=receiver_config.comms_port,
                            command="add_model",
                            payload=serialized_model,
                        )

                        # Verify response
                        assert response is not None
                        assert response["status"] == "success"

                        # Verify file was written to onnx_location subdirectory
                        saved_file = models_dir / "onnx_models" / "test_model"
                        assert saved_file.exists(), "Model file should be saved to onnx_location subdirectory"

                        # Verify file content
                        with open(saved_file, 'rb') as f:
                            saved_content = f.read()
                        assert saved_content == test_binary

                    finally:
                        sender_beacon.stop()
                        receiver_beacon.stop()
                        # Clean up
                        if saved_file.exists():
                            saved_file.unlink()
                        onnx_dir = models_dir / "onnx_models"
                        if onnx_dir.exists():
                            import shutil
                            shutil.rmtree(onnx_dir)

