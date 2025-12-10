"""Unit tests for mosaic_config.state_manager module."""

import pickle
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from mosaic_config.state_manager import HeartbeatStateManager, SessionStateManager
from mosaic_config.state import ReceiveHeartbeatStatus, SendHeartbeatStatus
from mosaic_config.config import MosaicConfig
from mosaic_config.state_utils import StateIdentifiers, read_state, save_state
from mosaic_config.state import Data, DataType, FileDefinition, Model, ModelType, Plan, Session, SessionStatus
from tests.conftest import create_test_config_with_state


class TestSessionStateManagerInitialization:
    """Test SessionStateManager initialization and loading."""

    def test_init_loads_existing_sessions(self, temp_state_dir):
        """Test that SessionStateManager loads existing sessions on initialization."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Create test sessions and save them manually
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
        sessions_list = [session1, session2]
        
        # Save sessions manually
        from mosaic_config.state_utils import save_state
        save_state(config, sessions_list, StateIdentifiers.SESSIONS)
        
        # Initialize manager - should load the sessions
        manager = SessionStateManager(config)
        
        # Verify sessions were loaded
        loaded_sessions = manager.get_sessions()
        assert len(loaded_sessions) == 2
        session_ids = [s.id for s in loaded_sessions]
        assert session1.id in session_ids
        assert session2.id in session_ids

    def test_init_creates_empty_list_when_no_state_file(self, temp_state_dir):
        """Test that SessionStateManager creates empty list when no state file exists."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Initialize manager - should create empty list
        manager = SessionStateManager(config)
        
        # Verify empty list
        sessions = manager.get_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) == 0

    def test_init_handles_invalid_state_file_gracefully(self, temp_state_dir):
        """Test that SessionStateManager handles invalid state file gracefully."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Create an invalid pickle file
        invalid_file = temp_state_dir / "sessions.pkl"
        invalid_file.write_text("not valid pickle data", encoding="utf-8")
        
        # Initialize manager - should handle error and create empty list
        manager = SessionStateManager(config)
        
        # Verify empty list was created
        sessions = manager.get_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) == 0


class TestSessionStateManagerReadWrite:
    """Test that read and write of lists of sessions work correctly."""

    def test_write_and_read_single_session(self, temp_state_dir):
        """Test that a single session can be written and read back exactly."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Create a test session
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1", "cpu": 4.0}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(
            plan=plan,
            status=SessionStatus.TRAINING,
            time_started=1234567890,
            time_ended=-1,
        )
        
        # Save model to state so it can be loaded later
        save_state(config, [model], StateIdentifiers.MODELS)
        
        # Add session (triggers save)
        manager.add_session(session)
        
        # Create model loader function
        def model_loader(model_id: str) -> Optional[Model]:
            models = read_state(config, StateIdentifiers.MODELS, default=[])
            for m in models:
                if m.id == model_id:
                    return m
            return None
        
        # Create a new manager to read from disk with model loader
        manager2 = SessionStateManager(config, model_loader=model_loader)
        loaded_sessions = manager2.get_sessions()
        
        # Verify exactly one session
        assert len(loaded_sessions) == 1
        loaded_session = loaded_sessions[0]
        
        # Verify all attributes match exactly
        assert loaded_session.id == session.id
        assert loaded_session.status == session.status
        assert loaded_session.time_started == session.time_started
        assert loaded_session.time_ended == session.time_ended
        assert loaded_session.plan.id == session.plan.id
        # Verify model_id is preserved
        assert loaded_session.plan.model_id == session.plan.model_id
        # Verify model can be loaded lazily
        assert loaded_session.plan.model is not None
        assert loaded_session.plan.model.name == session.plan.model.name

    def test_write_and_read_multiple_sessions(self, temp_state_dir):
        """Test that multiple sessions can be written and read back exactly."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Create multiple test sessions
        model = Model(name="test_model", model_type=ModelType.CNN)
        sessions = []
        for i in range(5):
            plan = Plan(
                stats_data=[{"host": f"node{i}", "cpu": float(i)}],
                distribution_plan=[{"host": f"node{i}", "allocated_samples": i * 10}],
                model=model,
            )
            session = Session(
                plan=plan,
                status=SessionStatus.TRAINING if i % 2 == 0 else SessionStatus.IDLE,
                time_started=1234567890 + i,
                time_ended=-1 if i < 3 else 1234567900 + i,
            )
            sessions.append(session)
            manager.add_session(session)
        
        # Create a new manager to read from disk
        manager2 = SessionStateManager(config)
        loaded_sessions = manager2.get_sessions()
        
        # Verify all sessions were loaded
        assert len(loaded_sessions) == 5
        
        # Verify each session matches exactly
        original_ids = {s.id for s in sessions}
        loaded_ids = {s.id for s in loaded_sessions}
        assert original_ids == loaded_ids
        
        # Verify each session's attributes match
        for original_session in sessions:
            loaded_session = next(s for s in loaded_sessions if s.id == original_session.id)
            assert loaded_session.id == original_session.id
            assert loaded_session.status == original_session.status
            assert loaded_session.time_started == original_session.time_started
            assert loaded_session.time_ended == original_session.time_ended
            assert loaded_session.plan.id == original_session.plan.id

    def test_write_and_read_sessions_with_data(self, temp_state_dir):
        """Test that sessions with Data objects can be written and read back exactly."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Create a session with Data
        model = Model(name="test_model", model_type=ModelType.CNN)
        file_def = FileDefinition(
            location="test_data/file.csv",
            data_type=DataType.CSV,
            is_segmentable=True,
        )
        data = Data(file_definitions=[file_def], training_task_type="classification")
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan, data=data, status=SessionStatus.TRAINING)
        
        # Add session (triggers save)
        manager.add_session(session)
        
        # Create a new manager to read from disk
        manager2 = SessionStateManager(config)
        loaded_sessions = manager2.get_sessions()
        
        # Verify session was loaded with data
        assert len(loaded_sessions) == 1
        loaded_session = loaded_sessions[0]
        assert loaded_session.data is not None
        assert len(loaded_session.data.file_definitions) == 1
        assert loaded_session.data.file_definitions[0].location == file_def.location
        assert loaded_session.data.file_definitions[0].data_type == file_def.data_type

    def test_write_and_read_sessions_preserves_all_attributes(self, temp_state_dir):
        """Test that all session attributes are preserved after write/read."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Create a session with all attributes set
        model = Model(name="test_model", model_type=ModelType.CNN, id="model-123")
        plan = Plan(
            stats_data=[{"host": "node1", "memory": 8192}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
            id="plan-456",
        )
        file_def = FileDefinition(
            location="test_data/file.csv",
            data_type=DataType.CSV,
            is_segmentable=True,
        )
        data = Data(file_definitions=[file_def])
        session = Session(
            plan=plan,
            data=data,
            model=model,
            status=SessionStatus.COMPLETE,
            time_started=1234567890,
            time_ended=1234567999,
            id="session-789",
        )
        
        # Save model to state so it can be loaded later
        save_state(config, [model], StateIdentifiers.MODELS)
        
        # Add session (triggers save)
        manager.add_session(session)
        
        # Create model loader function
        def model_loader(model_id: str) -> Optional[Model]:
            models = read_state(config, StateIdentifiers.MODELS, default=[])
            for m in models:
                if m.id == model_id:
                    return m
            return None
        
        # Create a new manager to read from disk with model loader
        manager2 = SessionStateManager(config, model_loader=model_loader)
        loaded_sessions = manager2.get_sessions()
        
        # Verify all attributes
        assert len(loaded_sessions) == 1
        loaded_session = loaded_sessions[0]
        
        # Session attributes
        assert loaded_session.id == "session-789"
        assert loaded_session.status == SessionStatus.COMPLETE
        assert loaded_session.time_started == 1234567890
        assert loaded_session.time_ended == 1234567999
        
        # Plan attributes
        assert loaded_session.plan.id == "plan-456"
        assert len(loaded_session.plan.stats_data) == 1
        assert loaded_session.plan.stats_data[0]["host"] == "node1"
        
        # Model attributes - verify model_id is preserved
        assert loaded_session.plan.model_id == "model-123"
        # Verify model can be loaded lazily
        assert loaded_session.plan.model is not None
        assert loaded_session.plan.model.id == "model-123"
        assert loaded_session.plan.model.name == "test_model"
        
        # Data attributes
        assert loaded_session.data is not None
        assert len(loaded_session.data.file_definitions) == 1

    def test_write_and_read_empty_list(self, temp_state_dir):
        """Test that an empty list of sessions can be written and read back."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Clear sessions (triggers save of empty list)
        manager.clear_sessions()
        
        # Create a new manager to read from disk
        manager2 = SessionStateManager(config)
        loaded_sessions = manager2.get_sessions()
        
        # Verify empty list
        assert isinstance(loaded_sessions, list)
        assert len(loaded_sessions) == 0

    def test_write_and_read_after_multiple_operations(self, temp_state_dir):
        """Test that sessions persist correctly after multiple add/remove operations."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Create and add multiple sessions
        model = Model(name="test_model", model_type=ModelType.CNN)
        session_ids = []
        for i in range(3):
            plan = Plan(
                stats_data=[{"host": f"node{i}"}],
                distribution_plan=[{"host": f"node{i}", "allocated_samples": i * 10}],
                model=model,
            )
            session = Session(plan=plan, status=SessionStatus.TRAINING)
            session_ids.append(session.id)
            manager.add_session(session)
        
        # Remove one session
        manager.remove_session(session_ids[1])
        
        # Create a new manager to read from disk
        manager2 = SessionStateManager(config)
        loaded_sessions = manager2.get_sessions()
        
        # Verify only 2 sessions remain
        assert len(loaded_sessions) == 2
        loaded_ids = {s.id for s in loaded_sessions}
        assert session_ids[0] in loaded_ids
        assert session_ids[1] not in loaded_ids
        assert session_ids[2] in loaded_ids

    def test_write_and_read_sessions_are_identical(self, temp_state_dir):
        """Test that written and read sessions are exactly equal (deep comparison)."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Create a complex session
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1", "cpu": 4.0, "memory": 8192}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        file_def = FileDefinition(
            location="test_data/file.csv",
            data_type=DataType.CSV,
            is_segmentable=True,
            input_shape=[224, 224, 3],
        )
        data = Data(
            file_definitions=[file_def],
            training_task_type="classification",
            batch_size_hint=32,
        )
        session = Session(
            plan=plan,
            data=data,
            model=model,
            status=SessionStatus.TRAINING,
            time_started=1234567890,
        )
        
        # Save model to state so it can be loaded later
        save_state(config, [model], StateIdentifiers.MODELS)
        
        # Add session (triggers save)
        manager.add_session(session)
        
        # Create model loader function
        def model_loader(model_id: str) -> Optional[Model]:
            models = read_state(config, StateIdentifiers.MODELS, default=[])
            for m in models:
                if m.id == model_id:
                    return m
            return None
        
        # Serialize original session for comparison
        original_pickled = pickle.dumps(session)
        
        # Create a new manager to read from disk with model loader
        manager2 = SessionStateManager(config, model_loader=model_loader)
        loaded_sessions = manager2.get_sessions()
        
        # Verify one session
        assert len(loaded_sessions) == 1
        loaded_session = loaded_sessions[0]
        
        # Serialize loaded session for comparison
        loaded_pickled = pickle.dumps(loaded_session)
        
        # The pickled representations should be identical (or at least equivalent)
        # We'll do a more detailed comparison
        assert loaded_session.id == session.id
        assert loaded_session.status == session.status
        assert loaded_session.time_started == session.time_started
        assert loaded_session.time_ended == session.time_ended
        assert loaded_session.plan.id == session.plan.id
        # Verify model_id is preserved
        assert loaded_session.plan.model_id == session.plan.model_id
        # Verify model can be loaded lazily
        assert loaded_session.plan.model is not None
        assert loaded_session.plan.model.name == session.plan.model.name
        assert loaded_session.data is not None
        assert len(loaded_session.data.file_definitions) == 1
        assert loaded_session.data.file_definitions[0].location == file_def.location
        assert loaded_session.data.file_definitions[0].data_type == file_def.data_type
        assert loaded_session.data.file_definitions[0].input_shape == file_def.input_shape


class TestSessionStateManagerOperations:
    """Test SessionStateManager operations (add, remove, update, get)."""

    def test_get_sessions_returns_copy(self, temp_state_dir):
        """Test that get_sessions returns a copy, not the internal list."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Add a session
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan)
        manager.add_session(session)
        
        # Get sessions and modify the returned list
        sessions = manager.get_sessions()
        assert len(sessions) == 1
        sessions.append("not a session")  # This should not affect internal state
        
        # Get sessions again - should still be 1
        sessions2 = manager.get_sessions()
        assert len(sessions2) == 1

    def test_get_session_by_id(self, temp_state_dir):
        """Test getting a session by ID."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Add multiple sessions
        model = Model(name="test_model", model_type=ModelType.CNN)
        session_ids = []
        for i in range(3):
            plan = Plan(
                stats_data=[{"host": f"node{i}"}],
                distribution_plan=[{"host": f"node{i}", "allocated_samples": i * 10}],
                model=model,
            )
            session = Session(plan=plan)
            session_ids.append(session.id)
            manager.add_session(session)
        
        # Get session by ID
        found_session = manager.get_session_by_id(session_ids[1])
        assert found_session is not None
        assert found_session.id == session_ids[1]
        
        # Get non-existent session
        not_found = manager.get_session_by_id("non-existent-id")
        assert not_found is None

    def test_update_session(self, temp_state_dir):
        """Test updating a session's attributes."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Add a session
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.IDLE)
        manager.add_session(session)
        
        # Update session
        success = manager.update_session(session.id, status=SessionStatus.COMPLETE, time_ended=1234567999)
        assert success is True
        
        # Verify update persisted
        manager2 = SessionStateManager(config)
        loaded_sessions = manager2.get_sessions()
        assert len(loaded_sessions) == 1
        assert loaded_sessions[0].status == SessionStatus.COMPLETE
        assert loaded_sessions[0].time_ended == 1234567999

    def test_update_nonexistent_session(self, temp_state_dir):
        """Test updating a non-existent session returns False."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Try to update non-existent session
        success = manager.update_session("non-existent-id", status=SessionStatus.COMPLETE)
        assert success is False

    def test_clear_sessions(self, temp_state_dir):
        """Test clearing all sessions."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Add multiple sessions
        model = Model(name="test_model", model_type=ModelType.CNN)
        for i in range(3):
            plan = Plan(
                stats_data=[{"host": f"node{i}"}],
                distribution_plan=[{"host": f"node{i}", "allocated_samples": i * 10}],
                model=model,
            )
            session = Session(plan=plan)
            manager.add_session(session)
        
        # Clear sessions
        manager.clear_sessions()
        
        # Verify cleared
        assert len(manager.get_sessions()) == 0
        
        # Verify persisted
        manager2 = SessionStateManager(config)
        assert len(manager2.get_sessions()) == 0


class TestMosaicStartupSessionLoading:
    """Test that sessions are loaded correctly when mosaic process starts."""

    def test_mosaic_startup_loads_sessions(self, temp_state_dir):
        """Test that mosaic startup process loads sessions from state correctly."""
        import mosaic.mosaic as mosaic_module
        
        # Create test config
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=6000,
            comms_port=6001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )
        
        # Create test sessions and save them to state
        model = Model(name="test_model", model_type=ModelType.CNN)
        # Save model to state so it can be loaded later
        save_state(config, [model], StateIdentifiers.MODELS)
        
        sessions = []
        for i in range(3):
            plan = Plan(
                stats_data=[{"host": f"node{i}", "cpu": float(i * 10)}],
                distribution_plan=[{"host": f"node{i}", "allocated_samples": i * 10}],
                model=model,
            )
            session = Session(
                plan=plan,
                status=SessionStatus.TRAINING if i % 2 == 0 else SessionStatus.IDLE,
                time_started=1234567890 + i,
            )
            sessions.append(session)
        
        # Save sessions to state manually (simulating a previous run)
        save_state(config, sessions, StateIdentifiers.SESSIONS)
        
        # Mock the heavy parts of mosaic startup
        mock_beacon = MagicMock()
        mock_beacon.register = MagicMock()
        mock_beacon.start = MagicMock()
        
        with patch("mosaic.mosaic.read_config", return_value=config), \
             patch("mosaic.mosaic.Beacon", return_value=mock_beacon), \
             patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class, \
             patch("mosaic.repl_commands.initialize") as mock_init_repl, \
             patch("sys.exit") as mock_exit:
            
            # Mock StatsCollector
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            # Reset global state
            mosaic_module._config = None
            mosaic_module._session_manager = None
            mosaic_module._beacon = None
            # Load models from state (simulating what main() does)
            loaded_models = read_state(config, StateIdentifiers.MODELS, default=[])
            mosaic_module._models = loaded_models if isinstance(loaded_models, list) else []
            
            # Simulate the startup sequence from main() up to session loading
            # Step 1: Create MosaicConfig (already done via mock)
            mosaic_module._config = config
            
            # Step 1.5: Initialize SessionStateManager (loads sessions automatically)
            # Create model loader function (simulating what main() does)
            def load_model_by_id(model_id: str) -> Optional[Model]:
                for m in mosaic_module._models:
                    if m.id == model_id:
                        return m
                return None
            
            mosaic_module._session_manager = SessionStateManager(config, model_loader=load_model_by_id)
            
            # Verify sessions were loaded
            loaded_sessions = mosaic_module._session_manager.get_sessions()
            assert len(loaded_sessions) == 3, "Should load 3 sessions from state"
            
            # Verify all sessions match
            original_ids = {s.id for s in sessions}
            loaded_ids = {s.id for s in loaded_sessions}
            assert original_ids == loaded_ids, "Session IDs should match"
            
            # Verify session attributes are preserved
            for original_session in sessions:
                loaded_session = next(s for s in loaded_sessions if s.id == original_session.id)
                assert loaded_session.id == original_session.id
                assert loaded_session.status == original_session.status
                assert loaded_session.time_started == original_session.time_started
                # Verify model_id is preserved
                assert loaded_session.plan.model_id == original_session.plan.model_id
                # Verify model can be loaded lazily
                assert loaded_session.plan.model is not None
                assert loaded_session.plan.model.name == original_session.plan.model.name

    def test_mosaic_startup_with_no_existing_sessions(self, temp_state_dir):
        """Test that mosaic startup handles no existing sessions gracefully."""
        import mosaic.mosaic as mosaic_module
        
        # Create test config with empty state
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=6002,
            comms_port=6003,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )
        
        # Mock the heavy parts
        mock_beacon = MagicMock()
        mock_beacon.register = MagicMock()
        mock_beacon.start = MagicMock()
        
        with patch("mosaic.mosaic.read_config", return_value=config), \
             patch("mosaic.mosaic.Beacon", return_value=mock_beacon), \
             patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class, \
             patch("mosaic.repl_commands.initialize") as mock_init_repl, \
             patch("sys.exit") as mock_exit:
            
            # Mock StatsCollector
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            # Reset global state
            mosaic_module._config = None
            mosaic_module._session_manager = None
            mosaic_module._beacon = None
            mosaic_module._models = []
            
            # Simulate startup
            mosaic_module._config = config
            mosaic_module._session_manager = SessionStateManager(config)
            
            # Verify empty list was created
            loaded_sessions = mosaic_module._session_manager.get_sessions()
            assert isinstance(loaded_sessions, list)
            assert len(loaded_sessions) == 0, "Should have empty list when no sessions exist"

    def test_mosaic_startup_sessions_accessible_via_global_manager(self, temp_state_dir):
        """Test that sessions loaded on startup are accessible via the global session manager."""
        import mosaic.mosaic as mosaic_module
        
        # Create test config
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=6004,
            comms_port=6005,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )
        
        # Create and save a session
        model = Model(name="test_model", model_type=ModelType.CNN)
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan, status=SessionStatus.TRAINING, id="test-session-123")
        save_state(config, [session], StateIdentifiers.SESSIONS)
        
        # Mock the heavy parts
        mock_beacon = MagicMock()
        mock_beacon.register = MagicMock()
        mock_beacon.start = MagicMock()
        
        with patch("mosaic.mosaic.read_config", return_value=config), \
             patch("mosaic.mosaic.Beacon", return_value=mock_beacon), \
             patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class, \
             patch("mosaic.repl_commands.initialize") as mock_init_repl, \
             patch("sys.exit") as mock_exit:
            
            # Mock StatsCollector
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            # Reset global state
            mosaic_module._config = None
            mosaic_module._session_manager = None
            mosaic_module._beacon = None
            mosaic_module._models = []
            
            # Simulate startup
            mosaic_module._config = config
            mosaic_module._session_manager = SessionStateManager(config)
            
            # Verify session is accessible via global manager
            assert mosaic_module._session_manager is not None
            loaded_sessions = mosaic_module._session_manager.get_sessions()
            assert len(loaded_sessions) == 1
            assert loaded_sessions[0].id == "test-session-123"
            
            # Verify session can be retrieved by ID
            found_session = mosaic_module._session_manager.get_session_by_id("test-session-123")
            assert found_session is not None
            assert found_session.id == "test-session-123"
            assert found_session.status == SessionStatus.TRAINING


class TestHeartbeatStateManagerInitialization:
    """Test HeartbeatStateManager initialization and loading."""

    def test_init_loads_existing_heartbeat_statuses(self, temp_state_dir):
        """Test that HeartbeatStateManager loads existing heartbeat statuses on initialization."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Create test heartbeat statuses and save them manually
        send_status1 = SendHeartbeatStatus(
            host="192.168.1.1",
            heartbeat_port=5000,
            last_time_sent=1234567890,
            connection_status="ok",
        )
        send_status2 = SendHeartbeatStatus(
            host="192.168.1.2",
            heartbeat_port=5001,
            last_time_sent=1234567891,
            connection_status="timeout",
        )
        receive_status1 = ReceiveHeartbeatStatus(
            host="192.168.1.3",
            heartbeat_port=5002,
            comms_port=5003,
            last_time_received=1234567892,
            connection_status="online",
        )
        
        send_statuses = {("192.168.1.1", 5000): send_status1, ("192.168.1.2", 5001): send_status2}
        receive_statuses = {("192.168.1.3", 5002): receive_status1}
        
        # Save heartbeat statuses manually
        save_state(config, send_statuses, StateIdentifiers.SEND_HEARTBEAT_STATUSES)
        save_state(config, receive_statuses, StateIdentifiers.RECEIVE_HEARTBEAT_STATUSES)
        
        # Initialize manager - should load the statuses
        manager = HeartbeatStateManager(config)
        
        # Verify statuses were loaded
        loaded_send = manager.get_send_heartbeat_statuses()
        loaded_receive = manager.get_receive_heartbeat_statuses()
        assert len(loaded_send) == 2
        assert len(loaded_receive) == 1
        assert ("192.168.1.1", 5000) in loaded_send
        assert ("192.168.1.2", 5001) in loaded_send
        assert ("192.168.1.3", 5002) in loaded_receive

    def test_init_creates_empty_dicts_when_no_state_file(self, temp_state_dir):
        """Test that HeartbeatStateManager creates empty dicts when no state file exists."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Initialize manager - should create empty dicts
        manager = HeartbeatStateManager(config)
        
        # Verify empty dicts
        send_statuses = manager.get_send_heartbeat_statuses()
        receive_statuses = manager.get_receive_heartbeat_statuses()
        assert isinstance(send_statuses, dict)
        assert isinstance(receive_statuses, dict)
        assert len(send_statuses) == 0
        assert len(receive_statuses) == 0

    def test_init_handles_invalid_state_file_gracefully(self, temp_state_dir):
        """Test that HeartbeatStateManager handles invalid state file gracefully."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Create an invalid pickle file
        invalid_file = temp_state_dir / "send_heartbeat_statuses.pkl"
        invalid_file.write_text("not valid pickle data", encoding="utf-8")
        
        # Initialize manager - should handle error and create empty dict
        manager = HeartbeatStateManager(config)
        
        # Verify empty dict was created
        send_statuses = manager.get_send_heartbeat_statuses()
        assert isinstance(send_statuses, dict)
        assert len(send_statuses) == 0


class TestHeartbeatStateManagerReadWrite:
    """Test that read and write of heartbeat statuses work correctly."""

    def test_write_and_read_single_send_heartbeat_status(self, temp_state_dir):
        """Test that a single send heartbeat status can be written and read back exactly."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = HeartbeatStateManager(config)
        
        # Create a test send heartbeat status
        send_status = SendHeartbeatStatus(
            host="192.168.1.1",
            heartbeat_port=5000,
            last_time_sent=1234567890,
            connection_status="ok",
        )
        
        # Update status (triggers internal update)
        manager.update_send_heartbeat_status("192.168.1.1", 5000, send_status)
        
        # Save state
        manager.save_heartbeat_statuses()
        
        # Create a new manager to read from disk
        manager2 = HeartbeatStateManager(config)
        loaded_send = manager2.get_send_heartbeat_statuses()
        
        # Verify exactly one status
        assert len(loaded_send) == 1
        loaded_status = loaded_send[("192.168.1.1", 5000)]
        
        # Verify all attributes match exactly
        assert loaded_status.host == send_status.host
        assert loaded_status.heartbeat_port == send_status.heartbeat_port
        assert loaded_status.last_time_sent == send_status.last_time_sent
        assert loaded_status.connection_status == send_status.connection_status

    def test_write_and_read_single_receive_heartbeat_status(self, temp_state_dir):
        """Test that a single receive heartbeat status can be written and read back exactly."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = HeartbeatStateManager(config)
        
        # Create a test receive heartbeat status
        receive_status = ReceiveHeartbeatStatus(
            host="192.168.1.2",
            heartbeat_port=5001,
            comms_port=5002,
            last_time_received=1234567891,
            connection_status="online",
            stats_payload={"cpu_percent": 45.3},
            delay=1000000,  # 1ms in nanoseconds
        )
        
        # Update status
        manager.update_receive_heartbeat_status("192.168.1.2", 5001, receive_status)
        
        # Save state
        manager.save_heartbeat_statuses()
        
        # Create a new manager to read from disk
        manager2 = HeartbeatStateManager(config)
        loaded_receive = manager2.get_receive_heartbeat_statuses()
        
        # Verify exactly one status
        assert len(loaded_receive) == 1
        loaded_status = loaded_receive[("192.168.1.2", 5001)]
        
        # Verify all attributes match exactly
        assert loaded_status.host == receive_status.host
        assert loaded_status.heartbeat_port == receive_status.heartbeat_port
        assert loaded_status.comms_port == receive_status.comms_port
        assert loaded_status.last_time_received == receive_status.last_time_received
        assert loaded_status.connection_status == receive_status.connection_status
        assert loaded_status.stats_payload == receive_status.stats_payload
        assert loaded_status.delay == receive_status.delay

    def test_write_and_read_multiple_heartbeat_statuses(self, temp_state_dir):
        """Test that multiple heartbeat statuses can be written and read back exactly."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = HeartbeatStateManager(config)
        
        # Create multiple test statuses
        send_statuses = []
        receive_statuses = []
        for i in range(5):
            send_status = SendHeartbeatStatus(
                host=f"192.168.1.{i+1}",
                heartbeat_port=5000 + i,
                last_time_sent=1234567890 + i,
                connection_status="ok" if i % 2 == 0 else "timeout",
            )
            receive_status = ReceiveHeartbeatStatus(
                host=f"192.168.1.{i+10}",
                heartbeat_port=6000 + i,
                comms_port=6001 + i,
                last_time_received=1234567890 + i,
                connection_status="online" if i % 2 == 0 else "stale",
            )
            send_statuses.append(send_status)
            receive_statuses.append(receive_status)
            manager.update_send_heartbeat_status(send_status.host, send_status.heartbeat_port, send_status)
            manager.update_receive_heartbeat_status(receive_status.host, receive_status.heartbeat_port, receive_status)
        
        # Save state
        manager.save_heartbeat_statuses()
        
        # Create a new manager to read from disk
        manager2 = HeartbeatStateManager(config)
        loaded_send = manager2.get_send_heartbeat_statuses()
        loaded_receive = manager2.get_receive_heartbeat_statuses()
        
        # Verify all statuses were loaded
        assert len(loaded_send) == 5
        assert len(loaded_receive) == 5
        
        # Verify each status matches exactly
        for send_status in send_statuses:
            key = (send_status.host, send_status.heartbeat_port)
            loaded_status = loaded_send[key]
            assert loaded_status.host == send_status.host
            assert loaded_status.heartbeat_port == send_status.heartbeat_port
            assert loaded_status.last_time_sent == send_status.last_time_sent
            assert loaded_status.connection_status == send_status.connection_status
        
        for receive_status in receive_statuses:
            key = (receive_status.host, receive_status.heartbeat_port)
            loaded_status = loaded_receive[key]
            assert loaded_status.host == receive_status.host
            assert loaded_status.heartbeat_port == receive_status.heartbeat_port
            assert loaded_status.comms_port == receive_status.comms_port
            assert loaded_status.last_time_received == receive_status.last_time_received
            assert loaded_status.connection_status == receive_status.connection_status

    def test_get_send_heartbeat_status(self, temp_state_dir):
        """Test getting a send heartbeat status by host and port."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = HeartbeatStateManager(config)
        
        # Add a status
        send_status = SendHeartbeatStatus(
            host="192.168.1.1",
            heartbeat_port=5000,
            last_time_sent=1234567890,
            connection_status="ok",
        )
        manager.update_send_heartbeat_status("192.168.1.1", 5000, send_status)
        
        # Get status
        found_status = manager.get_send_heartbeat_status("192.168.1.1", 5000)
        assert found_status is not None
        assert found_status.host == "192.168.1.1"
        assert found_status.heartbeat_port == 5000
        
        # Get non-existent status
        not_found = manager.get_send_heartbeat_status("192.168.1.99", 9999)
        assert not_found is None

    def test_get_receive_heartbeat_status(self, temp_state_dir):
        """Test getting a receive heartbeat status by host and port."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = HeartbeatStateManager(config)
        
        # Add a status
        receive_status = ReceiveHeartbeatStatus(
            host="192.168.1.2",
            heartbeat_port=5001,
            comms_port=5002,
            last_time_received=1234567891,
            connection_status="online",
        )
        manager.update_receive_heartbeat_status("192.168.1.2", 5001, receive_status)
        
        # Get status
        found_status = manager.get_receive_heartbeat_status("192.168.1.2", 5001)
        assert found_status is not None
        assert found_status.host == "192.168.1.2"
        assert found_status.heartbeat_port == 5001
        
        # Get non-existent status
        not_found = manager.get_receive_heartbeat_status("192.168.1.99", 9999)
        assert not_found is None

    def test_clear_heartbeat_statuses(self, temp_state_dir):
        """Test clearing all heartbeat statuses."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = HeartbeatStateManager(config)
        
        # Add multiple statuses
        for i in range(3):
            send_status = SendHeartbeatStatus(
                host=f"192.168.1.{i+1}",
                heartbeat_port=5000 + i,
                last_time_sent=1234567890 + i,
                connection_status="ok",
            )
            manager.update_send_heartbeat_status(send_status.host, send_status.heartbeat_port, send_status)
        
        # Clear statuses
        manager.clear_heartbeat_statuses()
        
        # Verify cleared
        assert len(manager.get_send_heartbeat_statuses()) == 0
        assert len(manager.get_receive_heartbeat_statuses()) == 0
        
        # Verify persisted
        manager2 = HeartbeatStateManager(config)
        assert len(manager2.get_send_heartbeat_statuses()) == 0
        assert len(manager2.get_receive_heartbeat_statuses()) == 0

    def test_write_and_read_heartbeat_statuses_are_identical(self, temp_state_dir):
        """Test that written and read heartbeat statuses are exactly equal."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = HeartbeatStateManager(config)
        
        # Create complex statuses
        send_status = SendHeartbeatStatus(
            host="192.168.1.1",
            heartbeat_port=5000,
            last_time_sent=1234567890,
            connection_status="ok",
        )
        receive_status = ReceiveHeartbeatStatus(
            host="192.168.1.2",
            heartbeat_port=5001,
            comms_port=5002,
            last_time_received=1234567891,
            connection_status="online",
            stats_payload={"cpu_percent": 45.3, "ram_percent": 67.8, "gpus": [{"gpu_id": 0, "utilization": 85.5}]},
            delay=1000000,
        )
        
        manager.update_send_heartbeat_status("192.168.1.1", 5000, send_status)
        manager.update_receive_heartbeat_status("192.168.1.2", 5001, receive_status)
        
        # Save state
        manager.save_heartbeat_statuses()
        
        # Create a new manager to read from disk
        manager2 = HeartbeatStateManager(config)
        loaded_send = manager2.get_send_heartbeat_statuses()
        loaded_receive = manager2.get_receive_heartbeat_statuses()
        
        # Verify statuses match
        assert len(loaded_send) == 1
        assert len(loaded_receive) == 1
        
        loaded_send_status = loaded_send[("192.168.1.1", 5000)]
        loaded_receive_status = loaded_receive[("192.168.1.2", 5001)]
        
        assert loaded_send_status.host == send_status.host
        assert loaded_send_status.heartbeat_port == send_status.heartbeat_port
        assert loaded_send_status.last_time_sent == send_status.last_time_sent
        assert loaded_send_status.connection_status == send_status.connection_status
        
        assert loaded_receive_status.host == receive_status.host
        assert loaded_receive_status.heartbeat_port == receive_status.heartbeat_port
        assert loaded_receive_status.comms_port == receive_status.comms_port
        assert loaded_receive_status.last_time_received == receive_status.last_time_received
        assert loaded_receive_status.connection_status == receive_status.connection_status
        assert loaded_receive_status.stats_payload == receive_status.stats_payload
        assert loaded_receive_status.delay == receive_status.delay


class TestLazyModelLoading:
    """Test that Model objects are not persisted with Sessions/Plans and are loaded lazily."""

    def test_model_objects_not_saved_with_session(self, temp_state_dir):
        """Test that Model objects are not saved when Sessions are persisted."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Create a model and save it separately
        model = Model(name="test_model", model_type=ModelType.CNN, id="model-123")
        save_state(config, [model], StateIdentifiers.MODELS)
        
        # Create a session with the model
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan, model=model)
        
        # Add session (should save only model_id, not Model object)
        manager.add_session(session)
        
        # Read the raw pickle file to verify Model objects are not in it
        import pickle
        sessions_file = temp_state_dir / "sessions.pkl"
        assert sessions_file.exists()
        
        with open(sessions_file, "rb") as f:
            loaded_data = pickle.load(f)
        
        # Verify it's a list of sessions
        assert isinstance(loaded_data, list)
        assert len(loaded_data) == 1
        
        loaded_session = loaded_data[0]
        # Verify model_id is present
        assert hasattr(loaded_session, 'model_id')
        assert loaded_session.model_id == "model-123"
        # Verify Model object is NOT present (should be None)
        assert loaded_session._model is None
        assert loaded_session.model is None  # Should be None without loader
        
        # Verify plan also has model_id but not Model object
        assert hasattr(loaded_session.plan, 'model_id')
        assert loaded_session.plan.model_id == "model-123"
        assert loaded_session.plan._model is None
        assert loaded_session.plan.model is None  # Should be None without loader

    def test_model_id_preserved_when_saving(self, temp_state_dir):
        """Test that model_id is correctly preserved when saving Sessions."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        manager = SessionStateManager(config)
        
        # Create models with specific IDs
        model1 = Model(name="model1", model_type=ModelType.CNN, id="model-id-1")
        model2 = Model(name="model2", model_type=ModelType.TRANSFORMER, id="model-id-2")
        save_state(config, [model1, model2], StateIdentifiers.MODELS)
        
        # Create sessions with models
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
        session1 = Session(plan=plan1, model=model1)
        session2 = Session(plan=plan2, model=model2)
        
        manager.add_session(session1)
        manager.add_session(session2)
        
        # Create new manager to load from disk
        def model_loader(model_id: str) -> Optional[Model]:
            models = read_state(config, StateIdentifiers.MODELS, default=[])
            for m in models:
                if m.id == model_id:
                    return m
            return None
        
        manager2 = SessionStateManager(config, model_loader=model_loader)
        loaded_sessions = manager2.get_sessions()
        
        # Verify model_ids are preserved
        assert len(loaded_sessions) == 2
        assert loaded_sessions[0].model_id == "model-id-1"
        assert loaded_sessions[0].plan.model_id == "model-id-1"
        assert loaded_sessions[1].model_id == "model-id-2"
        assert loaded_sessions[1].plan.model_id == "model-id-2"

    def test_models_loaded_lazily_when_accessed(self, temp_state_dir):
        """Test that Model objects are loaded lazily only when accessed."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Create and save a model
        model = Model(name="test_model", model_type=ModelType.CNN, id="model-456")
        save_state(config, [model], StateIdentifiers.MODELS)
        
        # Create a session with the model
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan, model=model)
        
        # Save session
        manager = SessionStateManager(config)
        manager.add_session(session)
        
        # Track model loader calls
        loader_calls = []
        
        def tracked_model_loader(model_id: str) -> Optional[Model]:
            loader_calls.append(model_id)
            models = read_state(config, StateIdentifiers.MODELS, default=[])
            for m in models:
                if m.id == model_id:
                    return m
            return None
        
        # Create new manager with tracked loader
        manager2 = SessionStateManager(config, model_loader=tracked_model_loader)
        loaded_sessions = manager2.get_sessions()
        
        # Initially, loader should not have been called
        assert len(loader_calls) == 0
        assert loaded_sessions[0]._model is None
        assert loaded_sessions[0].plan._model is None
        
        # Access session.model - should trigger lazy load
        loaded_model = loaded_sessions[0].model
        assert len(loader_calls) == 1
        assert loader_calls[0] == "model-456"
        assert loaded_model is not None
        assert loaded_model.id == "model-456"
        assert loaded_model.name == "test_model"
        
        # Access plan.model - should trigger lazy load
        plan_model = loaded_sessions[0].plan.model
        assert len(loader_calls) == 2  # Called again for plan
        assert plan_model is not None
        assert plan_model.id == "model-456"
        
        # Accessing again should not call loader (cached)
        cached_model = loaded_sessions[0].model
        assert len(loader_calls) == 2  # No new call
        assert cached_model is loaded_model  # Same object

    def test_model_binaries_saved_separately(self, temp_state_dir):
        """Test that model binaries are saved separately and not in session state."""
        from pathlib import Path
        
        config = create_test_config_with_state(state_dir=temp_state_dir)
        models_dir = Path(temp_state_dir) / "models"
        models_dir.mkdir(exist_ok=True)
        config.models_location = str(models_dir)
        
        # Create a model with binary data
        model_binary = b"fake_onnx_model_binary_data_12345"
        model = Model(
            name="test_model",
            model_type=ModelType.CNN,
            id="model-789",
            binary_rep=model_binary,
        )
        
        # Save model (this should save binary to models_location)
        from mosaic.mosaic import add_model
        import mosaic.mosaic as mosaic_module
        
        with patch("mosaic.mosaic._config", config):
            with patch("mosaic.mosaic._models", []):
                add_model(model)
        
        # Verify model binary was saved to models directory
        # add_model saves using sanitized model name and sets file_name to that name
        # The sanitized name for "test_model" is "test_model"
        assert model.file_name is not None, "add_model should set file_name"
        model_file = models_dir / model.file_name
        assert model_file.exists(), f"Model file should exist at {model_file}. File name was set to: {model.file_name}"
        with open(model_file, "rb") as f:
            saved_binary = f.read()
        assert saved_binary == model_binary
        
        # Create a session with the model
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model=model,
        )
        session = Session(plan=plan, model=model)
        
        # Save session
        manager = SessionStateManager(config)
        manager.add_session(session)
        
        # Verify session pickle file does NOT contain model binary
        import pickle
        sessions_file = temp_state_dir / "sessions.pkl"
        with open(sessions_file, "rb") as f:
            sessions_data = pickle.load(f)
        
        # Check that model binary is not in the pickle data
        sessions_str = str(sessions_data)
        # The binary data should not appear in the sessions file
        assert b"fake_onnx_model_binary_data" not in pickle.dumps(sessions_data)
        
        # Verify model_id is present
        assert sessions_data[0].model_id == "model-789"
        assert sessions_data[0].plan.model_id == "model-789"

    def test_model_loader_returns_none_for_invalid_id(self, temp_state_dir):
        """Test that model loader returns None for invalid model IDs."""
        config = create_test_config_with_state(state_dir=temp_state_dir)
        
        # Create a session with a model ID that doesn't exist
        plan = Plan(
            stats_data=[{"host": "node1"}],
            distribution_plan=[{"host": "node1", "allocated_samples": 10}],
            model_id="non-existent-model-id",
        )
        session = Session(plan=plan, model_id="non-existent-model-id")
        
        # Save session
        manager = SessionStateManager(config)
        manager.add_session(session)
        
        # Create model loader that returns None for invalid IDs
        def model_loader(model_id: str) -> Optional[Model]:
            return None  # Model not found
        
        # Load session
        manager2 = SessionStateManager(config, model_loader=model_loader)
        loaded_sessions = manager2.get_sessions()
        
        # Accessing model should return None
        assert loaded_sessions[0].model is None
        assert loaded_sessions[0].plan.model is None
        assert loaded_sessions[0].model_id == "non-existent-model-id"
        assert loaded_sessions[0].plan.model_id == "non-existent-model-id"

