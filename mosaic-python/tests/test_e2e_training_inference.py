"""End-to-end tests for training and inference across multiple mosaic nodes."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mosaic_comms.beacon import Beacon
from mosaic_config.config import Peer
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
import mosaic.mosaic as mosaic_module
from mosaic.mosaic import add_model
from mosaic.session_commands import (
    execute_use_session,
    execute_infer,
    initialize as init_session_commands,
)
from mosaic_planner.planner import plan_data_distribution
from mosaic_planner.model_planner import plan_model
from tests.conftest import create_test_config_with_state
from tests.create_dummy_data import create_dummy_images
from mosaic_model_runtime.predefined_models import create_resnet50_onnx
from mosaic_model_runtime.model_factory import create_resnet50_model


class TestE2ETrainingInference:
    """End-to-end tests for training and inference across 2 nodes."""
    
    def test_e2e_training_and_inference_two_nodes(self, temp_state_dir):
        """Test complete cycle: data/model distribution, training, and inference across 2 nodes."""
        # Set up test data directory
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        # Create dummy image data
        create_dummy_images()
        
        # Create configs for 2 beacons
        node1_state_dir = temp_state_dir / "node1"
        node2_state_dir = temp_state_dir / "node2"
        
        config1 = create_test_config_with_state(
            state_dir=node1_state_dir,
            host="127.0.0.1",
            heartbeat_port=7000,
            comms_port=7001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=30,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
            models_location=str(node1_state_dir / "models"),
        )
        
        config2 = create_test_config_with_state(
            state_dir=node2_state_dir,
            host="127.0.0.1",
            heartbeat_port=7002,
            comms_port=7003,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=30,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
            models_location=str(node2_state_dir / "models"),
        )
        
        # Add peers to each config
        config1.peers = [
            Peer(
                host=config2.host,
                heartbeat_port=config2.heartbeat_port,
                comms_port=config2.comms_port,
            )
        ]
        config2.peers = [
            Peer(
                host=config1.host,
                heartbeat_port=config1.heartbeat_port,
                comms_port=config1.comms_port,
            )
        ]
        
        # Mock stats collector for both nodes
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3, "memory_percent": 50.0}'
            mock_stats_class.return_value = mock_stats
            
            # Create beacons
            beacon1 = Beacon(config1)
            beacon2 = Beacon(config2)
            
            # Initialize mosaic module state for both nodes
            from mosaic_config.state_manager import SessionStateManager
            
            # Set up node1 (orchestrator node)
            mosaic_module._config = config1
            mosaic_module._beacon = beacon1
            mosaic_module._models = []
            mosaic_module._session_manager = SessionStateManager(config1)
            
            # Set up node2 (worker node) - in real scenario this would be separate process
            # For testing, we patch mosaic.mosaic when node2's handlers are called
            # The handlers will use the patched values
            
            try:
                # Start beacons
                beacon1.start()
                beacon2.start()
                
                # Wait for heartbeats to establish connection
                time.sleep(3)
                
                # Step 1: Create ResNet50 model on node1
                models_dir1 = Path(config1.models_location)
                models_dir1.mkdir(parents=True, exist_ok=True)
                
                # Create ResNet50 ONNX model
                onnx_filename = create_resnet50_onnx(models_dir1)
                model = create_resnet50_model(models_dir1)
                model.onnx_location = ""  # Root of models_location
                model.file_name = onnx_filename
                
                # Load model binary
                model_file = models_dir1 / onnx_filename
                with open(model_file, 'rb') as f:
                    model.binary_rep = f.read()
                
                # Add model to node1
                add_model(model)
                
                # Step 2: Create data
                file_def = FileDefinition(
                    location='dummy_images',
                    data_type=DataType.IMAGE,
                    is_segmentable=True,
                    input_shape=[3, 224, 224],
                )
                data = Data(
                    file_definitions=[file_def],
                    batch_size_hint=2,
                    data_loading_hints={"shuffle": False, "num_workers": 0},
                )
                
                # Step 3: Collect stats from both nodes
                stats_data = beacon1.collect_stats(include_self=True)
                assert len(stats_data) >= 2, "Should have stats from both nodes"
                
                # Step 4: Create distribution plan
                # Use capacity_fraction instead of allocated_samples
                distribution_plan = [
                    {
                        "host": config1.host,
                        "comms_port": config1.comms_port,
                        "capacity_fraction": 0.5,  # 50% to node1
                        "effective_score": 5.0,
                    },
                    {
                        "host": config2.host,
                        "comms_port": config2.comms_port,
                        "capacity_fraction": 0.5,  # 50% to node2
                        "effective_score": 5.0,
                    },
                ]
                
                # Step 5: Plan data distribution
                from mosaic_config.state import Project
                project = Project(name="test_project", config=config1, data=data)
                
                # Patch _get_file_metadata to use test helper that actually counts images
                from tests.test_data_segmentation import _get_file_metadata_for_test
                
                with patch('mosaic_planner.planner._get_file_metadata', _get_file_metadata_for_test):
                    plan = plan_data_distribution(
                        distribution_plan=distribution_plan,
                        project=project,
                        stats_data=stats_data,
                        model=model,
                    )
                
                assert plan.data_segmentation_plan is not None
                assert len(plan.data_segmentation_plan) == 2
                
                # Step 6: Plan model distribution
                # plan_model returns a dict with compression metadata, but doesn't modify the plan
                # The plan already has distribution_plan from plan_data_distribution
                model_plan_metadata = plan_model(model, plan, config1)
                assert model_plan_metadata is not None
                # model_plan_metadata is a dict with compression info per node
                assert isinstance(model_plan_metadata, dict)
                
                # Step 7: Create session using the plan (which already has distribution_plan)
                session = Session(
                    plan=plan,
                    data=data,
                    model_id=model.id,
                    status=SessionStatus.IDLE,
                )
                session.model = model
                
                # Add session to node1's session manager
                mosaic_module._session_manager.add_session(session)
                
                # Step 8: Execute data distribution plan
                beacon1.execute_data_plan(plan, data, session)
                
                # Wait for data transfer
                time.sleep(2)
                
                # Step 9: Execute model distribution plan
                beacon1.execute_model_plan(session, model)
                
                # Wait for model transfer
                time.sleep(2)
                
                # Step 10: Train model with reduced epochs
                # Initialize session commands for node1
                output_lines = []
                def output_fn(text: str) -> None:
                    output_lines.append(text)
                
                init_session_commands(beacon1, mosaic_module._session_manager, mosaic_module._models, config1)
                
                # Patch hyperparameters to use 1 epoch for speed
                from mosaic_planner import training_hyperparameters
                original_hyperparams = training_hyperparameters.DEFAULT_CNN_HYPERPARAMETERS.copy()
                training_hyperparameters.DEFAULT_CNN_HYPERPARAMETERS["epochs"] = 1
                
                try:
                    # Execute training
                    from mosaic.session_commands import execute_training
                    result = execute_training(session.id, output_fn, timeout=120.0, check_interval=1.0)
                    
                    # Verify training completed
                    assert result is not None
                    assert result.status == SessionStatus.COMPLETE
                    
                    # Check that training_nodes state is updated
                    assert "training_nodes" in result.data_distribution_state
                    training_nodes = result.data_distribution_state["training_nodes"]
                    assert len(training_nodes) >= 1  # At least one node completed
                finally:
                    # Restore original hyperparameters
                    training_hyperparameters.DEFAULT_CNN_HYPERPARAMETERS = original_hyperparams
                
                # Step 11: Wait a bit for training to fully complete
                time.sleep(2)
                
                # Step 12: Run federated inference
                # First, use the session
                output_lines.clear()
                execute_use_session(output_fn, session.id)
                
                # Verify session is set
                from mosaic.session_commands import _current_session_id
                assert _current_session_id == session.id
                
                # Create a test input image file for inference
                # create_dummy_images() creates images in subdirectories (dir_0/, dir_1/, etc.)
                # So we need to search recursively
                images_dir = test_data_dir / "dummy_images"
                image_files = list(images_dir.rglob("*.png"))  # Use rglob for recursive search
                
                if not image_files:
                    # If no images found, create a test image directly in dummy_images/
                    test_input_image = images_dir / "test_inference.png"
                    from PIL import Image
                    test_img = Image.new('RGB', (224, 224), color='red')
                    test_img.save(str(test_input_image))
                else:
                    # Use first available image
                    test_input_image = image_files[0]
                
                # Run inference (onnxruntime should be installed as a dependency)
                output_lines.clear()
                execute_infer(output_fn, str(test_input_image))
                
                # Verify inference completed successfully
                output_text = "".join(output_lines)
                assert "Error" not in output_text or "No predictions" not in output_text
                
                # Check for inference result indicators
                assert (
                    "Inference Complete" in output_text or
                    "Inference Result" in output_text or
                    "Prediction" in output_text or
                    "Nodes participated" in output_text
                ), f"Inference should complete. Output: {output_text}"
                
            finally:
                # Cleanup
                try:
                    beacon1.stop()
                    beacon2.stop()
                except Exception:
                    pass
                
                # Reset module state
                mosaic_module._config = None
                mosaic_module._beacon = None
                mosaic_module._models = []
                mosaic_module._session_manager = None

