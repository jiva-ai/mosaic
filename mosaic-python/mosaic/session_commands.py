"""Session management commands for Mosaic REPL."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from mosaic_config.config import MosaicConfig, read_config
from mosaic_config.state import (
    Data,
    DataType,
    FileDefinition,
    Model,
    ModelType,
    Plan,
    Project,
    Session,
    SessionStatus,
)
from mosaic_planner import plan_dynamic_weighted_batches, plan_static_weighted_shards
from mosaic_planner.model_planner import plan_model
from mosaic_planner.planner import plan_data_distribution

logger = logging.getLogger(__name__)

# Module-level references (set via initialize() function)
_beacon: Optional[Any] = None
_session_manager: Optional[Any] = None
_models: List[Model] = []
_config: Optional[MosaicConfig] = None
_input_fn: Optional[Callable[[str], str]] = None  # Function to get user input


def initialize(
    beacon: Any,
    session_manager: Any,
    models: List[Model],
    config: MosaicConfig,
    input_fn: Optional[Callable[[str], str]] = None,
) -> None:
    """
    Initialize the session_commands module with required dependencies.
    
    Args:
        beacon: Beacon instance
        session_manager: SessionStateManager instance
        models: List of loaded models
        config: MosaicConfig instance
        input_fn: Optional function to get user input (defaults to input())
    """
    global _beacon, _session_manager, _models, _config, _input_fn
    _beacon = beacon
    _session_manager = session_manager
    _models = models
    _config = config
    _input_fn = input_fn or input


def _discover_datasets(data_location: str, max_depth: int = 2) -> List[Dict[str, Any]]:
    """
    Discover available datasets in the data directory.
    
    Performs a quick, shallow search to find potential datasets.
    
    Args:
        data_location: Path to data directory
        max_depth: Maximum directory depth to search (default: 2)
    
    Returns:
        List of dataset information dictionaries
    """
    datasets = []
    data_path = Path(data_location)
    
    if not data_path.exists() or not data_path.is_dir():
        return datasets
    
    try:
        # Quick scan of top-level directories and files
        for item in data_path.iterdir():
            if item.is_dir():
                # Check if directory contains data files
                dataset_info = {
                    "name": item.name,
                    "path": str(item.relative_to(data_path)),
                    "type": "directory",
                    "files": [],
                }
                
                # Quick check for common data file patterns (limit to avoid slow scans)
                file_count = 0
                for file_item in item.iterdir():
                    if file_count >= 10:  # Limit to first 10 files for speed
                        break
                    if file_item.is_file():
                        ext = file_item.suffix.lower()
                        if ext in [".jpg", ".jpeg", ".png", ".csv", ".txt", ".jsonl", ".wav", ".flac"]:
                            dataset_info["files"].append(file_item.name)
                            file_count += 1
                
                if dataset_info["files"]:
                    dataset_info["file_count"] = len(list(item.rglob("*"))) if file_count > 0 else 0
                    datasets.append(dataset_info)
            elif item.is_file():
                # Single file dataset
                ext = item.suffix.lower()
                if ext in [".csv", ".txt", ".jsonl"]:
                    datasets.append({
                        "name": item.name,
                        "path": str(item.relative_to(data_path)),
                        "type": "file",
                        "files": [item.name],
                        "file_count": 1,
                    })
    except (OSError, PermissionError) as e:
        logger.warning(f"Error discovering datasets: {e}")
    
    return datasets


def _get_predefined_models() -> List[Dict[str, str]]:
    """
    Get list of predefined models that can be created.
    
    Returns:
        List of model info dictionaries
    """
    return [
        {"name": "resnet50", "type": "CNN", "description": "ResNet-50 for image classification"},
        {"name": "resnet101", "type": "CNN", "description": "ResNet-101 for image classification"},
        {"name": "wav2vec2", "type": "WAV2VEC", "description": "Wav2Vec2 for speech recognition"},
        {"name": "gpt-neo", "type": "TRANSFORMER", "description": "GPT-Neo for text generation"},
        {"name": "gcn-ogbn-arxiv", "type": "GNN", "description": "GCN for graph node classification"},
        {"name": "biggan", "type": "VAE", "description": "BigGAN for image generation"},
        {"name": "ppo", "type": "RL", "description": "PPO for reinforcement learning"},
    ]


def _format_plan_summary(plan: Plan, max_nodes: int = 20) -> str:
    """
    Format a plan summary for display, limiting output for large plans.
    
    Args:
        plan: Plan object to summarize
        max_nodes: Maximum number of nodes to show in detail
    
    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Distribution Plan Summary")
    lines.append("=" * 60)
    
    if plan.distribution_plan:
        lines.append(f"\nTotal Nodes: {len(plan.distribution_plan)}")
        lines.append("\nNode Allocations (showing first {}):".format(min(max_nodes, len(plan.distribution_plan))))
        lines.append("-" * 60)
        
        # Show first max_nodes nodes in detail
        for i, node in enumerate(plan.distribution_plan[:max_nodes]):
            host = node.get("host", "unknown")
            port = node.get("comms_port", 0)
            fraction = node.get("capacity_fraction", 0.0)
            allocated = node.get("allocated_samples") or node.get("allocated_batches", 0)
            lines.append(f"  {i+1:3d}. {host:20s}:{port:5d} - {fraction*100:5.1f}% ({allocated:6d} samples/batches)")
        
        if len(plan.distribution_plan) > max_nodes:
            remaining = len(plan.distribution_plan) - max_nodes
            lines.append(f"\n  ... and {remaining} more nodes (total: {len(plan.distribution_plan)})")
    
    if plan.data_segmentation_plan:
        lines.append(f"\nData Segmentation Plan:")
        lines.append("-" * 60)
        total_segments = sum(len(m.get("segments", [])) for m in plan.data_segmentation_plan)
        lines.append(f"  Total machines: {len(plan.data_segmentation_plan)}")
        lines.append(f"  Total segments: {total_segments}")
        
        # Show first few machines with more detail
        show_machines = min(5, len(plan.data_segmentation_plan))
        lines.append(f"\n  First {show_machines} machines:")
        for i, machine in enumerate(plan.data_segmentation_plan[:show_machines]):
            host = machine.get("host", "unknown")
            comms_port = machine.get("comms_port", 0)
            fraction = machine.get("fraction", 0.0)
            segments = machine.get("segments", [])
            lines.append(f"    {i+1}. {host}:{comms_port} - {fraction*100:.1f}% ({len(segments)} segments)")
        
        if len(plan.data_segmentation_plan) > show_machines:
            lines.append(f"  ... and {len(plan.data_segmentation_plan) - show_machines} more machines")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def create_session_interactive(output_fn: Callable[[str], None]) -> Optional[Session]:
    """
    Interactive session creation with Q&A flow.
    
    Uses simple text-based prompts. Can be enhanced with Textual dialogs in the future.
    
    Args:
        output_fn: Function to call with output text
    
    Returns:
        Created Session object, or None if cancelled
    """
    return _create_session_simple(output_fn)


def _create_session_simple(output_fn: Callable[[str], None]) -> Optional[Session]:
    """
    Simple text-based session creation flow with Q&A.
    
    Args:
        output_fn: Function to call with output text
    
    Returns:
        Created Session object, or None if cancelled
    """
    global _models  # Declare global at the start of the function
    
    if _beacon is None or _session_manager is None or _config is None:
        output_fn("Error: System not fully initialized\n")
        return None
    
    try:
        output_fn("\n" + "=" * 60 + "\n")
        output_fn("Creating New Session\n")
        output_fn("=" * 60 + "\n\n")
        
        # Step 1: Select Model
        output_fn("Step 1: Select a Model\n")
        output_fn("-" * 60 + "\n")
        output_fn("Choose a model for this session:\n\n")
        
        model_options: List[Tuple[str, Optional[Model], Optional[Dict[str, str]]]] = []
        
        # Show loaded models
        if _models:
            output_fn("Loaded Models:\n")
            for i, model in enumerate(_models):
                model_type_str = model.model_type.value if model.model_type else "Unknown"
                output_fn(f"  {i+1}. {model.name} ({model_type_str})\n")
                model_options.append((f"{i+1}", model, None))
            output_fn("\n")
        
        # Show predefined models
        predefined = _get_predefined_models()
        start_idx = len(_models) + 1
        if predefined:
            output_fn("Predefined Models (will be created on selection):\n")
            for i, model_info in enumerate(predefined):
                output_fn(f"  {start_idx + i}. {model_info['name']} - {model_info['description']}\n")
                model_options.append((f"{start_idx + i}", None, model_info))
            output_fn("\n")
        
        cancel_idx = len(model_options) + 1
        output_fn(f"  {cancel_idx}. Cancel\n")
        
        # Get model selection
        selection = _input_fn("\nEnter model number: ").strip()
        if not selection or selection == str(cancel_idx):
            output_fn("Session creation cancelled.\n")
            return None
        
        selected_model: Optional[Model] = None
        selected_option = None
        for opt_idx, model, model_info in model_options:
            if selection == opt_idx:
                if model is not None:
                    selected_model = model
                else:
                    selected_option = model_info
                break
        
        if selected_model is None and selected_option is None:
            output_fn("Invalid selection.\n")
            return None
        
        # Create predefined model if needed
        if selected_model is None and selected_option:
            output_fn(f"\nCreating {selected_option['name']} model...\n")
            try:
                from mosaic_model_runtime.model_factory import (
                    create_biggan_model,
                    create_gcn_model,
                    create_gpt_neo_model,
                    create_ppo_model,
                    create_resnet101_model,
                    create_resnet50_model,
                    create_wav2vec2_model,
                )
                
                model_name = selected_option['name']
                models_path = Path(_config.models_location) if _config.models_location else Path("models")
                models_path.mkdir(parents=True, exist_ok=True)
                
                if model_name == "resnet50":
                    selected_model = create_resnet50_model(models_path)
                elif model_name == "resnet101":
                    selected_model = create_resnet101_model(models_path)
                elif model_name == "wav2vec2":
                    selected_model = create_wav2vec2_model(models_path)
                elif model_name == "gpt-neo":
                    selected_model = create_gpt_neo_model(models_path)
                elif model_name == "gcn-ogbn-arxiv":
                    selected_model = create_gcn_model(models_path)
                elif model_name == "biggan":
                    selected_model = create_biggan_model(models_path)
                elif model_name == "ppo":
                    selected_model = create_ppo_model(models_path)
                else:
                    output_fn(f"Unknown predefined model: {model_name}\n")
                    return None
                
                # Add to models list
                _models.append(selected_model)
                # Save models state
                try:
                    from mosaic_config.state_utils import StateIdentifiers, save_state
                    save_state(_config, _models, StateIdentifiers.MODELS)
                except Exception as e:
                    logger.warning(f"Failed to save models state: {e}")
                output_fn(f"Model {selected_model.name} created and added.\n")
            except Exception as e:
                output_fn(f"Error creating model: {e}\n")
                logger.error(f"Error creating predefined model: {e}", exc_info=True)
                return None
        
        if selected_model is None:
            output_fn("Error: Could not get model.\n")
            return None
        
        # Step 2: Select Dataset
        output_fn("\n" + "-" * 60 + "\n")
        output_fn("Step 2: Select a Dataset\n")
        output_fn("-" * 60 + "\n")
        output_fn("Searching for datasets in data directory...\n")
        
        datasets = _discover_datasets(_config.data_location if _config.data_location else "")
        
        if not datasets:
            output_fn("No datasets found in data directory.\n")
            output_fn("You can manually specify a dataset path.\n")
            dataset_path = _input_fn("Enter dataset path (relative to data directory, or 'cancel'): ").strip()
            if not dataset_path or dataset_path.lower() == "cancel":
                output_fn("Session creation cancelled.\n")
                return None
            
            # Create a simple file definition
            data_type_str = _input_fn("Enter data type (image/audio/text/csv/dir/graph/rl): ").strip().lower()
            try:
                data_type = DataType(data_type_str)
            except ValueError:
                output_fn(f"Invalid data type: {data_type_str}\n")
                return None
            
            file_def = FileDefinition(location=dataset_path, data_type=data_type)
            data = Data(file_definitions=[file_def])
        else:
            output_fn(f"\nFound {len(datasets)} dataset(s):\n\n")
            for i, dataset in enumerate(datasets):
                file_count = dataset.get("file_count", len(dataset.get("files", [])))
                output_fn(f"  {i+1}. {dataset['name']} ({dataset['type']}, ~{file_count} files)\n")
            
            cancel_idx = len(datasets) + 1
            output_fn(f"  {cancel_idx}. Cancel\n")
            
            selection = _input_fn("\nEnter dataset number: ").strip()
            if not selection or selection == str(cancel_idx):
                output_fn("Session creation cancelled.\n")
                return None
            
            try:
                idx = int(selection) - 1
                if idx < 0 or idx >= len(datasets):
                    output_fn("Invalid selection.\n")
                    return None
                selected_dataset = datasets[idx]
            except ValueError:
                output_fn("Invalid selection.\n")
                return None
            
            # Infer data type from files
            dataset_path = selected_dataset['path']
            files = selected_dataset.get('files', [])
            data_type = DataType.DIR  # Default
            
            if files:
                ext = Path(files[0]).suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    data_type = DataType.IMAGE
                elif ext in ['.wav', '.flac']:
                    data_type = DataType.AUDIO
                elif ext in ['.txt', '.jsonl']:
                    data_type = DataType.TEXT
                elif ext == '.csv':
                    data_type = DataType.CSV
            
            file_def = FileDefinition(location=dataset_path, data_type=data_type)
            data = Data(file_definitions=[file_def])
        
        # Step 3: Create Plans
        output_fn("\n" + "-" * 60 + "\n")
        output_fn("Step 3: Create Distribution Plans\n")
        output_fn("-" * 60 + "\n")
        output_fn("Collecting peer statistics...\n")
        
        stats_data = _beacon.collect_stats()
        if not stats_data:
            output_fn("Error: No peer statistics available. Ensure peers are connected.\n")
            return None
        
        total_samples = len(stats_data)
        output_fn(f"Found {len(stats_data)} peer(s).\n")
        
        # Create data distribution plan
        output_fn("Calculating data distribution...\n")
        distribution_plan = plan_static_weighted_shards(stats_data, total_samples=total_samples)
        
        if not distribution_plan:
            output_fn("Error: Could not create distribution plan.\n")
            return None
        
        # Create project for data planning
        project = Project(name=f"session_{int(time.time())}", config=_config, data=data)
        
        # Create data segmentation plan
        output_fn("Planning data segmentation...\n")
        plan = plan_data_distribution(
            distribution_plan=distribution_plan,
            project=project,
            stats_data=stats_data,
            model=selected_model,
            overlap=0,
        )
        
        # Create model distribution plan
        output_fn("Planning model distribution...\n")
        try:
            model_plan = plan_model(selected_model, plan, _config)
            # Store model plan in plan (we can extend Plan if needed, or store separately)
            # For now, model planning is done but not stored in Plan object
        except Exception as e:
            output_fn(f"Warning: Model planning failed: {e}\n")
            logger.warning(f"Model planning failed: {e}", exc_info=True)
        
        # Show plan summary
        output_fn("\n" + _format_plan_summary(plan) + "\n")
        
        # Step 4: Confirmation
        output_fn("\n" + "-" * 60 + "\n")
        output_fn("Step 4: Confirm and Execute\n")
        output_fn("-" * 60 + "\n")
        confirm = _input_fn("Execute data and model distribution plans? (yes/no): ").strip().lower()
        
        if confirm not in ['yes', 'y']:
            output_fn("Session creation cancelled.\n")
            return None
        
        # Create session with IDLE status
        session = Session(
            plan=plan,
            data=data,
            model=selected_model,
            status=SessionStatus.IDLE,
        )
        
        # Add session to manager before distribution so state can be persisted
        _session_manager.add_session(session)
        
        # Set status to RUNNING
        session.status = SessionStatus.RUNNING
        _session_manager.update_session(session.id, status=SessionStatus.RUNNING)
        
        # Execute plans with progress indication and state tracking
        output_fn("\nExecuting data distribution plan...\n")
        try:
            if plan.data_segmentation_plan:
                total_machines = len(plan.data_segmentation_plan)
                output_fn(f"Distributing data to {total_machines} machine(s)...\n")
                
                # Execute the plan (now accepts session for state tracking)
                _beacon.execute_data_plan(plan, data, session)
                
                # Check final status
                if session.status == SessionStatus.ERROR_CORRECTION:
                    output_fn("⚠ Some distributions required retries. Checking final status...\n")
                
                # Show completion with progress bar
                progress_bar = "=" * 20
                successful = sum(
                    1 for m in session.data_distribution_state.get("machines", {}).values()
                    if m.get("status") == "success"
                )
                output_fn(f"Progress: [{progress_bar}] 100% - {successful}/{total_machines} machines succeeded\n")
                
                if session.status == SessionStatus.ERROR:
                    error_msg = session.data_distribution_state.get("final_error", "Unknown error")
                    output_fn(f"✗ Data distribution failed: {error_msg}\n")
                else:
                    output_fn("✓ Data distribution completed successfully.\n")
            else:
                output_fn("Warning: No data segmentation plan to execute.\n")
        except Exception as e:
            output_fn(f"✗ Error executing data plan: {e}\n")
            logger.error(f"Error executing data plan: {e}", exc_info=True)
            session.status = SessionStatus.ERROR
            _session_manager.update_session(session.id, status=SessionStatus.ERROR)
            return session
        
        output_fn("\nExecuting model distribution plan...\n")
        try:
            if plan.distribution_plan:
                total_nodes = len(plan.distribution_plan)
                output_fn(f"Distributing model to {total_nodes} node(s)...\n")
                
                _beacon.execute_model_plan(session, selected_model)
                
                # Check final status
                if session.status == SessionStatus.ERROR_CORRECTION:
                    output_fn("⚠ Some distributions required retries. Checking final status...\n")
                
                # Show completion with progress bar
                progress_bar = "=" * 20
                successful = sum(
                    1 for n in session.model_distribution_state.get("nodes", {}).values()
                    if n.get("status") == "success"
                )
                output_fn(f"Progress: [{progress_bar}] 100% - {successful}/{total_nodes} nodes succeeded\n")
                
                if session.status == SessionStatus.ERROR:
                    error_msg = session.model_distribution_state.get("final_error", "Unknown error")
                    output_fn(f"✗ Model distribution failed: {error_msg}\n")
                else:
                    output_fn("✓ Model distribution completed successfully.\n")
            else:
                output_fn("Warning: No distribution plan to execute.\n")
        except Exception as e:
            output_fn(f"✗ Error executing model plan: {e}\n")
            logger.error(f"Error executing model plan: {e}", exc_info=True)
            session.status = SessionStatus.ERROR
            _session_manager.update_session(session.id, status=SessionStatus.ERROR)
            return session
        
        # Final persistence (session already added, just update status)
        _session_manager.update_session(session.id, status=session.status)
        
        output_fn("\n" + "=" * 60 + "\n")
        output_fn(f"Session created successfully!\n")
        output_fn(f"Session ID: {session.id}\n")
        output_fn(f"Status: {session.status.value}\n")
        output_fn("=" * 60 + "\n\n")
        
        return session
        
    except KeyboardInterrupt:
        output_fn("\n\nSession creation cancelled by user.\n")
        return None
    except Exception as e:
        output_fn(f"\nError creating session: {e}\n")
        logger.error(f"Error creating session: {e}", exc_info=True)
        return None


def execute_create_session(output_fn: Callable[[str], None]) -> None:
    """
    Execute create_session command.
    
    Args:
        output_fn: Function to call with output text
    """
    try:
        session = create_session_interactive(output_fn)
        if session:
            # Success message is already printed in _create_session_simple
            pass
        else:
            # Cancellation message is already printed in _create_session_simple
            pass
    except Exception as e:
        output_fn(f"Error creating session: {e}\n")
        logger.error(f"Error in execute_create_session: {e}", exc_info=True)


def execute_delete_session(output_fn: Callable[[str], None], session_id: Optional[str] = None) -> None:
    """
    Execute delete_session command.
    
    Args:
        output_fn: Function to call with output text
        session_id: Optional session ID to delete (if not provided, will prompt)
    """
    if _session_manager is None:
        output_fn("Error: Session manager not initialized\n")
        return
    
    if session_id is None:
        # List sessions and let user choose
        sessions = _session_manager.get_sessions()
        if not sessions:
            output_fn("No sessions available to delete.\n")
            return
        
        output_fn("\n" + "=" * 60 + "\n")
        output_fn("Delete Session\n")
        output_fn("=" * 60 + "\n\n")
        output_fn("Available Sessions:\n")
        output_fn("-" * 60 + "\n")
        for i, session in enumerate(sessions):
            status = session.status.value if isinstance(session.status, SessionStatus) else str(session.status)
            model_id = session.model_id or "No model"
            output_fn(f"  {i+1}. {session.id[:8]}... - Status: {status}, Model: {model_id[:20]}...\n")
        
        cancel_idx = len(sessions) + 1
        output_fn(f"\n  {cancel_idx}. Cancel\n")
        
        try:
            selection = _input_fn("\nEnter session number to delete: ").strip()
            if not selection or selection == str(cancel_idx):
                output_fn("Deletion cancelled.\n")
                return
            
            idx = int(selection) - 1
            if idx < 0 or idx >= len(sessions):
                output_fn("Invalid selection.\n")
                return
            
            session_id = sessions[idx].id
        except (ValueError, IndexError) as e:
            output_fn(f"Invalid selection: {e}\n")
            return
        except KeyboardInterrupt:
            output_fn("\nDeletion cancelled.\n")
            return
    
    # Delete the specified session
    if _session_manager.remove_session(session_id):
        output_fn(f"\nSession {session_id} deleted successfully.\n")
    else:
        output_fn(f"\nSession {session_id} not found.\n")

