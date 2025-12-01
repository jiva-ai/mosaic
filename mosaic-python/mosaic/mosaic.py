"""Main entry point for Mosaic orchestrator."""

import argparse
import logging
import pickle
import re
import sys
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mosaic_comms.beacon import Beacon
from mosaic_config.config import MosaicConfig, read_config
from mosaic_config.state_utils import StateIdentifiers, read_state, save_state
from mosaic_planner import (
    plan_dynamic_weighted_batches,
    plan_static_weighted_shards,
)
from mosaic_planner.state import Model, ModelType, Plan, Session

# Set up logging
logger = logging.getLogger(__name__)

# Global beacon instance (set in main())
_beacon: Optional[Beacon] = None

# Global state lists (maintained at mosaic.py level)
_sessions: List[Session] = []
_models: List[Model] = []
_config: Optional[MosaicConfig] = None


def _format_send_heartbeat_table(statuses) -> str:
    """
    Format send heartbeat statuses as a table.
    
    Args:
        statuses: List of SendHeartbeatStatus objects
        
    Returns:
        Formatted table string
    """
    if not statuses:
        return "No send heartbeat statuses"
    
    # Table header
    header = f"{'Host':<20} {'Port':<8} {'Status':<12} {'Last Sent':<20}"
    separator = "-" * len(header)
    lines = [header, separator]
    
    # Table rows
    for status in statuses:
        last_sent_str = "Never" if status.last_time_sent == 0 else str(status.last_time_sent)
        row = f"{status.host:<20} {status.heartbeat_port:<8} {status.connection_status:<12} {last_sent_str:<20}"
        lines.append(row)
    
    return "\n".join(lines)


def _format_receive_heartbeat_table(statuses) -> str:
    """
    Format receive heartbeat statuses as a table.
    
    Args:
        statuses: List of ReceiveHeartbeatStatus objects
        
    Returns:
        Formatted table string
    """
    if not statuses:
        return "No receive heartbeat statuses"
    
    # Table header
    header = f"{'Host':<20} {'HB Port':<10} {'Comms Port':<12} {'Status':<12} {'Last Received':<20} {'Delay (ns)':<15}"
    separator = "-" * len(header)
    lines = [header, separator]
    
    # Table rows
    for status in statuses:
        last_received_str = "Never" if status.last_time_received == 0 else str(status.last_time_received)
        delay_str = str(status.delay) if status.delay is not None else "N/A"
        row = (
            f"{status.host:<20} {status.heartbeat_port:<10} {status.comms_port:<12} "
            f"{status.connection_status:<12} {last_received_str:<20} {delay_str:<15}"
        )
        lines.append(row)
    
    return "\n".join(lines)


def cmd_shb() -> None:
    """Show send heartbeat statuses."""
    from mosaic.repl_commands import execute_shb
    execute_shb(print)


def cmd_rhb() -> None:
    """Show receive heartbeat statuses."""
    from mosaic.repl_commands import execute_rhb
    execute_rhb(print)


def cmd_hb() -> None:
    """Show both send and receive heartbeat statuses."""
    from mosaic.repl_commands import execute_hb
    execute_hb(print)


def calculate_data_distribution(method: Optional[str] = None) -> None:
    """
    Calculate distribution of data (and therefore workloads) across peers.
    
    Args:
        method: Distribution method - "weighted_shard" or "weighted_batches"
    """
    if _beacon is None:
        print("Error: Beacon not initialized")
        return None
    
    # Collect stats from beacon
    stats_data = _beacon.collect_stats()
    
    if not stats_data:
        print("No stats data available")
        return None
    
    total_samples = len(stats_data)
    
    if method is None or method == "weighted_shard":
        result = plan_static_weighted_shards(stats_data, total_samples=total_samples)
        print(f"Static weighted shard allocation (method: {method or 'weighted_shard'}):")
    elif method == "weighted_batches":
        result = plan_dynamic_weighted_batches(stats_data, total_batches=total_samples)
        print(f"Dynamic weighted batch allocation (method: {method}):")
    else:
        print(f"Error: Unknown method '{method}'. Use 'weighted_shard' or 'weighted_batches'")
        return None
    
    if not result:
        print("No eligible peers for allocation")
        return None
    
    # Display results
    for allocation in result:
        if "allocated_samples" in allocation:
            print(f"  {allocation['host']}: {allocation['allocated_samples']} samples")
        elif "allocated_batches" in allocation:
            print(f"  {allocation['host']}: {allocation['allocated_batches']} batches")
    
    return result


def cmd_calcd(method: Optional[str] = None) -> None:
    """Calculate distribution of data/workloads."""
    from mosaic.repl_commands import execute_calcd
    execute_calcd(print, method)


def cmd_help() -> None:
    """Show help message with available commands."""
    from mosaic.repl_commands import execute_help
    execute_help(print)




def _convert_enums_to_values(obj: Any) -> Any:
    """
    Recursively convert Enum objects to their values for JSON serialization.
    
    Args:
        obj: Object that may contain Enum instances
    
    Returns:
        Object with Enums converted to their values
    """
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {key: _convert_enums_to_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_enums_to_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_enums_to_values(item) for item in obj)
    else:
        return obj


def _handle_sessions_command(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Handle 'sessions' command - returns the list of sessions as dictionaries.
    
    Args:
        payload: Command payload (not used, but required by handler signature)
    
    Returns:
        List of Session objects converted to dictionaries with Enums converted to values
    """
    # Convert Session dataclasses to dictionaries for JSON serialization
    sessions_dicts = [asdict(session) for session in _sessions]
    # Convert Enums to their values for JSON serialization
    return [_convert_enums_to_values(session_dict) for session_dict in sessions_dicts]


def _handle_add_model_command(payload: Union[Dict[str, Any], bytes]) -> Optional[Dict[str, Any]]:
    """
    Handle 'add_model' command - receives a Model object and adds it to the models list.
    
    Args:
        payload: Model object (can be dict or bytes/pickled)
    
    Returns:
        Dictionary with status information
    """
    try:
        # Deserialize model if it's bytes
        if isinstance(payload, bytes):
            model = pickle.loads(payload)
        else:
            # Reconstruct Model from dict
            model_type = None
            if payload.get("model_type"):
                model_type = ModelType(payload["model_type"])
            model = Model(
                name=payload["name"],
                model_type=model_type,
                onnx_location=payload.get("onnx_location"),
                binary_rep=payload.get("binary_rep"),
                file_name=payload.get("file_name"),
            )
        
        # Add model (this will save binary to disk if present)
        add_model(model)
        
        return {
            "status": "success",
            "message": f"Model {model.name} added successfully",
        }
    except Exception as e:
        logger.error(f"Error handling add_model: {e}")
        return {"status": "error", "message": str(e)}


def _save_sessions_state() -> None:
    """Save sessions list to persistent state."""
    if _config is None:
        logger.warning("Cannot save sessions state: config not initialized")
        return
    try:
        save_state(_config, _sessions, StateIdentifiers.SESSIONS)
        logger.debug("Saved sessions state")
    except Exception as e:
        logger.warning(f"Failed to save sessions state: {e}")


def add_session(session: Session) -> None:
    """
    Add a session to the sessions list and persist state.
    
    Args:
        session: Session instance to add
    """
    global _sessions
    _sessions.append(session)
    _save_sessions_state()
    logger.debug(f"Added session with ID: {session.id}")


def remove_session(session_id: str) -> bool:
    """
    Remove a session from the sessions list by ID and persist state.
    
    Args:
        session_id: ID of the session to remove
    
    Returns:
        True if session was found and removed, False otherwise
    """
    global _sessions
    initial_count = len(_sessions)
    _sessions = [s for s in _sessions if s.id != session_id]
    
    if len(_sessions) < initial_count:
        _save_sessions_state()
        logger.debug(f"Removed session with ID: {session_id}")
        return True
    else:
        logger.warning(f"Session with ID {session_id} not found")
        return False


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a filename for Unix filesystem compatibility.
    
    Replaces invalid characters (spaces, symbols) with underscores.
    Ensures the filename is valid for Unix filesystems.
    
    Args:
        name: Original filename/name
    
    Returns:
        Sanitized filename safe for Unix filesystems
    """
    # Replace spaces and invalid filename characters with underscore
    # Invalid characters for Unix: / \0 and any control characters
    # Also replace common problematic characters: spaces, < > : " | ? * and symbols
    # Keep only alphanumeric, underscores, hyphens, and dots (dots will be stripped later)
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    # Remove leading/trailing dots and spaces (converted to underscores)
    sanitized = sanitized.strip('._')
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed"
    # Limit length to reasonable size (255 chars is typical max)
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def _save_models_state() -> None:
    """Save models list to persistent state."""
    if _config is None:
        logger.warning("Cannot save models state: config not initialized")
        return
    try:
        save_state(_config, _models, StateIdentifiers.MODELS)
        logger.debug("Saved models state")
    except Exception as e:
        logger.warning(f"Failed to save models state: {e}")


def add_model(model: Model) -> None:
    """
    Add a model to the models list, save ONNX binary to disk if present, and persist state.
    
    If the model has binary_rep (ONNX binary data), it will be saved to disk at:
    - models_location/onnx_location/filename if onnx_location is not None
    - models_location/filename if onnx_location is None
    
    The binary_rep will be set to None after saving to conserve memory.
    The file_name field will be set to the sanitized filename.
    
    Args:
        model: Model instance to add
    """
    global _models
    
    # If model has binary_rep, save it to disk
    if model.binary_rep is not None:
        if _config is None:
            logger.warning("Cannot save model binary: config not initialized")
        elif not _config.models_location:
            logger.warning("Cannot save model binary: models_location not configured")
        else:
            try:
                # Sanitize the model name for use as filename
                sanitized_name = _sanitize_filename(model.name)
                
                # Determine the save location
                models_path = Path(_config.models_location)
                if model.onnx_location:
                    # Save to models_location/onnx_location
                    save_dir = models_path / model.onnx_location
                else:
                    # Save directly to models_location
                    save_dir = models_path
                
                # Ensure directory exists
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the binary data
                file_path = save_dir / sanitized_name
                with open(file_path, 'wb') as f:
                    f.write(model.binary_rep)
                
                # Update model: set file_name and clear binary_rep
                model.file_name = sanitized_name
                model.binary_rep = None
                
                logger.debug(f"Saved model binary to {file_path}")
            except Exception as e:
                logger.warning(f"Failed to save model binary: {e}")
    
    _models.append(model)
    _save_models_state()
    logger.debug(f"Added model: {model.name}")


def remove_model(model_name: str) -> bool:
    """
    Remove a model from the models list by name and persist state.
    
    Args:
        model_name: Name of the model to remove
    
    Returns:
        True if model was found and removed, False otherwise
    """
    global _models
    initial_count = len(_models)
    _models = [m for m in _models if m.name != model_name]
    
    if len(_models) < initial_count:
        _save_models_state()
        logger.debug(f"Removed model: {model_name}")
        return True
    else:
        logger.warning(f"Model with name {model_name} not found")
        return False


def interpret_command(command: str) -> None:
    """
    Interpret a command from the REPL.
    
    Parses the command string (command followed by space-separated arguments)
    and calls the appropriate handler function.
    
    Args:
        command: The command string to interpret
    """
    if not command:
        return
    
    from mosaic.repl_commands import process_command
    process_command(command, print)

def repl_loop() -> None:
    """
    Start the REPL loop.
    """
    while True:
        try:
            command = input("mosaic> ").strip()
            if not command:
                continue
            if command.lower() in ("exit", "quit", "q"):
                logger.info("Exiting REPL...")
                break
            interpret_command(command)
        except EOFError:
            logger.info("Exiting REPL (EOF)...")
            break
        except KeyboardInterrupt:
            logger.info("Exiting REPL (interrupted)...")
            break

def main() -> None:
    """Main entry point for Mosaic orchestrator."""
    parser = argparse.ArgumentParser(description="Mosaic Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (optional, MosaicConfig will search standard locations if not provided)",
    )
    parser.add_argument(
        "--repl",
        type=str,
        default="true",
        help="Start REPL (true/false, default: true)",
    )
    parser.add_argument(
        "--textual",
        action="store_true",
        help="Use Textual-based REPL instead of simple REPL",
    )
    
    args = parser.parse_args()
    
    # Parse --repl argument
    repl_enabled = args.repl.lower() in ("true", "1", "yes", "on")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Step 1: Create MosaicConfig
    logger.info("Initializing Mosaic configuration...")
    try:
        config = read_config(config_path=args.config)
        logger.info("Mosaic configuration loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load configuration file: {e}")
        logger.info("Using default Mosaic configuration")
        config = MosaicConfig()
    
    # Store config globally
    global _config
    _config = config
    
    # Step 1.5: Load Sessions and Models from persistent state
    logger.info("Loading Sessions from state...")
    try:
        global _sessions
        loaded_sessions = read_state(config, StateIdentifiers.SESSIONS, default=None)
        
        if isinstance(loaded_sessions, list):
            _sessions = loaded_sessions
            logger.info(f"Loaded {len(_sessions)} sessions from state")
        else:
            _sessions = []
            logger.info("No sessions found in state, initializing empty list")
    except Exception as e:
        logger.warning(f"Error loading Sessions from state: {e}")
        _sessions = []
    
    logger.info("Loading Models from state...")
    try:
        global _models
        loaded_models = read_state(config, StateIdentifiers.MODELS, default=None)
        
        if isinstance(loaded_models, list):
            _models = loaded_models
            logger.info(f"Loaded {len(_models)} models from state")
        else:
            _models = []
            logger.info("No models found in state, initializing empty list")
    except Exception as e:
        logger.warning(f"Error loading Models from state: {e}")
        _models = []
    
    # Step 2: Create Beacon
    logger.info("Creating Beacon instance...")
    try:
        global _beacon
        _beacon = Beacon(config)
        logger.info("Beacon instance created successfully")
    except Exception as e:
        logger.error(f"Error creating Beacon: {e}")
        sys.exit(1)
    
    # Step 2.5: Register command handlers for sessions and models
    logger.info("Registering command handlers...")
    try:
        _beacon.register("sessions", _handle_sessions_command)
        _beacon.register("add_model", _handle_add_model_command)
        logger.info("Command handlers registered successfully")
    except Exception as e:
        logger.warning(f"Error registering command handlers: {e}")
    
    # Step 3: Start Beacon
    logger.info("Starting Beacon...")
    try:
        _beacon.start()
        logger.info("Beacon started successfully")
    except Exception as e:
        logger.error(f"Error starting Beacon: {e}")
        sys.exit(1)
    
    # Step 4: Start REPL if enabled
    if repl_enabled:
        if args.textual:
            logger.info("Starting Textual-based REPL...")
            logger.info("Call 'help' to see available commands")
            logger.info("GitHub repo with README: https://github.com/manishjiva/mosaic [NB soon to be renamed]")
            try:
                from mosaic.textual_repl import start_textual_repl
                start_textual_repl()
            except ImportError:
                logger.error("Textual library not installed. Install with: pip install textual")
                logger.info("Falling back to simple REPL...")
                repl_loop()
            except Exception as e:
                logger.error(f"Error in Textual REPL: {e}")
                logger.info("Falling back to simple REPL...")
                repl_loop()
        else:
            logger.info("Starting simple REPL...")
            try:
                repl_loop()
            except Exception as e:
                logger.error(f"Error in REPL: {e}")
    else:
        logger.info("REPL disabled, running in background mode")
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == "__main__":
    main()

