"""State persistence utilities for Mosaic nodes."""

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from mosaic_config.config import MosaicConfig

logger = logging.getLogger(__name__)

# Thread-safe locks for each state file identifier
_state_locks: Dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


# State file identifiers - use these constants to avoid typos
class StateIdentifiers:
    """Constants for state file identifiers."""
    
    SEND_HEARTBEAT_STATUSES = "send_heartbeat_statuses"
    RECEIVE_HEARTBEAT_STATUSES = "receive_heartbeat_statuses"


def _get_state_directory(config: MosaicConfig) -> Path:
    """
    Get the state directory path from config, defaulting to current working directory.
    
    Args:
        config: MosaicConfig instance
        
    Returns:
        Path to state directory
    """
    state_location = config.state_location.strip()
    
    # Remove trailing slash if present
    if state_location.endswith("/") or state_location.endswith("\\"):
        state_location = state_location[:-1]
    
    # If empty, use current working directory
    if not state_location:
        return Path.cwd()
    
    return Path(state_location)


def _get_lock(identifier: str) -> threading.Lock:
    """
    Get or create a thread lock for a specific state identifier.
    
    Args:
        identifier: State file identifier
        
    Returns:
        Thread lock for the identifier
    """
    with _locks_lock:
        if identifier not in _state_locks:
            _state_locks[identifier] = threading.Lock()
        return _state_locks[identifier]


def save_state(config: MosaicConfig, obj: Any, identifier: str) -> None:
    """
    Save an object to a pickle file in the state directory (thread-safe).
    
    Args:
        config: MosaicConfig instance
        obj: Object to save (must be pickleable)
        identifier: File identifier (will be used as filename with .pkl extension)
        
    Raises:
        OSError: If the state directory cannot be created or written to
        pickle.PickleError: If the object cannot be pickled
    """
    state_dir = _get_state_directory(config)
    lock = _get_lock(identifier)
    
    # Ensure identifier is safe for filename
    safe_identifier = identifier.replace("/", "_").replace("\\", "_")
    file_path = state_dir / f"{safe_identifier}.pkl"
    
    with lock:
        try:
            # Create directory if it doesn't exist
            state_dir.mkdir(parents=True, exist_ok=True)
            
            # Write object to file atomically using a temporary file
            temp_path = file_path.with_suffix(".pkl.tmp")
            with open(temp_path, "wb") as f:
                pickle.dump(obj, f)
            
            # Atomic rename
            temp_path.replace(file_path)
            
            logger.debug(f"Saved state '{identifier}' to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save state '{identifier}': {e}")
            # Clean up temp file if it exists
            temp_path = file_path.with_suffix(".pkl.tmp")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise


def read_state(
    config: MosaicConfig, identifier: str, default: Optional[Any] = None
) -> Optional[Any]:
    """
    Read an object from a pickle file in the state directory (thread-safe).
    
    Args:
        config: MosaicConfig instance
        identifier: File identifier (will be used as filename with .pkl extension)
        default: Optional value to return if file doesn't exist or read fails
        
    Returns:
        Unpickled object, or default if file doesn't exist or read fails
    """
    state_dir = _get_state_directory(config)
    lock = _get_lock(identifier)
    
    # Ensure identifier is safe for filename
    safe_identifier = identifier.replace("/", "_").replace("\\", "_")
    file_path = state_dir / f"{safe_identifier}.pkl"
    
    with lock:
        if not file_path.exists():
            logger.debug(f"State file '{identifier}' not found at {file_path}")
            return default
        
        try:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            logger.debug(f"Loaded state '{identifier}' from {file_path}")
            return obj
        except Exception as e:
            logger.warning(f"Failed to read state '{identifier}': {e}")
            return default

