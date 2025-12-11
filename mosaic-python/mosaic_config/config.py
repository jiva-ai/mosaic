"""Configuration reading module for Mosaic."""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _expand_path(path_str: str) -> Path:
    """
    Expand a path string, resolving ~ and environment variables.

    Args:
        path_str: Path string that may contain ~ or environment variables

    Returns:
        Path: Expanded Path object
    """
    return Path(os.path.expanduser(os.path.expandvars(path_str)))


@dataclass
class Peer:
    """Represents a peer host and ports for communication."""

    host: str
    comms_port: int
    heartbeat_port: int


@dataclass
class MosaicConfig:
    """Configuration object for Mosaic processes."""

    host: str = "localhost"
    heartbeat_port: int = 5000
    comms_port: int = 5001
    peers: List[Peer] = field(default_factory=list)
    heartbeat_frequency: int = 5
    heartbeat_tolerance: int = 15
    heartbeat_report_length: int = 300
    heartbeat_wait_timeout: int = 2
    stats_request_timeout: int = 30
    server_crt: str = ""
    server_key: str = ""
    ca_crt: str = ""
    benchmark_data_location: str = ""
    run_benchmark_at_startup: bool = False
    data_location: str = ""
    plans_location: str = "plans"
    models_location: str = "models"
    state_location: str = ""
    data_chunk_size: int = 256  # Size in megabytes

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MosaicConfig":
        """
        Create a MosaicConfig instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            MosaicConfig: Configured instance with values from dict
        """
        # Extract peers and convert to Peer objects
        peers = []
        if "peers" in config_dict:
            for peer_data in config_dict["peers"]:
                if isinstance(peer_data, dict):
                    peers.append(
                        Peer(
                            host=peer_data["host"],
                            comms_port=peer_data["comms_port"],
                            heartbeat_port=peer_data["heartbeat_port"],
                        )
                    )

        # Extract heartbeat config (can be nested or flat)
        heartbeat_frequency = config_dict.get(
            "heartbeat_frequency",
            config_dict.get("heartbeat", {}).get("frequency", 5),
        )
        heartbeat_tolerance = config_dict.get(
            "heartbeat_tolerance",
            config_dict.get("heartbeat", {}).get("tolerance", 15),
        )
        heartbeat_report_length = config_dict.get(
            "heartbeat_report_length",
            config_dict.get("heartbeat", {}).get("report_length", 300),
        )
        heartbeat_wait_timeout = config_dict.get(
            "heartbeat_wait_timeout",
            config_dict.get("heartbeat", {}).get("wait_timeout", 2),
        )
        stats_request_timeout = config_dict.get("stats_request_timeout", 30)

        return cls(
            host=config_dict.get("host", "localhost"),
            heartbeat_port=config_dict.get("heartbeat_port", 5000),
            comms_port=config_dict.get("comms_port", 5001),
            peers=peers,
            heartbeat_frequency=heartbeat_frequency,
            heartbeat_tolerance=heartbeat_tolerance,
            heartbeat_report_length=heartbeat_report_length,
            heartbeat_wait_timeout=heartbeat_wait_timeout,
            stats_request_timeout=stats_request_timeout,
            server_crt=config_dict.get("server_crt", ""),
            server_key=config_dict.get("server_key", ""),
            ca_crt=config_dict.get("ca_crt", ""),
            benchmark_data_location=config_dict.get("benchmark_data_location", ""),
            run_benchmark_at_startup=bool(config_dict.get("run_benchmark_at_startup", False)),
            data_location=config_dict.get("data_location", ""),
            plans_location=config_dict.get("plans_location", "plans"),
            models_location=config_dict.get("models_location", "models"),
            state_location=config_dict.get("state_location", ""),
            data_chunk_size=config_dict.get("data_chunk_size", 256),
        )


def read_json_config() -> Dict[str, Any]:
    """
    Read JSON configuration from standard locations in priority order:
    1. Command-line switch --config
    2. Environment variable MOSAIC_CONFIG
    3. File mosaic.config in current working directory

    Paths support expansion of ~ (home directory) and environment variables
    (e.g., $HOME, ${VAR}).

    Returns:
        Dict[str, Any]: Parsed JSON configuration

    Raises:
        FileNotFoundError: If no configuration file is found
        json.JSONDecodeError: If the configuration file contains invalid JSON
    """
    config_path: Optional[Path] = None

    # 1. Check command-line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args, _ = parser.parse_known_args()

    if args.config:
        config_path = _expand_path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not config_path.is_file():
            raise ValueError(f"Configuration path is not a file: {config_path}")

    # 2. Check environment variable
    if config_path is None:
        env_config = os.environ.get("MOSAIC_CONFIG")
        if env_config:
            config_path = _expand_path(env_config)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            if not config_path.is_file():
                raise ValueError(f"Configuration path is not a file: {config_path}")

    # 3. Check current working directory for mosaic.config
    if config_path is None:
        cwd_config = Path.cwd() / "mosaic.config"
        if cwd_config.exists():
            config_path = cwd_config

    # If no config found, raise an error
    if config_path is None:
        raise FileNotFoundError(
            "No configuration file found. Tried:\n"
            "  1. --config command-line argument\n"
            "  2. MOSAIC_CONFIG environment variable\n"
            "  3. mosaic.config in current working directory"
        )

    # Read and parse JSON
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in configuration file {config_path}: {e.msg}",
            e.doc,
            e.pos,
        ) from e


def read_config(config_path: Optional[str] = None) -> MosaicConfig:
    """
    Read and parse Mosaic configuration from standard locations.

    Configuration is loaded from JSON in priority order:
    1. config_path parameter (if provided)
    2. Command-line switch --config
    3. Environment variable MOSAIC_CONFIG
    4. File mosaic.config in current working directory

    Paths support expansion of ~ (home directory) and environment variables
    (e.g., $HOME, ${VAR}).

    Args:
        config_path: Optional path to configuration file. If provided, this takes
                     precedence over other sources.

    Returns:
        MosaicConfig: Typed configuration object with default values
        overwritten by values from the configuration file

    Raises:
        FileNotFoundError: If no configuration file is found
        json.JSONDecodeError: If the configuration file contains invalid JSON
    """
    if config_path is not None:
        # If config_path is provided, use it directly
        expanded_path = _expand_path(config_path)
        if not expanded_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {expanded_path}")
        if not expanded_path.is_file():
            raise ValueError(f"Configuration path is not a file: {expanded_path}")
        
        # Read and parse JSON
        try:
            with open(expanded_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file {expanded_path}: {e.msg}",
                e.doc,
                e.pos,
            ) from e
        
        return MosaicConfig.from_dict(config_dict)
    else:
        # Use standard location search
        config_dict = read_json_config()
        return MosaicConfig.from_dict(config_dict)

