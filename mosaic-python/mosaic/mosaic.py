"""Main entry point for Mosaic orchestrator."""

import argparse
import logging
import sys
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from mosaic_comms.beacon import Beacon
from mosaic_config.config import MosaicConfig, read_config
from mosaic_config.state_utils import StateIdentifiers, read_state, save_state
from mosaic_planner import (
    plan_dynamic_weighted_batches,
    plan_static_weighted_shards,
)
from mosaic_planner.state import Plan, Session

# Set up logging
logger = logging.getLogger(__name__)

# Global beacon instance (set in main())
_beacon: Optional[Beacon] = None

# Global state lists (maintained at mosaic.py level)
_sessions: List[Session] = []
_plans: List[Plan] = []
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
    if _beacon is None:
        print("Error: Beacon not initialized")
        return
    
    statuses = _beacon.send_heartbeat_statuses
    print(_format_send_heartbeat_table(statuses))


def cmd_rhb() -> None:
    """Show receive heartbeat statuses."""
    if _beacon is None:
        print("Error: Beacon not initialized")
        return
    
    statuses = _beacon.receive_heartbeat_statuses
    print(_format_receive_heartbeat_table(statuses))


def cmd_hb() -> None:
    """Show both send and receive heartbeat statuses."""
    cmd_shb()
    print()  # Empty line between tables
    cmd_rhb()


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
    calculate_data_distribution(method)


def cmd_help() -> None:
    """Show help message with available commands."""
    help_text = """
Available commands:
  shb                    - Show send heartbeat statuses
  rhb                    - Show receive heartbeat statuses
  hb                     - Show both send and receive heartbeat statuses
  calcd [method]         - Calculate distribution (method: weighted_shard or weighted_batches)
  help                   - Show this help message
  exit/quit/q            - Exit the REPL
"""
    print(help_text.strip())


def show_usage() -> None:
    """Show usage message for unknown commands."""
    print("Unknown command. Type 'help' for available commands.")


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


def _handle_plans_command(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Handle 'plans' command - returns the list of plans as dictionaries.
    
    Args:
        payload: Command payload (not used, but required by handler signature)
    
    Returns:
        List of Plan objects converted to dictionaries with Enums converted to values
    """
    # Convert Plan dataclasses to dictionaries for JSON serialization
    plans_dicts = [asdict(plan) for plan in _plans]
    # Convert Enums to their values for JSON serialization
    return [_convert_enums_to_values(plan_dict) for plan_dict in plans_dicts]


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


def _save_plans_state() -> None:
    """Save plans list to persistent state."""
    if _config is None:
        logger.warning("Cannot save plans state: config not initialized")
        return
    try:
        save_state(_config, _plans, StateIdentifiers.PLANS)
        logger.debug("Saved plans state")
    except Exception as e:
        logger.warning(f"Failed to save plans state: {e}")


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
    
    # Split command and arguments
    parts = command.split()
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []
    
    # Route to appropriate command handler
    if cmd == "shb":
        cmd_shb()
    elif cmd == "rhb":
        cmd_rhb()
    elif cmd == "hb":
        cmd_hb()
    elif cmd == "calcd":
        cmd_calcd(args[0] if args else None)
    elif cmd == "help":
        cmd_help()
    else:
        show_usage()

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
    
    # Step 1.5: Load Sessions and Plans from persistent state
    logger.info("Loading Sessions and Plans from state...")
    try:
        global _sessions, _plans
        loaded_sessions = read_state(config, StateIdentifiers.SESSIONS, default=None)
        loaded_plans = read_state(config, StateIdentifiers.PLANS, default=None)
        
        if isinstance(loaded_sessions, list):
            _sessions = loaded_sessions
            logger.info(f"Loaded {len(_sessions)} sessions from state")
        else:
            _sessions = []
            logger.info("No sessions found in state, initializing empty list")
        
        if isinstance(loaded_plans, list):
            _plans = loaded_plans
            logger.info(f"Loaded {len(_plans)} plans from state")
        else:
            _plans = []
            logger.info("No plans found in state, initializing empty list")
    except Exception as e:
        logger.warning(f"Error loading Sessions/Plans from state: {e}")
        _sessions = []
        _plans = []
    
    # Step 2: Create Beacon
    logger.info("Creating Beacon instance...")
    try:
        global _beacon
        _beacon = Beacon(config)
        logger.info("Beacon instance created successfully")
    except Exception as e:
        logger.error(f"Error creating Beacon: {e}")
        sys.exit(1)
    
    # Step 2.5: Register command handlers for sessions and plans
    logger.info("Registering command handlers...")
    try:
        _beacon.register("sessions", _handle_sessions_command)
        _beacon.register("plans", _handle_plans_command)
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
        logger.info("Starting REPL...")
        try:
            repl_loop()
        except Exception as e:
            logger.error(f"Error in REPL: {e}")
    else:
        logger.info("REPL disabled, running in background mode")
        # Keep the main thread alive
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == "__main__":
    main()

