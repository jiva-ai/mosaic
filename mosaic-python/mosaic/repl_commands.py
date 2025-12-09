"""Shared REPL command execution functions for Mosaic orchestrator."""

from typing import Callable, Optional

import mosaic.mosaic as mosaic_module
from mosaic.mosaic import (
    _format_receive_heartbeat_table,
    _format_send_heartbeat_table,
    calculate_data_distribution,
)


def execute_shb(output_fn: Callable[[str], None]) -> None:
    """
    Execute shb command - show send heartbeat statuses.
    
    Args:
        output_fn: Function to call with output text
    """
    if mosaic_module._beacon is None:
        output_fn("Error: Beacon not initialized\n")
        return

    statuses = mosaic_module._beacon.send_heartbeat_statuses
    table = _format_send_heartbeat_table(statuses)
    output_fn(f"{table}\n")


def execute_rhb(output_fn: Callable[[str], None]) -> None:
    """
    Execute rhb command - show receive heartbeat statuses.
    
    Args:
        output_fn: Function to call with output text
    """
    if mosaic_module._beacon is None:
        output_fn("Error: Beacon not initialized\n")
        return

    statuses = mosaic_module._beacon.receive_heartbeat_statuses
    table = _format_receive_heartbeat_table(statuses)
    output_fn(f"{table}\n")


def execute_hb(output_fn: Callable[[str], None]) -> None:
    """
    Execute hb command - show both send and receive heartbeat statuses.
    
    Args:
        output_fn: Function to call with output text
    """
    execute_shb(output_fn)
    output_fn("\n")
    execute_rhb(output_fn)


def execute_calcd(output_fn: Callable[[str], None], method: Optional[str] = None) -> None:
    """
    Execute calcd command - calculate distribution of data/workloads.
    
    Args:
        output_fn: Function to call with output text
        method: Distribution method - "weighted_shard" or "weighted_batches"
    """
    if mosaic_module._beacon is None:
        output_fn("Error: Beacon not initialized\n")
        return

    # Collect output from calculate_data_distribution
    import io
    from contextlib import redirect_stdout

    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            calculate_data_distribution(method)
        output = output_buffer.getvalue()
        if output:
            output_fn(f"{output}\n")
    except Exception as e:
        output_fn(f"Error: {e}\n")


def execute_help(output_fn: Callable[[str], None]) -> None:
    """
    Execute help command - show help message.
    
    Args:
        output_fn: Function to call with output text
    """
    help_text = """
Available commands:
  shb                    - Show send heartbeat statuses
  rhb                    - Show receive heartbeat statuses
  hb                     - Show both send and receive heartbeat statuses
  calcd [method]         - Calculate distribution (method: weighted_shard or weighted_batches)
  help                   - Show this help message
  exit/quit/q            - Exit the REPL
"""
    output_fn(help_text.strip() + "\n")


def process_command(command: str, output_fn: Callable[[str], None]) -> None:
    """
    Process a command and display results.
    
    Args:
        command: The command string to interpret
        output_fn: Function to call with output text
    """
    parts = command.split()
    if not parts:
        return

    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    try:
        if cmd == "shb":
            execute_shb(output_fn)
        elif cmd == "rhb":
            execute_rhb(output_fn)
        elif cmd == "hb":
            execute_hb(output_fn)
        elif cmd == "calcd":
            method = args[0] if args else None
            execute_calcd(output_fn, method)
        elif cmd == "help":
            execute_help(output_fn)
        else:
            output_fn("Unknown command. Type 'help' for available commands.\n")
    except Exception as e:
        output_fn(f"Error: {e}\n")

