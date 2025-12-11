"""Shared REPL command execution functions for Mosaic orchestrator."""

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mosaic_comms.beacon import Beacon

# Module-level beacon reference (set via initialize() function)
_beacon: Optional["Beacon"] = None


def initialize(beacon: "Beacon") -> None:
    """
    Initialize the repl_commands module with the beacon instance.
    
    This should be called once from mosaic.py after the beacon is created.
    
    Args:
        beacon: The Beacon instance to use for REPL commands
    """
    global _beacon
    _beacon = beacon


def execute_shb(output_fn: Callable[[str], None]) -> None:
    """
    Execute shb command - show send heartbeat statuses.
    
    Args:
        output_fn: Function to call with output text
    """
    if _beacon is None:
        output_fn("Error: Beacon not initialized\n")
        return

    statuses = _beacon.send_heartbeat_statuses
    from mosaic.mosaic import _format_send_heartbeat_table
    table = _format_send_heartbeat_table(statuses)
    output_fn(f"{table}\n")


def execute_rhb(output_fn: Callable[[str], None]) -> None:
    """
    Execute rhb command - show receive heartbeat statuses.
    
    Args:
        output_fn: Function to call with output text
    """
    if _beacon is None:
        output_fn("Error: Beacon not initialized\n")
        return

    statuses = _beacon.receive_heartbeat_statuses
    from mosaic.mosaic import _format_receive_heartbeat_table
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
    if _beacon is None:
        output_fn("Error: Beacon not initialized\n")
        return

    # Collect output from calculate_data_distribution
    import io
    from contextlib import redirect_stdout
    from mosaic.mosaic import calculate_data_distribution

    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            calculate_data_distribution(method)
        output = output_buffer.getvalue()
        if output:
            output_fn(f"{output}\n")
    except Exception as e:
        output_fn(f"Error: {e}\n")


def execute_help(output_fn: Callable[[str], None], command: Optional[str] = None) -> None:
    """
    Execute help command - show help message or detailed command help.
    
    Args:
        output_fn: Function to call with output text
        command: Optional command name for detailed help
    """
    from mosaic.help_text import get_command_help, get_command_list_sorted, COMMAND_ALIASES
    
    if command is None:
        # Show list of all commands
        help_text = "Available commands:\n"
        commands = get_command_list_sorted()
        
        for cmd, short_desc in commands:
            # Format command name with appropriate spacing
            # Find aliases that map to this command
            aliases = [alias for alias, main in COMMAND_ALIASES.items() if main == cmd]
            if aliases:
                cmd_display = f"{cmd} ({', '.join(sorted(aliases))})"
            else:
                cmd_display = cmd
            
            help_text += f"  {cmd_display:<30} - {short_desc}\n"
        
        output_fn(help_text.strip() + "\n")
    else:
        # Show detailed help for specific command
        try:
            short_desc, long_desc = get_command_help(command)
            output_fn(long_desc + "\n")
        except KeyError:
            # Check if it's an alias
            if command.lower() in COMMAND_ALIASES:
                try:
                    short_desc, long_desc = get_command_help(COMMAND_ALIASES[command.lower()])
                    output_fn(long_desc + "\n")
                except KeyError:
                    output_fn(f"Unknown command: {command}\nType 'help' for available commands.\n")
            else:
                output_fn(f"Unknown command: {command}\nType 'help' for available commands.\n")


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
        elif cmd == "create_session" or cmd == "create-session":
            from mosaic.session_commands import execute_create_session
            execute_create_session(output_fn)
        elif cmd == "delete_session" or cmd == "delete-session":
            from mosaic.session_commands import execute_delete_session
            session_id = args[0] if args else None
            execute_delete_session(output_fn, session_id)
        elif cmd == "train_session" or cmd == "train-session":
            from mosaic.session_commands import execute_train_session
            session_id = args[0] if args else None
            execute_train_session(output_fn, session_id)
        elif cmd == "cancel_training" or cmd == "cancel-training":
            from mosaic.session_commands import execute_cancel_training
            session_id = args[0] if args else None
            hostname = args[1] if len(args) > 1 else None
            execute_cancel_training(output_fn, session_id, hostname)
        elif cmd == "use":
            from mosaic.session_commands import execute_use_session
            session_id = args[0] if args else None
            execute_use_session(output_fn, session_id)
        elif cmd == "infer":
            from mosaic.session_commands import execute_infer
            input_data = " ".join(args) if args else None
            execute_infer(output_fn, input_data)
        elif cmd == "set_infer_method" or cmd == "set-infer-method":
            from mosaic.session_commands import execute_set_infer_method
            method = args[0] if args else None
            execute_set_infer_method(output_fn, method)
        elif cmd == "help":
            help_command = args[0] if args else None
            execute_help(output_fn, help_command)
        else:
            output_fn("Unknown command. Type 'help' for available commands.\n")
    except Exception as e:
        output_fn(f"Error: {e}\n")

