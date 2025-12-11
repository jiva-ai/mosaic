"""Help text for MOSAIC REPL commands.

This module contains all help text for REPL commands, including short descriptions
for the help list and detailed man page-style descriptions for individual commands.
"""

from typing import Dict, Tuple

# Command help entries: (command, short_description, long_description)
COMMAND_HELP: Dict[str, Tuple[str, str]] = {
    "calcd": (
        "Calculate data distribution across available nodes",
        """CALCD - Calculate Data Distribution

SYNOPSIS
    calcd [method]

DESCRIPTION
    Calculates and displays the distribution of data/workloads across available
    nodes in the network. This command analyzes node capabilities and creates
    a distribution plan for data sharding.

    The distribution calculation considers:
    - Node capacity and resources (CPU, memory, disk)
    - Network connectivity and latency
    - Data size and characteristics
    - Current workload on each node

OPTIONS
    method
        Optional distribution method. If not specified, uses the default method
        configured in the system.
        
        Available methods:
        - weighted_shard: Distributes data into shards weighted by node capacity
        - weighted_batches: Distributes data into batches weighted by node capacity

EXAMPLES
    calcd
        Calculate distribution using default method
    
    calcd weighted_shard
        Calculate distribution using weighted shard method

SEE ALSO
    create_session, train_session
""",
    ),
    "cancel_training": (
        "Cancel training for a session",
        """CANCEL_TRAINING - Cancel Model Training

SYNOPSIS
    cancel_training [session_id] [hostname]

DESCRIPTION
    Cancels ongoing training for a session. This command sends cancellation
    requests to all nodes participating in the training, or optionally to a
    single node if hostname is specified.

    When training is cancelled:
    - Training threads on remote nodes are terminated
    - Model files written during training are deleted
    - Session status is set to IDLE (or ERROR if cancellation fails)
    - Training state is cleaned up

OPTIONS
    session_id
        Optional session ID to cancel. If not provided, you will be prompted
        to select from available sessions.

    hostname
        Optional hostname in format "host:port" to cancel training on a
        specific node only. If not provided, cancellation is sent to all
        training nodes.

EXAMPLES
    cancel_training
        Cancel training (will prompt for session ID)
    
    cancel_training session_123
        Cancel training for session_123 on all nodes
    
    cancel_training session_123 192.168.1.1:7001
        Cancel training for session_123 on node 192.168.1.1:7001 only

SEE ALSO
    train_session, create_session
""",
    ),
    "create_session": (
        "Create a new training/inference session",
        """CREATE_SESSION - Create a New Session

SYNOPSIS
    create_session

DESCRIPTION
    Creates a new session through an interactive Q&A flow. A session is a
    container that associates a model with a dataset and manages the
    distribution and execution of training or inference across the network.

    The interactive flow guides you through:
    1. Model selection - Choose from loaded models or predefined models
    2. Dataset selection - Select from discovered datasets or provide a path
    3. Distribution planning - Review and confirm data and model distribution plans
    4. Execution - Execute data and model distribution across nodes

    After creation, you can:
    - Train the model using train_session
    - Run inference using infer (after setting the session with 'use')

    The session status progresses through: IDLE -> RUNNING -> TRAINING/INFERRING -> COMPLETE/ERROR

EXAMPLES
    create_session
        Start the interactive session creation flow

SEE ALSO
    train_session, use, infer, delete_session
""",
    ),
    "delete_session": (
        "Delete a session",
        """DELETE_SESSION - Delete a Session

SYNOPSIS
    delete_session [session_id]

DESCRIPTION
    Deletes a session from the system. This removes the session and all
    associated state, including distribution plans and training statistics.

    WARNING: This action cannot be undone. All session data will be permanently
    removed.

OPTIONS
    session_id
        Optional session ID to delete. If not provided, you will be prompted
        to select from available sessions.

EXAMPLES
    delete_session
        Delete a session (will prompt for session ID)
    
    delete_session session_123
        Delete session_123

SEE ALSO
    create_session, train_session
""",
    ),
    "exit": (
        "Exit the REPL",
        """EXIT - Exit the REPL

SYNOPSIS
    exit
    quit
    q

DESCRIPTION
    Exits the MOSAIC REPL. All three commands (exit, quit, q) perform the
    same action.

    The REPL will cleanly shut down and return to the system shell.

EXAMPLES
    exit
    quit
    q
        All three commands exit the REPL

SEE ALSO
    help
""",
    ),
    "hb": (
        "Show both send and receive heartbeat statuses",
        """HB - Show Heartbeat Statuses

SYNOPSIS
    hb

DESCRIPTION
    Displays both send and receive heartbeat statuses in a single view.
    This is equivalent to running both 'shb' and 'rhb' commands.

    Heartbeats are used to monitor network connectivity and node health.
    The status shows:
    - Send heartbeats: Nodes this orchestrator is sending heartbeats to
    - Receive heartbeats: Nodes that are sending heartbeats to this orchestrator
    - Connection status, latency, and last heartbeat time

EXAMPLES
    hb
        Show all heartbeat statuses

SEE ALSO
    shb, rhb
""",
    ),
    "help": (
        "Show help message or detailed command help",
        """HELP - Show Help

SYNOPSIS
    help [command]

DESCRIPTION
    Displays help information. Without arguments, shows a list of all
    available commands with short descriptions.

    With a command name, displays detailed man page-style help for that
    specific command.

OPTIONS
    command
        Optional command name to get detailed help for. If not provided,
        shows the list of all available commands.

EXAMPLES
    help
        Show list of all commands
    
    help create_session
        Show detailed help for create_session command
    
    help shb
        Show detailed help for shb command

SEE ALSO
    All commands support help via 'help <command>'
""",
    ),
    "infer": (
        "Run inference on the current session",
        """INFER - Run Inference

SYNOPSIS
    infer [input]

DESCRIPTION
    Runs federated inference on the currently active session. This command
    sends inference requests to all nodes that have model shards for the
    session, collects predictions, and aggregates them using the configured
    aggregation method.

    The input data is pre-processed locally before being sent to remote nodes,
    ensuring consistent preprocessing across all nodes. Results are aggregated
    and displayed or saved to file based on the data type.

    If no input is provided, displays advice on what input format is expected
    based on the session's model and data types.

OPTIONS
    input
        Optional input data. Can be:
        - File path to an image, audio, or text file
        - If not provided, shows input format advice

    The input is automatically preprocessed based on the session's data type:
    - IMAGE: Loaded and resized to model input dimensions
    - AUDIO: Loaded and resampled/padded to expected length
    - TEXT: Tokenized based on model type

EXAMPLES
    infer
        Show input format advice
    
    infer /path/to/image.jpg
        Run inference on an image file
    
    infer /path/to/audio.wav
        Run inference on an audio file

SEE ALSO
    use, set_infer_method, create_session
""",
    ),
    "ls": (
        "List sessions, models, or data files",
        """LS - List Resources

SYNOPSIS
    ls <parameter>

DESCRIPTION
    Lists and displays information about sessions, models, or data files in a
    formatted table. This command provides a quick overview of available
    resources in the MOSAIC system.

    The command queries:
    - State manager for session information
    - In-memory model registry and on-disk model files
    - Data directory for available datasets

OPTIONS
    parameter
        What to list. Must be one of:
        - sessions: List all sessions with their status, model IDs, and metadata
        - models: List all models in memory and on disk
        - data: List all data files and directories in the data location

EXAMPLES
    ls sessions
        Display all sessions in a table format showing ID, status, model ID,
        start time, and parent session ID (for shard sessions)
    
    ls models
        Display all models in memory, showing ID, name, type, file name, and
        location. Also shows additional model files found on disk.
    
    ls data
        Display all data files and directories in the configured data location,
        showing name, type (file/directory), size, and relative path.

OUTPUT FORMAT
    All outputs are formatted as tables with aligned columns for easy reading.
    Long values are truncated with "..." to maintain table formatting.

    For sessions:
    - ID: Session unique identifier
    - Status: Current session status (idle, running, training, etc.)
    - Model ID: Associated model identifier
    - Started: Timestamp when session was created
    - Parent ID: Parent session ID for shard sessions (N/A for main sessions)

    For models:
    - ID: Model unique identifier
    - Name: Model name
    - Type: Model type (CNN, TRANSFORMER, etc.)
    - File: Model file name
    - Location: Subdirectory within models_location (or "root")

    For data:
    - Name: File or directory name
    - Type: "file" or "directory"
    - Size: File size (for files) or file count (for directories)
    - Path: Relative path from data location root

SEE ALSO
    create_session, delete_session, use
""",
    ),
    "rhb": (
        "Show receive heartbeat statuses",
        """RHB - Show Receive Heartbeat Statuses

SYNOPSIS
    rhb

DESCRIPTION
    Displays the status of heartbeats received from other nodes in the network.
    This shows which nodes are sending heartbeats to this orchestrator and
    their connection status.

    The output includes:
    - Node hostname and port
    - Connection status (online/offline)
    - Last heartbeat timestamp
    - Latency information

    This is useful for monitoring which nodes are actively connected and
    communicating with this orchestrator.

EXAMPLES
    rhb
        Show receive heartbeat statuses

SEE ALSO
    shb, hb
""",
    ),
    "set_infer_method": (
        "Set inference aggregation method",
        """SET_INFER_METHOD - Set Inference Aggregation Method

SYNOPSIS
    set_infer_method [method]

DESCRIPTION
    Sets the aggregation method used when combining predictions from multiple
    nodes during federated inference. The method determines how predictions
    from different nodes are combined into a final result.

    Available methods:
    - fedavg: Federated Averaging - weighted average of predictions
    - fedprox: Federated Proximal - similar to FedAvg with proximity term
    - weighted_average: Weighted average based on node reliability/data size
    - majority_vote: Majority voting for classification tasks
    - max_vote: Maximum vote aggregation
    - min_vote: Minimum vote aggregation

OPTIONS
    method
        Optional method name. If not provided, you will be prompted to select
        from available methods. The current method is marked in the list.

EXAMPLES
    set_infer_method
        Set method (will prompt for selection)
    
    set_infer_method fedavg
        Set aggregation method to FedAvg
    
    set_infer_method majority_vote
        Set aggregation method to majority voting

SEE ALSO
    infer, use
""",
    ),
    "shb": (
        "Show send heartbeat statuses",
        """SHB - Show Send Heartbeat Statuses

SYNOPSIS
    shb

DESCRIPTION
    Displays the status of heartbeats being sent to other nodes in the network.
    This shows which nodes this orchestrator is sending heartbeats to and
    their connection status.

    The output includes:
    - Node hostname and port
    - Connection status (online/offline)
    - Last heartbeat timestamp
    - Latency information

    This is useful for monitoring network connectivity and identifying nodes
    that may be unreachable or experiencing issues.

EXAMPLES
    shb
        Show send heartbeat statuses

SEE ALSO
    rhb, hb
""",
    ),
    "train_session": (
        "Train a model using a session",
        """TRAIN_SESSION - Train Model Using Session

SYNOPSIS
    train_session [session_id]

DESCRIPTION
    Starts model training for a session. This command orchestrates distributed
    training across all nodes that received data and model shards during
    session creation.

    The training process:
    1. Sets session status to TRAINING
    2. Sends training commands to all participating nodes
    3. Monitors training progress from all nodes
    4. Collects training statistics (loss, epochs, training time)
    5. Updates session status to COMPLETE or ERROR

    Training runs asynchronously on each node, allowing parallel execution.
    The orchestrator tracks progress and collects statistics from all nodes.
    At completion, training statistics are displayed and stored on the session.

    The session must be in RUNNING or IDLE status to start training.

OPTIONS
    session_id
        Optional session ID to train. If not provided, you will be prompted
        to select from available sessions.

EXAMPLES
    train_session
        Train a session (will prompt for session ID)
    
    train_session session_123
        Train session_123

SEE ALSO
    create_session, cancel_training
""",
    ),
    "use": (
        "Set active session for inference",
        """USE - Set Active Session

SYNOPSIS
    use [session_id]

DESCRIPTION
    Sets the active session for inference operations. Only one session can be
    active at a time. The 'infer' command operates on the currently active
    session.

    When a session is set, information about the session is displayed:
    - Session ID and status
    - Associated model ID
    - Data types in the session

    The session must exist and be accessible. Setting a new session replaces
    the previously active session.

OPTIONS
    session_id
        Optional session ID to use. If not provided, you will be prompted to
        select from available sessions.

EXAMPLES
    use
        Set active session (will prompt for session ID)
    
    use session_123
        Set session_123 as the active session

SEE ALSO
    infer, create_session
""",
    ),
}

# Aliases for commands
COMMAND_ALIASES: Dict[str, str] = {
    "quit": "exit",
    "q": "exit",
    "create-session": "create_session",
    "delete-session": "delete_session",
    "train-session": "train_session",
    "cancel-training": "cancel_training",
    "set-infer-method": "set_infer_method",
}


def get_command_help(command: str) -> Tuple[str, str]:
    """
    Get help text for a command.
    
    Args:
        command: Command name (handles aliases)
        
    Returns:
        Tuple of (short_description, long_description)
        
    Raises:
        KeyError: If command not found
    """
    # Check aliases first
    command = COMMAND_ALIASES.get(command.lower(), command.lower())
    
    if command not in COMMAND_HELP:
        raise KeyError(f"Command '{command}' not found")
    
    return COMMAND_HELP[command]


def get_all_commands() -> Dict[str, str]:
    """
    Get all commands with their short descriptions.
    
    Returns:
        Dictionary mapping command names to short descriptions
    """
    return {cmd: desc[0] for cmd, desc in COMMAND_HELP.items()}


def get_command_list_sorted() -> list:
    """
    Get list of all commands sorted alphabetically.
    
    Returns:
        List of (command, short_description) tuples, sorted alphabetically
    """
    commands = list(COMMAND_HELP.items())
    commands.sort(key=lambda x: x[0])
    return [(cmd, desc[0]) for cmd, desc in commands]

