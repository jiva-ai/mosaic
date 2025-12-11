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


def execute_ls(output_fn: Callable[[str], None], parameter: Optional[str] = None) -> None:
    """
    Execute ls command - list sessions, models, or data.
    
    Args:
        output_fn: Function to call with output text
        parameter: What to list - "sessions", "models", or "data"
    """
    from mosaic.mosaic import _session_manager, _models, _config
    from pathlib import Path
    from datetime import datetime
    from mosaic_config.state import SessionStatus
    
    if parameter is None:
        output_fn("Usage: ls <parameter>\n")
        output_fn("Parameters: sessions, models, data\n")
        return
    
    param = parameter.lower()
    
    if param == "sessions":
        if _session_manager is None:
            output_fn("Error: Session manager not initialized\n")
            return
        
        sessions = _session_manager.get_sessions()
        
        if not sessions:
            output_fn("No sessions found.\n")
            return
        
        # Format sessions table
        header = f"{'ID':<38} {'Status':<15} {'Model ID':<38} {'Started':<20} {'Parent ID':<38}"
        separator = "-" * len(header)
        lines = [header, separator]
        
        for session in sessions:
            session_id = session.id[:36] + "..." if len(session.id) > 36 else session.id
            status = session.status.value if isinstance(session.status, SessionStatus) else str(session.status)
            model_id = (session.model_id[:36] + "..." if session.model_id and len(session.model_id) > 36 else session.model_id) or "N/A"
            
            # Format timestamp
            try:
                started_time = datetime.fromtimestamp(session.time_started / 1000.0)
                started_str = started_time.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OSError):
                started_str = "N/A"
            
            parent_id = (session.parent_id[:36] + "..." if session.parent_id and len(session.parent_id) > 36 else session.parent_id) or "N/A"
            
            row = f"{session_id:<38} {status:<15} {model_id:<38} {started_str:<20} {parent_id:<38}"
            lines.append(row)
        
        output_fn("\n".join(lines) + "\n")
    
    elif param == "models":
        if _models is None:
            output_fn("Error: Models not initialized\n")
            return
        
        if not _models:
            output_fn("No models found in memory.\n")
            # Also check disk
            if _config and _config.models_location:
                models_path = Path(_config.models_location)
                if models_path.exists():
                    model_files = list(models_path.rglob("*.onnx"))
                    if model_files:
                        output_fn(f"\nFound {len(model_files)} model file(s) on disk:\n")
                        header = f"{'File Path':<60} {'Size':<15}"
                        separator = "-" * len(header)
                        lines = [header, separator]
                        for model_file in sorted(model_files):
                            try:
                                size = model_file.stat().st_size
                                size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.2f} KB"
                            except OSError:
                                size_str = "N/A"
                            rel_path = str(model_file.relative_to(models_path))
                            rel_path = (rel_path[:57] + "...") if len(rel_path) > 57 else rel_path
                            lines.append(f"{rel_path:<60} {size_str:<15}")
                        output_fn("\n".join(lines) + "\n")
            return
        
        # Format models table
        header = f"{'ID':<38} {'Name':<30} {'Type':<15} {'File':<40} {'Location':<30}"
        separator = "-" * len(header)
        lines = [header, separator]
        
        for model in _models:
            model_id = model.id[:36] + "..." if len(model.id) > 36 else model.id
            name = (model.name[:28] + "...") if len(model.name) > 28 else model.name
            model_type = model.model_type.value if model.model_type else "N/A"
            file_name = (model.file_name[:38] + "...") if model.file_name and len(model.file_name) > 38 else (model.file_name or "N/A")
            onnx_location = (model.onnx_location[:28] + "...") if model.onnx_location and len(model.onnx_location) > 28 else (model.onnx_location or "root")
            
            row = f"{model_id:<38} {name:<30} {model_type:<15} {file_name:<40} {onnx_location:<30}"
            lines.append(row)
        
        output_fn("\n".join(lines) + "\n")
        
        # Also show disk files if config available
        if _config and _config.models_location:
            models_path = Path(_config.models_location)
            if models_path.exists():
                model_files = list(models_path.rglob("*.onnx"))
                if model_files:
                    output_fn(f"\nAdditional model files on disk ({len(model_files)} total):\n")
                    # Show first 10 files
                    for model_file in sorted(model_files)[:10]:
                        rel_path = str(model_file.relative_to(models_path))
                        try:
                            size = model_file.stat().st_size
                            size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.2f} KB"
                        except OSError:
                            size_str = "N/A"
                        output_fn(f"  {rel_path:<60} {size_str}\n")
                    if len(model_files) > 10:
                        output_fn(f"  ... and {len(model_files) - 10} more\n")
    
    elif param == "data":
        if _config is None or not _config.data_location:
            output_fn("Error: Data location not configured\n")
            return
        
        data_path = Path(_config.data_location)
        
        if not data_path.exists():
            output_fn(f"Data directory does not exist: {data_path}\n")
            return
        
        # List data files and directories
        data_items = []
        
        # Get all files and directories
        try:
            for item in sorted(data_path.iterdir()):
                if item.is_dir():
                    # Count files in directory
                    file_count = len(list(item.rglob("*"))) - len(list(item.rglob("*/")))
                    data_items.append({
                        "name": item.name,
                        "type": "directory",
                        "size": f"{file_count} files",
                        "path": str(item.relative_to(data_path))
                    })
                elif item.is_file():
                    try:
                        size = item.stat().st_size
                        size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.2f} KB"
                    except OSError:
                        size_str = "N/A"
                    data_items.append({
                        "name": item.name,
                        "type": "file",
                        "size": size_str,
                        "path": str(item.relative_to(data_path))
                    })
        except PermissionError:
            output_fn(f"Error: Permission denied accessing {data_path}\n")
            return
        
        if not data_items:
            output_fn("No data files or directories found.\n")
            return
        
        # Format data table
        header = f"{'Name':<40} {'Type':<12} {'Size':<15} {'Path':<50}"
        separator = "-" * len(header)
        lines = [header, separator]
        
        for item in data_items[:50]:  # Limit to 50 items
            name = (item["name"][:38] + "...") if len(item["name"]) > 38 else item["name"]
            item_type = item["type"]
            size = item["size"]
            path = (item["path"][:48] + "...") if len(item["path"]) > 48 else item["path"]
            
            row = f"{name:<40} {item_type:<12} {size:<15} {path:<50}"
            lines.append(row)
        
        output_fn("\n".join(lines) + "\n")
        
        if len(data_items) > 50:
            output_fn(f"... and {len(data_items) - 50} more items\n")
    
    else:
        output_fn(f"Unknown parameter: {parameter}\n")
        output_fn("Usage: ls <parameter>\n")
        output_fn("Parameters: sessions, models, data\n")


def execute_quickstart(output_fn: Callable[[str], None]) -> None:
    """
    Execute quickstart command - show getting started guide.
    
    Args:
        output_fn: Function to call with output text
    """
    guide = """MOSAIC - Quick Start Guide
================================================================================

Welcome to MOSAIC! This guide will walk you through the basic workflow of
creating a session, training a model, and running inference.

WORKFLOW OVERVIEW
================================================================================

The typical MOSAIC workflow consists of three main steps:

1. CREATE A SESSION
   - Sets up a container for your model and data
   - Distributes data and model shards across available nodes
   - Prepares the system for training

2. TRAIN THE MODEL
   - Trains the model using distributed data shards
   - Each node trains on its assigned data portion
   - Training statistics are collected and displayed

3. RUN INFERENCE
   - Uses the trained model to make predictions
   - Aggregates results from all participating nodes
   - Supports various aggregation methods (FedAvg, Majority Vote, etc.)

STEP-BY-STEP COMMANDS
================================================================================

Step 1: Create a Session
-------------------------
Command: create_session

This interactive command will guide you through:
  • Selecting a model (from disk or predefined models)
  • Choosing a dataset (searches your data directory)
  • Reviewing the distribution plan
  • Confirming data and model distribution

Example:
  mosaic> create_session

After creation, you'll be asked if you want to train immediately.
Answer 'yes' to proceed to training, or 'no' to train later.

Step 2: Train the Model
------------------------
Command: train_session [session_id]

If you didn't train during session creation, use this command to start training.
You can specify a session ID, or the system will prompt you to select one.

Example:
  mosaic> train_session
  mosaic> train_session abc-123-def-456

Training runs asynchronously across all nodes. You'll see status updates
as each node reports progress. Training statistics are displayed at completion.

Step 3: Run Inference
----------------------
Commands: use [session_id]
          infer [input_file]

First, set the active session for inference:
  mosaic> use
  mosaic> use abc-123-def-456

Then run inference on your data:
  mosaic> infer /path/to/image.jpg
  mosaic> infer /path/to/audio.wav
  mosaic> infer /path/to/text.txt

The system will:
  • Preprocess your input based on the model type
  • Send inference requests to all participating nodes
  • Aggregate predictions using the configured method
  • Display or save results

PREDEFINED MODELS AND DATASETS
================================================================================

MOSAIC includes several predefined models. Here's a quick reference:

  ResNet-50/101 (CNN) - Image Classification
    Models: resnet50, resnet101
    Datasets: ImageNet, CIFAR-10, CIFAR-100
    Download: https://www.image-net.org/ (ImageNet)
              https://www.cs.toronto.edu/~kriz/cifar.html (CIFAR)
    Data: Organize images in class folders or flat directory
    Format: .jpg, .jpeg, .png

  Wav2Vec2 - Speech Recognition
    Model: wav2vec2
    Datasets: LibriSpeech, Common Voice
    Download: https://www.openslr.org/12/ (LibriSpeech)
    Data: Audio files (.wav, .flac) with optional transcriptions
    Format: 16kHz sample rate

  GPT-Neo - Text Generation
    Model: gpt-neo
    Datasets: The Pile, C4, WikiText-103
    Download: https://pile.eleuther.ai/ (The Pile)
              https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    Data: Text files (.txt) or JSONL format
    Format: One document per file or JSONL line

  GCN - Graph Node Classification
    Model: gcn-ogbn-arxiv
    Dataset: OGB ArXiv
    Download: pip install ogb
              from ogb.nodeproppred import PygNodePropPredDataset
              dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
    Data: Graph JSON files with nodes, edges, features
    Format: .json, .graphml, .pkl

  BigGAN - Image Generation
    Model: biggan
    Datasets: ImageNet, CelebA
    Download: https://www.image-net.org/ (ImageNet)
    Data: Images organized by class folders
    Format: .jpg, .jpeg, .png

  PPO - Reinforcement Learning
    Model: ppo
    Datasets: OpenAI Gym environments
    Download: pip install gym
    Data: Trajectory JSON files or generated on-the-fly
    Format: .json with observations, actions, rewards

For detailed download instructions and data organization, see:
  help train_session

USEFUL COMMANDS
================================================================================

List Resources:
  ls sessions    - List all sessions
  ls models      - List all models
  ls data        - List data files

Session Management:
  delete_session [id]  - Delete a session
  cancel_training [id]  - Cancel ongoing training

Inference Configuration:
  set_infer_method [method]  - Set aggregation method (fedavg, majority_vote, etc.)

Help:
  help              - List all commands
  help <command>    - Detailed help for a specific command
  help infer <model> - Model-specific inference help
  quickstart        - Show this guide again

EXAMPLES
================================================================================

Complete workflow example:

  mosaic> create_session
  [Interactive session creation...]
  Train model now? (yes/no): yes
  [Training starts...]

  mosaic> use
  [Select session for inference]
  mosaic> infer /path/to/test_image.jpg
  [Inference results displayed]

Viewing resources:

  mosaic> ls sessions
  mosaic> ls models
  mosaic> ls data

Getting help:

  mosaic> help create_session
  mosaic> help train_session
  mosaic> help infer

ADDITIONAL RESOURCES
================================================================================

For more detailed documentation, examples, and advanced features, visit:
  https://github.com/jiva-ai/mosaic

The repository includes:
  • Complete API documentation
  • Configuration guides
  • Advanced usage examples
  • Troubleshooting tips

================================================================================
"""
    output_fn(guide)


def _get_model_specific_inference_help(model_name_or_id: str, output_fn: Callable[[str], None]) -> None:
    """
    Get model-specific inference help based on model name or ID.
    
    Args:
        model_name_or_id: Model name or ID to look up
        output_fn: Function to call with output text
    """
    from mosaic.mosaic import _models, _session_manager
    from mosaic_config.state import ModelType, DataType
    from mosaic.inference_advice import get_inference_advice
    
    if _models is None:
        output_fn("Error: Models not initialized\n")
        return
    
    # Find model by name or ID
    model = None
    for m in _models:
        if m.name.lower() == model_name_or_id.lower() or m.id == model_name_or_id:
            model = m
            break
    
    if model is None:
        output_fn(f"Error: Model '{model_name_or_id}' not found.\n")
        output_fn("Use 'ls models' to see available models.\n")
        return
    
    # Get model type
    model_type = model.model_type
    
    # Try to infer data type from sessions using this model
    data_type = None
    if _session_manager:
        sessions = _session_manager.get_sessions()
        for session in sessions:
            if session.model_id == model.id and session.data and session.data.file_definitions:
                data_type = session.data.file_definitions[0].data_type
                break
    
    # If no data type from sessions, infer from model type
    if data_type is None and model_type:
        # Map model types to likely data types
        model_to_data_type = {
            ModelType.CNN: DataType.IMAGE,
            ModelType.VIT: DataType.IMAGE,
            ModelType.VAE: DataType.IMAGE,
            ModelType.DIFFUSION: DataType.IMAGE,
            ModelType.WAV2VEC: DataType.AUDIO,
            ModelType.TRANSFORMER: DataType.TEXT,
            ModelType.BERT: DataType.TEXT,
            ModelType.RNN: DataType.TEXT,
            ModelType.LSTM: DataType.TEXT,
            ModelType.GNN: DataType.GRAPH,
            ModelType.RL: DataType.RL,
        }
        data_type = model_to_data_type.get(model_type)
    
    # Generate comprehensive help
    help_text = f"""INFER - Model-Specific Inference Help
{'=' * 70}

Model Information:
  Name: {model.name}
  ID: {model.id}
  Type: {model_type.value if model_type else 'Unknown'}
  Trained: {'Yes' if model.trained else 'No'}

"""
    
    # Training requirement warning
    if not model.trained:
        help_text += """⚠️  IMPORTANT: This model has not been trained yet!
   Before running inference, you must:
   1. Create a session with this model: create_session
   2. Train the model: train_session [session_id]
   3. Then you can run inference: infer [input]

"""
    else:
        help_text += """✓ Model is trained and ready for inference.

"""
    
    # Data type information
    if data_type:
        help_text += f"Expected Data Type: {data_type.value.upper()}\n\n"
    else:
        help_text += "Expected Data Type: Unknown (check model documentation)\n\n"
    
    # Get inference advice
    advice = get_inference_advice(model_type, data_type)
    help_text += "Input Format Requirements:\n"
    help_text += "-" * 70 + "\n"
    help_text += advice + "\n\n"
    
    # Additional guidance
    help_text += "Usage Steps:\n"
    help_text += "-" * 70 + "\n"
    help_text += "1. Ensure you have a session with this model:\n"
    help_text += "   ls sessions\n"
    help_text += "\n"
    help_text += "2. Set the active session for inference:\n"
    help_text += "   use [session_id]\n"
    help_text += "\n"
    help_text += "3. Run inference with your input:\n"
    help_text += "   infer /path/to/input_file\n"
    help_text += "   or\n"
    help_text += "   infer <input_data>\n"
    help_text += "\n"
    
    # Model-specific examples
    if model_type == ModelType.CNN and data_type == DataType.IMAGE:
        help_text += "Example:\n"
        help_text += "  infer /path/to/image.jpg\n"
        help_text += "  infer /path/to/image.png\n"
    elif model_type == ModelType.WAV2VEC and data_type == DataType.AUDIO:
        help_text += "Example:\n"
        help_text += "  infer /path/to/audio.wav\n"
        help_text += "  infer /path/to/audio.flac\n"
    elif model_type in (ModelType.TRANSFORMER, ModelType.BERT) and data_type == DataType.TEXT:
        help_text += "Example:\n"
        help_text += "  infer /path/to/text.txt\n"
        help_text += "  infer 'Your text here'\n"
    
    help_text += "\n" + "=" * 70 + "\n"
    help_text += "For general inference help, type: help infer\n"
    help_text += "To see all models, type: ls models\n"
    
    output_fn(help_text)


def execute_help(output_fn: Callable[[str], None], command: Optional[str] = None) -> None:
    """
    Execute help command - show help message or detailed command help.
    
    Args:
        output_fn: Function to call with output text
        command: Optional command name for detailed help, or "infer <model_name>" for model-specific help
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
        # Check if this is "infer <model_name>" format
        parts = command.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "infer":
            # Model-specific inference help
            model_name_or_id = parts[1]
            _get_model_specific_inference_help(model_name_or_id, output_fn)
            return
        
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
        elif cmd == "ls":
            parameter = args[0] if args else None
            execute_ls(output_fn, parameter)
        elif cmd == "set_infer_method" or cmd == "set-infer-method":
            from mosaic.session_commands import execute_set_infer_method
            method = args[0] if args else None
            execute_set_infer_method(output_fn, method)
        elif cmd == "help":
            # Handle "help infer <model_name>" format
            if args and len(args) >= 2 and args[0].lower() == "infer":
                help_command = f"infer {args[1]}"
            else:
                help_command = " ".join(args) if args else None
            execute_help(output_fn, help_command)
        elif cmd == "quickstart":
            execute_quickstart(output_fn)
        else:
            output_fn("Unknown command. Type 'help' for available commands.\n")
    except Exception as e:
        output_fn(f"Error: {e}\n")

