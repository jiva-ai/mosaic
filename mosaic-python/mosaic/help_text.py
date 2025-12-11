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
    help infer <model_name>
    help infer <model_id>

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

    For model-specific inference guidance, use:
        help infer <model_name>
        help infer <model_id>
    
    This will show detailed information about:
    - Expected input format for that specific model
    - Data type requirements
    - Training status and requirements
    - Usage examples

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
    
    help infer my_model
        Show model-specific inference help for a model named "my_model"
    
    help infer abc-123-def-456
        Show model-specific inference help for a model with ID "abc-123-def-456"

IMPORTANT NOTES
    - The model must be trained before running inference
    - Use 'ls models' to see available models and their IDs
    - Use 'use [session_id]' to set the active session before running inference
    - Model-specific help shows training status and input requirements

SEE ALSO
    use, set_infer_method, create_session, train_session, ls models
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
    "quickstart": (
        "Show getting started guide for MOSAIC",
        """QUICKSTART - Getting Started Guide

SYNOPSIS
    quickstart

DESCRIPTION
    Displays a comprehensive getting started guide that walks you through the
    basic MOSAIC workflow. This guide is perfect for new users who want to
    understand how to use MOSAIC for federated machine learning.

    The guide covers:
    - Overview of the MOSAIC workflow (create session, train, infer)
    - Step-by-step instructions for each major command
    - Common use cases and examples
    - Links to additional resources

WORKFLOW OVERVIEW
    The typical MOSAIC workflow consists of three main steps:

    1. CREATE A SESSION
       Sets up a container for your model and data, distributes data and model
       shards across available nodes, and prepares the system for training.

    2. TRAIN THE MODEL
       Trains the model using distributed data shards. Each node trains on its
       assigned data portion, and training statistics are collected.

    3. RUN INFERENCE
       Uses the trained model to make predictions, aggregates results from all
       participating nodes using various aggregation methods.

PREDEFINED MODELS AND DATASETS
    MOSAIC includes several predefined models. Below is information about each
    model, recommended datasets, where to download them, and how to organize
    them in your data directory:

    ResNet-50 / ResNet-101 (CNN - Image Classification):
    ----------------------------------------------------
    Model: resnet50, resnet101
    Data Type: IMAGE
    Recommended Datasets:
      - ImageNet: https://www.image-net.org/download.php
        Large-scale image classification (1000 classes, ~1.2M images)
      - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
        Smaller dataset (10 classes, 60K images) - good for testing
      - CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
        Extended version (100 classes, 60K images)
    
    Data Organization:
        data/
        └── imagenet/
            ├── train/
            │   ├── n01440764/
            │   │   ├── n01440764_18.JPEG
            │   │   └── ...
            │   └── n01443537/
            │       └── ...
            └── val/
                └── ...
    
    For CIFAR-10/100, extract to:
        data/
        └── cifar10/
            ├── data_batch_1
            ├── data_batch_2
            └── ...
    
    Supported formats: .jpg, .jpeg, .png
    Input shape: [batch, 3, 224, 224] (automatically resized)

    Wav2Vec2 (Speech Recognition):
    -------------------------------
    Model: wav2vec2
    Data Type: AUDIO
    Recommended Datasets:
      - LibriSpeech: https://www.openslr.org/12/
        Large-scale speech recognition corpus (1000+ hours)
      - Common Voice: https://commonvoice.mozilla.org/
        Multilingual speech dataset (open source)
      - TIMIT: https://catalog.ldc.upenn.edu/LDC93S1
        Phoneme recognition dataset
    
    Data Organization:
        data/
        └── librispeech/
            ├── train-clean-100/
            │   ├── speaker_id/
            │   │   ├── chapter_id/
            │   │   │   ├── audio1.wav
            │   │   │   └── audio1.txt  (transcription)
            │   │   │   └── ...
            │   │   └── ...
            │   └── ...
            └── dev-clean/
                └── ...
    
    Supported formats: .wav, .flac
    Input: Audio waveform at 16kHz sample rate
    Transcripts: Optional .txt files with same name as audio files

    GPT-Neo (Text Generation):
    ---------------------------
    Model: gpt-neo
    Data Type: TEXT
    Recommended Datasets:
      - The Pile: https://pile.eleuther.ai/
        Large diverse text corpus for language modeling
      - C4 (Colossal Clean Crawled Corpus): https://github.com/allenai/allennlp
        Cleaned web text corpus
      - OpenWebText: https://github.com/jcpeterson/openwebtext
        Open-source recreation of WebText dataset
    
    Data Organization:
        data/
        └── text_corpus/
            ├── document1.txt
            ├── document2.txt
            └── ...
    
    Or JSONL format:
        data/
        └── text_corpus.jsonl
            {"text": "First document text..."}
            {"text": "Second document text..."}
    
    Supported formats: .txt, .jsonl
    Each file or JSONL line contains text samples

    GCN (Graph Node Classification - ogbn-arxiv):
    ----------------------------------------------
    Model: gcn-ogbn-arxiv
    Data Type: GRAPH
    Recommended Dataset:
      - OGB ArXiv: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
        Academic paper citation network (169K nodes, 1.1M edges)
        Download: pip install ogb
        Then: from ogb.nodeproppred import PygNodePropPredDataset
              dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    
    Data Organization:
        data/
        └── ogbn_arxiv/
            ├── graph.json
            └── metadata.json
    
    Graph JSON format:
        {
          "nodes": [{"id": 0, "features": [...]}, ...],
          "edges": [[source, target], ...],
          "labels": [0, 1, ...]
        }
    
    Supported formats: .json, .graphml, .pkl

    BigGAN (Image Generation):
    ---------------------------
    Model: biggan
    Data Type: IMAGE
    Recommended Datasets:
      - ImageNet: https://www.image-net.org/download.php
        Same as ResNet - used for training generative models
      - CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        Face images dataset (200K+ celebrity images)
    
    Data Organization:
        data/
        └── imagenet/
            └── train/
                ├── class1/
                │   ├── img1.jpg
                │   └── ...
                └── class2/
                    └── ...
    
    Supported formats: .jpg, .jpeg, .png
    Input shape: [batch, 3, 128, 128] for generation
    Class labels: Required for conditional generation

    PPO (Reinforcement Learning):
    ------------------------------
    Model: ppo
    Data Type: RL
    Recommended Datasets:
      - OpenAI Gym environments: https://gym.openai.com/
        Standard RL environments (CartPole, Atari, etc.)
      - Custom trajectory data: Recorded episodes from RL environments
    
    Data Organization:
        data/
        └── rl_trajectories/
            ├── trajectory_001.json
            ├── trajectory_002.json
            └── ...
    
    Trajectory JSON format:
        {
          "observations": [[obs1], [obs2], ...],
          "actions": [action1, action2, ...],
          "rewards": [reward1, reward2, ...],
          "dones": [False, False, ..., True]
        }
    
    Supported formats: .json
    Each file contains one or more episodes

DOWNLOADING AND SETUP
    For most datasets:
    1. Download the dataset from the provided links
    2. Extract/unpack to your data directory (as configured in config)
    3. Organize according to the structure shown above
    4. Ensure file formats match supported extensions
    5. For classification tasks, organize by class folders if possible
    
    Quick setup examples:
      # ImageNet (example)
      mkdir -p data/imagenet
      # Download and extract ImageNet to data/imagenet/
      
      # LibriSpeech (example)
      mkdir -p data/librispeech
      # Download LibriSpeech and extract to data/librispeech/
      
      # Text corpus (example)
      mkdir -p data/text_corpus
      # Download text files and place in data/text_corpus/

EXAMPLES
    quickstart
        Display the complete getting started guide

SEE ALSO
    create_session, train_session, infer, help, ls data
    https://github.com/jiva-ai/mosaic
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

DATA DIRECTORY STRUCTURE
    The data directory (as defined in your config) should be organized based on
    the model type and data type you're using. Below are the recommended
    structures for common combinations:

    IMAGE DATA (CNN, VIT, VAE, DIFFUSION models):
    ---------------------------------------------
    Directory structure:
        data/
        └── your_dataset/
            ├── image1.jpg
            ├── image2.png
            ├── image3.jpg
            └── ...
    
    Supported formats: .jpg, .jpeg, .png
    - Can be a single directory containing all images
    - Images are automatically loaded and preprocessed
    - For classification: labels typically inferred from directory structure
      or provided separately
    
    Example:
        data/
        └── cats_and_dogs/
            ├── cat_001.jpg
            ├── cat_002.jpg
            ├── dog_001.jpg
            └── dog_002.jpg

    AUDIO DATA (WAV2VEC models):
    -----------------------------
    Directory structure:
        data/
        └── your_dataset/
            ├── audio1.wav
            ├── audio2.flac
            ├── audio3.wav
            └── ...
    
    Supported formats: .wav, .flac
    - Can be a single directory containing all audio files
    - Audio files are loaded and resampled to expected sample rate
    - For speech recognition: transcriptions typically in separate .txt files
      or embedded in filenames
    
    Example:
        data/
        └── speech_data/
            ├── sample_001.wav
            ├── sample_002.flac
            └── transcriptions.txt  (optional, separate file)

    TEXT DATA (TRANSFORMER, BERT, RNN, LSTM models):
    ------------------------------------------------
    Directory structure:
        data/
        └── your_dataset/
            ├── text1.txt
            ├── text2.txt
            ├── text3.jsonl
            └── ...
    
    Supported formats: .txt, .jsonl
    - Can be a single directory containing text files
    - Each file can contain one or multiple samples
    - For .jsonl: each line is a JSON object with text data
    - Text is tokenized based on model requirements
    
    Example (single file):
        data/
        └── corpus.txt  (one large text file)
    
    Example (multiple files):
        data/
        └── documents/
            ├── doc1.txt
            ├── doc2.txt
            └── doc3.txt
    
    Example (JSONL):
        data/
        └── dataset.jsonl
            {"text": "First sample text..."}
            {"text": "Second sample text..."}

    GRAPH DATA (GNN models):
    ------------------------
    Directory structure:
        data/
        └── your_dataset/
            ├── graph1.json
            ├── graph2.json
            └── ...
    
    Supported formats: .json, .graphml, .pkl
    - Each file represents a graph with nodes, edges, and features
    - JSON format should contain:
      {
        "nodes": [...],
        "edges": [...],
        "features": [...],
        "labels": [...]  (optional)
      }
    
    Example:
        data/
        └── graphs/
            ├── graph_001.json
            └── graph_002.json

    CSV DATA (CNN, TRANSFORMER models):
    ------------------------------------
    Directory structure:
        data/
        └── your_dataset.csv
    
    Supported formats: .csv
    - Single CSV file with features and optional target column
    - First row typically contains column headers
    - Target column should be specified during session creation
    
    Example:
        data/
        └── tabular_data.csv
            feature1,feature2,feature3,target
            1.0,2.0,3.0,0
            4.0,5.0,6.0,1

    RL DATA (RL models):
    --------------------
    Directory structure:
        data/
        └── your_dataset/
            ├── trajectory1.json
            ├── trajectory2.json
            └── ...
    
    Supported formats: .json
    - Each file contains trajectory data (observations, actions, rewards)
    - Format should include:
      {
        "observations": [...],
        "actions": [...],
        "rewards": [...],
        "dones": [...]  (optional)
      }
    
    Example:
        data/
        └── rl_trajectories/
            ├── traj_001.json
            └── traj_002.json

    DIRECTORY (DIR data type):
    ---------------------------
    Directory structure:
        data/
        └── your_dataset/
            └── subdirectory/
                ├── file1.ext
                ├── file2.ext
                └── ...
    
    - Points to a directory containing mixed or organized data
    - System will discover and process files based on extensions
    - Useful for datasets with nested organization
    
    Example:
        data/
        └── mixed_dataset/
            ├── images/
            │   ├── img1.jpg
            │   └── img2.png
            └── metadata/
                └── labels.csv

GENERAL GUIDELINES
    - All paths are relative to the data_location configured in your config
    - Use descriptive directory/file names
    - Ensure sufficient disk space for your dataset
    - For large datasets, consider organizing into subdirectories
    - File extensions must match the expected formats for your data type
    - Labels/targets can be:
      * Embedded in directory structure (e.g., class folders)
      * In separate annotation files
      * Inferred from filenames
      * Specified during session creation

PREDEFINED MODELS AND DATASET DOWNLOADS
    MOSAIC includes several predefined models. Below is information about each
    model, recommended datasets, where to download them, and how to organize
    them in your data directory:

    ResNet-50 / ResNet-101 (CNN - Image Classification):
    ----------------------------------------------------
    Model: resnet50, resnet101
    Data Type: IMAGE
    Recommended Datasets:
      - ImageNet: https://www.image-net.org/download.php
        Large-scale image classification (1000 classes, ~1.2M images)
        Requires registration and agreement to terms
      - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
        Smaller dataset (10 classes, 60K images) - good for testing
        Direct download: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
      - CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
        Extended version (100 classes, 60K images)
        Direct download: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    
    Data Organization for ImageNet:
        data/
        └── imagenet/
            ├── train/
            │   ├── n01440764/  (class folder)
            │   │   ├── n01440764_18.JPEG
            │   │   └── ...
            │   └── n01443537/
            │       └── ...
            └── val/
                └── ...
    
    Data Organization for CIFAR-10/100:
        data/
        └── cifar10/
            ├── data_batch_1
            ├── data_batch_2
            └── ...
        # Or extract images to:
        data/
        └── cifar10/
            ├── airplane/
            │   ├── img1.png
            │   └── ...
            └── automobile/
                └── ...
    
    Supported formats: .jpg, .jpeg, .png
    Input shape: [batch, 3, 224, 224] (automatically resized)

    Wav2Vec2 (Speech Recognition):
    -------------------------------
    Model: wav2vec2
    Data Type: AUDIO
    Recommended Datasets:
      - LibriSpeech: https://www.openslr.org/12/
        Large-scale speech recognition corpus (1000+ hours)
        Direct download links:
          * train-clean-100: http://www.openslr.org/resources/12/train-clean-100.tar.gz
          * train-clean-360: http://www.openslr.org/resources/12/train-clean-360.tar.gz
          * dev-clean: http://www.openslr.org/resources/12/dev-clean.tar.gz
          * test-clean: http://www.openslr.org/resources/12/test-clean.tar.gz
      - Common Voice: https://commonvoice.mozilla.org/
        Multilingual speech dataset (open source, requires registration)
      - TIMIT: https://catalog.ldc.upenn.edu/LDC93S1
        Phoneme recognition dataset (requires LDC membership)
    
    Data Organization for LibriSpeech:
        data/
        └── librispeech/
            ├── train-clean-100/
            │   ├── 19/
            │   │   ├── 198/
            │   │   │   ├── 19-198-0001.flac
            │   │   │   ├── 19-198-0001.txt  (transcription)
            │   │   │   ├── 19-198-0002.flac
            │   │   │   └── ...
            │   │   └── ...
            │   └── ...
            └── dev-clean/
                └── ...
    
    Supported formats: .wav, .flac
    Input: Audio waveform at 16kHz sample rate
    Transcripts: Optional .txt files with same name as audio files

    GPT-Neo (Text Generation):
    ---------------------------
    Model: gpt-neo
    Data Type: TEXT
    Recommended Datasets:
      - The Pile: https://pile.eleuther.ai/
        Large diverse text corpus for language modeling
        Download: Requires access request
      - C4 (Colossal Clean Crawled Corpus): 
        https://github.com/allenai/allennlp
        Cleaned web text corpus
        Available via TensorFlow Datasets: tfds.load('c4')
      - OpenWebText: https://github.com/jcpeterson/openwebtext
        Open-source recreation of WebText dataset
        Download: Follow instructions in repository
      - WikiText-103: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
        Wikipedia text dataset
        Direct download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    
    Data Organization (single directory):
        data/
        └── text_corpus/
            ├── document1.txt
            ├── document2.txt
            └── ...
    
    Data Organization (JSONL format):
        data/
        └── text_corpus.jsonl
            {"text": "First document text..."}
            {"text": "Second document text..."}
    
    Supported formats: .txt, .jsonl
    Each file or JSONL line contains text samples

    GCN (Graph Node Classification - ogbn-arxiv):
    ----------------------------------------------
    Model: gcn-ogbn-arxiv
    Data Type: GRAPH
    Recommended Dataset:
      - OGB ArXiv: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
        Academic paper citation network (169K nodes, 1.1M edges)
        
        Download via Python:
          pip install ogb
          from ogb.nodeproppred import PygNodePropPredDataset
          dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
          # Dataset will be downloaded to data/ogbg_nodeproppred/ogbn-arxiv/
    
    Data Organization:
        data/
        └── ogbn_arxiv/
            ├── graph.json
            └── metadata.json
    
    Graph JSON format:
        {
          "nodes": [{"id": 0, "features": [...]}, ...],
          "edges": [[source, target], ...],
          "labels": [0, 1, ...]
        }
    
    Supported formats: .json, .graphml, .pkl
    The OGB library handles download and preprocessing automatically

    BigGAN (Image Generation):
    ---------------------------
    Model: biggan
    Data Type: IMAGE
    Recommended Datasets:
      - ImageNet: https://www.image-net.org/download.php
        Same as ResNet - used for training generative models
        See ResNet-50 section above for download and organization
      - CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        Face images dataset (200K+ celebrity images)
        Direct download: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        (Requires Google Drive access)
    
    Data Organization for ImageNet:
        data/
        └── imagenet/
            └── train/
                ├── n01440764/  (class folder)
                │   ├── img1.jpg
                │   └── ...
                └── n01443537/
                    └── ...
    
    Data Organization for CelebA:
        data/
        └── celeba/
            ├── img_align_celeba/
            │   ├── 000001.jpg
            │   ├── 000002.jpg
            │   └── ...
            └── list_attr_celeba.txt  (optional attributes)
    
    Supported formats: .jpg, .jpeg, .png
    Input shape: [batch, 3, 128, 128] for generation
    Class labels: Required for conditional generation (from ImageNet classes)

    PPO (Reinforcement Learning):
    ------------------------------
    Model: ppo
    Data Type: RL
    Recommended Datasets/Environments:
      - OpenAI Gym: https://gym.openai.com/
        Standard RL environments (CartPole, Atari, etc.)
        Install: pip install gym
        Environments generate data on-the-fly during training
      - Custom trajectory data: Recorded episodes from RL environments
    
    Data Organization (for pre-recorded trajectories):
        data/
        └── rl_trajectories/
            ├── trajectory_001.json
            ├── trajectory_002.json
            └── ...
    
    Trajectory JSON format:
        {
          "observations": [[obs1], [obs2], ...],
          "actions": [action1, action2, ...],
          "rewards": [reward1, reward2, ...],
          "dones": [False, False, ..., True]
        }
    
    Supported formats: .json
    Each file contains one or more episodes
    Note: For most RL training, data is generated during training from
          the environment, so pre-recorded data is optional

DOWNLOADING AND SETUP INSTRUCTIONS
    General Steps:
    1. Identify your model type and required data type
    2. Download the recommended dataset from the links above
    3. Extract/unpack the dataset
    4. Organize files according to the structure shown above
    5. Place in your configured data directory (see config.data_location)
    6. Verify file formats match supported extensions
    7. For classification tasks, organize by class folders when possible
    
    Quick Setup Examples:
    
      # ImageNet (requires registration)
      mkdir -p data/imagenet
      # Download ImageNet from https://www.image-net.org/
      # Extract and organize into train/val folders with class subfolders
      
      # CIFAR-10 (quick test)
      mkdir -p data/cifar10
      wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
      tar -xzf cifar-10-python.tar.gz -C data/cifar10
      
      # LibriSpeech
      mkdir -p data/librispeech
      wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
      tar -xzf train-clean-100.tar.gz -C data/librispeech
      
      # WikiText-103 (text)
      mkdir -p data/wikitext
      wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
      unzip wikitext-103-v1.zip -d data/wikitext
      
      # OGB ArXiv (graph - automatic download)
      pip install ogb
      python -c "from ogb.nodeproppred import PygNodePropPredDataset; \
                 dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')"
    
    Verification:
      Use 'ls data' to verify your data is organized correctly
      Check that file extensions match supported formats
      Ensure directory structure matches the examples above

EXAMPLES
    train_session
        Train a session (will prompt for session ID)
    
    train_session session_123
        Train session_123

SEE ALSO
    create_session, cancel_training, ls data, help infer <model_name>
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

