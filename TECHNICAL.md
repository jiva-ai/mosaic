# MOSAIC Technical Documentation

This document provides comprehensive technical documentation about how MOSAIC processes work, component architecture, and implementation details. The information here applies to both the Python implementation (currently available) and the Java implementation (in development), as both implementations follow the same logical architecture.

## Table of Contents

1. [Configuration](#configuration)
2. [Models, Sessions, and Plans](#models-sessions-and-plans)
3. [Models Represented by ONNX](#models-represented-by-onnx)
4. [Out-of-the-Box Models](#out-of-the-box-models)
5. [Communication and Beacon System](#communication-and-beacon-system)
6. [Planner and Model Runtime](#planner-and-model-runtime)
7. [Data Sharding Algorithms](#data-sharding-algorithms)

---

## Configuration

MOSAIC is designed to be **entirely configuration-driven**. All aspects of node behavior, network topology, resource management, and data handling are controlled through configuration files, making the system highly flexible and adaptable to different deployment scenarios.

### Configuration File Structure

Configuration is provided as a JSON file that defines:

- **Network Settings**: Host address, ports for heartbeat and general communication
- **Peer Configuration**: List of peer nodes to communicate with, including their hostnames and ports
- **Heartbeat Parameters**: Frequency, tolerance thresholds, report length, and timeout values
- **Security**: Paths to SSL/TLS certificates (server certificate, private key, and CA certificate)
- **Resource Management**: Benchmark data location, whether to run benchmarks at startup
- **Data Management**: Locations for data, models, plans, and state files
- **Data Transmission**: Chunk size for data transfer operations

### Configuration Priority

The system searches for configuration files in the following priority order:

1. **Command-line argument**: Explicitly specified via `--config` parameter
2. **Environment variable**: `MOSAIC_CONFIG` environment variable
3. **Current working directory**: `mosaic.config` file in the current directory

This allows flexibility in deployment scenarios, from development (local config file) to containerized environments (environment variables) to explicit control (command-line).

### Path Expansion

Configuration paths support:
- **Home directory expansion**: `~/config/mosaic.json` expands to the user's home directory
- **Environment variable expansion**: `$HOME/config/mosaic.json` or `${VAR}/path` syntax

### Configuration-Driven Behavior

All major system behaviors are controlled through configuration:

- **Network topology**: Which nodes communicate with which other nodes
- **Heartbeat behavior**: How often nodes report their status and capabilities
- **Resource allocation**: How the system determines node capabilities and allocates workloads
- **Data distribution**: How data is segmented and distributed across the network
- **Model distribution**: How models are compressed and distributed based on node capabilities
- **Security**: SSL/TLS certificate locations and validation settings

This configuration-driven approach ensures that the same codebase can be deployed across diverse environments—from edge devices to cloud clusters—simply by changing configuration files.

For more detailed configuration information, see [CONFIG.md](CONFIG.md).

---

## Models, Sessions, and Plans

MOSAIC operates on three fundamental abstractions that represent different stages of the machine learning workflow: **Models**, **Plans**, and **Sessions**.

### Models

A **Model** represents a machine learning model in the MOSAIC network. Each model has:

- **Name**: A unique identifier for the model
- **Model Type**: Classification of the model architecture (CNN, Transformer, GNN, etc.)
- **ONNX Representation**: The model stored in ONNX format, either as a file location or binary data
- **Training Status**: Whether the model has been trained

Models are the fundamental units of computation in MOSAIC. They can be predefined (created from standard architectures) or custom (loaded from ONNX files).

### Plans

A **Plan** represents a distribution strategy for training or inference. A plan contains:

- **Distribution Plan**: The result of calculating how data and workloads should be distributed across nodes based on their capabilities
- **Model**: The model instance that will be used
- **Stats Data**: Machine statistics from all nodes in the network, used to make distribution decisions
- **Data Segmentation Plan**: Detailed plan of how data files are segmented and which segments go to which nodes

Plans are created by analyzing:
1. The capabilities of all nodes in the network (from heartbeat data)
2. The model architecture and requirements
3. The data to be processed

The planning process determines:
- Which nodes will participate in training/inference
- How much data each node will process
- How models should be modified (compressed) for each node's capabilities
- How data should be segmented and distributed

### Sessions

A **Session** represents an active execution of a plan. A session tracks:

- **Plan**: The distribution plan being executed
- **Data**: The data instance being processed (if applicable)
- **Model**: The model instance being used (if applicable)
- **Status**: Current state of the session (IDLE, TRAINING, INFERRING, COMPLETE, etc.)
- **Timing**: Start and end times for the session
- **Unique ID**: Identifier for tracking the session

Sessions provide the execution context for training or inference operations. They maintain state throughout the lifecycle of a distributed ML operation, from initial planning through data distribution, model training/inference, and result aggregation.

### Relationship Between Components

The workflow typically follows this pattern:

1. **Model Creation**: A model is created or loaded into the system
2. **Plan Creation**: A plan is generated that determines how to distribute the model and data across the network
3. **Session Initiation**: A session is created to execute the plan
4. **Execution**: The session coordinates data distribution, model deployment, training/inference, and result collection
5. **Completion**: The session tracks completion and results

This three-tier abstraction (Model → Plan → Session) allows MOSAIC to:
- Separate model definition from distribution strategy
- Reuse models across multiple plans
- Track and manage multiple concurrent operations
- Provide clear state management for distributed operations

---

## Models Represented by ONNX

MOSAIC uses **ONNX (Open Neural Network Exchange)** as the standard format for representing all machine learning models. This choice provides several critical advantages:

### Why ONNX?

1. **Framework Agnostic**: ONNX models can be created from PyTorch, TensorFlow, Keras, and other frameworks, then executed across different runtimes
2. **Portability**: Models can be transferred between nodes without requiring the original training framework
3. **Optimization**: ONNX provides a standardized format for model optimization and compression
4. **Interoperability**: Enables the same model to work across different implementations (Python, Java, etc.)

### Model Factory Pattern

MOSAIC uses a factory pattern for model creation and loading:

- **Model Loading**: Models can be loaded from ONNX files stored on disk or from binary representations in memory
- **Path Resolution**: The system resolves model paths relative to a configured models directory
- **Validation**: ONNX models are validated to ensure they are properly formatted
- **Binary Representation**: Models can be stored as binary data for efficient transmission across the network

### Model Types

MOSAIC supports several model types, each with specific characteristics:

- **CNN (Convolutional Neural Networks)**: Image classification models like ResNet
- **WAV2VEC**: Audio processing models for speech recognition
- **TRANSFORMER**: Language models like GPT-Neo
- **GNN (Graph Neural Networks)**: Graph-based models for node classification
- **VAE (Variational Autoencoders)**: Generative models like BigGAN
- **RL (Reinforcement Learning)**: Policy models like PPO

Each model type has specific planning and execution strategies tailored to its architecture.

---

## Out-of-the-Box Models

MOSAIC provides predefined model creation functions for common architectures. These allow users to quickly get started without needing to convert their own models to ONNX format.

### Supported Predefined Models

1. **ResNet-50**: Standard ResNet-50 architecture for image classification
2. **ResNet-101**: Deeper ResNet variant for more complex image classification
3. **Wav2Vec2**: Audio model for speech recognition tasks
4. **GPT-Neo**: Transformer-based language model for text generation
5. **GCN (Graph Convolutional Network)**: Graph neural network for node classification (ogbn-arxiv variant)
6. **BigGAN**: Generative adversarial network for image generation
7. **PPO (Proximal Policy Optimization)**: Reinforcement learning policy model

### Model Creation Process

When creating a predefined model:

1. The system loads the model architecture from a standard library (e.g., torchvision, transformers)
2. The model is converted to ONNX format with appropriate input/output specifications
3. The ONNX model is saved to the configured models directory
4. A Model instance is created with the appropriate model type and file references

### Custom Models

Users can also provide their own ONNX models by:
- Placing ONNX files in the configured models directory
- Creating Model instances that reference these files
- Specifying the appropriate model type

This flexibility allows MOSAIC to work with any model that can be represented in ONNX format.

---

## Communication and Beacon System

The **Beacon** system is the core communication infrastructure of MOSAIC. It manages all inter-node communication, including heartbeats, command execution, and data transmission.

### Beacon Architecture

The Beacon system operates through multiple concurrent threads:

1. **Send Heartbeat Thread**: Periodically sends heartbeat messages to configured peer nodes
2. **Receive Heartbeat UDP Listener**: Listens for incoming heartbeat messages via UDP
3. **Receive Comms TCP Listener**: Listens for incoming commands and data via TCP
4. **Stale Heartbeat Checker**: Monitors received heartbeats and marks nodes as stale if they haven't been heard from

### Heartbeat System

**Heartbeats** are periodic messages that nodes send to inform other nodes of their status and capabilities.

#### Heartbeat Content

Each heartbeat message contains:

- **Machine Statistics**: Current CPU, GPU, RAM, and disk utilization
- **Capability Information**: Hardware specifications (GPUs, CPUs, memory capacity)
- **Benchmark Data**: Performance benchmarks for CPU, GPU, RAM, and disk
- **Connection Status**: Current state of the node (online, offline, etc.)
- **Timestamp**: When the heartbeat was generated

#### Directional Heartbeats

Heartbeats in MOSAIC are **directional** by design:

- **One-Way Communication**: Node A can send heartbeats to Node B without Node B necessarily sending heartbeats back to Node A
- **Asymmetric Topology**: This enables hub-and-spoke topologies where worker nodes heartbeat to a central controller, but the controller doesn't heartbeat back to each worker
- **Configurable**: While directional by default, bidirectional heartbeats can be configured if needed

#### Advantages of Directional Heartbeats

1. **Simplified Topology**: Clear hierarchy with controller nodes and worker nodes
2. **Reduced Network Traffic**: Fewer heartbeat messages in large networks
3. **Scalability**: Worker nodes can be added without requiring controller reconfiguration
4. **Fault Tolerance**: Controller can detect worker failures without maintaining bidirectional state

#### Disadvantages of Directional Heartbeats

1. **Limited Visibility**: Worker nodes may not know about other workers directly
2. **Central Dependency**: Workers depend on the controller for network-wide information
3. **Asymmetric Failure Detection**: Controller can detect worker failures, but workers may not detect controller failures as quickly

### Command Execution

The Beacon system supports **remote command execution** between nodes:

- **Command Registry**: Nodes can register handlers for specific commands
- **Command Transmission**: Commands can be sent from one node to another via TCP
- **Response Handling**: Commands can return results that are transmitted back to the sender
- **SSL/TLS Security**: All command transmission is secured with SSL/TLS certificates

This allows nodes to:
- Request statistics from other nodes
- Trigger operations on remote nodes
- Coordinate distributed operations
- Execute data and model distribution plans

### Data Transmission

The Beacon system handles data transmission for:

- **Model Distribution**: Sending ONNX models to nodes that need them
- **Data Distribution**: Segmenting and sending training/inference data to nodes
- **Result Collection**: Gathering results from distributed operations
- **Chunked Transfer**: Large data is automatically chunked for efficient transmission

Data transmission uses:
- **Compression**: Data is compressed (gzip) before transmission
- **Chunking**: Large files are split into configurable chunk sizes
- **Streaming**: Data can be streamed to avoid memory issues
- **Threading**: Multiple transfers can occur in parallel based on network capacity

---

## Planner and Model Runtime

The **Planner** and **Model Runtime** are the core components that enable MOSAIC's composite learning strategies. They work together to distribute models and data across the network in ways that optimize for different constraints and capabilities.

### Composite Learning Strategies

MOSAIC implements two primary strategies for composite learning:

#### 1. Data Parallel Strategy

In the **data parallel** strategy:

- **Model Requirements**: All models must fit on each machine without modification
- **Training Approach**: Each node trains on only the data that is on that machine
- **Data Distribution**: Data is striped and sharded across the network
- **Consolidation**: Federated Averaging (FedAvg)-like techniques are used to consolidate training and inference results

**How it works:**
1. The original model is distributed to all participating nodes unchanged
2. Data is segmented and distributed so each node has a subset
3. Each node trains the model on its local data subset
4. Model weights are aggregated using weighted averaging based on data distribution
5. The aggregated model is used for inference or further training rounds

**Use Cases:**
- When all nodes have sufficient resources to run the full model
- When model size is not a constraint
- When the primary goal is to distribute data across nodes

#### 2. Data Parallel + Simplification Strategy

In the **data parallel + simplification** strategy:

- **Model Requirements**: Models must fit on machines, but can be simplified for less capable nodes
- **Training Approach**: Each node trains on only the data on that machine, but with different model complexities
- **Data Distribution**: Data is sharded similarly to pure data parallel
- **Model Simplification**: More capable machines run the entire model with full complexity, while less capable machines run simplified versions
- **Consolidation**: FedAvg-like techniques consolidate results, accounting for model differences

**How it works:**
1. Node capabilities are assessed (CPU, GPU, memory, etc.)
2. The model is analyzed to identify simplification opportunities
3. Different model variants are created for different capability levels:
   - High-capability nodes: Full model or minimal compression
   - Medium-capability nodes: Moderate compression/simplification
   - Low-capability nodes: Significant compression/simplification
4. Data is distributed across nodes
5. Each node trains its model variant on its local data
6. Results are consolidated, accounting for model differences

**Advantages over other techniques:**

The key advantage of this approach is **vastly reduced communication overhead**:

- **Traditional Federated Learning**: Requires sending full model weights between nodes in each round
- **MOSAIC Approach**: Only aggregated results need to be communicated, not individual model weights
- **Bandwidth Savings**: Instead of transmitting large model parameter sets, only smaller aggregated results are sent
- **Scalability**: Network bandwidth becomes less of a bottleneck as the network grows

### Model Types and Simplification Strategies

Each model type has specific simplification strategies:

#### CNN Models (ResNet)

- **Simplification Method**: Magnitude-based channel pruning
- **Strategy**: 
  - Identify bottleneck layers (middle layers, not first/last)
  - More capable nodes keep more channels in bottleneck layers
  - Less capable nodes reduce channels in bottleneck layers
  - First and last layers are typically kept intact for all nodes
- **Compression Ratio**: Based on node capability score, ranging from 20% to 90% of original channels

#### Wav2Vec2 Models

- **Simplification Method**: Structured encoder layer dropping and hidden dimension compression
- **Strategy**:
  - More capable nodes keep more encoder layers
  - Less capable nodes drop encoder layers
  - Hidden dimensions can be reduced for less capable nodes
- **Adaptation**: Maintains audio processing capabilities while reducing computational requirements

#### Transformer Models (GPT-Neo)

- **Simplification Method**: Attention head reduction and layer dropping
- **Strategy**:
  - More capable nodes keep full attention mechanisms
  - Less capable nodes reduce attention heads per layer
  - Some transformer layers can be dropped entirely for very low-capability nodes
- **Trade-off**: Maintains language modeling capabilities with reduced parameters

#### GNN Models (Graph Convolutional Networks)

- **Simplification Method**: Graph convolution layer reduction and feature dimension compression
- **Strategy**:
  - More capable nodes use deeper networks with more layers
  - Less capable nodes use fewer convolution layers
  - Feature dimensions can be reduced
- **Consideration**: Graph structure is preserved, but processing depth is reduced

#### VAE Models (BigGAN)

- **Simplification Method**: Generator and discriminator complexity reduction
- **Strategy**:
  - More capable nodes run full generator/discriminator networks
  - Less capable nodes reduce network width and depth
  - Channel numbers in convolutional layers are reduced
- **Balance**: Maintains generation quality while reducing computational load

#### RL Models (PPO)

- **Simplification Method**: Policy network compression
- **Strategy**:
  - More capable nodes use full policy networks
  - Less capable nodes reduce policy network layers and hidden dimensions
- **Consideration**: Action space and observation space are typically preserved

### Planning Process

The planning process involves several steps:

1. **Capability Assessment**: Collect statistics from all nodes via heartbeats
2. **Eligibility Filtering**: Filter nodes based on:
   - Connection status (must be online)
   - Staleness (must have recent heartbeats)
   - Resource availability (CPU, GPU, memory)
3. **Capacity Scoring**: Calculate a capacity score for each node based on:
   - CPU performance (GFLOPS)
   - GPU performance (GFLOPS per GPU)
   - RAM bandwidth
   - Disk I/O performance
   - Weighted combination of these factors
4. **Network Factor Calculation**: Assess network performance between nodes
5. **Effective Score Calculation**: Combine capacity and network factors
6. **Model Planning**: For simplification strategy, determine compression ratios and methods for each node
7. **Data Distribution Planning**: Determine how data should be segmented and distributed

### Model Runtime

The Model Runtime executes the planned operations:

1. **Model Loading**: Loads ONNX models (original or compressed variants)
2. **Model Conversion**: Converts ONNX models to the execution framework (PyTorch)
3. **Data Loading**: Loads and preprocesses data according to the segmentation plan
4. **Training Execution**: Executes training with appropriate hyperparameters
5. **Inference Execution**: Executes inference on test data
6. **Result Aggregation**: Collects and aggregates results from all nodes

### Hyperparameters

Each model type has default hyperparameters optimized for distributed training:

- **Learning Rate**: Adjusted for federated learning scenarios
- **Batch Size**: Optimized for each node's capabilities
- **Optimizer Settings**: Adam, AdamW, or SGD with appropriate parameters
- **Loss Functions**: Task-specific loss functions
- **Regularization**: Dropout, weight decay, etc. appropriate for the model type

Hyperparameters can be customized per model type and adjusted based on node capabilities.

---

## Data Sharding Algorithms

MOSAIC implements sophisticated algorithms for distributing data across the network. The goal is to allocate data segments to nodes in a way that:
- Balances workload based on node capabilities
- Minimizes data transfer overhead
- Ensures all eligible nodes participate
- Accounts for network conditions

### Data Distribution Planning

The data distribution planning process takes a **Plan** (which contains the distribution strategy) and a **Data** instance (which contains file definitions) and creates a detailed segmentation plan.

#### Key Steps

1. **Normalize Capacity Fractions**: Convert effective scores or capacity fractions into normalized proportions
2. **File Analysis**: For each file in the data:
   - Determine file size
   - Check if file is segmentable (some file types cannot be split)
   - Calculate metadata (number of samples, data type, etc.)
3. **Cumulative Distribution**: Create a cumulative distribution function based on capacity fractions
4. **Segment Calculation**: For each node:
   - Calculate start and end positions in the cumulative distribution
   - Map these to specific file segments
   - Handle edge cases (very small files, non-segmentable files)
5. **Overlap Handling**: If overlap is specified, add overlap regions between segments
6. **Segment Metadata**: Create metadata for each segment including:
   - Source file information
   - Start and end positions
   - Target node information
   - Segment size

### Static Weighted Sharding

The **static weighted sharding** algorithm allocates fixed data shards to nodes based on their capabilities.

#### Algorithm Steps

1. **Eligibility Filtering**: Filter nodes based on:
   - Connection status must be "online"
   - Heartbeat must be recent (within stale threshold)
   - Must have benchmark data available

2. **Capacity Scoring**: Calculate capacity score for each node:
   - Normalize CPU, GPU, RAM, and disk scores using min-max normalization
   - Apply weights: typically GPU (0.75), CPU (0.15), RAM (0.05), Disk (0.05)
   - Combine into a single capacity score

3. **Network Factor Calculation**: For each node:
   - Assess network performance (bandwidth, latency)
   - Calculate network factor (0.0 to 1.0) based on relative network performance
   - Apply minimum network factor threshold

4. **Effective Score**: Calculate effective score = capacity_score × network_factor

5. **Normalization**: Normalize effective scores to sum to 1.0 (capacity fractions)

6. **Integer Allocation**: Convert capacity fractions to integer sample counts:
   - Calculate base allocation: floor(fraction × total_samples)
   - Calculate remainder: fraction × total_samples - base
   - Distribute remainders to nodes with largest remainders first
   - Ensure all nodes with valid data get at least 1 sample

7. **Result**: Return allocation plan with:
   - Host, ports for each node
   - Allocated sample count
   - Capacity fraction
   - Effective score
   - Network factor

#### Characteristics

- **Static**: Allocation is determined once and remains fixed
- **Proportional**: Allocation is proportional to node capabilities
- **Deterministic**: Same inputs produce same outputs
- **Fair**: All eligible nodes participate

### Dynamic Weighted Batching

The **dynamic weighted batching** algorithm allocates batches dynamically based on current node load and capabilities.

#### Algorithm Steps

1. **Enhanced Eligibility Filtering**: In addition to basic eligibility:
   - Check CPU utilization (must be below threshold, e.g., 90%)
   - Check RAM utilization (must be below threshold, e.g., 95%)
   - Verify benchmark data exists
   - Check heartbeat freshness

2. **Capacity Scoring**: Same as static sharding

3. **Load Factor Calculation**: For each node:
   - Assess current CPU and RAM utilization
   - Calculate load factor: lower utilization = higher factor
   - Formula: load_factor = 1.0 - (α × cpu_util + (1-α) × ram_util)
   - Where α is typically 0.7 (more weight on CPU)

4. **Freshness Factor**: Account for heartbeat age:
   - More recent heartbeats get higher freshness factor
   - Older heartbeats (but still within threshold) get lower factor
   - Encourages use of nodes with up-to-date information

5. **Network Factor**: Same as static sharding

6. **Effective Score**: Calculate effective score = capacity_score × load_factor × freshness_factor × network_factor

7. **Normalization and Allocation**: Similar to static sharding, but:
   - Allocation is in terms of "batches" rather than "samples"
   - Can be more dynamic as node loads change

#### Characteristics

- **Dynamic**: Allocation can change based on current node state
- **Load-Aware**: Accounts for current CPU and RAM utilization
- **Freshness-Aware**: Prefers nodes with recent status updates
- **Adaptive**: Responds to changing network conditions

### Data Segmentation

Once allocation is determined, data files must be segmented:

#### Segmentation Strategies

1. **File-Based Segmentation**: 
   - For segmentable files, split files into chunks
   - Each chunk goes to a different node
   - Maintains file boundaries where possible

2. **Sample-Based Segmentation**:
   - For data with known sample counts, segment by sample indices
   - More precise allocation
   - Requires sample-level metadata

3. **Size-Based Segmentation**:
   - Segment files by byte ranges
   - Simpler but may split individual samples
   - Used when sample boundaries are unknown

#### Overlap Handling

Segments can have overlap regions:
- **Purpose**: Ensure boundary samples are available to multiple nodes
- **Use Case**: Important for models that use sliding windows or context
- **Implementation**: Add overlap regions at segment boundaries
- **Trade-off**: Increases data transfer but improves model performance

### Data Transmission

Once segmentation is planned, data is transmitted:

1. **Chunking**: Large files are split into configurable chunk sizes (default: 256 MB)
2. **Compression**: Data is compressed (gzip) before transmission
3. **Parallel Transfer**: Multiple transfers occur in parallel based on network capacity
4. **Streaming**: Large files are streamed to avoid memory issues
5. **Verification**: Transmitted data is verified at the receiving end

### Algorithm Comparison

| Aspect | Static Weighted Sharding | Dynamic Weighted Batching |
|--------|-------------------------|--------------------------|
| **Allocation Basis** | Fixed capability scores | Current load + capability |
| **Adaptability** | Low (static) | High (dynamic) |
| **Complexity** | Lower | Higher |
| **Use Case** | Stable networks, predictable loads | Variable loads, dynamic networks |
| **Overhead** | Lower (calculated once) | Higher (recalculated periodically) |

Both algorithms use the same underlying capacity scoring and network factor calculation, ensuring consistency in how nodes are evaluated. The choice between them depends on the stability and predictability of the network environment.

---

## Conclusion

This technical documentation covers the core architectural components and algorithms that make MOSAIC work. The system is designed to be:

- **Configuration-Driven**: All behavior controlled through configuration
- **Flexible**: Supports multiple model types and distribution strategies
- **Efficient**: Minimizes communication overhead through intelligent model simplification
- **Scalable**: Handles networks from a few nodes to many nodes
- **Robust**: Fault-tolerant with heartbeat monitoring and automatic redistribution

The Python implementation provides a complete, production-ready system, and the Java implementation will replicate this architecture to provide language choice and interoperability.

