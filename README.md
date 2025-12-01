# MOSAIC: Model Orchestration & Synthesis for Adaptive Intelligent Computation

MOSAIC is an innovative open-source framework for composite learning, developed as part of the [SPRIN-D](https://www.sprind.org/) Composite Learning Challenge. MOSAIC reimagines distributed AI by orchestrating many independently trained, small-scale models—enabling efficient, robust, and scalable machine learning across heterogeneous hardware environments. It is an attempt to enable *fusion* of models that have been learned at edge into a single model, thereby enabling learning to be transferred without transfer of data itself. 


## Project Overview

**MOSAIC challenges the monolithic AI paradigm.**  
Instead of relying on massive, centralized models, MOSAIC fuses the outputs of distributed, independently trained models (pattern recognizers) into a unified, high-performing system. This approach democratizes AI, making advanced machine learning accessible and efficient across a wide range of devices and environments.

**Key Goals:**
- **Efficiency:** Achieve superior model performance with less computational resource.
- **Decentralization:** Distribute learning and inference tasks across diverse, often resource-constrained, devices.
- **Scalability & Robustness:** Dynamically adapt to changing compute resources and network conditions.


## Technical Components

MOSAIC is built around a modular, extensible architecture:

- **Independent Network Nodes:**  
  Each Mosaic node is an independent entity in the network that can function as both a workhorse (executing ML operations and computations) and a controller (where users interact with the network to issue commands). Nodes can be dynamically added or removed, allowing the system to scale and adapt to available resources.

- **Implementation Languages:**  
  MOSAIC is implemented in both Python and Java. The Python implementation is currently available and actively maintained, while the Java implementation is still in development.

- **Decentralized, Secure Architecture:**  
  Peer-to-peer connections between nodes, secured with SSL. Each node can independently participate in the network, supporting robust, fault-tolerant operations.

- **Heterogeneous Hardware Optimization:**  
  Automatic detection and utilization of available hardware (CPUs, GPUs, accelerators), with containerized deployment (Docker, Podman, Kubernetes) for seamless integration across cloud, edge, and on-premise environments.

- **Parallelization & Communication:**  
  Supports both data and model parallelization (e.g., for ResNet, GPT-Neo), with efficient, adaptive communication protocols and gradient compression to minimize network overhead.

- **Robustness & Fault Tolerance:**  
  Dynamic resource monitoring, checkpointing, and local ledgers ensure resilience to node failures and fluctuating network conditions.

- **Advanced Features:**  
  - Encryption and privacy-preserving techniques
  - MLOps integration and monitoring (Coming Soon)
  
# Understanding the Architecture

![Mosaic Network](readme_image1.png "Mosaic Network")

A MOSAIC network is a network of independent nodes where at least one node (such as Node 1 in the diagram above) serves as the **controller**—the point where users interact with the network to issue commands for operations like training or inference. While any node can function as both a workhorse and a controller, having a designated controller node simplifies network management and command coordination.

## Heartbeat Communication

**Heartbeats are directional** in MOSAIC. For example, in the diagram, Node 2 reports its status to Node 1, but Node 1 does not necessarily send heartbeats back to Node 2 (though this can be configured if bidirectional communication is desired).

A heartbeat serves more than just a "keep-alive" signal. Each heartbeat message includes **capability information** about the sending node, such as:
- Available GPUs and their specifications
- CPU resources and cores
- Memory capacity
- Current workload status

MOSAIC uses this capability information to intelligently calculate how to distribute workloads across the network, ensuring optimal resource utilization and performance.

*NOTE* Please be aware port blocks on the firewall, or sealed off subnets, etc, will block Mosaic nodes from connecting with one another.

## Fault Tolerance

When a node breaks down or becomes unreachable, connected nodes detect the failure through missed heartbeats. The network automatically redistributes workloads from the failed node to available nodes, ensuring continuous operation and resilience to individual node failures.

## Recommended Setup

For the simplest setup, configure your architecture as shown in the diagram: **one central node connected to many others**, with the other nodes sending heartbeats to the central node. This hub-and-spoke topology provides a clear control point while maintaining the flexibility to scale horizontally.

# BEFORE YOU START: Core Requirements 

You'll need the following on every server to be able to run Mosaic:

- Docker (runs Mosaic)
- Python 3.11 or higher (only required once to create SSL certificates)

# Quick Installation Guidelines

## Download and Install



## Requirements

### System Requirements
- **Operating System**: Linux (recommended), Unix-like systems
- **Network**: TCP/UDP ports for communication (default: 5000 for heartbeats, 5001 for general communications)
- **SSL/TLS**: Python's built-in `ssl` module (no additional installation required)
- **Memory**: Sufficient RAM for model loading and inference (varies by model size)
- **Storage**: Space for models, data, and state files (varies by use case)

### GPU Support
If you plan to use GPU acceleration (which are usually installed when you run these servers from a cloud provider):
- **NVIDIA**: CUDA-compatible GPU with appropriate CUDA drivers installed
- **AMD**: ROCm-compatible GPU with ROCm drivers installed
- **Intel**: Intel GPU with appropriate drivers
- **Habana**: Habana Gaudi accelerator with appropriate drivers

GPU libraries (CUDA, cuDNN, etc.) are automatically included with PyTorch when installed with GPU support.



## Contributors and Acknowledgements

**Lead Institution & Development Team:**  
Jiva.ai Ltd., Tramshed Tech, Pendyris Street, Cardiff, Wales, CF11 6BH

**Special Thanks:**  
The MOSAIC project is proudly supported by [SPRIN-D](https://www.sprind.org/), the German Federal Agency for Disruptive Innovation. SPRIN-D’s Composite Learning Challenge funds pioneering research and development in distributed, decentralized, and robust AI systems. Their support enables us to push the boundaries of what’s possible in democratized, efficient, and secure machine learning. It is thanks to their initial funding that enabled the Jiva.ai team to create the first iteration of this project. 

Thank you to SPRIN-D and its mentors, as well as all contributors for making MOSAIC possible.

---

*This README is a living document and will be updated as the project evolves.*

