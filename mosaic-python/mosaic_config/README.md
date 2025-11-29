# MOSAIC Configuration Module

This module handles configuration reading for MOSAIC (Model Orchestration & Synthesis for Adaptive Intelligent Computation) nodes. It provides a flexible way to configure MOSAIC processes through JSON configuration files.

## Configuration File Priority

The configuration system searches for configuration files in the following priority order:

1. **Command-line argument** (`--config`)
   ```bash
   python your_script.py --config /path/to/config.json
   ```

2. **Environment variable** (`MOSAIC_CONFIG`)
   ```bash
   export MOSAIC_CONFIG=/path/to/config.json
   python your_script.py
   ```

3. **Current working directory** (`mosaic.config`)
   ```bash
   # Looks for mosaic.config in the directory where you run the script
   python your_script.py
   ```

If no configuration file is found in any of these locations, the program will raise a `FileNotFoundError` with a helpful message indicating which locations were checked.

## Path Expansion

Configuration file paths support:
- **Home directory expansion**: `~/config/mosaic.json` expands to `/home/username/config/mosaic.json`
- **Environment variables**: `$HOME/config/mosaic.json` or `${HOME}/config/mosaic.json`

## Configuration File Structure

The configuration file must be valid JSON with the following structure:

```json
{
  "host": "localhost",
  "heartbeat_port": 5000,
  "comms_port": 5001,
  "peers": [
    {"host": "node1.example.com", "comms_port": 5001, "heartbeat_port": 5000},
    {"host": "node2.example.com", "comms_port": 5001, "heartbeat_port": 5000}
  ],
  "heartbeat_frequency": 5,
  "heartbeat_tolerance": 15,
  "heartbeat_report_length": 300,
  "heartbeat_wait_timeout": 2,
  "stats_request_timeout": 30,
  "server_crt": "/path/to/server.crt",
  "server_key": "/path/to/server.key",
  "ca_crt": "/path/to/ca.crt",
  "benchmark_data_location": "/path/to/benchmark/data",
  "run_benchmark_at_startup": false,
  "data_location": "/path/to/data",
  "plans_location": "plans",
  "models_location": "models"
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | string | `"localhost"` | The local hostname or IP address for this MOSAIC node |
| `heartbeat_port` | integer | `5000` | UDP port number for heartbeat communication with other nodes |
| `comms_port` | integer | `5001` | TCP port number for general communications between nodes |
| `peers` | array | `[]` | List of peer nodes to communicate with (see Peers section below) |
| `heartbeat_frequency` | integer | `5` | How often (in seconds) to send heartbeats to peer nodes |
| `heartbeat_tolerance` | integer | `15` | How long (in seconds) to wait before considering a peer stale (effectively offline) |
| `heartbeat_report_length` | integer | `300` | Duration (in seconds) to retain machine statistics records |
| `heartbeat_wait_timeout` | integer | `2` | How long (in seconds) to wait for an initial connection to timeout |
| `stats_request_timeout` | integer | `30` | How long (in seconds) to wait for incoming statistics requested by any Mosaic node to the whole connected network |
| `server_crt` | string | `""` | Path to the SSL server certificate file |
| `server_key` | string | `""` | Path to the SSL server private key file |
| `ca_crt` | string | `""` | Path to the SSL Certificate Authority certificate file |
| `benchmark_data_location` | string | `""` | Directory path where benchmark data is stored |
| `run_benchmark_at_startup` | boolean | `false` | Whether to run benchmarks when the node starts |
| `data_location` | string | `""` | Directory path where operational data can be found |
| `plans_location` | string | `"plans"` | Directory path where plans for data and model distribution are kept |
| `models_location` | string | `"models"` | Directory path where model data is stored, such as ONNX files |

### Peers Configuration

Each peer must specify both communication ports:

```json
"peers": [
  {"host": "192.168.1.10", "comms_port": 5001, "heartbeat_port": 5000},
  {"host": "node2.local", "comms_port": 5001, "heartbeat_port": 5000}
]
```

Each peer object requires:
- `host`: The hostname or IP address of the peer node
- `comms_port`: The TCP port number for general communications
- `heartbeat_port`: The UDP port number for heartbeat communication

### Alternative Heartbeat Configuration Format

The heartbeat parameters can also be nested under a `heartbeat` object (for backward compatibility):

```json
{
  "heartbeat": {
    "frequency": 5,
    "tolerance": 15,
    "report_length": 300,
    "wait_timeout": 2
  }
}
```

Both flat and nested formats are supported; flat format takes precedence if both are present.

## Usage Examples

### Basic Usage

```python
from mosaic_config.config import read_config

# Read configuration from standard locations
config = read_config()

print(f"Node host: {config.host}")
print(f"Heartbeat port: {config.heartbeat_port}")
print(f"Number of peers: {len(config.peers)}")
for peer in config.peers:
    print(f"  Peer: {peer.host} (comms: {peer.comms_port}, heartbeat: {peer.heartbeat_port})")
```

### Reading Raw JSON

If you need the raw configuration dictionary:

```python
from mosaic_config.config import read_json_config

config_dict = read_json_config()
```

### Creating Configuration from Dictionary

```python
from mosaic_config.config import MosaicConfig

config_dict = {
    "host": "192.168.1.100",
    "heartbeat_port": 6000,
    "peers": [{"host": "192.168.1.101", "comms_port": 6001, "heartbeat_port": 6000}]
}

config = MosaicConfig.from_dict(config_dict)
```

## Error Handling

The module raises the following exceptions:

- **`FileNotFoundError`**: When no configuration file is found in any of the standard locations
- **`json.JSONDecodeError`**: When the configuration file contains invalid JSON
- **`ValueError`**: When a specified configuration path exists but is not a file

## Minimal Configuration Example

Here's a minimal configuration file to get started:

```json
{
  "host": "localhost",
  "peers": []
}
```

All other parameters will use their default values. This is suitable for testing a single node without peer connections.

## Complete Configuration Example

```json
{
  "host": "mosaic-node-01.local",
  "heartbeat_port": 5000,
  "comms_port": 5001,
  "peers": [
    {"host": "mosaic-node-02.local", "comms_port": 5001, "heartbeat_port": 5000},
    {"host": "mosaic-node-03.local", "comms_port": 5001, "heartbeat_port": 5000}
  ],
  "heartbeat_frequency": 10,
  "heartbeat_tolerance": 30,
  "heartbeat_report_length": 600,
  "heartbeat_wait_timeout": 2,
  "stats_request_timeout": 30,
  "server_crt": "/etc/mosaic/certs/server.crt",
  "server_key": "/etc/mosaic/certs/server.key",
  "ca_crt": "/etc/mosaic/certs/ca.crt",
  "benchmark_data_location": "/var/mosaic/benchmarks",
  "run_benchmark_at_startup": true,
  "data_location": "/var/mosaic/data",
  "plans_location": "plans",
  "models_location": "models"
}
```

## Notes

- All file paths in the configuration support `~` and environment variable expansion
- Empty strings (`""`) for optional path parameters indicate those features are disabled
- The configuration object uses Python dataclasses for type safety and validation
- Port numbers should be in the valid range (1-65535)
- Heartbeat parameters should be positive integers