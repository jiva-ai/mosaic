# Running Mosaic Docker Containers Across Multiple Machines

This guide explains how to run Mosaic Docker containers on different machines and ensure they can communicate with each other.

## Two Approaches

### Option 1: Host Network Mode (Recommended for Linux)
Uses the host's network stack directly - container inherits host IP automatically.

### Option 2: Bridge Network Mode
Uses Docker's bridge network with port mapping - requires manual IP configuration.

---

## Option 1: Host Network Mode (Simplest)

**Advantages:**
- Container automatically uses host's IP address
- No port mapping needed
- Simpler configuration
- Better performance (no NAT overhead)

**Limitations:**
- Only works on Linux (not Docker Desktop for Mac/Windows)
- Less network isolation
- Port conflicts with host services

### Configuration Setup

**mosaic.config (same on all machines):**
```json
{
  "host": "0.0.0.0",
  "heartbeat_port": 5000,
  "comms_port": 5001,
  "peers": [
    {"host": "192.168.1.101", "comms_port": 5001, "heartbeat_port": 5000}
  ],
  "data_location": "/app/data",
  "models_location": "/app/models",
  "plans_location": "/app/plans",
  "state_location": "/app/state"
}
```

**Run command:**
```bash
docker run -d \
  --name mosaic-node \
  --restart unless-stopped \
  --network host \
  -v $(pwd)/mosaic.config:/app/mosaic.config:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/plans:/app/plans \
  -v $(pwd)/state:/app/state \
  mosaic-python:latest
```

**Key points:**
- `--network host`: Container uses host's network stack
- Container sees the host's actual IP address
- No `-p` port mapping flags needed
- Use the host machine's IP in peer configs (e.g., `192.168.1.100`)

---

## Option 2: Bridge Network Mode (Works Everywhere)

**Advantages:**
- Works on all platforms (Linux, Mac, Windows)
- Better network isolation
- More flexible port configuration

**Limitations:**
- Requires port mapping
- Need to manually specify host IP in peer configs

## Key Requirements

1. **Container Binding**: Inside the container, bind to `0.0.0.0` to listen on all interfaces
2. **Peer Configuration**: Use the **host machine's IP address** (not container IP) in peer configurations
3. **Port Mapping**: Map the required ports from host to container
4. **Network Accessibility**: Ensure machines can reach each other (firewall rules, network routing)

## Configuration Setup

### Machine 1 (e.g., IP: 192.168.1.100)

**mosaic.config:**
```json
{
  "host": "0.0.0.0",
  "heartbeat_port": 5000,
  "comms_port": 5001,
  "peers": [
    {"host": "192.168.1.101", "comms_port": 5001, "heartbeat_port": 5000}
  ],
  "data_location": "/app/data",
  "models_location": "/app/models",
  "plans_location": "/app/plans",
  "state_location": "/app/state"
}
```

**Run command:**
```bash
docker run -d \
  --name mosaic-node-1 \
  --restart unless-stopped \
  -p 5000:5000/udp \
  -p 5001:5001/tcp \
  -p 49152-65535:49152-65535/tcp \
  -p 49152-65535:49152-65535/udp \
  -v $(pwd)/mosaic.config:/app/mosaic.config:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/plans:/app/plans \
  -v $(pwd)/state:/app/state \
  mosaic-python:latest
```

### Machine 2 (e.g., IP: 192.168.1.101)

**mosaic.config:**
```json
{
  "host": "0.0.0.0",
  "heartbeat_port": 5000,
  "comms_port": 5001,
  "peers": [
    {"host": "192.168.1.100", "comms_port": 5001, "heartbeat_port": 5000}
  ],
  "data_location": "/app/data",
  "models_location": "/app/models",
  "plans_location": "/app/plans",
  "state_location": "/app/state"
}
```

**Run command:**
```bash
docker run -d \
  --name mosaic-node-2 \
  --restart unless-stopped \
  -p 5000:5000/udp \
  -p 5001:5001/tcp \
  -p 49152-65535:49152-65535/tcp \
  -p 49152-65535:49152-65535/udp \
  -v $(pwd)/mosaic.config:/app/mosaic.config:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/plans:/app/plans \
  -v $(pwd)/state:/app/state \
  mosaic-python:latest
```

## Important Points

### 1. Host Configuration (`"host": "0.0.0.0"`)

- **Inside the container**: Use `"host": "0.0.0.0"` to bind to all network interfaces
- This allows the container to receive connections from outside the container
- The container will listen on all interfaces, and Docker port mapping will forward traffic

### 2. Peer Configuration (Use Host Machine IP)

- **Peer `host` field**: Use the **actual IP address of the host machine** (e.g., `192.168.1.100`)
- **NOT** the container's internal IP (e.g., `172.17.0.2`)
- **NOT** `localhost` or `127.0.0.1` (these won't work across machines)
- Use the IP address that other machines can reach over the network

### 3. Port Mapping

The Docker port mappings (`-p host:container`) ensure:
- Traffic to the host machine's ports is forwarded to the container
- Ephemeral ports (49152-65535) are available for dynamic connections
- Both UDP (heartbeat) and TCP (comms) ports are accessible

### 4. Network Connectivity

Ensure:
- Machines are on the same network or can route to each other
- Firewall rules allow:
  - UDP port 5000 (heartbeat)
  - TCP port 5001 (comms)
  - TCP/UDP ports 49152-65535 (ephemeral range)
- If using cloud providers, check security groups/network ACLs

## Verification Steps

### 1. Check Container is Running
```bash
docker ps | grep mosaic
```

### 2. Check Logs for Connection Status
```bash
docker logs mosaic-node-1
docker logs mosaic-node-2
```

Look for:
- `UDP heartbeat listener started on 0.0.0.0:5000`
- `TCP comms listener started on 0.0.0.0:5001`
- Heartbeat status messages showing peer connections

### 3. Test Network Connectivity

From Machine 1, test connection to Machine 2:
```bash
# Test UDP (heartbeat port)
nc -u -v 192.168.1.101 5000

# Test TCP (comms port)
nc -v 192.168.1.101 5001
```

### 4. Use REPL to Check Status

Attach to container and use REPL commands:
```bash
docker attach mosaic-node-1
# Then in REPL:
hb          # Check heartbeat status
rhb         # Check received heartbeats
shb         # Check sent heartbeats
```

## Troubleshooting

### Issue: Nodes can't see each other

**Check:**
1. `host` in config is `0.0.0.0` (not `localhost`)
2. Peer `host` is the actual machine IP (not container IP)
3. Ports are properly mapped with `-p` flags
4. Firewall allows the ports
5. Machines can ping each other

### Issue: "Connection refused" errors

**Check:**
1. Container is running: `docker ps`
2. Ports are exposed: `docker port <container-name>`
3. Host firewall allows connections
4. Network routing is correct

### Issue: Heartbeats not received

**Check:**
1. UDP port 5000 is mapped: `-p 5000:5000/udp`
2. Firewall allows UDP traffic
3. Peer configuration has correct IP and port
4. Check logs: `docker logs <container-name>`

## Example: Three Machines

### Machine 1 (192.168.1.100)
```json
{
  "host": "0.0.0.0",
  "heartbeat_port": 5000,
  "comms_port": 5001,
  "peers": [
    {"host": "192.168.1.101", "comms_port": 5001, "heartbeat_port": 5000},
    {"host": "192.168.1.102", "comms_port": 5001, "heartbeat_port": 5000}
  ]
}
```

### Machine 2 (192.168.1.101)
```json
{
  "host": "0.0.0.0",
  "heartbeat_port": 5000,
  "comms_port": 5001,
  "peers": [
    {"host": "192.168.1.100", "comms_port": 5001, "heartbeat_port": 5000},
    {"host": "192.168.1.102", "comms_port": 5001, "heartbeat_port": 5000}
  ]
}
```

### Machine 3 (192.168.1.102)
```json
{
  "host": "0.0.0.0",
  "heartbeat_port": 5000,
  "comms_port": 5001,
  "peers": [
    {"host": "192.168.1.100", "comms_port": 5001, "heartbeat_port": 5000},
    {"host": "192.168.1.101", "comms_port": 5001, "heartbeat_port": 5000}
  ]
}
```

## Summary

### Option 1: Host Network Mode (Linux)
✅ **Run with**: `--network host`  
✅ **Container config**: `"host": "0.0.0.0"`  
✅ **Peer config**: Use host machine's IP address  
✅ **No port mapping needed**  

### Option 2: Bridge Network Mode (All Platforms)
✅ **Container config**: `"host": "0.0.0.0"` (bind to all interfaces)  
✅ **Peer config**: Use host machine's IP address (reachable from network)  
✅ **Port mapping**: Map all required ports (`-p` flags)  
✅ **Network**: Ensure machines can reach each other (firewall, routing)  

The Dockerfile is correctly configured - the key is proper configuration files and network setup!

## Recommendation

- **Linux servers**: Use `--network host` (Option 1) for simplicity
- **Docker Desktop (Mac/Windows)**: Use bridge mode (Option 2) with port mapping
- **Cloud deployments**: Either works, but host mode is simpler if supported

