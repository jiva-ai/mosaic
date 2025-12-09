#!/bin/bash

# Mosaic Docker Runner Script
# Automatically detects network mode and handles configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="mosaic-python:latest"
CONTAINER_NAME="mosaic-node"
CONFIG_FILE="mosaic.config"
CONFIG_BASENAME=""
CONFIG_ABS_PATH=""
FORCE_READ_CONFIG_NETWORK=false
HEARTBEAT_PORT=5000
COMMS_PORT=5001

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --bridge-network|--use-config-ports)
                FORCE_READ_CONFIG_NETWORK=true
                shift
                ;;
            --name)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --config PATH           Path to mosaic configuration file"
                echo "  --name NAME             Container name (default: mosaic-node)"
                echo "  --bridge-network         Use bridge network mode with ports from config (default: host network)"
                echo "  --use-config-ports       Alias for --bridge-network"
                echo "  --help, -h              Show this help message"
                echo ""
                echo "If --config is not provided, looks for mosaic.config in current directory"
                echo ""
                echo "Network Modes:"
                echo "  Host network (default):"
                echo "    - Container shares the host's network stack directly"
                echo "    - No port mapping needed - all ports are directly accessible"
                echo "    - Best performance, recommended for Linux servers"
                echo "    - Works on native Linux; may not work on Docker Desktop/Mac/Windows"
                echo ""
                echo "  Bridge network (--bridge-network):"
                echo "    - Container runs in isolated network with port forwarding"
                echo "    - Requires explicit port mappings (heartbeat_port and comms_port from config)"
                echo "    - More portable, works on all Docker platforms"
                echo "    - Slightly higher overhead due to network translation"
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Expand ~ and environment variables in config path
    CONFIG_FILE=$(eval echo "$CONFIG_FILE")
}


# Parse config file and extract paths (pure bash, no dependencies)
parse_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        return 1
    fi
    
    local base_dir=$(pwd)
    
    # Extract JSON value for a given key (handles basic JSON with quoted strings)
    extract_json_value() {
        local key="$1"
        local file="$2"
        # Remove comments and whitespace, then extract value for key
        # Pattern: "key": "value" or "key":"value"
        grep -o "\"$key\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$file" 2>/dev/null | \
        sed -n 's/.*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1
    }
    
    # Extract JSON numeric value for a given key (handles unquoted numbers)
    extract_json_numeric() {
        local key="$1"
        local file="$2"
        # Pattern: "key": number or "key":number (handles whitespace)
        grep -o "\"$key\"[[:space:]]*:[[:space:]]*[0-9]\+" "$file" 2>/dev/null | \
        sed -n 's/.*:[[:space:]]*\([0-9]\+\).*/\1/p' | head -1
    }
    
    # Expand path (~ and environment variables) and make absolute
    expand_path() {
        local path="$1"
        [ -z "$path" ] && return 1
        
        # Expand ~ to $HOME
        path="${path/#\~/$HOME}"
        
        # Expand environment variables ($VAR and ${VAR})
        while [[ "$path" =~ \$\{?([A-Za-z_][A-Za-z0-9_]*)\}? ]]; do
            local var="${BASH_REMATCH[1]}"
            local value="${!var}"
            if [ -z "$value" ]; then
                # Variable not set, leave as-is or remove
                path="${path//\$\{$var\}/}"
                path="${path//\$$var/}"
            else
                path="${path//\$\{$var\}/$value}"
                path="${path//\$$var/$value}"
            fi
        done
        
        # Remove empty path
        [ -z "$path" ] && return 1
        
        # Make absolute if relative
        if [[ "$path" != /* ]]; then
            path="$base_dir/$path"
        fi
        
        # Normalize path (remove .. and .)
        # Use realpath if available, otherwise use cd/pwd trick
        if command -v realpath >/dev/null 2>&1; then
            path=$(realpath "$path" 2>/dev/null || echo "$path")
        else
            # Use cd/pwd to normalize path (handles .. and . correctly)
            if [ -d "$path" ] || [ -f "$path" ]; then
                path=$(cd "$(dirname "$path")" 2>/dev/null && pwd)/$(basename "$path") 2>/dev/null || echo "$path"
            else
                # Path doesn't exist yet, use dirname/basename approach
                local dir_part=$(dirname "$path")
                local file_part=$(basename "$path")
                if [ -d "$dir_part" ]; then
                    path=$(cd "$dir_part" 2>/dev/null && pwd)/"$file_part" 2>/dev/null || echo "$path"
                else
                    # Can't normalize, return as-is
                    path="$path"
                fi
            fi
        fi
        
        echo "$path"
    }
    
    # Get parent directory of a file path
    get_parent_dir() {
        local file_path="$1"
        [ -z "$file_path" ] && return 1
        
        local expanded=$(expand_path "$file_path")
        [ -z "$expanded" ] && return 1
        
        if [ -f "$expanded" ]; then
            dirname "$expanded"
        else
            # If it's a directory or doesn't exist, return the directory part
            dirname "$expanded"
        fi
    }
    
    # Extract and process directory paths
    echo "DIRS_START"
    for key in data_location models_location plans_location state_location benchmark_data_location; do
        value=$(extract_json_value "$key" "$CONFIG_FILE")
        if [ -n "$value" ]; then
            expanded=$(expand_path "$value")
            if [ -n "$expanded" ]; then
                echo "${key}=${expanded}"
            fi
        fi
    done
    
    # Extract certificate file paths and their parent directories
    # Track which directories we've already output to avoid duplicates
    local cert_dirs_output=""
    for key in server_crt server_key ca_crt; do
        value=$(extract_json_value "$key" "$CONFIG_FILE")
        if [ -n "$value" ]; then
            expanded=$(expand_path "$value")
            if [ -n "$expanded" ]; then
                # Store cert file path for CERTS_START section
                eval "cert_${key}=\"${expanded}\""
                
                # Get parent directory and add to dirs if not already output
                parent_dir=$(get_parent_dir "$expanded")
                if [ -n "$parent_dir" ]; then
                    # Simple check: if parent_dir is not in our output string, add it
                    if [[ ! "|${cert_dirs_output}|" =~ "|${parent_dir}|" ]]; then
                        echo "${key}_dir=${parent_dir}"
                        if [ -z "$cert_dirs_output" ]; then
                            cert_dirs_output="${parent_dir}"
                        else
                            cert_dirs_output="${cert_dirs_output}|${parent_dir}"
                        fi
                    fi
                fi
            fi
        fi
    done
    echo "DIRS_END"
    
    # Output certificate file paths (expanded and original)
    echo "CERTS_START"
    for key in server_crt server_key ca_crt; do
        eval "expanded=\$cert_${key}"
        if [ -n "$expanded" ]; then
            echo "${key}=${expanded}"
            # Also output original path from config for comparison
            original=$(extract_json_value "$key" "$CONFIG_FILE")
            if [ -n "$original" ]; then
                echo "${key}_original=${original}"
            fi
        fi
    done
    echo "CERTS_END"
    
    # Extract port numbers
    echo "PORTS_START"
    heartbeat_port=$(extract_json_numeric "heartbeat_port" "$CONFIG_FILE")
    comms_port=$(extract_json_numeric "comms_port" "$CONFIG_FILE")
    if [ -n "$heartbeat_port" ]; then
        echo "heartbeat_port=${heartbeat_port}"
    fi
    if [ -n "$comms_port" ]; then
        echo "comms_port=${comms_port}"
    fi
    echo "PORTS_END"
}

# Check if config file exists and parse it
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${YELLOW}Warning: Config file not found: $CONFIG_FILE${NC}"
        echo -e "${YELLOW}Mosaic will use default configuration or look for config via:${NC}"
        echo -e "${YELLOW}  1. --config command-line argument${NC}"
        echo -e "${YELLOW}  2. MOSAIC_CONFIG environment variable${NC}"
        echo -e "${YELLOW}  3. mosaic.config in current working directory${NC}"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        CONFIG_MOUNT=""
        CONFIG_PARSED=false
    else
        # Get absolute path of config file
        if command -v realpath >/dev/null 2>&1; then
            CONFIG_ABS_PATH=$(realpath "$CONFIG_FILE")
        elif command -v readlink >/dev/null 2>&1; then
            CONFIG_ABS_PATH=$(readlink -f "$CONFIG_FILE")
        else
            # Fallback: construct absolute path manually
            if [[ "$CONFIG_FILE" == /* ]]; then
                CONFIG_ABS_PATH="$CONFIG_FILE"
            else
                CONFIG_ABS_PATH="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
            fi
        fi
        
        CONFIG_DIR=$(dirname "$CONFIG_ABS_PATH")
        CONFIG_BASENAME=$(basename "$CONFIG_ABS_PATH")
        
        CONFIG_MOUNT="-v $CONFIG_ABS_PATH:/app/$CONFIG_BASENAME:ro"
        echo -e "${GREEN}Found config file: $CONFIG_ABS_PATH${NC}"
        CONFIG_PARSED=true
        
        # Parse config to get paths
        echo -e "${GREEN}Parsing configuration...${NC}"
        export CONFIG_FILE
        CONFIG_OUTPUT=$(parse_config)
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error parsing config file${NC}"
            echo ""
            echo -e "${YELLOW}Here's an example Docker command with all configurable elements:${NC}"
            echo ""
            echo "docker run -it \\"
            echo "  --name mosaic-node \\"
            echo "  --restart unless-stopped \\"
            echo "  --network host \\"
            echo "  -v \$(pwd)/mosaic.config:/app/mosaic.config:ro \\"
            echo "  -v /path/to/data:/app/data \\"
            echo "  -v /path/to/models:/app/models \\"
            echo "  -v /path/to/plans:/app/plans \\"
            echo "  -v /path/to/state:/app/state \\"
            echo "  -v /path/to/benchmark_data:/app/benchmark_data \\"
            echo "  -v /path/to/certs:/path/to/certs:ro \\"
            echo "  -e MOSAIC_CONFIG=/app/mosaic.config \\"
            echo "  mosaic-python:latest"
            echo ""
            echo -e "${YELLOW}Replace the following placeholders:${NC}"
            echo "  - /path/to/data          -> Your data_location from config"
            echo "  - /path/to/models        -> Your models_location from config"
            echo "  - /path/to/plans         -> Your plans_location from config"
            echo "  - /path/to/state         -> Your state_location from config"
            echo "  - /path/to/benchmark_data -> Your benchmark_data_location from config (if set)"
            echo "  - /path/to/certs         -> Directory containing server_crt, server_key, ca_crt (if set)"
            echo ""
            echo -e "${YELLOW}For host network mode (Linux), remove the -p port mapping lines${NC}"
            echo -e "${YELLOW}For bridge mode (Mac/Windows), keep the -p port mapping lines${NC}"
            exit 1
        fi
        
        # Extract directory and cert mappings using associative arrays
        in_dirs=false
        in_certs=false
        in_ports=false
        declare -gA CONFIG_DIRS
        declare -gA CONFIG_CERTS
        declare -gA CONFIG_CERTS_ORIGINAL  # Store original paths from config file
        
        while IFS= read -r line; do
            if [ "$line" = "DIRS_START" ]; then
                in_dirs=true
                continue
            elif [ "$line" = "DIRS_END" ]; then
                in_dirs=false
                continue
            elif [ "$line" = "CERTS_START" ]; then
                in_certs=true
                continue
            elif [ "$line" = "CERTS_END" ]; then
                in_certs=false
                continue
            elif [ "$line" = "PORTS_START" ]; then
                in_ports=true
                continue
            elif [ "$line" = "PORTS_END" ]; then
                in_ports=false
                continue
            fi
            
            if [ "$in_dirs" = true ] && [[ "$line" == *"="* ]]; then
                key="${line%%=*}"
                value="${line#*=}"
                CONFIG_DIRS["$key"]="$value"
            elif [ "$in_certs" = true ] && [[ "$line" == *"="* ]]; then
                key="${line%%=*}"
                value="${line#*=}"
                if [[ "$key" == *_original ]]; then
                    # Store original path (remove _original suffix for key)
                    orig_key="${key%_original}"
                    CONFIG_CERTS_ORIGINAL["$orig_key"]="$value"
                else
                    CONFIG_CERTS["$key"]="$value"
                fi
            elif [ "$in_ports" = true ] && [[ "$line" == *"="* ]]; then
                key="${line%%=*}"
                value="${line#*=}"
                if [ "$key" = "heartbeat_port" ]; then
                    HEARTBEAT_PORT="$value"
                elif [ "$key" = "comms_port" ]; then
                    COMMS_PORT="$value"
                fi
            fi
        done <<< "$CONFIG_OUTPUT"
    fi
}

# Create necessary directories from config
create_directories() {
    if [ "$CONFIG_PARSED" = true ]; then
        echo -e "${GREEN}Creating directories from config...${NC}"
        for key in "${!CONFIG_DIRS[@]}"; do
            path="${CONFIG_DIRS[$key]}"
            # Only create if it's a directory path (not cert file dirs)
            if [[ "$key" != *_dir ]]; then
                if [ -n "$path" ]; then
                    mkdir -p "$path"
                    echo -e "  ${GREEN}✓${NC} $key: $path"
                fi
            fi
        done
    else
        # Fallback to default directories
        echo -e "${GREEN}Creating default data directories...${NC}"
        mkdir -p data models plans state
    fi
}

# Build Docker run command as array
build_docker_command() {
    # Use array to build command (avoids eval issues with special characters)
    DOCKER_ARGS=()
    
    # Base command parts
    DOCKER_ARGS+=("docker")
    DOCKER_ARGS+=("run")
    DOCKER_ARGS+=("-it")
    DOCKER_ARGS+=("--name" "$CONTAINER_NAME")
    DOCKER_ARGS+=("--restart" "unless-stopped")
    
    if [ "$FORCE_READ_CONFIG_NETWORK" = true ]; then
        DOCKER_ARGS+=("-p" "${HEARTBEAT_PORT}:${HEARTBEAT_PORT}/udp") 
        DOCKER_ARGS+=("-p" "${COMMS_PORT}:${COMMS_PORT}/tcp")
        echo -e "${GREEN}Using bridge network mode (ports: ${HEARTBEAT_PORT}/udp, ${COMMS_PORT}/tcp)${NC}"
    else
        DOCKER_ARGS+=("--network" "host")
        echo -e "${GREEN}Using host network mode (default)${NC}"
    fi

    # Config file mount
    if [ -n "$CONFIG_MOUNT" ]; then
        # CONFIG_MOUNT is like "-v /path:/path:ro", split it properly
        # Split on spaces, but preserve quoted strings
        read -ra mount_parts <<< "$CONFIG_MOUNT"
        DOCKER_ARGS+=("${mount_parts[@]}")
    fi
    
    # Mount directories and files from config
    if [ "$CONFIG_PARSED" = true ]; then
        echo -e "${GREEN}Mounting paths from config...${NC}"
        
        # Mount directory paths
        for key in "${!CONFIG_DIRS[@]}"; do
            host_path="${CONFIG_DIRS[$key]}"
            
            # Skip cert file directories (we'll handle them separately)
            if [[ "$key" == *_dir ]]; then
                continue
            fi
            
            if [ -n "$host_path" ]; then
                # Determine container path based on key
                case "$key" in
                    data_location)
                        container_path="/app/data"
                        ;;
                    models_location)
                        container_path="/app/models"
                        ;;
                    plans_location)
                        container_path="/app/plans"
                        ;;
                    state_location)
                        container_path="/app/state"
                        ;;
                    benchmark_data_location)
                        container_path="/app/benchmark_data"
                        ;;
                    *)
                        # Use key name as container path
                        container_path="/app/${key}"
                        ;;
                esac
                
                DOCKER_ARGS+=("-v" "$host_path:$container_path")
                echo -e "  ${GREEN}✓${NC} $key: $host_path -> $container_path"
            fi
        done
        
        # Mount certificate file directories
        # Collect unique directories to mount
        declare -A cert_dirs_seen
        for key in "${!CONFIG_CERTS[@]}"; do
            cert_file="${CONFIG_CERTS[$key]}"
            original_path="${CONFIG_CERTS_ORIGINAL[$key]}"
            
            if [ -n "$cert_file" ]; then
                cert_dir=$(dirname "$cert_file")
                
                # Check for path mismatch between config and mount location
                if [ -n "$original_path" ]; then
                    original_dir=$(dirname "$original_path")
                    if [ "$cert_dir" != "$original_dir" ]; then
                        echo -e "  ${YELLOW}⚠${NC} ${RED}WARNING:${NC} Certificate path mismatch detected for $key"
                        echo -e "     Config file path: ${original_path}"
                        echo -e "     Host file path:   ${cert_file}"
                        echo -e "     ${YELLOW}The container will mount: ${cert_dir} -> ${cert_dir}${NC}"
                        echo -e "     ${YELLOW}But the app will look for: ${original_path}${NC}"
                        echo -e "     ${YELLOW}This will cause SSL certificate errors!${NC}"
                        echo -e "     ${YELLOW}Solution: Update your config file to use the host path:${NC}"
                        echo -e "     ${YELLOW}  \"$key\": \"${cert_file}\"${NC}"
                        echo ""
                    fi
                fi
                
                # Only mount if we haven't seen this directory
                if [ -z "${cert_dirs_seen[$cert_dir]}" ]; then
                    DOCKER_ARGS+=("-v" "$cert_dir:$cert_dir:ro")
                    cert_dirs_seen["$cert_dir"]=1
                    echo -e "  ${GREEN}✓${NC} certs: $cert_dir -> $cert_dir (ro)"
                fi
            fi
        done
    else
        # Fallback to default directory mounts
        echo -e "${YELLOW}Using default directory mounts${NC}"
        DOCKER_ARGS+=("-v" "$(pwd)/data:/app/data")
        DOCKER_ARGS+=("-v" "$(pwd)/models:/app/models")
        DOCKER_ARGS+=("-v" "$(pwd)/plans:/app/plans")
        DOCKER_ARGS+=("-v" "$(pwd)/state:/app/state")
    fi
    
    # Set MOSAIC_CONFIG environment variable if config file exists
    if [ "$CONFIG_PARSED" = true ]; then
        DOCKER_ARGS+=("-e" "MOSAIC_CONFIG=/app/$CONFIG_BASENAME")
    fi
    
    # Image name
    DOCKER_ARGS+=("$IMAGE_NAME")
}

# Main execution
main() {
    # Parse command-line arguments first
    parse_args "$@"
    
    echo -e "${GREEN}=== Mosaic Docker Runner ===${NC}"
    echo ""
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Container $CONTAINER_NAME already exists${NC}"
        read -p "Remove existing container and start new one ('y' to remove, 'n' to attach to existing container)? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}Removing existing container...${NC}"
            docker rm -f $CONTAINER_NAME 2>/dev/null || true
        else
            echo -e "${YELLOW}Attaching to existing container...${NC}"
            docker attach $CONTAINER_NAME
            exit 0
        fi
    fi
    
    # Check if image exists
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
        echo -e "${RED}Image $IMAGE_NAME not found${NC}"
        echo -e "${YELLOW}Please pull or build the image before running this script${NC}"
        echo -e "${YELLOW}Example: docker pull $IMAGE_NAME${NC}"
        exit 1
    fi
    
    # Check config
    check_config
    
    # Create directories
    create_directories
    
    # Build command array
    build_docker_command
    
    echo ""
    echo -e "${GREEN}Starting Mosaic container...${NC}"
    echo -e "${YELLOW}Command: ${DOCKER_ARGS[*]}${NC}"
    echo ""
    
    # Execute using array (properly handles special characters)
    "${DOCKER_ARGS[@]}"
}

# Run main function with all arguments
main "$@"

