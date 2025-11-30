"""Beacon class for managing heartbeat communications."""

import json
import logging
import pickle
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from mosaic_config.config import MosaicConfig, Peer
from mosaic_config.state_utils import (
    StateIdentifiers,
    read_state,
    save_state,
)
from mosaic_planner.planner import (
    deserialize_plan_with_data,
    prepare_file_data_for_transmission,
    serialize_plan_with_data,
)
from mosaic_planner.state import Data, FileDefinition, Plan
from mosaic_stats.benchmark import load_benchmarks
from mosaic_stats.stats_collector import StatsCollector

logger = logging.getLogger(__name__)


@dataclass
class SendHeartbeatStatus:
    """Status of heartbeats being sent to a peer."""

    host: str
    heartbeat_port: int
    last_time_sent: int = 0  # milliseconds since epoch, 0 means never sent
    connection_status: str = "nascent"  # cert_error, nascent, ok, timeout

    def __post_init__(self):
        """Validate connection status."""
        valid_statuses = {"cert_error", "nascent", "ok", "timeout"}
        if self.connection_status not in valid_statuses:
            raise ValueError(
                f"Invalid connection_status: {self.connection_status}. "
                f"Must be one of {valid_statuses}"
            )


@dataclass
class ReceiveHeartbeatStatus:
    """Status of heartbeats being received from a peer."""

    host: str
    heartbeat_port: int
    comms_port: int = 0  # Comms port for the peer, 0 means not set
    last_time_received: int = 0  # milliseconds since epoch, 0 means never received
    connection_status: str = "nascent"  # nascent, online, stale
    stats_payload: Optional[Dict[str, Any]] = None
    delay: Optional[int] = None  # Delay in nanoseconds (receive_time_ns - send_time_ns)

    def __post_init__(self):
        """Validate connection status."""
        valid_statuses = {"nascent", "online", "stale"}
        if self.connection_status not in valid_statuses:
            raise ValueError(
                f"Invalid connection_status: {self.connection_status}. "
                f"Must be one of {valid_statuses}"
            )


class Beacon:
    """Manages heartbeat communications with peers."""

    def __init__(self, config: MosaicConfig):
        """
        Initialize the Beacon.

        Args:
            config: MosaicConfig instance for configuration
        """
        self.config = config
        self.stats_collector = StatsCollector(config)

        # Dictionary keyed by (host, heartbeat_port) for efficient lookup
        # Try to load from saved state, otherwise initialize empty
        loaded_send = read_state(
            config, StateIdentifiers.SEND_HEARTBEAT_STATUSES, default=None
        )
        loaded_receive = read_state(
            config, StateIdentifiers.RECEIVE_HEARTBEAT_STATUSES, default=None
        )
        
        # Initialize from loaded state or empty dict
        if isinstance(loaded_send, dict):
            self._send_heartbeat_statuses: Dict[tuple[str, int], SendHeartbeatStatus] = loaded_send
            logger.info(f"Loaded {len(loaded_send)} send heartbeat statuses from state")
        else:
            self._send_heartbeat_statuses: Dict[tuple[str, int], SendHeartbeatStatus] = {}
        
        if isinstance(loaded_receive, dict):
            self._receive_heartbeat_statuses: Dict[tuple[str, int], ReceiveHeartbeatStatus] = loaded_receive
            logger.info(f"Loaded {len(loaded_receive)} receive heartbeat statuses from state")
        else:
            self._receive_heartbeat_statuses: Dict[tuple[str, int], ReceiveHeartbeatStatus] = {}

        # Command handler registry
        # Handlers can accept Dict (JSON) or bytes (binary/pickled) payloads
        self._command_handlers: Dict[str, Callable[[Union[Dict[str, Any], bytes]], Optional[Any]]] = {}
        
        # Register default command handlers
        self.register("add_peer", self._handle_add_peer)
        self.register("ping", self._handle_ping_test)
        self.register("stats", self._handle_stats)
        self.register("exdplan", self._handle_execute_data_plan)

        # Thread control
        self._stop_event = threading.Event()
        self._running = False
        self._send_heartbeat_thread: Optional[threading.Thread] = None
        self._receive_heartbeat_thread: Optional[threading.Thread] = None
        self._receive_comms_thread: Optional[threading.Thread] = None
        self._check_stale_heartbeats_thread: Optional[threading.Thread] = None

    @property
    def send_heartbeat_statuses(self) -> List[SendHeartbeatStatus]:
        """
        Get list of all send-heartbeat statuses.

        Returns:
            List of SendHeartbeatStatus objects
        """
        return list(self._send_heartbeat_statuses.values())

    @property
    def receive_heartbeat_statuses(self) -> List[ReceiveHeartbeatStatus]:
        """
        Get list of all receive-heartbeat statuses.

        Returns:
            List of ReceiveHeartbeatStatus objects
        """
        return list(self._receive_heartbeat_statuses.values())

    def get_send_heartbeat_status(
        self, host: str, heartbeat_port: int
    ) -> Optional[SendHeartbeatStatus]:
        """
        Get send-heartbeat status for a specific peer.

        Args:
            host: Peer host address
            heartbeat_port: Peer heartbeat port

        Returns:
            SendHeartbeatStatus if found, None otherwise
        """
        key = (host, heartbeat_port)
        return self._send_heartbeat_statuses.get(key)

    def get_receive_heartbeat_status(
        self, host: str, heartbeat_port: int
    ) -> Optional[ReceiveHeartbeatStatus]:
        """
        Get receive-heartbeat status for a specific peer.

        Args:
            host: Peer host address
            heartbeat_port: Peer heartbeat port

        Returns:
            ReceiveHeartbeatStatus if found, None otherwise
        """
        key = (host, heartbeat_port)
        return self._receive_heartbeat_statuses.get(key)

    def register(self, command: str, handler: Callable[[Dict[str, Any]], Optional[Any]]) -> None:
        """
        Register a command handler function.

        Args:
            command: Command name (string) to register
            handler: Function to call when command is received.
                     The function will be called with the command payload as argument.
                     Can return a value to send back to the caller.
        """
        if not isinstance(command, str):
            raise TypeError(f"Command must be a string, got {type(command)}")
        if not callable(handler):
            raise TypeError(f"Handler must be callable, got {type(handler)}")
        
        self._command_handlers[command] = handler
        logger.debug(f"Registered command handler for '{command}'")

    def start(self) -> None:
        """
        Start the beacon and all communication threads.

        Creates four threads:
        1. Send heartbeat thread (runs periodically)
        2. Receive heartbeat UDP listener thread
        3. Receive comms TCP listener thread
        4. Check stale heartbeats thread (runs periodically)

        Also starts the stats collector.
        """
        if self._running:
            logger.warning("Beacon is already running")
            return

        # Start stats collector
        self.stats_collector.start()

        self._stop_event.clear()
        self._running = True

        # Thread 1: Send heartbeats periodically
        self._send_heartbeat_thread = threading.Thread(
            target=self._send_heartbeat_loop, daemon=True, name="SendHeartbeat"
        )
        self._send_heartbeat_thread.start()

        # Thread 2: Receive heartbeat UDP listener
        self._receive_heartbeat_thread = threading.Thread(
            target=self._receive_heartbeat_listener, daemon=True, name="ReceiveHeartbeat"
        )
        self._receive_heartbeat_thread.start()

        # Thread 3: Receive comms TCP listener
        self._receive_comms_thread = threading.Thread(
            target=self._receive_comms_listener, daemon=True, name="ReceiveComms"
        )
        self._receive_comms_thread.start()

        # Thread 4: Check stale heartbeats periodically
        self._check_stale_heartbeats_thread = threading.Thread(
            target=self._check_stale_heartbeats_loop, daemon=True, name="CheckStaleHeartbeats"
        )
        self._check_stale_heartbeats_thread.start()

        logger.info("Beacon started with all threads running")

    def _send_heartbeat_loop(self) -> None:
        """Loop that runs send heartbeats periodically."""
        while not self._stop_event.is_set():
            try:
                self.run_send_heartbeats()
                
                # Save state after sending heartbeats
                try:
                    save_state(
                        self.config,
                        self._send_heartbeat_statuses,
                        StateIdentifiers.SEND_HEARTBEAT_STATUSES,
                    )
                    save_state(
                        self.config,
                        self._receive_heartbeat_statuses,
                        StateIdentifiers.RECEIVE_HEARTBEAT_STATUSES,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save heartbeat state: {e}")
            except Exception as e:
                logger.error(f"Error in send heartbeat loop: {e}")

            # Wait for heartbeat_frequency seconds or until stop event
            self._stop_event.wait(self.config.heartbeat_frequency)

    def _check_stale_heartbeats_loop(self) -> None:
        """Loop that checks for stale heartbeats periodically."""
        while not self._stop_event.is_set():
            try:
                self._check_stale_heartbeats()
            except Exception as e:
                logger.error(f"Error in stale heartbeat check loop: {e}")

            # Wait for heartbeat_frequency seconds or until stop event
            self._stop_event.wait(self.config.heartbeat_frequency)

    def _check_stale_heartbeats(self) -> None:
        """Check all receive-heartbeat statuses and mark stale ones."""
        current_time_s = time.time()
        tolerance_seconds = self.config.heartbeat_tolerance
        cutoff_time_ms = int((current_time_s - tolerance_seconds) * 1000)

        for status_entry in self._receive_heartbeat_statuses.values():
            if status_entry.last_time_received < cutoff_time_ms:
                status_entry.connection_status = "stale"

    def _receive_heartbeat_listener(self) -> None:
        """UDP listener for receiving heartbeats on heartbeat_port."""
        sock = None
        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.config.host, self.config.heartbeat_port))

            logger.info(f"UDP heartbeat listener started on {self.config.host}:{self.config.heartbeat_port}")

            # Set up SSL context if certificates are provided
            ssl_context = None
            if self.config.server_crt and self.config.server_key:
                try:
                    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                    ssl_context.load_cert_chain(
                        certfile=self.config.server_crt, keyfile=self.config.server_key
                    )
                    if self.config.ca_crt:
                        ssl_context.load_verify_locations(cafile=self.config.ca_crt)
                    logger.debug("SSL context created for UDP heartbeat listener")
                except Exception as e:
                    logger.error(f"Failed to create SSL context for UDP listener: {e}")

            while not self._stop_event.is_set():
                try:
                    sock.settimeout(1.0)  # Check stop event periodically
                    data, addr = sock.recvfrom(65535)  # Max UDP packet size

                    # Try to parse as JSON
                    payload = None
                    try:
                        payload = json.loads(data.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # If not JSON, pass raw bytes or empty dict
                        payload = data

                    # Call handler in try/except as requested
                    try:
                        self.run_receive_heartbeat(payload)
                    except Exception as e:
                        logger.error(f"Error in run_receive_heartbeat: {e}")

                except socket.timeout:
                    # Timeout is expected, continue loop
                    continue
                except OSError as e:
                    if not self._stop_event.is_set():
                        logger.error(f"UDP listener error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in UDP listener: {e}")

        except Exception as e:
            logger.error(f"Failed to start UDP heartbeat listener: {e}")
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
            logger.info("UDP heartbeat listener stopped")

    def _receive_comms_listener(self) -> None:
        """TCP listener for receiving comms on comms_port."""
        server_sock = None
        try:
            # Create TCP socket
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.config.host, self.config.comms_port))
            server_sock.listen(10)  # Allow up to 10 pending connections
            server_sock.settimeout(1.0)  # Check stop event periodically

            logger.info(f"TCP comms listener started on {self.config.host}:{self.config.comms_port}")

            # Set up SSL context if certificates are provided
            ssl_context = None
            if self.config.server_crt and self.config.server_key:
                try:
                    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                    ssl_context.load_cert_chain(
                        certfile=self.config.server_crt, keyfile=self.config.server_key
                    )
                    if self.config.ca_crt:
                        ssl_context.load_verify_locations(cafile=self.config.ca_crt)
                    logger.debug("SSL context created for TCP comms listener")
                except Exception as e:
                    logger.error(f"Failed to create SSL context for TCP listener: {e}")

            while not self._stop_event.is_set():
                try:
                    client_sock, addr = server_sock.accept()
                    logger.debug(f"Accepted connection from {addr}")

                    # Wrap with SSL if context is available
                    if ssl_context:
                        try:
                            client_sock = ssl_context.wrap_socket(
                                client_sock, server_side=True
                            )
                            logger.debug(f"SSL handshake completed with {addr}")
                        except ssl.SSLError as e:
                            logger.error(f"SSL handshake failed with {addr}: {e}")
                            try:
                                client_sock.close()
                            except Exception:
                                pass
                            continue

                    # Handle connection in a separate thread to avoid blocking
                    client_thread = threading.Thread(
                        target=self._handle_comms_connection,
                        args=(client_sock, addr),
                        daemon=True,
                    )
                    client_thread.start()

                except socket.timeout:
                    # Timeout is expected, continue loop
                    continue
                except OSError as e:
                    if not self._stop_event.is_set():
                        logger.error(f"TCP listener error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in TCP listener: {e}")

        except Exception as e:
            logger.error(f"Failed to start TCP comms listener: {e}")
        finally:
            if server_sock:
                try:
                    server_sock.close()
                except Exception:
                    pass
            logger.info("TCP comms listener stopped")

    def _receive_chunked_data(
        self, client_sock: socket.socket, addr: tuple, timeout: float = 30.0
    ) -> Optional[bytes]:
        """
        Receive all data from a TCP socket in chunks.
        
        Reads data in chunks until the connection is closed or a complete
        JSON message is detected. Includes safety limits and timeout handling.
        
        Args:
            client_sock: The client socket to receive data from
            addr: Client address tuple for logging
            timeout: Socket timeout in seconds (default: 30.0)
        
        Returns:
            bytes: Complete received data, or None if error or empty
        """
        try:
            # Set a timeout to avoid blocking indefinitely
            client_sock.settimeout(timeout)
            
            # Receive all data in chunks
            chunks = []
            chunk_size = 1024 * 1024  # Read in 1MB chunks
            
            while True:
                chunk = client_sock.recv(chunk_size)
                if not chunk:
                    # Empty chunk means connection closed by peer
                    break
                chunks.append(chunk)
                
                # Try to parse as JSON after each chunk to detect complete message
                # This allows us to handle cases where the message is complete
                # but the connection isn't closed yet
                try:
                    data = b"".join(chunks)
                    decoded = data.decode("utf-8")
                    # Attempt to parse - if successful, we have complete message
                    json.loads(decoded)
                    # If we get here, JSON is valid and complete, stop reading
                    break
                except json.JSONDecodeError:
                    # JSON not complete yet or invalid, continue reading
                    continue
                except UnicodeDecodeError:
                    # Not valid UTF-8 yet, might be incomplete, continue reading
                    continue
            
            # Combine all chunks
            data = b"".join(chunks)
            
            if not data:
                logger.warning(f"Received empty data from {addr}")
                return None
            
            return data
            
        except socket.timeout:
            logger.error(f"Timeout receiving data from {addr}")
            return None
        except Exception as e:
            logger.error(f"Error receiving chunked data from {addr}: {e}")
            return None

    def _handle_comms_connection(self, client_sock: socket.socket, addr: tuple) -> None:
        """
        Handle a single TCP comms connection.
        
        Always expects header format: 4-byte header length (big-endian) + JSON header + payload
        Header: {"command": "...", "payload_type": "json"|"binary", "payload_length": N}
        """
        try:
            # Always expect header format: read 4-byte header length first
            header_length_bytes = b""
            while len(header_length_bytes) < 4:
                chunk = client_sock.recv(4 - len(header_length_bytes))
                if not chunk:
                    logger.warning(f"Connection closed before header length from {addr}")
                    return
                header_length_bytes += chunk
            
            # Parse header length
            try:
                header_length = int.from_bytes(header_length_bytes, byteorder="big")
                # Reasonable check: header should be between 10 and 1MB
                if not (10 <= header_length <= 1024 * 1024):
                    logger.warning(f"Invalid header length: {header_length}")
                    return
            except (ValueError, OverflowError):
                logger.warning(f"Failed to parse header length from {addr}")
                return
            
            # Read the JSON header
            header_bytes = b""
            while len(header_bytes) < header_length:
                chunk = client_sock.recv(header_length - len(header_bytes))
                if not chunk:
                    logger.warning(f"Connection closed before header from {addr}")
                    return
                header_bytes += chunk
            
            # Parse header
            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to parse header JSON from {addr}: {e}")
                return
            
            # Extract command and payload type
            command = header.get("command")
            payload_type = header.get("payload_type")
            payload_length = header.get("payload_length", 0)
            
            if command is None:
                logger.warning(f"Header missing 'command' field from {addr}")
                return
            
            # Read payload based on type
            if payload_type == "binary":
                # Read binary payload
                payload_data = b""
                while len(payload_data) < payload_length:
                    chunk = client_sock.recv(min(1024 * 1024, payload_length - len(payload_data)))
                    if not chunk:
                        logger.warning(f"Connection closed before complete binary payload from {addr}")
                        return
                    payload_data += chunk
                # Construct payload dict with command and binary payload
                payload = {
                    "command": command,
                    "payload": payload_data,
                }
            elif payload_type == "json":
                # Read JSON payload
                payload_data = b""
                while len(payload_data) < payload_length:
                    chunk = client_sock.recv(min(1024 * 1024, payload_length - len(payload_data)))
                    if not chunk:
                        logger.warning(f"Connection closed before complete JSON payload from {addr}")
                        return
                    payload_data += chunk
                # Parse JSON payload
                try:
                    json_payload = json.loads(payload_data.decode("utf-8"))
                    payload = {
                        "command": command,
                        "payload": json_payload,
                    }
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse JSON payload from {addr}: {e}")
                    return
            else:
                logger.warning(f"Unknown payload_type '{payload_type}' from {addr}")
                return

            # Call handler in try/except as requested
            try:
                result = self.run_receive_comms(payload)
                
                # If run_receive_comms returns a non-None value, send it back
                # Always use header format: 4-byte header length + JSON header + payload
                if result is not None:
                    try:
                        # Determine payload type and serialize
                        if isinstance(result, bytes):
                            payload_bytes = result
                            payload_type = "binary"
                        elif isinstance(result, (dict, list)):
                            payload_bytes = json.dumps(result).encode("utf-8")
                            payload_type = "json"
                        elif isinstance(result, str):
                            payload_bytes = result.encode("utf-8")
                            payload_type = "json"  # Strings are sent as JSON-encoded
                        else:
                            # Convert other types to JSON
                            payload_bytes = json.dumps(result).encode("utf-8")
                            payload_type = "json"
                        
                        # Create header
                        header = {
                            "payload_type": payload_type,
                            "payload_length": len(payload_bytes),
                        }
                        header_bytes = json.dumps(header).encode("utf-8")
                        # Send header length (4 bytes, big-endian), then header, then payload
                        header_length = len(header_bytes).to_bytes(4, byteorder="big")
                        response_message = header_length + header_bytes + payload_bytes
                        client_sock.sendall(response_message)
                    except Exception as e:
                        logger.error(f"Error sending response to {addr}: {e}")
            except Exception as e:
                logger.error(f"Error in run_receive_comms: {e}")

        except Exception as e:
            logger.error(f"Error handling comms connection from {addr}: {e}")
        finally:
            try:
                client_sock.close()
            except Exception:
                pass

    def run_send_heartbeats(self) -> None:
        """
        Run send heartbeats logic.

        Iterates through all peers in MosaicConfig and sends heartbeats to each.
        This method is called periodically by the send heartbeat thread.
        """
        # Get the last stats as JSON string and parse it
        stats_json_str = self.stats_collector.get_last_stats_json()
        try:
            stats = json.loads(stats_json_str) if stats_json_str else {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse stats JSON, using empty dict")
            stats = {}

        # Build the payload
        payload = {
            "host": self.config.host,
            "port": self.config.heartbeat_port,
            "comms_port": self.config.comms_port,
            "stats": stats,
        }

        # Iterate through all peers and send heartbeats
        for peer in self.config.peers:
            try:
                self.send_heartbeat(
                    host=peer.host,
                    port=peer.heartbeat_port,
                    heartbeat_wait_timeout=self.config.heartbeat_wait_timeout,
                    server_crt=self.config.server_crt,
                    server_key=self.config.server_key,
                    ca_crt=self.config.ca_crt,
                    json_payload=payload,
                )
            except Exception as e:
                logger.error(f"Failed to send heartbeat to {peer.host}:{peer.heartbeat_port}: {e}")

    def run_receive_heartbeat(self, payload: Any) -> None:
        """
        Handle received heartbeat payload.

        Updates or creates receive-heartbeat status based on host/port match.
        Sets status to "online" and updates timestamp. Marks stale entries.

        Args:
            payload: The received payload (JSON dict or bytes)
        """
        # Capture receive time at the earliest possible moment
        receive_time_ns = time.time_ns()
        
        # Parse payload if it's bytes or ensure it's a dict
        if isinstance(payload, bytes):
            try:
                payload = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Received heartbeat payload is not valid JSON")
                return

        if not isinstance(payload, dict):
            logger.warning("Received heartbeat payload is not a dictionary")
            return
        
        # Add receive_time_ns to payload
        payload["receive_time_ns"] = receive_time_ns

        # Extract host, port, and comms_port from payload
        host = payload.get("host")
        port = payload.get("port")
        comms_port = payload.get("comms_port")
        stats = payload.get("stats")
        send_time_ns = payload.get("send_time_ns")

        if host is None or port is None:
            logger.warning("Received heartbeat payload missing host or port")
            return
        
        # Calculate delay if both timestamps are present
        delay = None
        if send_time_ns is not None and receive_time_ns is not None:
            try:
                # Ensure both are integers (JSON might return floats for large numbers)
                send_time_ns_int = int(send_time_ns)
                receive_time_ns_int = int(receive_time_ns)
                delay = receive_time_ns_int - send_time_ns_int
            except (TypeError, ValueError, OverflowError):
                # If timestamps are not valid integers, delay remains None
                logger.warning(f"Failed to calculate delay: send_time_ns={send_time_ns}, receive_time_ns={receive_time_ns}")
                pass

        # Ensure port is an integer
        try:
            port = int(port)
        except (ValueError, TypeError):
            logger.warning(f"Received heartbeat payload has invalid port: {port}")
            return

        # Ensure comms_port is an integer if provided
        if comms_port is not None:
            try:
                comms_port = int(comms_port)
            except (ValueError, TypeError):
                logger.warning(f"Received heartbeat payload has invalid comms_port: {comms_port}")
                comms_port = 0
        else:
            comms_port = 0

        # Get current timestamp in milliseconds
        current_time_ms = int(time.time() * 1000)

        # Find or create status entry
        key = (host, port)
        if key in self._receive_heartbeat_statuses:
            # Update existing entry
            status = self._receive_heartbeat_statuses[key]
            status.last_time_received = current_time_ms
            status.connection_status = "online"
            status.delay = delay

            # Handle stats: if payload has stats, use it; otherwise keep existing if present
            if stats is not None:
                status.stats_payload = stats
            # Update comms_port if provided
            if comms_port != 0:
                status.comms_port = comms_port
            # If payload doesn't have stats and existing has stats, keep existing (no change needed)
        else:
            # Create new entry
            status = ReceiveHeartbeatStatus(
                host=host,
                heartbeat_port=port,
                comms_port=comms_port,
                last_time_received=current_time_ms,
                connection_status="online",
                stats_payload=stats if stats is not None else None,
                delay=delay,
            )
            self._receive_heartbeat_statuses[key] = status

    def run_receive_comms(self, payload: Any) -> Optional[Any]:
        """
        Handle received comms payload.

        Parses JSON payload with structure: {command: <string>, payload: {}}
        Processes commands and acts accordingly.

        Args:
            payload: The received payload (JSON dict with "command" and "payload" keys)
        
        Returns:
            Optional value that will be sent back to the caller if not None
        """
        # Parse payload if it's bytes or ensure it's a dict
        if isinstance(payload, bytes):
            try:
                payload = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Received comms payload is not valid JSON")
                return None

        if not isinstance(payload, dict):
            logger.warning("Received comms payload is not a dictionary")
            return None

        # Extract command and payload
        command = payload.get("command")
        command_payload = payload.get("payload")

        if command is None:
            logger.warning("Received comms payload missing 'command' field")
            return None

        if command_payload is None:
            logger.warning("Received comms payload missing 'payload' field")
            return None

        # Handle commands using registered handlers
        handler = self._command_handlers.get(command)
        if handler:
            try:
                # Handler receives the command_payload directly (can be dict or bytes)
                result = handler(command_payload)
                return result
            except Exception as e:
                logger.error(f"Error executing command handler for '{command}': {e}")
                return None
        else:
            logger.warning(f"Unknown command received: {command}")
            return None

    def send_command(
        self,
        host: str,
        port: int,
        command: str,
        payload: Union[Dict[str, Any], bytes],
        timeout: float = 30.0,
    ) -> Optional[Any]:
        """
        Send a command to another Beacon instance and return the response.

        Connects to the specified host/port, sends the command and payload,
        and returns the response from the remote Beacon.

        Args:
            host: Target host address
            port: Target comms port
            command: Command name to send
            payload: Command payload - either a Dict (JSON) or bytes (pickled/binary)
            timeout: Connection and receive timeout in seconds (default: 30.0)

        Returns:
            Response value from the remote Beacon, or None if error or no response
        """
        sock = None
        try:
            # Create TCP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)

            # Set up SSL context if certificates are provided
            ssl_context = None
            if self.config.server_crt and self.config.server_key:
                try:
                    ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                    ssl_context.load_cert_chain(
                        certfile=self.config.server_crt, keyfile=self.config.server_key
                    )
                    if self.config.ca_crt:
                        ssl_context.load_verify_locations(cafile=self.config.ca_crt)
                except Exception as e:
                    logger.error(f"Failed to create SSL context for send_command: {e}")
                    return None

            # Connect to remote host
            try:
                sock.connect((host, port))
            except (socket.timeout, OSError) as e:
                logger.error(f"Failed to connect to {host}:{port}: {e}")
                return None

            # Wrap with SSL if context is available
            if ssl_context:
                try:
                    sock = ssl_context.wrap_socket(sock, server_hostname=host)
                except ssl.SSLError as e:
                    logger.error(f"SSL handshake failed with {host}:{port}: {e}")
                    return None

            # Prepare command message
            # Always use header format: 4-byte header length + JSON header + payload
            if isinstance(payload, bytes):
                # Binary payload
                payload_bytes = payload
                payload_type = "binary"
            else:
                # JSON payload - serialize to bytes
                payload_bytes = json.dumps(payload).encode("utf-8")
                payload_type = "json"
            
            # Create header with command, payload type, and payload length
            header = {
                "command": command,
                "payload_type": payload_type,
                "payload_length": len(payload_bytes),
            }
            header_bytes = json.dumps(header).encode("utf-8")
            # Send header length (4 bytes, big-endian), then header, then payload
            header_length = len(header_bytes).to_bytes(4, byteorder="big")
            message_bytes = header_length + header_bytes + payload_bytes

            # Send command
            try:
                sock.sendall(message_bytes)
            except Exception as e:
                logger.error(f"Failed to send command to {host}:{port}: {e}")
                return None

            # Receive response - always expect header format
            try:
                # Read header length (4 bytes)
                header_length_bytes = b""
                while len(header_length_bytes) < 4:
                    chunk = sock.recv(4 - len(header_length_bytes))
                    if not chunk:
                        logger.warning(f"Connection closed before response header length from {host}:{port}")
                        return None
                    header_length_bytes += chunk
                
                # Parse header length
                header_length = int.from_bytes(header_length_bytes, byteorder="big")
                if not (10 <= header_length <= 1024 * 1024):
                    logger.error(f"Invalid response header length: {header_length}")
                    return None
                
                # Read the JSON header
                header_bytes = b""
                while len(header_bytes) < header_length:
                    chunk = sock.recv(header_length - len(header_bytes))
                    if not chunk:
                        logger.warning(f"Connection closed before response header from {host}:{port}")
                        return None
                    header_bytes += chunk
                
                # Parse header
                header = json.loads(header_bytes.decode("utf-8"))
                payload_type = header.get("payload_type")
                payload_length = header.get("payload_length", 0)
                
                if payload_type is None:
                    logger.warning(f"Response header missing 'payload_type' from {host}:{port}")
                    return None
                
                # Read payload based on type
                if payload_type == "binary":
                    # Read binary payload
                    response_data = b""
                    while len(response_data) < payload_length:
                        chunk = sock.recv(min(1024 * 1024, payload_length - len(response_data)))
                        if not chunk:
                            logger.warning(f"Connection closed before complete binary response from {host}:{port}")
                            return None
                        response_data += chunk
                    return response_data
                elif payload_type == "json":
                    # Read JSON payload
                    response_data = b""
                    while len(response_data) < payload_length:
                        chunk = sock.recv(min(1024 * 1024, payload_length - len(response_data)))
                        if not chunk:
                            logger.warning(f"Connection closed before complete JSON response from {host}:{port}")
                            return None
                        response_data += chunk
                    # Parse JSON
                    try:
                        response = json.loads(response_data.decode("utf-8"))
                        return response
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.error(f"Failed to parse JSON response from {host}:{port}: {e}")
                        return None
                else:
                    logger.warning(f"Unknown response payload_type '{payload_type}' from {host}:{port}")
                    return None
                    
            except socket.timeout:
                logger.warning(f"Timeout receiving response from {host}:{port}")
                return None
            except (ValueError, OverflowError) as e:
                logger.error(f"Error parsing response header length from {host}:{port}: {e}")
                return None
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"Error parsing response header from {host}:{port}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error receiving response from {host}:{port}: {e}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in send_command to {host}:{port}: {e}")
            return None
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def _handle_add_peer(self, payload: Dict[str, Any]) -> None:
        """
        Handle add_peer command.

        Args:
            payload: Dictionary with host, comms_port, and heartbeat_port
        """
        # Extract required fields
        host = payload.get("host")
        comms_port = payload.get("comms_port")
        heartbeat_port = payload.get("heartbeat_port")

        if host is None or comms_port is None or heartbeat_port is None:
            logger.warning("add_peer payload missing required fields (host, comms_port, heartbeat_port)")
            return

        # Validate and convert ports to integers
        try:
            comms_port = int(comms_port)
            heartbeat_port = int(heartbeat_port)
        except (ValueError, TypeError) as e:
            logger.warning(f"add_peer payload has invalid port values: {e}")
            return

        # Check if peer already exists
        for existing_peer in self.config.peers:
            if (
                existing_peer.host == host
                and existing_peer.comms_port == comms_port
                and existing_peer.heartbeat_port == heartbeat_port
            ):
                logger.info(f"Peer {host}:{comms_port}/{heartbeat_port} already exists, skipping")
                return

        # Create and add new peer
        new_peer = Peer(host=host, comms_port=comms_port, heartbeat_port=heartbeat_port)
        self.config.peers.append(new_peer)
        logger.info(f"Added new peer: {host}:{comms_port}/{heartbeat_port}")

    def _handle_ping_test(self, payload: Union[Dict[str, Any], bytes]) -> Union[Dict[str, Any], bytes]:
        """
        Handle ping command.

        Args:
            payload: Dictionary or bytes payload to echo back

        Returns:
            The same payload (dict or bytes)
        """
        return payload

    def _handle_stats(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle stats command.

        Returns a copy of all receive-heartbeat statuses.

        Args:
            payload: Command payload (not used, but required by handler signature)

        Returns:
            List of receive-heartbeat status dictionaries, each including benchmark data
        """
        # Load benchmark data once for all statuses
        benchmark_data = load_benchmarks(self.config.benchmark_data_location)
        
        # Return a copy of all receive-heartbeat statuses as dictionaries
        status_list = []
        for status in self._receive_heartbeat_statuses.values():
            status_dict = {
                "host": status.host,
                "heartbeat_port": status.heartbeat_port,
                "comms_port": status.comms_port,
                "last_time_received": status.last_time_received,
                "connection_status": status.connection_status,
                "stats_payload": status.stats_payload,
                "benchmark": benchmark_data,
            }
            status_list.append(status_dict)
        return status_list

    def collect_stats(self, include_self: bool = True) -> List[Dict[str, Any]]:
        """
        Collect stats from all peers by sending stats commands.

        Iterates through all receive-heartbeat statuses and requests stats
        from each peer. Stops if the timeout is exceeded.

        Args:
            include_self: If True, append local node stats to the result (default: True)

        Returns:
            List of stat dictionaries collected from peers (and optionally local node)
        """
        start_time = int(time.time() * 1000)  # Current millis since epoch
        status_request_cache: List[Dict[str, Any]] = []

        for status in self._receive_heartbeat_statuses.values():
            # Check timeout before processing
            current_time = int(time.time() * 1000)
            if (current_time - start_time) > (self.config.stats_request_timeout * 1000):
                # Timeout exceeded, break out of loop
                break

            # Skip if comms_port is not set (0 means not set)
            if status.comms_port == 0:
                logger.warning(f"Skipping {status.host}:{status.comms_port} because comms_port is not set")
                continue

            try:
                # Send stats command to the peer
                result = self.send_command(
                    host=status.host,
                    port=status.comms_port,
                    command="stats",
                    payload={},
                )

                # Add each element of the result to status_request_cache
                if result is not None and isinstance(result, list):
                    for stat_item in result:
                        if isinstance(stat_item, dict):
                            status_request_cache.append(stat_item)
            except Exception as e:
                logger.error(f"Error collecting stats from {status.host}:{status.comms_port}: {e}")
                continue

            # Check timeout after each iteration
            current_time = int(time.time() * 1000)
            if (current_time - start_time) > (self.config.stats_request_timeout * 1000):
                # Timeout exceeded, break out of loop
                break

        # Add local node stats if requested
        if include_self:
            try:
                # Get local stats from stats_collector
                stats_json = self.stats_collector.get_last_stats_json()
                stats_payload = None
                if stats_json:
                    try:
                        stats_payload = json.loads(stats_json)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse local stats JSON")
                        stats_payload = {}

                # Load benchmark data
                benchmark_data = load_benchmarks(self.config.benchmark_data_location)

                # Create local node stats entry
                local_stats = {
                    "host": self.config.host,
                    "heartbeat_port": self.config.heartbeat_port,
                    "comms_port": self.config.comms_port,
                    "last_time_received": int(time.time() * 1000),
                    "connection_status": "online",  # Local node is always "online"
                    "stats_payload": stats_payload,
                    "benchmark": benchmark_data,
                }
                status_request_cache.append(local_stats)
            except Exception as e:
                logger.error(f"Error adding local node stats: {e}")

        return status_request_cache

    def _is_self_host(self, host: str, port: int) -> bool:
        """
        Check if the given host and port refer to the current Beacon instance.
        
        Args:
            host: Host address to check
            port: Port to check
        
        Returns:
            True if host/port matches this Beacon instance, False otherwise
        """
        # Check if host matches localhost variants
        localhost_variants = ["localhost", "127.0.0.1", "::1", "0.0.0.0"]
        if host.lower() in localhost_variants or host in localhost_variants:
            # Check if port matches
            if port == self.config.comms_port:
                return True
        
        # Check if host matches configured host
        if host == self.config.host:
            if port == self.config.comms_port:
                return True
        
        # Check if host is one of the local IP addresses
        try:
            import socket as sock
            # Get local hostname
            local_hostname = sock.gethostname()
            if host == local_hostname and port == self.config.comms_port:
                return True
            
            # Get local IP addresses
            local_ips = []
            try:
                # Try to get primary IP
                s = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ips.append(s.getsockname()[0])
                s.close()
            except Exception:
                pass
            
            # Check all local IPs
            for local_ip in local_ips:
                if host == local_ip and port == self.config.comms_port:
                    return True
        except Exception:
            pass
        
        return False

    def execute_data_plan(self, plan: Plan, data: Data) -> None:
        """
        Execute a data distribution plan by sending data segments to nodes.
        
        Reads the data_segmentation_plan from the Plan, prepares file data,
        and distributes it to the appropriate nodes via host/comms_port.
        If a node is the current host, calls the handler directly.
        
        Args:
            plan: Plan object with data_segmentation_plan
            data: Data object with file definitions
        """
        if not plan.data_segmentation_plan:
            logger.warning("Plan has no data_segmentation_plan")
            return
        
        # Create a mapping of file location to FileDefinition for quick lookup
        file_def_map = {fd.location: fd for fd in data.file_definitions}
        
        # Process each machine's segments
        for machine_plan in plan.data_segmentation_plan:
            host = machine_plan.get("host")
            comms_port = machine_plan.get("comms_port")
            segments = machine_plan.get("segments", [])
            
            if not host or not comms_port:
                logger.warning(f"Invalid machine plan: missing host or comms_port")
                continue
            
            # Prepare data for this machine
            machine_data = Data(file_definitions=[])
            
            for segment in segments:
                file_location = segment.get("file_location")
                if not file_location:
                    continue
                
                # Find corresponding FileDefinition
                file_def = file_def_map.get(file_location)
                if not file_def:
                    logger.warning(f"FileDefinition not found for location: {file_location}")
                    continue
                
                # Prepare file data for transmission
                try:
                    binary_data = prepare_file_data_for_transmission(
                        file_def=file_def,
                        segment_info=segment,
                        data_folder=self.config.data_location,
                    )
                    
                    # Create new FileDefinition with binary data
                    new_file_def = FileDefinition(
                        location=file_def.location,
                        data_type=file_def.data_type,
                        is_segmentable=file_def.is_segmentable,
                        binary_data=binary_data,
                    )
                    machine_data.file_definitions.append(new_file_def)
                except Exception as e:
                    logger.error(f"Error preparing data for {file_location}: {e}")
                    continue
            
            # Serialize plan and data for this machine
            try:
                # Create a machine-specific plan (just the segments for this machine)
                machine_plan_obj = Plan(
                    stats_data=plan.stats_data,
                    distribution_plan=plan.distribution_plan,
                    model=plan.model,
                    data_segmentation_plan=[machine_plan],  # Only this machine's plan
                )
                
                serialized_data = serialize_plan_with_data(machine_plan_obj, machine_data)
            except Exception as e:
                logger.error(f"Error serializing data for {host}:{comms_port}: {e}")
                continue
            
            # Send to node (or call handler directly if self)
            if self._is_self_host(host, comms_port):
                # Call handler directly
                try:
                    logger.info(f"Executing data plan locally for {host}:{comms_port}")
                    self._handle_execute_data_plan(serialized_data)
                except Exception as e:
                    logger.error(f"Error executing data plan locally: {e}")
            else:
                # Send to remote node
                try:
                    logger.info(f"Sending data plan to {host}:{comms_port}")
                    result = self.send_command(
                        host=host,
                        port=comms_port,
                        command="exdplan",
                        payload=serialized_data,
                        timeout=300.0,  # Longer timeout for data transfer
                    )
                    if result is None:
                        logger.warning(f"No response from {host}:{comms_port} for data plan")
                except Exception as e:
                    logger.error(f"Error sending data plan to {host}:{comms_port}: {e}")

    def _handle_execute_data_plan(self, payload: Union[Dict[str, Any], bytes]) -> Optional[Dict[str, Any]]:
        """
        Handle execute data plan command.
        
        Receives serialized Plan and Data, deserializes them, and stores
        the binary data in FileDefinition objects.
        
        Args:
            payload: Serialized bytes containing Plan and Data
        
        Returns:
            Dictionary with status information
        """
        if not isinstance(payload, bytes):
            logger.error("execute_data_plan payload must be bytes")
            return {"status": "error", "message": "Payload must be bytes"}
        
        try:
            # Deserialize plan and data
            plan, data = deserialize_plan_with_data(payload, compressed=True)
            
            logger.info(f"Received data plan with {len(data.file_definitions)} file definitions")
            
            # The binary_data is already set in FileDefinition objects
            # Store or process the data as needed
            # For now, just log the receipt
            
            return {
                "status": "success",
                "message": f"Received {len(data.file_definitions)} file definitions",
                "file_count": len(data.file_definitions),
            }
        except Exception as e:
            logger.error(f"Error handling execute_data_plan: {e}")
            return {"status": "error", "message": str(e)}

    def stop(self) -> None:
        """
        Stop the beacon and all threads.

        Also stops the stats collector.
        """
        if not self._running:
            return

        logger.info("Stopping beacon...")
        self._stop_event.set()
        self._running = False

        # Wait for threads to finish (with timeout)
        if self._send_heartbeat_thread:
            self._send_heartbeat_thread.join(timeout=5.0)
        if self._receive_heartbeat_thread:
            self._receive_heartbeat_thread.join(timeout=5.0)
        if self._receive_comms_thread:
            self._receive_comms_thread.join(timeout=5.0)
        if self._check_stale_heartbeats_thread:
            self._check_stale_heartbeats_thread.join(timeout=5.0)

        # Stop stats collector
        self.stats_collector.stop()

        logger.info("Beacon stopped")

    def send_heartbeat(
        self,
        host: str,
        port: int,
        heartbeat_wait_timeout: int,
        server_crt: str,
        server_key: str,
        ca_crt: str,
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send a heartbeat to a peer over secure UDP.

        Args:
            host: Target host address
            port: Target heartbeat port
            heartbeat_wait_timeout: Connection timeout in seconds
            server_crt: Path to server certificate file
            server_key: Path to server private key file
            ca_crt: Path to CA certificate file
            json_payload: Optional JSON payload to send

        Updates the send-heartbeat status for the host/port combination.
        """
        key = (host, port)
        current_time_ms = int(time.time() * 1000)

        # Initialize status if it doesn't exist
        if key not in self._send_heartbeat_statuses:
            self._send_heartbeat_statuses[key] = SendHeartbeatStatus(
                host=host, heartbeat_port=port
            )

        status = self._send_heartbeat_statuses[key]

        # Validate certificate files exist if provided
        cert_error = False
        if server_crt or server_key or ca_crt:
            if server_crt and not Path(server_crt).exists():
                logger.error(f"Server certificate not found: {server_crt}")
                cert_error = True
            if server_key and not Path(server_key).exists():
                logger.error(f"Server key not found: {server_key}")
                cert_error = True
            if ca_crt and not Path(ca_crt).exists():
                logger.error(f"CA certificate not found: {ca_crt}")
                cert_error = True

        if cert_error:
            status.connection_status = "cert_error"
            status.last_time_sent = current_time_ms
            return

        # Create UDP socket
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(heartbeat_wait_timeout)

            # Prepare payload
            payload_bytes = b""
            if json_payload is not None:
                # Make a copy of the payload to avoid modifying the original
                # Add send_time_ns at the very last moment before serialization
                payload_to_send = json_payload.copy()
                payload_to_send["send_time_ns"] = time.time_ns()
                try:
                    payload_bytes = json.dumps(payload_to_send).encode("utf-8")
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to serialize JSON payload: {e}")
                    status.connection_status = "cert_error"  # Treat as error
                    status.last_time_sent = current_time_ms
                    return

            # Attempt secure connection setup
            # Note: Python's standard library doesn't fully support DTLS
            # This validates certificates and sends UDP
            # For full DTLS encryption, consider using a library like python-dtls
            try:
                # Validate SSL context can be created with certificates
                # This validates certificate files are readable and properly formatted
                if server_crt or server_key or ca_crt:
                    try:
                        context = ssl.create_default_context(
                            ssl.Purpose.SERVER_AUTH, cafile=ca_crt if ca_crt else None
                        )
                        if server_crt and server_key:
                            context.load_cert_chain(certfile=server_crt, keyfile=server_key)
                    except ssl.SSLError as e:
                        logger.error(f"SSL certificate error: {e}")
                        status.connection_status = "cert_error"
                        status.last_time_sent = current_time_ms
                        return
                    except FileNotFoundError as e:
                        logger.error(f"Certificate file not found: {e}")
                        status.connection_status = "cert_error"
                        status.last_time_sent = current_time_ms
                        return
                    except Exception as e:
                        logger.error(f"Certificate loading error: {e}")
                        status.connection_status = "cert_error"
                        status.last_time_sent = current_time_ms
                        return

                # Send the payload over UDP
                # Note: For true DTLS encryption, wrap the socket with a DTLS library
                if payload_bytes:
                    sock.sendto(payload_bytes, (host, port))
                else:
                    # Send empty heartbeat
                    sock.sendto(b"", (host, port))

                # Connection successful
                status.connection_status = "ok"
                status.last_time_sent = current_time_ms
                logger.debug(f"Successfully sent heartbeat to {host}:{port}")

            except ssl.SSLError as e:
                logger.error(f"SSL error sending heartbeat to {host}:{port}: {e}")
                status.connection_status = "cert_error"
                status.last_time_sent = current_time_ms
            except socket.timeout:
                logger.warning(f"Timeout sending heartbeat to {host}:{port}")
                status.connection_status = "timeout"
                status.last_time_sent = current_time_ms
            except OSError as e:
                # Network errors (connection refused, etc.)
                logger.error(f"Network error sending heartbeat to {host}:{port}: {e}")
                status.connection_status = "timeout"
                status.last_time_sent = current_time_ms
            except Exception as e:
                logger.error(f"Unexpected error sending heartbeat to {host}:{port}: {e}")
                status.connection_status = "timeout"
                status.last_time_sent = current_time_ms

        except Exception as e:
            logger.error(f"Failed to create socket for {host}:{port}: {e}")
            status.connection_status = "cert_error"
            status.last_time_sent = current_time_ms
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

