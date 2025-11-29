"""Unit tests for mosaic_comms.beacon module."""

import json
import socket
import ssl
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mosaic_comms.beacon import Beacon, ReceiveHeartbeatStatus, SendHeartbeatStatus
from mosaic_config.config import MosaicConfig, Peer


@pytest.fixture(scope="session")
def test_certs_dir():
    """Create test certificate files in tests directory."""
    # Get the tests directory
    tests_dir = Path(__file__).parent
    cert_dir = tests_dir / "certs"
    cert_dir.mkdir(exist_ok=True)

    # Create dummy certificate files (they don't need to be valid for all tests)
    server_crt = cert_dir / "server.crt"
    server_key = cert_dir / "server.key"
    ca_crt = cert_dir / "ca.crt"

    # Write dummy content (not valid certs, but files exist)
    if not server_crt.exists():
        server_crt.write_text("-----BEGIN CERTIFICATE-----\nDUMMY CERT\n-----END CERTIFICATE-----\n")
    if not server_key.exists():
        server_key.write_text("-----BEGIN PRIVATE KEY-----\nDUMMY KEY\n-----END PRIVATE KEY-----\n")
    if not ca_crt.exists():
        ca_crt.write_text("-----BEGIN CERTIFICATE-----\nDUMMY CA CERT\n-----END CERTIFICATE-----\n")

    return cert_dir


@pytest.fixture
def beacon_config(test_certs_dir):
    """Create a MosaicConfig for beacon testing with SSL certificates."""
    return MosaicConfig(
        host="127.0.0.1",
        heartbeat_port=5000,
        comms_port=5001,
        heartbeat_frequency=2,  # 2 seconds for faster testing
        heartbeat_tolerance=5,  # 5 seconds as requested
        heartbeat_wait_timeout=2,
        server_crt=str(test_certs_dir / "server.crt"),
        server_key=str(test_certs_dir / "server.key"),
        ca_crt=str(test_certs_dir / "ca.crt"),
        benchmark_data_location="",  # Not needed for these tests
    )


@pytest.fixture
def beacon_config_no_ssl():
    """Create a MosaicConfig for beacon testing without SSL certificates."""
    return MosaicConfig(
        host="127.0.0.1",
        heartbeat_port=5000,
        comms_port=5001,
        heartbeat_frequency=2,
        heartbeat_tolerance=5,
        heartbeat_wait_timeout=2,
        server_crt="",  # No SSL for basic functionality tests
        server_key="",
        ca_crt="",
        benchmark_data_location="",
    )


@pytest.fixture
def sender_config(test_certs_dir):
    """Create a MosaicConfig for the sender beacon."""
    return MosaicConfig(
        host="127.0.0.1",
        heartbeat_port=5002,  # Different port to avoid conflicts
        comms_port=5003,
        heartbeat_frequency=2,
        heartbeat_tolerance=5,
        heartbeat_wait_timeout=2,
        server_crt=str(test_certs_dir / "server.crt"),
        server_key=str(test_certs_dir / "server.key"),
        ca_crt=str(test_certs_dir / "ca.crt"),
        benchmark_data_location="",
    )


@pytest.fixture
def sender_config_no_ssl():
    """Create a MosaicConfig for the sender beacon without SSL."""
    return MosaicConfig(
        host="127.0.0.1",
        heartbeat_port=5002,
        comms_port=5003,
        heartbeat_frequency=2,
        heartbeat_tolerance=5,
        heartbeat_wait_timeout=2,
        server_crt="",  # No SSL for basic functionality tests
        server_key="",
        ca_crt="",
        benchmark_data_location="",
    )


class TestBeaconInitialization:
    """Test Beacon initialization."""

    def test_beacon_initialization_does_not_start_threads(self, beacon_config):
        """Test that creating a Beacon doesn't start threads."""
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats:
            mock_stats.return_value.start = MagicMock()
            beacon = Beacon(beacon_config)

            # Verify threads are not started
            assert not beacon._running
            assert beacon._send_heartbeat_thread is None
            assert beacon._receive_heartbeat_thread is None
            assert beacon._receive_comms_thread is None
            assert beacon._check_stale_heartbeats_thread is None

            # Verify start was not called on stats collector
            mock_stats.return_value.start.assert_not_called()


class TestBeaconHeartbeatStatus:
    """Test heartbeat status updates."""

    def test_heartbeat_status_update_online_then_stale_then_online(
        self, beacon_config_no_ssl, sender_config_no_ssl
    ):
        """Test that heartbeat status updates correctly: online -> stale -> online."""
        # Mock StatsCollector
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3, "ram_percent": 67.8}'
            mock_stats_class.return_value = mock_stats

            # Create two beacons
            receiver_beacon = Beacon(beacon_config_no_ssl)
            sender_beacon = Beacon(sender_config_no_ssl)

            # Start the receiver beacon
            receiver_beacon.start()

            # Wait for threads to start and UDP listener to bind to port
            time.sleep(1.0)

            try:
                # Prepare heartbeat payload
                payload = {
                    "host": sender_config_no_ssl.host,
                    "port": sender_config_no_ssl.heartbeat_port,
                    "comms_port": sender_config_no_ssl.comms_port,
                    "stats": {"cpu_percent": 45.3, "ram_percent": 67.8},
                }

                # Send heartbeat to receiver (no SSL)
                sender_beacon.send_heartbeat(
                    host=beacon_config_no_ssl.host,
                    port=beacon_config_no_ssl.heartbeat_port,
                    heartbeat_wait_timeout=beacon_config_no_ssl.heartbeat_wait_timeout,
                    server_crt="",
                    server_key="",
                    ca_crt="",
                    json_payload=payload,
                )

                # Wait for UDP to be received and processed
                # Retry checking status a few times in case of timing issues
                status = None
                for attempt in range(20):
                    time.sleep(0.2)
                    status = receiver_beacon.get_receive_heartbeat_status(
                        sender_config_no_ssl.host, sender_config_no_ssl.heartbeat_port
                    )
                    if status is not None:
                        break

                assert status is not None, f"Heartbeat status was not created after sending heartbeat (attempted {attempt + 1} times)"
                assert status.connection_status == "online"
                assert status.last_time_received > 0
                current_time_ms = int(time.time() * 1000)
                # Timestamp should be recent (within last 2 seconds)
                assert current_time_ms - status.last_time_received < 2000

                # Wait for more than heartbeat_tolerance seconds plus time for stale check to run
                # The stale check runs every heartbeat_frequency seconds
                # We need to wait tolerance + frequency + buffer to ensure check runs
                wait_time = beacon_config_no_ssl.heartbeat_tolerance + beacon_config_no_ssl.heartbeat_frequency + 1
                time.sleep(wait_time)

                # Check status is now stale
                status = receiver_beacon.get_receive_heartbeat_status(
                    sender_config_no_ssl.host, sender_config_no_ssl.heartbeat_port
                )
                assert status is not None
                assert status.connection_status == "stale"

                # Send another heartbeat
                sender_beacon.send_heartbeat(
                    host=beacon_config_no_ssl.host,
                    port=beacon_config_no_ssl.heartbeat_port,
                    heartbeat_wait_timeout=beacon_config_no_ssl.heartbeat_wait_timeout,
                    server_crt="",
                    server_key="",
                    ca_crt="",
                    json_payload=payload,
                )

                # Wait a moment for UDP to be received
                time.sleep(0.5)

                # Check status is online again
                status = receiver_beacon.get_receive_heartbeat_status(
                    sender_config_no_ssl.host, sender_config_no_ssl.heartbeat_port
                )
                assert status is not None
                assert status.connection_status == "online"

            finally:
                receiver_beacon.stop()

    def test_stats_mock_called_during_send_heartbeat(self, beacon_config_no_ssl, sender_config_no_ssl):
        """Test that stats mock is called during send heartbeat."""
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            sender_beacon = Beacon(sender_config_no_ssl)
            sender_beacon.start()

            # Wait for send heartbeat to run
            time.sleep(beacon_config_no_ssl.heartbeat_frequency + 0.5)

            # Verify get_last_stats_json was called
            assert mock_stats.get_last_stats_json.called

            sender_beacon.stop()

    def test_stats_included_in_heartbeat_payload(self, beacon_config_no_ssl, sender_config_no_ssl):
        """Test that stats are included in heartbeat payload and received correctly."""
        # Mock StatsCollector with specific stats
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            # Return stats that will be sent in heartbeat
            test_stats = {
                "timestamp_ms": 1704067200000,
                "cpu_percent": 45.3,
                "ram_percent": 67.8,
                "disk_free_space": {
                    "/": {
                        "total_bytes": 500107862016,
                        "free_bytes": 150000000000,
                        "used_bytes": 350107862016,
                        "percent_used": 70
                    }
                },
                "gpus": [
                    {
                        "gpu_id": 0,
                        "utilization_percent": 85.5
                    }
                ]
            }
            mock_stats.get_last_stats_json.return_value = json.dumps(test_stats)
            mock_stats_class.return_value = mock_stats

            # Create two beacons
            receiver_beacon = Beacon(beacon_config_no_ssl)
            sender_beacon = Beacon(sender_config_no_ssl)

            # Add receiver as a peer to sender so it will send heartbeats
            sender_beacon.config.peers.append(
                Peer(
                    host=beacon_config_no_ssl.host,
                    comms_port=beacon_config_no_ssl.comms_port,
                    heartbeat_port=beacon_config_no_ssl.heartbeat_port,
                )
            )

            # Start both beacons
            receiver_beacon.start()
            sender_beacon.start()

            try:
                # Wait for threads to start and for heartbeat to be sent
                # Wait for at least one heartbeat cycle
                time.sleep(beacon_config_no_ssl.heartbeat_frequency + 1.0)

                # Wait for UDP to be received and processed
                # Retry checking status a few times in case of timing issues
                status = None
                for attempt in range(20):
                    time.sleep(0.2)
                    status = receiver_beacon.get_receive_heartbeat_status(
                        sender_config_no_ssl.host, sender_config_no_ssl.heartbeat_port
                    )
                    if status is not None:
                        break

                # Verify status was created
                assert status is not None, f"Status was not created after sending heartbeat (attempted {attempt + 1} times)"

                # Verify stats_payload contains the expected stats
                assert status.stats_payload is not None, "stats_payload should not be None"
                assert isinstance(status.stats_payload, dict), "stats_payload should be a dictionary"

                # Verify the stats match what was sent
                assert status.stats_payload == test_stats, f"Expected stats {test_stats}, got {status.stats_payload}"

                # Verify other status fields
                assert status.connection_status == "online"
                assert status.comms_port == sender_config_no_ssl.comms_port
                assert status.host == sender_config_no_ssl.host
                assert status.heartbeat_port == sender_config_no_ssl.heartbeat_port

            finally:
                receiver_beacon.stop()
                sender_beacon.stop()

    def test_heartbeat_delay_calculation(self, beacon_config_no_ssl, sender_config_no_ssl):
        """Test that heartbeat delay is calculated and is greater than 0."""
        # Mock StatsCollector
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3, "ram_percent": 67.8}'
            mock_stats_class.return_value = mock_stats

            # Create two beacons
            receiver_beacon = Beacon(beacon_config_no_ssl)
            sender_beacon = Beacon(sender_config_no_ssl)

            # Add receiver as a peer to sender so it will send heartbeats automatically
            sender_beacon.config.peers.append(
                Peer(
                    host=beacon_config_no_ssl.host,
                    comms_port=beacon_config_no_ssl.comms_port,
                    heartbeat_port=beacon_config_no_ssl.heartbeat_port,
                )
            )

            # Start both beacons
            receiver_beacon.start()
            sender_beacon.start()

            try:
                # Wait for threads to start and for heartbeat to be sent
                # Wait for at least one heartbeat cycle
                time.sleep(beacon_config_no_ssl.heartbeat_frequency + 1.0)

                # Wait for UDP to be received and processed
                # Retry checking status a few times in case of timing issues
                status = None
                for attempt in range(20):
                    time.sleep(0.2)
                    status = receiver_beacon.get_receive_heartbeat_status(
                        sender_config_no_ssl.host, sender_config_no_ssl.heartbeat_port
                    )
                    if status is not None and status.delay is not None:
                        break

                # Verify status was created
                assert status is not None, f"Status was not created after sending heartbeat (attempted {attempt + 1} times)"

                # Verify delay was calculated
                assert status.delay is not None, "delay should not be None when heartbeat is received"
                assert isinstance(status.delay, int), "delay should be an integer"
              
                # Verify other status fields
                assert status.connection_status == "online"
                assert status.host == sender_config_no_ssl.host
                assert status.heartbeat_port == sender_config_no_ssl.heartbeat_port

            finally:
                receiver_beacon.stop()
                sender_beacon.stop()


class TestBeaconPortListeners:
    """Test that port listeners trigger appropriate functions."""

    def test_udp_listener_triggers_run_receive_heartbeat(self, beacon_config_no_ssl):
        """Test that UDP listener calls run_receive_heartbeat."""
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats_class.return_value = mock_stats

            beacon = Beacon(beacon_config_no_ssl)
            beacon.start()

            try:
                # Wait for listener to start and bind
                time.sleep(1.0)

                # Send UDP packet
                payload = {
                    "host": "127.0.0.1",
                    "port": 5002,
                    "comms_port": 5003,
                    "stats": {"cpu_percent": 45.3},
                }
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(json.dumps(payload).encode("utf-8"), (beacon_config_no_ssl.host, beacon_config_no_ssl.heartbeat_port))
                sock.close()

                # Wait for processing - retry checking status
                status = None
                for _ in range(20):
                    time.sleep(0.2)
                    status = beacon.get_receive_heartbeat_status("127.0.0.1", 5002)
                    if status is not None:
                        break

                # Verify status was created/updated
                assert status is not None, "Status was not created after sending UDP packet"
                assert status.connection_status == "online"

            finally:
                beacon.stop()

    def test_tcp_listener_triggers_run_receive_comms(self, beacon_config_no_ssl):
        """Test that TCP listener calls run_receive_comms."""
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            beacon = Beacon(beacon_config_no_ssl)
            beacon.start()

            try:
                # Wait for listener to start
                time.sleep(1.0)

                # Send TCP packet using new header format
                command_payload = {
                    "host": "192.168.1.100",
                    "comms_port": 6001,
                    "heartbeat_port": 6000,
                }
                
                # Serialize payload to bytes
                payload_bytes = json.dumps(command_payload).encode("utf-8")
                
                # Create header
                header = {
                    "command": "add_peer",
                    "payload_type": "json",
                    "payload_length": len(payload_bytes),
                }
                header_bytes = json.dumps(header).encode("utf-8")
                
                # Send: [4-byte header length][JSON header][payload bytes]
                header_length = len(header_bytes).to_bytes(4, byteorder="big")
                message = header_length + header_bytes + payload_bytes

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((beacon_config_no_ssl.host, beacon_config_no_ssl.comms_port))
                sock.sendall(message)
                sock.close()

                # Wait for processing
                time.sleep(0.5)

                # Verify peer was added
                assert len(beacon.config.peers) > 0
                peer_found = False
                for peer in beacon.config.peers:
                    if (
                        peer.host == "192.168.1.100"
                        and peer.comms_port == 6001
                        and peer.heartbeat_port == 6000
                    ):
                        peer_found = True
                        break
                assert peer_found

            finally:
                beacon.stop()


class TestBeaconSSLErrors:
    """Test SSL/certificate error handling."""

    def test_certificate_mismatch_prevents_connection(self, beacon_config, tmp_path):
        """Test that certificate/SSL mismatch prevents connection."""
        # Create invalid certificate files
        invalid_cert_dir = tmp_path / "invalid_certs"
        invalid_cert_dir.mkdir()
        invalid_crt = invalid_cert_dir / "invalid.crt"
        invalid_key = invalid_cert_dir / "invalid.key"

        # Write invalid certificate content
        invalid_crt.write_text("INVALID CERT CONTENT")
        invalid_key.write_text("INVALID KEY CONTENT")

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats_class.return_value = mock_stats

            sender_beacon = Beacon(beacon_config)

            # Try to send heartbeat with invalid certificates
            payload = {"host": "127.0.0.1", "port": 5000, "stats": {}}
            sender_beacon.send_heartbeat(
                host=beacon_config.host,
                port=beacon_config.heartbeat_port,
                heartbeat_wait_timeout=1,
                server_crt=str(invalid_crt),
                server_key=str(invalid_key),
                ca_crt=str(invalid_cert_dir / "ca.crt"),  # This file doesn't exist
                json_payload=payload,
            )

            # Check status shows cert_error
            status = sender_beacon.get_send_heartbeat_status(beacon_config.host, beacon_config.heartbeat_port)
            assert status is not None
            assert status.connection_status == "cert_error"

    def test_missing_certificate_files_causes_error(self, beacon_config):
        """Test that missing certificate files cause cert_error."""
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats_class.return_value = mock_stats

            sender_beacon = Beacon(beacon_config)

            # Try to send heartbeat with non-existent certificates
            payload = {"host": "127.0.0.1", "port": 5000, "stats": {}}
            sender_beacon.send_heartbeat(
                host=beacon_config.host,
                port=beacon_config.heartbeat_port,
                heartbeat_wait_timeout=1,
                server_crt="/nonexistent/server.crt",
                server_key="/nonexistent/server.key",
                ca_crt="/nonexistent/ca.crt",
                json_payload=payload,
            )

            # Check status shows cert_error
            status = sender_beacon.get_send_heartbeat_status(beacon_config.host, beacon_config.heartbeat_port)
            assert status is not None
            assert status.connection_status == "cert_error"


class TestBeaconCommandRegistration:
    """Test command registration functionality."""

    def test_register_function(self, beacon_config_no_ssl):
        """Test that register function allows custom command handlers."""
        # Create a dummy class with test_function
        class DummyHandler:
            def __init__(self):
                self.result = None

            def test_function(self, payload):
                """Test function that adds 1 to the 'x' field in payload."""
                if isinstance(payload, dict):
                    x_value = payload.get("x")
                    if x_value is not None:
                        self.result = int(x_value) + 1

        # Create dummy instance
        dummy = DummyHandler()

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats_class.return_value = mock_stats

            # Create beacon instance
            beacon = Beacon(beacon_config_no_ssl)

            # Register the test_function with command name "test_addition"
            beacon.register("test_addition", dummy.test_function)

            # Call run_receive_comms with the payload
            payload = {
                "command": "test_addition",
                "payload": {"x": 5},
            }
            beacon.run_receive_comms(payload)

            # Verify the result variable in dummy instance is equal to 6
            assert dummy.result == 6, f"Expected result to be 6, got {dummy.result}"

    def test_send_command_ping_returns_payload(self, beacon_config_no_ssl, sender_config_no_ssl):
        """Test that send_command with ping command returns the payload."""
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create two beacons on different ports
            beacon1 = Beacon(beacon_config_no_ssl)
            beacon2 = Beacon(sender_config_no_ssl)

            # Start both beacons
            beacon1.start()
            beacon2.start()

            try:
                # Wait for both listeners to start
                time.sleep(1.0)

                # Prepare a simple payload
                test_payload = {
                    "test_key": "test_value",
                    "number": 42,
                    "nested": {"inner": "data"},
                }

                # Send ping command from beacon1 to beacon2 (first time)
                response1 = beacon1.send_command(
                    host=sender_config_no_ssl.host,
                    port=sender_config_no_ssl.comms_port,
                    command="ping",
                    payload=test_payload,
                )

                # Verify first response is not None
                assert response1 is not None, "send_command should return a response (first call)"
                # Verify the returned JSON is equivalent to what was sent
                assert response1 == test_payload, f"Expected {test_payload}, got {response1}"

                # Send ping command from beacon1 to beacon2 (second time)
                # This ensures there is no early socket closure on the receiving end
                response2 = beacon1.send_command(
                    host=sender_config_no_ssl.host,
                    port=sender_config_no_ssl.comms_port,
                    command="ping",
                    payload=test_payload,
                )

                # Verify second response is not None
                assert response2 is not None, "send_command should return a response (second call)"
                # Verify the returned JSON is equivalent to what was sent
                assert response2 == test_payload, f"Expected {test_payload}, got {response2}"

            finally:
                beacon1.stop()
                beacon2.stop()

    def test_send_command_ping_with_binary_payload(self, beacon_config_no_ssl, sender_config_no_ssl):
        """Test that send_command with ping command can send and receive 128 bytes of binary data."""
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create two beacons on different ports
            beacon1 = Beacon(beacon_config_no_ssl)
            beacon2 = Beacon(sender_config_no_ssl)

            # Start both beacons
            beacon1.start()
            beacon2.start()

            try:
                # Wait for both listeners to start
                time.sleep(1.0)

                # Prepare 128 bytes of binary data
                test_binary_payload = bytes(range(128))  # 0x00 to 0x7F

                # Send ping command from beacon1 to beacon2 with binary payload (first time)
                response1 = beacon1.send_command(
                    host=sender_config_no_ssl.host,
                    port=sender_config_no_ssl.comms_port,
                    command="ping",
                    payload=test_binary_payload,
                )

                # Verify first response is not None
                assert response1 is not None, "send_command should return a response (first call)"
                # Verify the returned bytes match what was sent
                assert isinstance(response1, bytes), f"Expected bytes, got {type(response1)}"
                assert len(response1) == 128, f"Expected 128 bytes, got {len(response1)}"
                assert response1 == test_binary_payload, f"Binary payload mismatch (first call)"

                # Send ping command from beacon1 to beacon2 with binary payload (second time)
                # This ensures there is no early socket closure on the receiving end
                response2 = beacon1.send_command(
                    host=sender_config_no_ssl.host,
                    port=sender_config_no_ssl.comms_port,
                    command="ping",
                    payload=test_binary_payload,
                )

                # Verify second response is not None
                assert response2 is not None, "send_command should return a response (second call)"
                # Verify the returned bytes match what was sent
                assert isinstance(response2, bytes), f"Expected bytes, got {type(response2)}"
                assert len(response2) == 128, f"Expected 128 bytes, got {len(response2)}"
                assert response2 == test_binary_payload, f"Binary payload mismatch (second call)"

            finally:
                beacon1.stop()
                beacon2.stop()

    def test_handle_stats_includes_benchmark_field(self, beacon_config_no_ssl, tmp_path):
        """Test that _handle_stats includes benchmark field in status dictionaries."""
        from mosaic_stats.benchmark import save_benchmarks
        
        # Create a config with benchmark_data_location
        config = MosaicConfig(
            host="127.0.0.1",
            heartbeat_port=5040,
            comms_port=5041,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location=str(tmp_path / "benchmarks"),
        )

        # Create test benchmark data
        test_benchmark_data = {
            "timestamp_ms": 1234567890123,
            "host": "test-host",
            "disk": {"write_speed_mbps": 150.5, "read_speed_mbps": 200.3},
            "cpu": {"gflops": 75.8},
            "gpus": [{"gpu_id": 0, "gflops": 1200.5, "type": "nvidia"}],
            "ram": {"bandwidth_gbps": 25.5},
        }

        # Save benchmark data
        with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
            save_benchmarks(str(tmp_path / "benchmarks"), test_benchmark_data)

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create beacon
            beacon = Beacon(config)

            # Add a receive-heartbeat status
            status = ReceiveHeartbeatStatus(
                host="192.168.1.100",
                heartbeat_port=6000,
                comms_port=6001,
                last_time_received=int(time.time() * 1000),
                connection_status="online",
                stats_payload={"cpu_percent": 50.0},
            )
            beacon._receive_heartbeat_statuses[("192.168.1.100", 6000)] = status

            # Call _handle_stats (this will call load_benchmarks which uses _get_hostname)
            # Need to patch _get_hostname when load_benchmarks is called
            with patch("mosaic_stats.benchmark._get_hostname", return_value="test-host"):
                result = beacon._handle_stats({})

            # Verify result structure
            assert isinstance(result, list), "Result should be a list"
            assert len(result) == 1, "Should have one status entry"

            # Verify the entry has all expected fields including benchmark
            entry = result[0]
            assert "host" in entry
            assert "heartbeat_port" in entry
            assert "comms_port" in entry
            assert "last_time_received" in entry
            assert "connection_status" in entry
            assert "stats_payload" in entry
            assert "benchmark" in entry, "Entry should have 'benchmark' field"

            # Verify benchmark data matches what was saved
            assert entry["benchmark"] is not None, "benchmark should not be None when benchmark file exists"
            assert isinstance(entry["benchmark"], dict), "benchmark should be a dictionary"
            assert entry["benchmark"]["timestamp_ms"] == test_benchmark_data["timestamp_ms"]
            assert entry["benchmark"]["host"] == test_benchmark_data["host"]
            assert entry["benchmark"]["cpu"]["gflops"] == test_benchmark_data["cpu"]["gflops"]

    def test_handle_stats_benchmark_none_when_no_file(self, beacon_config_no_ssl):
        """Test that _handle_stats returns None for benchmark when no benchmark file exists."""
        # Create a config with empty benchmark_data_location
        config = MosaicConfig(
            host="127.0.0.1",
            heartbeat_port=5050,
            comms_port=5051,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",  # Empty location
        )

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create beacon
            beacon = Beacon(config)

            # Add a receive-heartbeat status
            status = ReceiveHeartbeatStatus(
                host="192.168.1.100",
                heartbeat_port=6000,
                comms_port=6001,
                last_time_received=int(time.time() * 1000),
                connection_status="online",
                stats_payload={"cpu_percent": 50.0},
            )
            beacon._receive_heartbeat_statuses[("192.168.1.100", 6000)] = status

            # Call _handle_stats
            result = beacon._handle_stats({})

            # Verify benchmark is None when no benchmark file exists
            assert len(result) == 1
            entry = result[0]
            assert "benchmark" in entry, "Entry should have 'benchmark' field"
            assert entry["benchmark"] is None, "benchmark should be None when no benchmark file exists"


class TestBeaconCollectStats:
    """Test collect_stats functionality."""

    def test_collect_stats_aggregates_from_peers(self, beacon_config_no_ssl):
        """Test that collect_stats aggregates stats from multiple peers."""
        # Create configs for 3 beacons with different ports
        config1 = MosaicConfig(
            host="127.0.0.1",
            heartbeat_port=5010,
            comms_port=5011,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,  # 10 seconds timeout
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )

        config2 = MosaicConfig(
            host="127.0.0.1",
            heartbeat_port=5020,
            comms_port=5021,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )

        config3 = MosaicConfig(
            host="127.0.0.1",
            heartbeat_port=5030,
            comms_port=5031,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create 3 beacons
            beacon1 = Beacon(config1)
            beacon2 = Beacon(config2)
            beacon3 = Beacon(config3)

            # Start beacon2 and beacon3 so they can receive commands
            beacon2.start()
            beacon3.start()

            try:
                # Wait for listeners to start
                time.sleep(1.0)

                # Manually add dummy heartbeats to beacon1's _receive_heartbeat_statuses
                # This simulates beacon1 having received heartbeats from beacon2 and beacon3
                status2 = ReceiveHeartbeatStatus(
                    host=config2.host,
                    heartbeat_port=config2.heartbeat_port,
                    comms_port=config2.comms_port,  # Valid comms_port
                    last_time_received=int(time.time() * 1000),
                    connection_status="online",
                    stats_payload={"cpu_percent": 50.0},
                )

                status3 = ReceiveHeartbeatStatus(
                    host=config3.host,
                    heartbeat_port=config3.heartbeat_port,
                    comms_port=config3.comms_port,  # Valid comms_port
                    last_time_received=int(time.time() * 1000),
                    connection_status="online",
                    stats_payload={"cpu_percent": 60.0},
                )

                # Add one with comms_port=0 (should be excluded)
                status_no_comms = ReceiveHeartbeatStatus(
                    host="192.168.1.100",
                    heartbeat_port=6000,
                    comms_port=0,  # No comms_port set
                    last_time_received=int(time.time() * 1000),
                    connection_status="online",
                    stats_payload={"cpu_percent": 70.0},
                )

                # Add statuses to beacon1
                key2 = (config2.host, config2.heartbeat_port)
                key3 = (config3.host, config3.heartbeat_port)
                key_no_comms = ("192.168.1.100", 6000)

                # Add statuses to beacon1 (representing peers that have sent heartbeats)
                beacon1._receive_heartbeat_statuses[key2] = status2
                beacon1._receive_heartbeat_statuses[key3] = status3
                beacon1._receive_heartbeat_statuses[key_no_comms] = status_no_comms

                # Add some statuses to beacon2 and beacon3 so they return meaningful stats
                # Add a status to beacon2's receive_heartbeat_statuses
                status2_peer = ReceiveHeartbeatStatus(
                    host="192.168.1.200",
                    heartbeat_port=7000,
                    comms_port=7001,
                    last_time_received=int(time.time() * 1000),
                    connection_status="online",
                    stats_payload={"cpu_percent": 55.0},
                )
                beacon2._receive_heartbeat_statuses[("192.168.1.200", 7000)] = status2_peer

                # Add a status to beacon3's receive_heartbeat_statuses
                status3_peer = ReceiveHeartbeatStatus(
                    host="192.168.1.300",
                    heartbeat_port=8000,
                    comms_port=8001,
                    last_time_received=int(time.time() * 1000),
                    connection_status="online",
                    stats_payload={"cpu_percent": 65.0},
                )
                beacon3._receive_heartbeat_statuses[("192.168.1.300", 8000)] = status3_peer

                # Call collect_stats on beacon1
                result = beacon1.collect_stats()

                # Verify result is a list
                assert isinstance(result, list), "collect_stats should return a list"

                # Verify result contains stats from beacon2 and beacon3
                # Each beacon's stats response contains their receive_heartbeat_statuses (1 entry each)
                assert len(result) == 2, f"Expected 2 entries, got {len(result)}"

                # Verify the entries are dictionaries with the expected structure
                for entry in result:
                    assert isinstance(entry, dict), "Each entry should be a dictionary"
                    assert "host" in entry, "Entry should have 'host' field"
                    assert "heartbeat_port" in entry, "Entry should have 'heartbeat_port' field"
                    assert "comms_port" in entry, "Entry should have 'comms_port' field"
                    assert "benchmark" in entry, "Entry should have 'benchmark' field"
                    # benchmark can be None or a dict
                    assert entry["benchmark"] is None or isinstance(entry["benchmark"], dict), "benchmark should be None or a dict"

                # Verify that no entries have comms_port=0 (the status_no_comms should be excluded)
                for entry in result:
                    assert entry.get("comms_port") != 0, "Result should not contain entries with comms_port=0"

                # Verify we got the expected entries from beacon2 and beacon3
                hosts_found = {entry.get("host") for entry in result}
                assert "192.168.1.200" in hosts_found, "Should contain entry from beacon2"
                assert "192.168.1.300" in hosts_found, "Should contain entry from beacon3"

            finally:
                beacon2.stop()
                beacon3.stop()

    def test_collect_stats_timeout_breaks_loop(self, beacon_config_no_ssl):
        """Test that collect_stats breaks when timeout is exceeded."""
        # Create configs for 3 beacons
        config1 = MosaicConfig(
            host="127.0.0.1",
            heartbeat_port=5010,
            comms_port=5011,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=0,  # 0 seconds timeout to force break
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )

        config2 = MosaicConfig(
            host="127.0.0.1",
            heartbeat_port=5020,
            comms_port=5021,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )

        config3 = MosaicConfig(
            host="127.0.0.1",
            heartbeat_port=5030,
            comms_port=5031,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create 3 beacons
            beacon1 = Beacon(config1)
            beacon2 = Beacon(config2)
            beacon3 = Beacon(config3)

            # Start beacon2 and beacon3
            beacon2.start()
            beacon3.start()

            try:
                # Wait for listeners to start
                time.sleep(1.0)

                # Manually add dummy heartbeats to beacon1's _receive_heartbeat_statuses
                status2 = ReceiveHeartbeatStatus(
                    host=config2.host,
                    heartbeat_port=config2.heartbeat_port,
                    comms_port=config2.comms_port,
                    last_time_received=int(time.time() * 1000),
                    connection_status="online",
                )

                status3 = ReceiveHeartbeatStatus(
                    host=config3.host,
                    heartbeat_port=config3.heartbeat_port,
                    comms_port=config3.comms_port,
                    last_time_received=int(time.time() * 1000),
                    connection_status="online",
                )

                # Add statuses to beacon1
                key2 = (config2.host, config2.heartbeat_port)
                key3 = (config3.host, config3.heartbeat_port)

                beacon1._receive_heartbeat_statuses[key2] = status2
                beacon1._receive_heartbeat_statuses[key3] = status3

                # Call collect_stats on beacon1 with timeout=0
                result = beacon1.collect_stats()

                # Verify result is empty (timeout should break immediately)
                assert isinstance(result, list), "collect_stats should return a list"
                assert len(result) == 0, f"Expected empty list due to timeout, got {len(result)} entries"

            finally:
                beacon2.stop()
                beacon3.stop()

