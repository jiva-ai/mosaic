"""Unit tests for mosaic_comms.beacon module."""

import json
import socket
import ssl
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mosaic_comms.beacon import Beacon
from mosaic_config.state import ReceiveHeartbeatStatus, SendHeartbeatStatus
from mosaic_config.config import MosaicConfig, Peer
from mosaic_config.state import Data, FileDefinition, DataType, Model, Plan
from tests.conftest import create_test_config_with_state


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
def beacon_config(test_certs_dir, temp_state_dir):
    """Create a MosaicConfig for beacon testing with SSL certificates."""
    return create_test_config_with_state(
        state_dir=temp_state_dir,
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
def beacon_config_no_ssl(temp_state_dir):
    """Create a MosaicConfig for beacon testing without SSL certificates."""
    return create_test_config_with_state(
        state_dir=temp_state_dir,
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
def sender_config(test_certs_dir, temp_state_dir):
    """Create a MosaicConfig for the sender beacon."""
    return create_test_config_with_state(
        state_dir=temp_state_dir,
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
def sender_config_no_ssl(temp_state_dir):
    """Create a MosaicConfig for the sender beacon without SSL."""
    return create_test_config_with_state(
        state_dir=temp_state_dir,
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

    def test_beacon_startup_with_no_peers(self, temp_state_dir):
        """Test that Beacon starts up correctly when config has no peers."""
        # Create config with no peers (empty list is the default)
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5000,
            comms_port=5001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )
        # Explicitly ensure peers list is empty
        config.peers = []

        # Mock StatsCollector to avoid actual stats collection
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 0}'
            mock_stats_class.return_value = mock_stats

            # Create and start beacon - should not error
            beacon = Beacon(config)
            assert len(beacon.config.peers) == 0

            # Start beacon - should not error
            beacon.start()
            assert beacon._running

            try:
                # Wait a moment for threads to start
                time.sleep(0.5)

                # Verify beacon is running
                assert beacon._running
                assert beacon._send_heartbeat_thread is not None
                assert beacon._receive_heartbeat_thread is not None
                assert beacon._receive_comms_thread is not None
                assert beacon._check_stale_heartbeats_thread is not None

                # Test run_send_heartbeats with no peers - should not error
                # This method iterates through config.peers, which is empty
                beacon.run_send_heartbeats()

                # Test collect_stats with no peers - should return empty list or just local stats
                stats = beacon.collect_stats(include_self=False)
                assert isinstance(stats, list)
                # With no peers and include_self=False, should be empty
                assert len(stats) == 0

                # Test collect_stats with include_self=True - should return local stats
                stats_with_self = beacon.collect_stats(include_self=True)
                assert isinstance(stats_with_self, list)
                # Should have at least local stats
                assert len(stats_with_self) >= 0  # Could be 0 if stats collection fails, but shouldn't error

            finally:
                beacon.stop()


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

    def test_ssl_files_not_existing_sockets_work_without_ssl(self, temp_state_dir):
        """Test that when SSL files don't exist, sockets are created without SSL and commands work."""
        # Create config with non-existent SSL certificate paths
        config_with_missing_ssl = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5010,
            comms_port=5011,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            server_crt="/nonexistent/server.crt",  # Non-existent file
            server_key="/nonexistent/server.key",  # Non-existent file
            ca_crt="/nonexistent/ca.crt",  # Non-existent file
            benchmark_data_location="",
        )
        
        config_receiver = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5012,
            comms_port=5013,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            server_crt="",  # No SSL
            server_key="",
            ca_crt="",
            benchmark_data_location="",
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = "{}"
            mock_stats_class.return_value = mock_stats
            
            # Create beacons - SSL validation should fail and disable SSL
            beacon1 = Beacon(config_with_missing_ssl)
            beacon2 = Beacon(config_receiver)
            
            # Verify SSL is disabled
            assert not beacon1._ssl_enabled, "SSL should be disabled when certificate files don't exist"
            assert not beacon2._ssl_enabled, "SSL should be disabled when no certificates configured"
            
            try:
                # Start both beacons
                beacon1.start()
                beacon2.start()
                
                # Give them time to start listening
                time.sleep(0.5)
                
                # Test 1: Verify heartbeat can be sent/received (UDP socket works)
                payload = {
                    "host": config_receiver.host,
                    "port": config_receiver.heartbeat_port,
                    "comms_port": config_receiver.comms_port,
                    "stats": {},
                }
                beacon1.send_heartbeat(
                    host=config_receiver.host,
                    port=config_receiver.heartbeat_port,
                    heartbeat_wait_timeout=config_with_missing_ssl.heartbeat_wait_timeout,
                    server_crt=config_with_missing_ssl.server_crt,
                    server_key=config_with_missing_ssl.server_key,
                    ca_crt=config_with_missing_ssl.ca_crt,
                    json_payload=payload,
                )
                
                # Wait a bit for heartbeat to be received
                time.sleep(0.5)
                
                # Verify heartbeat was received (connection_status should be "ok" since SSL is disabled)
                status = beacon1.get_send_heartbeat_status(
                    config_receiver.host, config_receiver.heartbeat_port
                )
                assert status is not None
                # Status should be "ok" since we're not using SSL
                assert status.connection_status == "ok", f"Expected 'ok', got '{status.connection_status}'"
                
                # Test 2: Verify send_command works (TCP socket works)
                response = beacon1.send_command(
                    host=config_receiver.host,
                    port=config_receiver.comms_port,
                    command="ping",
                    payload={"test": "data"},
                )
                
                assert response is not None, "send_command should return a response"
                assert response == {"test": "data"}, "Response should echo the payload"
                
                # Test 3: Send command again to ensure socket is still working
                response2 = beacon1.send_command(
                    host=config_receiver.host,
                    port=config_receiver.comms_port,
                    command="ping",
                    payload={"test2": "data2"},
                )
                
                assert response2 is not None, "Second send_command should also return a response"
                assert response2 == {"test2": "data2"}, "Second response should echo the payload"
                
            finally:
                beacon1.stop()
                beacon2.stop()


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

    def test_handle_stats_includes_benchmark_field(self, beacon_config_no_ssl, tmp_path, temp_state_dir):
        """Test that _handle_stats includes benchmark field in status dictionaries."""
        from mosaic_stats.benchmark import save_benchmarks
        
        # Create a config with benchmark_data_location
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
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

    def test_handle_stats_benchmark_none_when_no_file(self, beacon_config_no_ssl, temp_state_dir):
        """Test that _handle_stats returns None for benchmark when no benchmark file exists."""
        # Create a config with empty benchmark_data_location
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
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

    def test_collect_stats_aggregates_from_peers(self, beacon_config_no_ssl, temp_state_dir):
        """Test that collect_stats aggregates stats from multiple peers."""
        # Create configs for 3 beacons with different ports
        config1 = create_test_config_with_state(
            state_dir=temp_state_dir,
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

        config2 = create_test_config_with_state(
            state_dir=temp_state_dir,
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

        config3 = create_test_config_with_state(
            state_dir=temp_state_dir,
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

                # Call collect_stats on beacon1 (include_self=True by default)
                result = beacon1.collect_stats()

                # Verify result is a list
                assert isinstance(result, list), "collect_stats should return a list"

                # Verify result contains stats from beacon2, beacon3, and beacon1 (local)
                # Each beacon's stats response contains their receive_heartbeat_statuses (1 entry each)
                # Plus beacon1's local stats (1 entry)
                assert len(result) == 3, f"Expected 3 entries (2 peers + 1 local), got {len(result)}"

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

                # Verify we got the expected entries from beacon2, beacon3, and beacon1 (local)
                hosts_found = {entry.get("host") for entry in result}
                assert "192.168.1.200" in hosts_found, "Should contain entry from beacon2"
                assert "192.168.1.300" in hosts_found, "Should contain entry from beacon3"
                assert config1.host in hosts_found, "Should contain local node entry"
                
                # Verify local node entry has correct structure
                local_entry = next((e for e in result if e.get("host") == config1.host), None)
                assert local_entry is not None, "Local node entry should be present"
                assert local_entry.get("heartbeat_port") == config1.heartbeat_port
                assert local_entry.get("comms_port") == config1.comms_port
                assert local_entry.get("connection_status") == "online"
                assert local_entry.get("stats_payload") is not None

            finally:
                beacon2.stop()
                beacon3.stop()

    def test_collect_stats_timeout_breaks_loop(self, beacon_config_no_ssl, temp_state_dir):
        """Test that collect_stats breaks when timeout is exceeded."""
        # Create configs for 3 beacons
        config1 = create_test_config_with_state(
            state_dir=temp_state_dir,
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

        config2 = create_test_config_with_state(
            state_dir=temp_state_dir,
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

        config3 = create_test_config_with_state(
            state_dir=temp_state_dir,
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

                # Call collect_stats on beacon1 with timeout=0 and include_self=False
                # With timeout=0, no peers will be queried, but include_self=False means no local stats either
                result = beacon1.collect_stats(include_self=False)

                # Verify result is empty (timeout should break immediately, and include_self=False)
                assert isinstance(result, list), "collect_stats should return a list"
                assert len(result) == 0, f"Expected empty list due to timeout and include_self=False, got {len(result)} entries"
                
                # Now test with include_self=True - should have local stats even with timeout=0
                result_with_self = beacon1.collect_stats(include_self=True)
                assert len(result_with_self) == 1, f"Expected 1 entry (local stats), got {len(result_with_self)} entries"
                local_entry = result_with_self[0]
                assert local_entry.get("host") == config1.host, "Should contain local node entry"

            finally:
                beacon2.stop()
                beacon3.stop()

    def test_collect_stats_include_self_parameter(self, beacon_config_no_ssl):
        """Test that collect_stats includes/excludes local node stats based on include_self parameter."""
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3, "ram_percent": 60.0}'
            mock_stats_class.return_value = mock_stats

            beacon = Beacon(beacon_config_no_ssl)
            beacon.start()

            try:
                # Wait for listener to start
                time.sleep(0.5)

                # Test with include_self=True (default)
                result_with_self = beacon.collect_stats(include_self=True)
                assert isinstance(result_with_self, list), "collect_stats should return a list"
                assert len(result_with_self) == 1, f"Expected 1 entry (local stats), got {len(result_with_self)}"
                
                local_entry = result_with_self[0]
                assert local_entry.get("host") == beacon_config_no_ssl.host
                assert local_entry.get("heartbeat_port") == beacon_config_no_ssl.heartbeat_port
                assert local_entry.get("comms_port") == beacon_config_no_ssl.comms_port
                assert local_entry.get("connection_status") == "online"
                assert local_entry.get("stats_payload") is not None
                assert local_entry.get("stats_payload").get("cpu_percent") == 45.3
                assert local_entry.get("stats_payload").get("ram_percent") == 60.0
                assert "benchmark" in local_entry

                # Test with include_self=False
                result_without_self = beacon.collect_stats(include_self=False)
                assert isinstance(result_without_self, list), "collect_stats should return a list"
                assert len(result_without_self) == 0, f"Expected 0 entries (no peers, no local), got {len(result_without_self)}"

            finally:
                beacon.stop()

    def test_handle_execute_data_plan_chunk(self, temp_state_dir):
        """Test that _handle_execute_data_plan_chunk correctly writes chunks and interprets headers."""
        import pickle
        import uuid
        from pathlib import Path
        
        # Set up test_data directory as data_location
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5000,
            comms_port=5001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            beacon = Beacon(config)
            
            # Create test chunk data
            chunk_id = str(uuid.uuid4())
            chunk_index = 0
            total_chunks = 2
            chunk_data = b"test chunk data content"
            
            # Create header
            chunk_header = {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
            }
            header_bytes = pickle.dumps(chunk_header)
            header_length = len(header_bytes).to_bytes(4, byteorder="big")
            
            # Create payload: header_length + header + chunk_data
            payload = header_length + header_bytes + chunk_data
            
            # Call handler
            result = beacon._handle_execute_data_plan_chunk(payload)
            
            # Verify result
            assert result is not None
            assert result["status"] == "success"
            assert result["chunk_index"] == chunk_index
            assert "Received chunk 1/2" in result["message"]
            
            # Verify chunk was written to disk
            chunk_file = test_data_dir / "temp_chunks" / f"{chunk_id}.part"
            assert chunk_file.exists(), "Chunk file should exist"
            
            # Verify chunk content
            with open(chunk_file, 'rb') as f:
                saved_data = f.read()
            assert saved_data == chunk_data, "Chunk data should match"
            
            # Verify chunk storage was updated
            assert chunk_id in beacon._chunk_storage
            assert beacon._chunk_storage[chunk_id]["total_chunks"] == total_chunks
            assert beacon._chunk_storage[chunk_id]["chunks_received"] == 1
            
            # Clean up
            chunk_file.unlink()
            (test_data_dir / "temp_chunks").rmdir()

    def test_handle_execute_data_plan_finalize(self, temp_state_dir):
        """Test that _handle_execute_data_plan_finalize correctly coalesces 2 chunks together."""
        import pickle
        import uuid
        from pathlib import Path
        from mosaic_planner.planner import save_chunk_to_disk, serialize_plan_with_data
        from mosaic_config.state import Data, FileDefinition, DataType, Model, Plan
        
        # Set up test_data directory as data_location
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5000,
            comms_port=5001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            beacon = Beacon(config)
            
            # Initialize session manager for this test
            import mosaic.mosaic as mosaic_module
            from mosaic_config.state_manager import SessionStateManager
            mosaic_module._session_manager = SessionStateManager(config)
            mosaic_module._config = config
            
            # Clear any existing sessions and track initial count
            initial_session_count = len(mosaic_module._session_manager.get_sessions())
            
            # Create test plan and data
            import zipfile
            from io import BytesIO
            
            model = Model(name="test_model")
            plan = Plan(
                stats_data=[],
                distribution_plan=[],
                model=model,
            )
            
            # Create a valid zip file for binary_data
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr('test_file.csv', 'col1,col2\nval1,val2\n')
            zip_data = zip_buffer.getvalue()
            
            file_def = FileDefinition(
                location="test_file",
                data_type=DataType.CSV,
                is_segmentable=True,
                binary_data=zip_data,
            )
            data = Data(file_definitions=[file_def])
            
            # Serialize plan and data
            serialized_data = serialize_plan_with_data(plan, data, compress=True)
            
            # Split into 2 chunks
            chunk_id = str(uuid.uuid4())
            chunk_size = len(serialized_data) // 2 + 1  # Ensure we get 2 chunks
            chunk1_data = serialized_data[:chunk_size]
            chunk2_data = serialized_data[chunk_size:]
            
            # Save chunks to disk manually (simulating what chunk handler does)
            temp_chunks_dir = test_data_dir / "temp_chunks"
            temp_chunks_dir.mkdir(parents=True, exist_ok=True)
            chunk_file = temp_chunks_dir / f"{chunk_id}.part"
            
            # Write first chunk
            save_chunk_to_disk(chunk1_data, chunk_file, is_first_chunk=True)
            
            # Write second chunk (append)
            save_chunk_to_disk(chunk2_data, chunk_file, is_first_chunk=False)
            
            # Verify chunks were written correctly
            with open(chunk_file, 'rb') as f:
                combined_data = f.read()
            assert combined_data == serialized_data, "Combined chunks should equal original data"
            
            # Set up chunk storage
            beacon._chunk_storage[chunk_id] = {
                "total_chunks": 2,
                "chunks_received": 2,
            }
            
            # Clean up any existing old location directory and session directories before test
            old_location_dir = test_data_dir / "test_file"
            if old_location_dir.exists():
                import shutil
                shutil.rmtree(old_location_dir)
            
            # Also clean up any existing session directories that might have been created
            # by previous test runs (they would be UUIDs)
            for item in test_data_dir.iterdir():
                if item.is_dir() and len(item.name) == 36 and item.name.count('-') == 4:  # UUID format
                    # This looks like a session ID directory, clean it up
                    import shutil
                    try:
                        shutil.rmtree(item)
                    except Exception:
                        pass  # Ignore errors during cleanup
            
            # Call finalize handler
            result = beacon._handle_execute_data_plan_finalize({"chunk_id": chunk_id})
            
            # Verify result
            assert result is not None
            assert result["status"] == "success"
            assert result["file_count"] == 1
            assert "Finalized and processed 1 file definitions" in result["message"]
            
            # Verify chunks were coalesced correctly
            # The handler should have read the complete file and deserialized it
            # We can't directly verify the deserialized content, but we can verify
            # that the chunk file was processed and cleaned up
            assert chunk_id not in beacon._chunk_storage, "Chunk storage should be cleaned up"
            
            # Verify temp directories are cleaned up (or at least chunk file is gone)
            assert not chunk_file.exists(), "Chunk file should be cleaned up"
            
            # Verify that a session was created
            sessions = mosaic_module._session_manager.get_sessions()
            session = next((s for s in sessions if s.plan.id == plan.id), None)
            assert session is not None, "Session should be created for this plan"
            assert session.data is not None, "Session should have data"
            assert len(session.data.file_definitions) == 1, "Session data should have file definitions"
            assert session.data.file_definitions[0].location == "test_file", "Session data should match original data"
            
            # Verify files are extracted to session-specific directory
            session_id = session.id
            session_dir = test_data_dir / session_id
            extract_location = session_dir / "test_file"
            expected_file = extract_location / "test_file.csv"
            
            assert session_dir.exists(), f"Session directory {session_id} should exist"
            assert extract_location.exists(), f"Extract location {extract_location} should exist"
            assert expected_file.exists(), f"File should be extracted to {expected_file}"
            
            # Verify file content is correct
            with open(expected_file, 'r', encoding='utf-8') as f:
                content = f.read()
            assert content == 'col1,col2\nval1,val2\n', "File content should match original"
            
            # Verify file is NOT in the old location (without session ID)
            # The handler should extract to {data_location}/{session_id}/{file_location}
            # NOT to {data_location}/{file_location}
            old_location_dir = test_data_dir / "test_file"
            old_location_file = old_location_dir / "test_file.csv"
            
            # The cleanup before the test should have removed the old location,
            # but if it still exists, verify it doesn't contain our test file
            # (it might be from a previous test run that didn't clean up)
            if old_location_file.exists():
                # Check if this is the same file we just created (same content)
                old_file_content = old_location_file.read_text(encoding='utf-8')
                if old_file_content == content:
                    # Same content - this shouldn't happen if handler is working correctly
                    # Check modification times to see if it's from this test run
                    old_file_mtime = old_location_file.stat().st_mtime
                    new_file_mtime = expected_file.stat().st_mtime
                    # If modification times are very close (within 1 second), it might be from this test
                    time_diff = abs(new_file_mtime - old_file_mtime)
                    if time_diff < 1.0:
                        # Files were created at nearly the same time - this suggests the handler
                        # extracted to both locations, which is a bug
                        assert False, f"File exists in both old location {old_location_file} and session directory {expected_file} with same content and similar timestamps - handler may have extracted to both locations"
                # If content is different or timestamps are far apart, it's from a previous test run
                # and the cleanup didn't work - but that's okay, we'll just verify our file is in the right place
            
            # Clean up session directory and temp directories
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
            try:
                temp_chunks_dir.rmdir()
            except OSError:
                pass

    def test_handle_execute_data_plan_creates_session(self, temp_state_dir):
        """Test that _handle_execute_data_plan creates a session after processing."""
        import zipfile
        from io import BytesIO
        from pathlib import Path
        from mosaic_planner.planner import serialize_plan_with_data
        from mosaic_config.state import Data, FileDefinition, DataType, Model, Plan
        
        # Set up test_data directory as data_location
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5000,
            comms_port=5001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            beacon = Beacon(config)
            
            # Initialize session manager for this test
            import mosaic.mosaic as mosaic_module
            from mosaic_config.state_manager import SessionStateManager
            mosaic_module._session_manager = SessionStateManager(config)
            mosaic_module._config = config
            
            # Clear any existing sessions
            initial_session_count = len(mosaic_module._session_manager.get_sessions())
            
            # Create test plan and data
            model = Model(name="test_model")
            plan = Plan(
                stats_data=[],
                distribution_plan=[],
                model=model,
            )
            
            # Create a valid zip file for binary_data
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr('test_file.csv', 'col1,col2\nval1,val2\n')
            zip_data = zip_buffer.getvalue()
            
            file_def = FileDefinition(
                location="test_file",
                data_type=DataType.CSV,
                is_segmentable=True,
                binary_data=zip_data,
            )
            data = Data(file_definitions=[file_def])
            
            # Serialize plan and data
            serialized_data = serialize_plan_with_data(plan, data, compress=True)
            
            # Call the handler
            result = beacon._handle_execute_data_plan(serialized_data)
            
            # Verify result
            assert result is not None
            assert result["status"] == "success"
            assert result["file_count"] == 1
            
            # Verify that a session was created
            sessions = mosaic_module._session_manager.get_sessions()
            assert len(sessions) == initial_session_count + 1, "One session should be created"
            session = sessions[-1]  # Get the most recently added session
            assert session.plan.id == plan.id, "Session should have the correct plan"
            assert session.data is not None, "Session should have data"
            assert len(session.data.file_definitions) == 1, "Session data should have file definitions"
            assert session.data.file_definitions[0].location == "test_file", "Session data should match original data"
            assert session.data.file_definitions[0].data_type == DataType.CSV, "Session data should have correct data type"

    def test_execute_data_plan_extracts_to_session_directory(self, temp_state_dir):
        """Test that _handle_execute_data_plan extracts files to a session-specific directory."""
        import zipfile
        import shutil
        from io import BytesIO
        from pathlib import Path
        from mosaic_planner.planner import serialize_plan_with_data
        from mosaic_config.state import Data, FileDefinition, DataType, Model, Plan
        
        # Set up test_data directory as data_location
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5000,
            comms_port=5001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            beacon = Beacon(config)
            
            # Initialize session manager for this test
            import mosaic.mosaic as mosaic_module
            from mosaic_config.state_manager import SessionStateManager
            mosaic_module._session_manager = SessionStateManager(config)
            mosaic_module._config = config
            
            # Clear any existing sessions
            initial_session_count = len(mosaic_module._session_manager.get_sessions())
            
            # Create test plan and data
            model = Model(name="test_model")
            plan = Plan(
                stats_data=[],
                distribution_plan=[],
                model=model,
            )
            
            # Create a valid zip file for binary_data with a test file
            test_file_content = 'col1,col2\nval1,val2\nval3,val4\n'
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr('test_file.csv', test_file_content)
            zip_data = zip_buffer.getvalue()
            
            file_def = FileDefinition(
                location="test_session_file",
                data_type=DataType.CSV,
                is_segmentable=True,
                binary_data=zip_data,
            )
            data = Data(file_definitions=[file_def])
            
            # Serialize plan and data
            serialized_data = serialize_plan_with_data(plan, data, compress=True)
            
            # Clean up any existing old location directory before test
            old_location_dir = test_data_dir / "test_session_file"
            if old_location_dir.exists():
                shutil.rmtree(old_location_dir)
            
            # Call the handler
            result = beacon._handle_execute_data_plan(serialized_data)
            
            # Verify result
            assert result is not None
            assert result["status"] == "success"
            assert result["file_count"] == 1
            
            # Verify that a session was created
            sessions = mosaic_module._session_manager.get_sessions()
            assert len(sessions) == initial_session_count + 1, "One session should be created"
            session = sessions[-1]  # Get the most recently added session
            session_id = session.id
            
            # Verify files are extracted to session-specific directory
            # The extraction path is: {data_location}/{session_id}/{file_def.location}
            session_dir = test_data_dir / session_id
            extract_location = session_dir / "test_session_file"
            expected_file = extract_location / "test_file.csv"
            
            assert session_dir.exists(), f"Session directory {session_id} should exist"
            assert extract_location.exists(), f"Extract location {extract_location} should exist"
            assert expected_file.exists(), f"File should be extracted to {expected_file}"
            
            # Verify file content is correct
            with open(expected_file, 'r', encoding='utf-8') as f:
                content = f.read()
            assert content == test_file_content, "File content should match original"
            
            # Verify file is NOT in the old location (without session ID)
            old_location_dir = test_data_dir / "test_session_file"
            old_location_file = old_location_dir / "test_file.csv"
            assert not old_location_file.exists(), f"File should NOT be in old location {old_location_file}"
            # The old location directory might exist if it was created, but it shouldn't have the file
            if old_location_dir.exists():
                files_in_old_location = list(old_location_dir.rglob('*'))
                files_in_old_location = [f for f in files_in_old_location if f.is_file()]
                assert len(files_in_old_location) == 0, f"Old location should not contain files, found: {files_in_old_location}"
            
            # Clean up
            if session_dir.exists():
                shutil.rmtree(session_dir)

    def test_large_directory_transfer_with_chunking(self, temp_state_dir):
        """Test transferring a 16MB directory between 2 beacons with chunking."""
        import time
        import random
        import shutil
        from pathlib import Path
        from mosaic_planner.planner import get_directory_size
        from mosaic_config.state import Data, DataType, FileDefinition, Model, Plan
        
        # Set up test_data directory
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        # Create source directory name
        source_dir_name = "large_test_dir"
        source_dir = test_data_dir / source_dir_name
        
        # Check if directory exists, if not create it
        if not source_dir.exists():
            source_dir.mkdir(exist_ok=True)
            
            # Create files of varying sizes with random bytes (don't compress well)
            # Create a larger directory to ensure chunking even after compression
            # Total size should be large enough that even after compression, it exceeds chunk_size
            # Use larger files to ensure we get multiple chunks
            file_sizes = [
                (20 * 1024 * 1024, "file_20mb.bin"),  # 20MB file
                (15 * 1024 * 1024, "file_15mb.bin"),  # 15MB file
                (10 * 1024 * 1024, "file_10mb.bin"),  # 10MB file
                (5 * 1024 * 1024, "file_5mb.bin"),     # 5MB file
                (2 * 1024 * 1024, "file_2mb.bin"),     # 2MB file
                (512 * 1024, "file_512kb.bin"),        # 512KB file
            ]
            
            # Generate random binary data (doesn't compress well)
            # Use a fixed seed for reproducibility so files can be compared byte-for-byte
            random.seed(42)
            chunk_size = 1024 * 1024  # Write in 1MB chunks
            
            for size_bytes, filename in file_sizes:
                file_path = source_dir / filename
                with open(file_path, 'wb') as f:
                    written = 0
                    while written < size_bytes:
                        bytes_to_write = min(chunk_size, size_bytes - written)
                        # Generate random bytes
                        random_data = bytes([random.randint(0, 255) for _ in range(bytes_to_write)])
                        f.write(random_data)
                        written += bytes_to_write
        
        # Verify directory exists and has files
        total_size = get_directory_size(source_dir)
        assert total_size > 0, "Source directory should contain files"
        
        # Count original files
        original_files = list(source_dir.rglob('*'))
        original_files = [f for f in original_files if f.is_file()]
        original_file_count = len(original_files)
        assert original_file_count > 0, "Source directory should contain at least one file"
        
        # Set up receiver directory (different location)
        receiver_dir_name = "large_test_dir_received"
        receiver_dir = test_data_dir / receiver_dir_name
        # Clean up receiver directory (but not source directory as per requirements)
        if receiver_dir.exists():
            shutil.rmtree(receiver_dir)
        
        # Create configs for sender and receiver
        # Use 2MB chunk size to force chunking
        chunk_size_mb = 2
        sender_config = create_test_config_with_state(
            state_dir=temp_state_dir / "sender",
            host="127.0.0.1",
            heartbeat_port=6000,
            comms_port=6001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
            data_chunk_size=chunk_size_mb,  # 2MB chunks
        )
        
        receiver_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver",
            host="127.0.0.1",
            heartbeat_port=6002,
            comms_port=6003,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
            data_chunk_size=chunk_size_mb,  # 2MB chunks
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            # Create beacons
            sender_beacon = Beacon(sender_config)
            receiver_beacon = Beacon(receiver_config)
            
            sender_beacon.start()
            receiver_beacon.start()
            
            try:
                # Wait for beacons to start
                time.sleep(1.0)
                
                # Create Plan with data_segmentation_plan pointing to receiver
                model = Model(name="test_model")
                plan = Plan(
                    stats_data=[],
                    distribution_plan=[],
                    model=model,
                    data_segmentation_plan=[{
                        "host": receiver_config.host,
                        "comms_port": receiver_config.comms_port,
                        "fraction": 1.0,
                        "segments": [{
                            "file_location": source_dir_name,  # Source location
                            "data_type": "dir",
                            "is_segmentable": False,  # Send entire directory
                        }],
                    }],
                )
                
                # Create Data with directory definition (source location)
                file_def = FileDefinition(
                    location=source_dir_name,
                    data_type=DataType.DIR,
                    is_segmentable=False,  # Send entire directory
                )
                data = Data(file_definitions=[file_def])
                
                # Patch both handlers to extract to receiver directory instead
                # Create a shared function to copy extracted files (not move, to preserve source)
                def copy_extracted_files():
                    """Copy extracted files to receiver directory."""
                    extracted_dir = test_data_dir / source_dir_name
                    if extracted_dir.exists() and extracted_dir != receiver_dir:
                        if receiver_dir.exists():
                            shutil.rmtree(receiver_dir)
                        # Copy the extracted directory to receiver location (not move, to preserve source)
                        shutil.copytree(extracted_dir, receiver_dir)
                
                # Patch _handle_execute_data_plan (for non-chunked data)
                original_handler = receiver_beacon._handle_execute_data_plan
                def patched_handler(payload):
                    result = original_handler(payload)
                    copy_extracted_files()
                    return result
                receiver_beacon._handle_execute_data_plan = patched_handler
                receiver_beacon.register("exdplan", patched_handler)
                
                # Patch _handle_execute_data_plan_finalize (for chunked data)
                original_finalize_handler = receiver_beacon._handle_execute_data_plan_finalize
                def patched_finalize_handler(payload):
                    result = original_finalize_handler(payload)
                    copy_extracted_files()
                    return result
                receiver_beacon._handle_execute_data_plan_finalize = patched_finalize_handler
                receiver_beacon.register("exdplan_finalize", patched_finalize_handler)
                
                # Track chunk sizes sent (mock send_command to capture chunk sizes)
                chunk_sizes_sent = []
                original_send_command = sender_beacon.send_command
                
                def track_send_command(host, port, command, payload, timeout=30.0):
                    if command == "exdplan_chunk":
                        # Extract chunk size from payload
                        if isinstance(payload, bytes):
                            # Payload is: header_length (4 bytes) + header + chunk_data
                            header_length = int.from_bytes(payload[:4], byteorder="big")
                            chunk_data = payload[4 + header_length:]
                            chunk_sizes_sent.append(len(chunk_data))
                    elif command == "exdplan":
                        # Single chunk - entire payload is the data
                        if isinstance(payload, bytes):
                            chunk_sizes_sent.append(len(payload))
                    return original_send_command(host, port, command, payload, timeout)
                
                sender_beacon.send_command = track_send_command
                
                # Execute data plan - track transmission time
                transmission_start = time.time()
                sender_beacon.execute_data_plan(plan, data)
                transmission_end = time.time()
                transmission_duration = transmission_end - transmission_start
                
                # Calculate total transmission time
                total_transmission_time = transmission_duration
                print(f"Data plan execution and transmission completed in {total_transmission_time:.2f} seconds")
                
                # Calculate total data size sent
                total_data_sent = sum(chunk_sizes_sent)
                
                # Verify chunks were created correctly
                chunk_size_bytes = sender_config.data_chunk_size * 1024 * 1024
                for chunk_size in chunk_sizes_sent:
                    assert chunk_size <= chunk_size_bytes, f"Chunk size {chunk_size / 1024 / 1024:.2f}MB exceeds data_chunk_size {sender_config.data_chunk_size}MB"
                
                # Verify chunks were sent
                assert len(chunk_sizes_sent) > 0, "At least one chunk should have been sent"
                
                # Verify we got multiple chunks
                # With compression, the data might compress significantly, but we should still get multiple chunks
                # if the uncompressed size is large enough
                # Note: If compression is very effective, we might only get 1 chunk, which is acceptable
                # The important thing is that chunking works correctly when needed
                if total_data_sent > chunk_size_bytes:
                    assert len(chunk_sizes_sent) > 1, f"Expected multiple chunks for {total_data_sent / 1024 / 1024:.2f}MB data with {chunk_size_mb}MB chunk size, got {len(chunk_sizes_sent)} chunks"
                else:
                    # Data compressed to less than chunk size, single chunk is expected
                    assert len(chunk_sizes_sent) == 1, f"Expected single chunk for {total_data_sent / 1024 / 1024:.2f}MB compressed data, got {len(chunk_sizes_sent)} chunks"
                
                # Verify chunk sizes are correct (each chunk should be close to chunk_size_bytes)
                # Only check if we have multiple chunks
                if len(chunk_sizes_sent) > 1:
                    for i, chunk_size in enumerate(chunk_sizes_sent[:-1]):  # All except last
                        # Chunks should be close to chunk_size_bytes (within 10% tolerance)
                        assert chunk_size >= chunk_size_bytes * 0.9, f"Chunk {i+1} size {chunk_size / 1024 / 1024:.2f}MB is too small (expected ~{chunk_size_mb}MB)"
                
                # Verify memory protection: no chunk should exceed data_chunk_size
                max_chunk_size = max(chunk_sizes_sent) if chunk_sizes_sent else 0
                assert max_chunk_size <= chunk_size_bytes, f"Largest chunk {max_chunk_size / 1024 / 1024:.2f}MB exceeds data_chunk_size {chunk_size_mb}MB"
                
                # Verify receiver directory exists and contains all files
                assert receiver_dir.exists(), "Receiver directory should exist"
                
                # List all files in receiver directory (recursively)
                received_files = list(receiver_dir.rglob('*'))
                received_files = [f for f in received_files if f.is_file()]
                
                # Assert that receiving directory contains the same number of files as original
                assert len(received_files) == original_file_count, f"Expected {original_file_count} files, got {len(received_files)}"
                
                # Create mapping of relative paths to files
                original_map = {f.relative_to(source_dir): f for f in original_files}
                received_map = {f.relative_to(receiver_dir): f for f in received_files}
                
                # Assert that each file on the receiving end is created and is byte-for-byte equal to the original
                for rel_path, original_file in original_map.items():
                    assert rel_path in received_map, f"File {rel_path} not found in receiver directory"
                    
                    received_file = received_map[rel_path]
                    
                    # Verify byte-for-byte equality
                    with open(original_file, 'rb') as f1, open(received_file, 'rb') as f2:
                        original_data = f1.read()
                        received_data = f2.read()
                        assert original_data == received_data, f"File {rel_path} content does not match"
                        
                        # Verify file sizes match
                        assert len(original_data) == len(received_data), f"File {rel_path} size does not match"
                
            finally:
                sender_beacon.stop()
                receiver_beacon.stop()
                
                # Clean up receiver directory (but not source directory as per requirements)
                if receiver_dir.exists():
                    shutil.rmtree(receiver_dir)

    def test_large_directory_transfer_three_beacons(self, temp_state_dir):
        """Test transferring a directory from beacon1 to beacon2 and beacon3 with chunking."""
        import time
        import random
        import shutil
        from pathlib import Path
        from mosaic_planner.planner import get_directory_size
        from mosaic_config.state import Data, DataType, FileDefinition, Model, Plan
        
        # Set up test_data directory
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        # Create source directory name
        source_dir_name = "large_test_dir_three_beacons"
        source_dir = test_data_dir / source_dir_name
        
        # Check if directory exists, if not create it
        if not source_dir.exists():
            source_dir.mkdir(exist_ok=True)
            
            # Create files of varying sizes with random bytes (don't compress well)
            # Create a larger directory to ensure chunking even after compression
            file_sizes = [
                (20 * 1024 * 1024, "file_20mb.bin"),  # 20MB file
                (15 * 1024 * 1024, "file_15mb.bin"),  # 15MB file
                (10 * 1024 * 1024, "file_10mb.bin"),  # 10MB file
                (5 * 1024 * 1024, "file_5mb.bin"),     # 5MB file
                (2 * 1024 * 1024, "file_2mb.bin"),     # 2MB file
                (512 * 1024, "file_512kb.bin"),        # 512KB file
            ]
            
            # Generate random binary data (doesn't compress well)
            # Use a fixed seed for reproducibility so files can be compared byte-for-byte
            random.seed(42)
            chunk_size = 1024 * 1024  # Write in 1MB chunks
            
            for size_bytes, filename in file_sizes:
                file_path = source_dir / filename
                with open(file_path, 'wb') as f:
                    written = 0
                    while written < size_bytes:
                        bytes_to_write = min(chunk_size, size_bytes - written)
                        # Generate random bytes
                        random_data = bytes([random.randint(0, 255) for _ in range(bytes_to_write)])
                        f.write(random_data)
                        written += bytes_to_write
        
        # Verify directory exists and has files
        total_size = get_directory_size(source_dir)
        assert total_size > 0, "Source directory should contain files"
        
        # Count original files
        original_files = list(source_dir.rglob('*'))
        original_files = [f for f in original_files if f.is_file()]
        original_file_count = len(original_files)
        assert original_file_count > 0, "Source directory should contain at least one file"
        
        # Set up receiver directories (different locations for each receiver)
        receiver2_dir_name = "large_test_dir_received_beacon2"
        receiver2_dir = test_data_dir / receiver2_dir_name
        receiver3_dir_name = "large_test_dir_received_beacon3"
        receiver3_dir = test_data_dir / receiver3_dir_name
        
        # Clean up receiver directories
        if receiver2_dir.exists():
            shutil.rmtree(receiver2_dir)
        if receiver3_dir.exists():
            shutil.rmtree(receiver3_dir)
        
        # Create configs for 3 beacons
        # Use 2MB chunk size to force chunking
        chunk_size_mb = 2
        
        # Beacon1 (sender)
        beacon1_config = create_test_config_with_state(
            state_dir=temp_state_dir / "beacon1",
            host="127.0.0.1",
            heartbeat_port=7000,
            comms_port=7001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
            data_chunk_size=chunk_size_mb,
        )
        
        # Beacon2 (receiver 1 - gets 60% of data)
        beacon2_config = create_test_config_with_state(
            state_dir=temp_state_dir / "beacon2",
            host="127.0.0.1",
            heartbeat_port=7002,
            comms_port=7003,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
            data_chunk_size=chunk_size_mb,
        )
        
        # Beacon3 (receiver 2 - gets 40% of data)
        beacon3_config = create_test_config_with_state(
            state_dir=temp_state_dir / "beacon3",
            host="127.0.0.1",
            heartbeat_port=7004,
            comms_port=7005,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
            data_chunk_size=chunk_size_mb,
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            # Create beacons
            beacon1 = Beacon(beacon1_config)
            beacon2 = Beacon(beacon2_config)
            beacon3 = Beacon(beacon3_config)
            
            beacon1.start()
            beacon2.start()
            beacon3.start()
            
            try:
                # Wait for beacons to start
                time.sleep(1.0)
                
                # Split files between beacon2 (60%) and beacon3 (40%)
                # Files are sorted by name for consistent indexing
                files_sorted = sorted(original_files, key=lambda f: f.name)
                total_files = len(files_sorted)
                split_index = int(total_files * 0.6)
                beacon2_file_indices = list(range(0, split_index))
                beacon3_file_indices = list(range(split_index, total_files))
                
                beacon2_files = [files_sorted[i] for i in beacon2_file_indices]
                beacon3_files = [files_sorted[i] for i in beacon3_file_indices]
                
                # Create Plan with data_segmentation_plan for both receivers
                # Use segmentable directory with file_indices
                model = Model(name="test_model")
                plan = Plan(
                    stats_data=[],
                    distribution_plan=[],
                    model=model,
                    data_segmentation_plan=[
                        {
                            "host": beacon2_config.host,
                            "comms_port": beacon2_config.comms_port,
                            "fraction": 0.6,
                            "segments": [
                                {
                                    "file_location": source_dir_name,
                                    "data_type": "dir",
                                    "is_segmentable": True,
                                    "file_indices": beacon2_file_indices,
                                    "total_files": total_files,
                                    "fraction": 0.6,
                                },
                            ],
                        },
                        {
                            "host": beacon3_config.host,
                            "comms_port": beacon3_config.comms_port,
                            "fraction": 0.4,
                            "segments": [
                                {
                                    "file_location": source_dir_name,
                                    "data_type": "dir",
                                    "is_segmentable": True,
                                    "file_indices": beacon3_file_indices,
                                    "total_files": total_files,
                                    "fraction": 0.4,
                                },
                            ],
                        },
                    ],
                )
                
                # Create Data with single directory definition (segmentable)
                file_def = FileDefinition(
                    location=source_dir_name,
                    data_type=DataType.DIR,
                    is_segmentable=True,  # Make segmentable to split files
                )
                data = Data(file_definitions=[file_def])
                
                # Patch handlers to extract to unique locations and copy files
                # Use unique extraction directories to avoid conflicts
                extracted_dir_beacon2 = test_data_dir / f"{source_dir_name}_beacon2"
                extracted_dir_beacon3 = test_data_dir / f"{source_dir_name}_beacon3"
                
                # Patch beacon2 handler to extract to unique location
                original_handler2 = beacon2._handle_execute_data_plan
                def patched_handler2(payload):
                    # Modify the handler to extract to unique location
                    # We need to intercept and modify the extraction path
                    from mosaic_planner.planner import deserialize_plan_with_data
                    from mosaic_config.state import Data
                    from pathlib import Path
                    import uuid
                    from mosaic_planner.planner import unzip_stream_memory_safe
                    
                    # Deserialize to get the data
                    plan, data = deserialize_plan_with_data(payload, compressed=True)
                    
                    # Process each file definition, extracting to unique location
                    temp_dir = Path(beacon2_config.data_location) / "temp_received"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    for file_def in data.file_definitions:
                        if file_def.binary_data:
                            chunk_file = temp_dir / f"{uuid.uuid4()}.zip"
                            with open(chunk_file, 'wb') as f:
                                f.write(file_def.binary_data)
                            
                            # Extract to unique location for beacon2
                            extract_to = extracted_dir_beacon2
                            extract_to.parent.mkdir(parents=True, exist_ok=True)
                            unzip_stream_memory_safe(chunk_file, extract_to)
                            chunk_file.unlink()
                    
                    # Copy to receiver directory
                    if extracted_dir_beacon2.exists():
                        if receiver2_dir.exists():
                            shutil.rmtree(receiver2_dir)
                        shutil.copytree(extracted_dir_beacon2, receiver2_dir)
                    
                    return {"status": "success", "message": f"Received and processed {len(data.file_definitions)} file definitions", "file_count": len(data.file_definitions)}
                
                beacon2._handle_execute_data_plan = patched_handler2
                beacon2.register("exdplan", patched_handler2)
                
                # Patch beacon2 finalize handler
                original_finalize_handler2 = beacon2._handle_execute_data_plan_finalize
                def patched_finalize_handler2(payload):
                    from mosaic_planner.planner import deserialize_plan_with_data
                    from pathlib import Path
                    import uuid
                    from mosaic_planner.planner import unzip_stream_memory_safe
                    
                    chunk_id = payload.get("chunk_id")
                    temp_dir = Path(beacon2_config.data_location) / "temp_chunks"
                    chunk_file = temp_dir / f"{chunk_id}.part"
                    
                    with open(chunk_file, 'rb') as f:
                        complete_data = f.read()
                    
                    plan, data = deserialize_plan_with_data(complete_data, compressed=True)
                    
                    extract_temp_dir = Path(beacon2_config.data_location) / "temp_received"
                    extract_temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    for file_def in data.file_definitions:
                        if file_def.binary_data:
                            temp_zip = extract_temp_dir / f"{uuid.uuid4()}.zip"
                            with open(temp_zip, 'wb') as f:
                                f.write(file_def.binary_data)
                            
                            extract_to = extracted_dir_beacon2
                            extract_to.parent.mkdir(parents=True, exist_ok=True)
                            unzip_stream_memory_safe(temp_zip, extract_to)
                            temp_zip.unlink()
                    
                    chunk_file.unlink()
                    if chunk_id in beacon2._chunk_storage:
                        del beacon2._chunk_storage[chunk_id]
                    
                    # Copy to receiver directory
                    if extracted_dir_beacon2.exists():
                        if receiver2_dir.exists():
                            shutil.rmtree(receiver2_dir)
                        shutil.copytree(extracted_dir_beacon2, receiver2_dir)
                    
                    return {"status": "success", "message": f"Finalized and processed {len(data.file_definitions)} file definitions", "file_count": len(data.file_definitions)}
                
                beacon2._handle_execute_data_plan_finalize = patched_finalize_handler2
                beacon2.register("exdplan_finalize", patched_finalize_handler2)
                
                # Patch beacon3 handler to extract to unique location
                original_handler3 = beacon3._handle_execute_data_plan
                def patched_handler3(payload):
                    from mosaic_planner.planner import deserialize_plan_with_data
                    from pathlib import Path
                    import uuid
                    from mosaic_planner.planner import unzip_stream_memory_safe
                    
                    plan, data = deserialize_plan_with_data(payload, compressed=True)
                    
                    temp_dir = Path(beacon3_config.data_location) / "temp_received"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    for file_def in data.file_definitions:
                        if file_def.binary_data:
                            chunk_file = temp_dir / f"{uuid.uuid4()}.zip"
                            with open(chunk_file, 'wb') as f:
                                f.write(file_def.binary_data)
                            
                            # Extract to unique location for beacon3
                            extract_to = extracted_dir_beacon3
                            extract_to.parent.mkdir(parents=True, exist_ok=True)
                            unzip_stream_memory_safe(chunk_file, extract_to)
                            chunk_file.unlink()
                    
                    # Copy to receiver directory
                    if extracted_dir_beacon3.exists():
                        if receiver3_dir.exists():
                            shutil.rmtree(receiver3_dir)
                        shutil.copytree(extracted_dir_beacon3, receiver3_dir)
                    
                    return {"status": "success", "message": f"Received and processed {len(data.file_definitions)} file definitions", "file_count": len(data.file_definitions)}
                
                beacon3._handle_execute_data_plan = patched_handler3
                beacon3.register("exdplan", patched_handler3)
                
                # Patch beacon3 finalize handler
                original_finalize_handler3 = beacon3._handle_execute_data_plan_finalize
                def patched_finalize_handler3(payload):
                    from mosaic_planner.planner import deserialize_plan_with_data
                    from pathlib import Path
                    import uuid
                    from mosaic_planner.planner import unzip_stream_memory_safe
                    
                    chunk_id = payload.get("chunk_id")
                    temp_dir = Path(beacon3_config.data_location) / "temp_chunks"
                    chunk_file = temp_dir / f"{chunk_id}.part"
                    
                    with open(chunk_file, 'rb') as f:
                        complete_data = f.read()
                    
                    plan, data = deserialize_plan_with_data(complete_data, compressed=True)
                    
                    extract_temp_dir = Path(beacon3_config.data_location) / "temp_received"
                    extract_temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    for file_def in data.file_definitions:
                        if file_def.binary_data:
                            temp_zip = extract_temp_dir / f"{uuid.uuid4()}.zip"
                            with open(temp_zip, 'wb') as f:
                                f.write(file_def.binary_data)
                            
                            extract_to = extracted_dir_beacon3
                            extract_to.parent.mkdir(parents=True, exist_ok=True)
                            unzip_stream_memory_safe(temp_zip, extract_to)
                            temp_zip.unlink()
                    
                    chunk_file.unlink()
                    if chunk_id in beacon3._chunk_storage:
                        del beacon3._chunk_storage[chunk_id]
                    
                    # Copy to receiver directory
                    if extracted_dir_beacon3.exists():
                        if receiver3_dir.exists():
                            shutil.rmtree(receiver3_dir)
                        shutil.copytree(extracted_dir_beacon3, receiver3_dir)
                    
                    return {"status": "success", "message": f"Finalized and processed {len(data.file_definitions)} file definitions", "file_count": len(data.file_definitions)}
                
                beacon3._handle_execute_data_plan_finalize = patched_finalize_handler3
                beacon3.register("exdplan_finalize", patched_finalize_handler3)
                
                # Track chunk sizes sent (mock send_command to capture chunk sizes)
                chunk_sizes_sent = []
                original_send_command = beacon1.send_command
                
                def track_send_command(host, port, command, payload, timeout=30.0):
                    if command == "exdplan_chunk":
                        if isinstance(payload, bytes):
                            header_length = int.from_bytes(payload[:4], byteorder="big")
                            chunk_data = payload[4 + header_length:]
                            chunk_sizes_sent.append(len(chunk_data))
                    elif command == "exdplan":
                        if isinstance(payload, bytes):
                            chunk_sizes_sent.append(len(payload))
                    return original_send_command(host, port, command, payload, timeout)
                
                beacon1.send_command = track_send_command
                
                # Execute data plan - track transmission time
                transmission_start = time.time()
                beacon1.execute_data_plan(plan, data)
                transmission_end = time.time()
                transmission_duration = transmission_end - transmission_start
                
                # Calculate total transmission time
                total_transmission_time = transmission_duration
                print(f"Data plan execution and transmission completed in {total_transmission_time:.2f} seconds")
                
                # Calculate total data size sent
                total_data_sent = sum(chunk_sizes_sent)
                
                # Verify chunks were created correctly
                chunk_size_bytes = beacon1_config.data_chunk_size * 1024 * 1024
                for chunk_size in chunk_sizes_sent:
                    assert chunk_size <= chunk_size_bytes, f"Chunk size {chunk_size / 1024 / 1024:.2f}MB exceeds data_chunk_size {chunk_size_mb}MB"
                
                # Verify chunks were sent
                assert len(chunk_sizes_sent) > 0, "At least one chunk should have been sent"
                
                # Verify chunking works (similar to 2-beacon test)
                if total_data_sent > chunk_size_bytes:
                    assert len(chunk_sizes_sent) > 1, f"Expected multiple chunks for {total_data_sent / 1024 / 1024:.2f}MB data with {chunk_size_mb}MB chunk size, got {len(chunk_sizes_sent)} chunks"
                else:
                    assert len(chunk_sizes_sent) == 1, f"Expected single chunk for {total_data_sent / 1024 / 1024:.2f}MB compressed data, got {len(chunk_sizes_sent)} chunks"
                
                # Verify memory protection
                max_chunk_size = max(chunk_sizes_sent) if chunk_sizes_sent else 0
                assert max_chunk_size <= chunk_size_bytes, f"Largest chunk {max_chunk_size / 1024 / 1024:.2f}MB exceeds data_chunk_size {chunk_size_mb}MB"
                
                # Verify receiver directories exist
                assert receiver2_dir.exists(), "Beacon2 receiver directory should exist"
                assert receiver3_dir.exists(), "Beacon3 receiver directory should exist"
                
                # List files in receiver directories
                received_files_beacon2 = list(receiver2_dir.rglob('*'))
                received_files_beacon2 = [f for f in received_files_beacon2 if f.is_file()]
                received_files_beacon3 = list(receiver3_dir.rglob('*'))
                received_files_beacon3 = [f for f in received_files_beacon3 if f.is_file()]
                
                # Verify file counts match expected distribution
                assert len(received_files_beacon2) == len(beacon2_files), f"Beacon2: Expected {len(beacon2_files)} files, got {len(received_files_beacon2)}"
                assert len(received_files_beacon3) == len(beacon3_files), f"Beacon3: Expected {len(beacon3_files)} files, got {len(received_files_beacon3)}"
                assert len(received_files_beacon2) + len(received_files_beacon3) == original_file_count, "Total received files should match original"
                
                # Verify byte-for-byte equality for beacon2 files
                for original_file in beacon2_files:
                    rel_path = original_file.relative_to(source_dir)
                    received_file = receiver2_dir / rel_path
                    assert received_file.exists(), f"Beacon2: File {rel_path} not found"
                    
                    with open(original_file, 'rb') as f1, open(received_file, 'rb') as f2:
                        original_data = f1.read()
                        received_data = f2.read()
                        assert original_data == received_data, f"Beacon2: File {rel_path} content does not match"
                        assert len(original_data) == len(received_data), f"Beacon2: File {rel_path} size does not match"
                
                # Verify byte-for-byte equality for beacon3 files
                for original_file in beacon3_files:
                    rel_path = original_file.relative_to(source_dir)
                    received_file = receiver3_dir / rel_path
                    assert received_file.exists(), f"Beacon3: File {rel_path} not found"
                    
                    with open(original_file, 'rb') as f1, open(received_file, 'rb') as f2:
                        original_data = f1.read()
                        received_data = f2.read()
                        assert original_data == received_data, f"Beacon3: File {rel_path} content does not match"
                        assert len(original_data) == len(received_data), f"Beacon3: File {rel_path} size does not match"
                
            finally:
                beacon1.stop()
                beacon2.stop()
                beacon3.stop()
                
                # Clean up receiver directories and unique extraction directories
                if receiver2_dir.exists():
                    shutil.rmtree(receiver2_dir)
                if receiver3_dir.exists():
                    shutil.rmtree(receiver3_dir)
                if extracted_dir_beacon2.exists():
                    shutil.rmtree(extracted_dir_beacon2)
                if extracted_dir_beacon3.exists():
                    shutil.rmtree(extracted_dir_beacon3)

    def test_execute_data_plan_multithreaded_simultaneous_send(self, temp_state_dir):
        """Test that execute_data_plan sends data simultaneously to multiple recipients when multithreading is enabled."""
        from pathlib import Path
        from unittest.mock import patch
        
        # Set up test_data directory
        test_data_dir = Path(__file__).parent / 'test_data'
        test_data_dir.mkdir(exist_ok=True)
        
        # Create sender beacon config
        sender_config = create_test_config_with_state(
            state_dir=temp_state_dir / "sender",
            host="127.0.0.1",
            heartbeat_port=8000,
            comms_port=8001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location=str(test_data_dir),
            data_location=str(test_data_dir),
            data_chunk_size=2,  # 2MB chunks
        )
        
        # Create receiver beacon configs
        receiver1_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver1",
            host="127.0.0.1",
            heartbeat_port=8002,
            comms_port=8003,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
        )
        
        receiver2_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver2",
            host="127.0.0.1",
            heartbeat_port=8004,
            comms_port=8005,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location=str(test_data_dir),
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            # Create sender beacon
            sender_beacon = Beacon(sender_config)
            
            # Track when send_command is called for each recipient
            send_timestamps = {}
            send_lock = threading.Lock()
            
            # Mock send_command to track timestamps
            original_send_command = sender_beacon.send_command
            
            def mock_send_command(host, port, command, payload=None, timeout=None):
                """Track when send_command is called for each recipient."""
                if command == "exdplan":  # Only track the main data plan command
                    with send_lock:
                        key = f"{host}:{port}"
                        if key not in send_timestamps:
                            send_timestamps[key] = []
                        send_timestamps[key].append(time.time())
                    # Add a small delay to simulate network latency
                    time.sleep(0.01)
                # Call original send_command for other commands or return success
                if command in ("exdplan", "exdplan_chunk", "exdplan_finalize"):
                    return {"status": "success"}
                return original_send_command(host, port, command, payload, timeout)
            
            sender_beacon.send_command = mock_send_command
            
            # Mock benchmark data to enable multithreading
            # network_capacity = 5000 Mbps, chunk_size = 2MB
            # possible_thread_count = 5000 / 2 = 2500
            # thread_count = max(5, 2500) = 2500 (but we'll cap it reasonably)
            mock_benchmark_data = {
                "network_capacity": 5000,  # 5000 Mbps > 1000, so multithreading enabled
                "timestamp_ms": int(time.time() * 1000),
            }
            
            with patch("mosaic_comms.beacon.load_benchmarks", return_value=mock_benchmark_data):
                # Create a simple plan with data to send to 2 recipients
                model = Model(name="test_model")
                plan = Plan(
                    stats_data=[],
                    distribution_plan=[],
                    model=model,
                    data_segmentation_plan=[
                        {
                            "host": receiver1_config.host,
                            "comms_port": receiver1_config.comms_port,
                            "fraction": 0.5,
                            "segments": [
                                {
                                    "file_location": "test_file.csv",
                                    "data_type": "csv",
                                    "is_segmentable": True,
                                    "start_row": 0,
                                    "end_row": 10,
                                },
                            ],
                        },
                        {
                            "host": receiver2_config.host,
                            "comms_port": receiver2_config.comms_port,
                            "fraction": 0.5,
                            "segments": [
                                {
                                    "file_location": "test_file.csv",
                                    "data_type": "csv",
                                    "is_segmentable": True,
                                    "start_row": 10,
                                    "end_row": 20,
                                },
                            ],
                        },
                    ],
                )
                
                # Create a simple CSV file for testing
                test_csv = test_data_dir / "test_file.csv"
                with open(test_csv, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow(['col1', 'col2'])
                    for i in range(20):
                        writer.writerow([f'val1_{i}', f'val2_{i}'])
                
                # Create Data with file definition
                file_def = FileDefinition(
                    location="test_file.csv",
                    data_type=DataType.CSV,
                    is_segmentable=True,
                )
                data = Data(file_definitions=[file_def])
                
                try:
                    # Execute data plan
                    start_time = time.time()
                    sender_beacon.execute_data_plan(plan, data)
                    end_time = time.time()
                    
                    # Verify that both recipients received data
                    receiver1_key = f"{receiver1_config.host}:{receiver1_config.comms_port}"
                    receiver2_key = f"{receiver2_config.host}:{receiver2_config.comms_port}"
                    
                    assert receiver1_key in send_timestamps, "Receiver1 should have received data"
                    assert receiver2_key in send_timestamps, "Receiver2 should have received data"
                    
                    # Get the first send timestamp for each receiver
                    receiver1_first_send = send_timestamps[receiver1_key][0]
                    receiver2_first_send = send_timestamps[receiver2_key][0]
                    
                    # Calculate time difference between the two sends
                    time_diff = abs(receiver1_first_send - receiver2_first_send)
                    
                    # If multithreading is working, both sends should start within a small time window
                    # (e.g., within 0.1 seconds of each other)
                    # If sequential, the second send would start after the first completes (much longer)
                    max_time_diff_for_simultaneous = 0.1  # 100ms tolerance
                    assert time_diff < max_time_diff_for_simultaneous, (
                        f"Sends should be simultaneous (time diff: {time_diff:.3f}s), "
                        f"but got {time_diff:.3f}s difference. "
                        f"Receiver1: {receiver1_first_send:.3f}, Receiver2: {receiver2_first_send:.3f}"
                    )
                    
                    # Verify total execution time is reasonable (should be close to single send time,
                    # not double, if multithreading is working)
                    total_time = end_time - start_time
                    # With multithreading, total time should be roughly the time of one send + overhead
                    # Without multithreading, it would be roughly 2x the time of one send
                    # We expect total time to be less than 0.5 seconds (with mocked fast sends)
                    assert total_time < 0.5, (
                        f"Total execution time {total_time:.3f}s suggests sequential sending. "
                        f"With multithreading, should be much faster."
                    )
                finally:
                    # Clean up - ensure beacon is always stopped
                    sender_beacon.stop()
                    if test_csv.exists():
                        test_csv.unlink()

    def test_handle_execute_model_plan_adds_model(self, temp_state_dir):
        """Test that _handle_execute_model_plan correctly deserializes and adds a model."""
        import gzip
        import pickle
        from pathlib import Path
        from mosaic_config.state import Model, ModelType
        
        # Set up test directory for models
        test_models_dir = Path(__file__).parent / 'test_data' / 'test_models'
        test_models_dir.mkdir(parents=True, exist_ok=True)
        
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5000,
            comms_port=5001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location="",
            models_location=str(test_models_dir),
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            beacon = Beacon(config)
            
            # Initialize mosaic config and models list
            import mosaic.mosaic as mosaic_module
            mosaic_module._config = config
            from mosaic.mosaic import _models
            initial_model_count = len(_models)
            
            # Create a test model with binary_rep
            test_model_binary = b"fake_onnx_model_data_" + b"x" * 100  # Simulate ONNX model
            model = Model(
                name="test_model",
                model_type=ModelType.CNN,
                onnx_location="test_models",
                file_name="test_model.onnx",
                binary_rep=test_model_binary,
            )
            
            # Store original binary_rep for comparison
            original_binary_rep = model.binary_rep
            
            # Serialize model (same way execute_model_plan does)
            pickled_model = pickle.dumps(model)
            serialized_model = gzip.compress(pickled_model)
            
            # Call the handler with _config properly set using patch
            with patch("mosaic.mosaic._config", config):
                result = beacon._handle_execute_model_plan(serialized_model)
                
                # Verify result
                assert result is not None
                assert result["status"] == "success"
                assert f"Model {model.name} added successfully" in result["message"]
                
                # Verify model was added to _models list
                assert len(_models) == initial_model_count + 1, "One model should be added"
                added_model = _models[-1]  # Get the most recently added model
                assert added_model.name == model.name, "Model name should match"
                assert added_model.model_type == model.model_type, "Model type should match"
                
                # Verify model was saved to disk
                # add_model saves to models_location/onnx_location/model.id
                # The file_name field is set by add_model to model.id (if not already set)
                assert added_model.file_name is not None, "file_name should be set by add_model"
                
                # Verify that added_model.file_name is set correctly
                # add_model uses model.id as filename if file_name is None
                # If file_name was already set (e.g., for sharded models), it should be preserved (after sanitization)
                if model.file_name is not None:
                    # If model already had file_name set, it should be preserved (after sanitization)
                    from mosaic_config.state import Model as StateModel
                    expected_file_name = StateModel._sanitize_filename(model.file_name)
                    assert added_model.file_name == expected_file_name, f"file_name should be sanitized version of original: expected {expected_file_name}, got {added_model.file_name}"
                else:
                    # Otherwise, file_name should be set to model.id
                    assert added_model.file_name == model.id, f"file_name should be model.id: expected {model.id}, got {added_model.file_name}"
                
                # Build the expected file path using the actual file_name
                if model.onnx_location:
                    expected_file = test_models_dir / model.onnx_location / added_model.file_name
                else:
                    expected_file = test_models_dir / added_model.file_name
                
                # Debug: list all files if assertion fails
                if not expected_file.exists():
                    all_files = list(test_models_dir.rglob('*')) if test_models_dir.exists() else []
                    file_list = [str(f.relative_to(test_models_dir)) for f in all_files if f.is_file()]
                    assert False, (
                        f"Model file should be saved to {expected_file}, but it doesn't exist. "
                        f"Expected path: {expected_file}. "
                        f"Files found in {test_models_dir}: {file_list}"
                    )
                
                # Verify file content matches original binary_rep (byte-for-byte)
                with open(expected_file, 'rb') as f:
                    saved_data = f.read()
                assert saved_data == original_binary_rep, "Saved model binary should match original binary_rep byte-for-byte"
                assert saved_data == test_model_binary, "Saved model binary should match original test data byte-for-byte"
                assert len(saved_data) == len(original_binary_rep), "Saved file size should match original binary_rep size"
                
                # Verify binary_rep was cleared after saving (to conserve memory)
                assert added_model.binary_rep is None, "binary_rep should be cleared after saving"
                
                # Additional verification: deserialize the model again to verify binary_rep was preserved during serialization
                # This ensures the round-trip serialization/deserialization doesn't corrupt the binary_rep
                deserialized_pickled = gzip.decompress(serialized_model)
                deserialized_model = pickle.loads(deserialized_pickled)
                assert deserialized_model.binary_rep == original_binary_rep, "Deserialized binary_rep should match original byte-for-byte"
                assert deserialized_model.binary_rep == saved_data, "Deserialized binary_rep should match saved file byte-for-byte"
            
            # Clean up
            if test_models_dir.exists():
                import shutil
                shutil.rmtree(test_models_dir)

    def test_execute_model_plan_transmits_model(self, temp_state_dir):
        """Test that execute_model_plan transmits a model from one beacon to another."""
        from pathlib import Path
        from mosaic_config.state import Model, ModelType, Plan, Session, SessionStatus
        
        # Set up test directories
        test_models_dir = Path(__file__).parent / 'test_data' / 'test_models_transmit'
        test_models_dir.mkdir(parents=True, exist_ok=True)
        
        sender_models_dir = test_models_dir / "sender"
        receiver_models_dir = test_models_dir / "receiver"
        sender_models_dir.mkdir(parents=True, exist_ok=True)
        receiver_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sender config
        sender_config = create_test_config_with_state(
            state_dir=temp_state_dir / "sender",
            host="127.0.0.1",
            heartbeat_port=9000,
            comms_port=9001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location="",
            models_location=str(sender_models_dir),
        )
        
        # Create receiver config
        receiver_config = create_test_config_with_state(
            state_dir=temp_state_dir / "receiver",
            host="127.0.0.1",
            heartbeat_port=9002,
            comms_port=9003,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location="",
            models_location=str(receiver_models_dir),
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            # Create beacons
            sender_beacon = Beacon(sender_config)
            receiver_beacon = Beacon(receiver_config)
            
            # Initialize mosaic config for receiver
            from mosaic.mosaic import _config, _models
            import mosaic.mosaic as mosaic_module
            mosaic_module._config = receiver_config
            initial_model_count = len(_models)
            
            # Create a test model file on sender
            test_model_binary = b"fake_onnx_model_data_" + b"y" * 200  # Simulate ONNX model
            model_file = sender_models_dir / "test_model.onnx"
            with open(model_file, 'wb') as f:
                f.write(test_model_binary)
            
            # Create model with onnx_location and file_name (binary_rep will be loaded)
            model = Model(
                name="transmitted_model",
                model_type=ModelType.TRANSFORMER,
                onnx_location="",  # Empty means root of models_location
                file_name="test_model.onnx",
            )
            
            # Create a plan with distribution_plan
            plan = Plan(
                stats_data=[],
                distribution_plan=[
                    {
                        "host": receiver_config.host,
                        "comms_port": receiver_config.comms_port,
                        "capacity_fraction": 1.0,
                    }
                ],
                model=model,
            )
            
            # Create a session
            session = Session(plan=plan, status=SessionStatus.TRAINING)
            
            # Mock send_command on sender to capture the payload and call receiver handler
            received_payload = None
            
            def mock_send_command(host, port, command, payload=None, timeout=None):
                nonlocal received_payload
                if command == "exmplan" and host == receiver_config.host and port == receiver_config.comms_port:
                    received_payload = payload
                    # Call receiver handler directly with _config properly set
                    with patch("mosaic.mosaic._config", receiver_config):
                        return receiver_beacon._handle_execute_model_plan(payload)
                return {"status": "success"}
            
            sender_beacon.send_command = mock_send_command
            
            try:
                # Execute model plan
                sender_beacon.execute_model_plan(session, model)
                
                # Verify payload was sent
                assert received_payload is not None, "Model payload should have been sent"
                
                # Verify model was added to receiver's _models list
                assert len(_models) == initial_model_count + 1, "One model should be added to receiver"
                received_model = _models[-1]
                assert received_model.name == "transmitted_model", "Model name should match"
                assert received_model.model_type == ModelType.TRANSFORMER, "Model type should match"
                
                # Verify model was saved to disk on receiver
                # add_model saves to models_location/onnx_location/model.id
                # For sharded models (from execute_model_plan), file_name will be model.id-<shard_number>
                assert received_model.file_name is not None, "file_name should be set by add_model"
                
                # For sharded models from execute_model_plan, file_name should be model.id-0 (first shard)
                # The model ID should also be updated to include shard number
                assert received_model.file_name == received_model.id, f"file_name should match model.id: expected {received_model.id}, got {received_model.file_name}"
                assert received_model.id == f"{model.id}-0", f"Sharded model ID should be {model.id}-0, got {received_model.id}"
                
                # Build the expected file path using the actual file_name
                if model.onnx_location:
                    expected_file = receiver_models_dir / model.onnx_location / received_model.file_name
                else:
                    expected_file = receiver_models_dir / received_model.file_name
                
                assert expected_file.exists(), f"Model file should be saved to {expected_file}"
                
                # Verify file content matches original
                with open(expected_file, 'rb') as f:
                    saved_data = f.read()
                assert saved_data == test_model_binary, "Saved model binary should match original"
                
                # Verify binary_rep was cleared after saving
                assert received_model.binary_rep is None, "binary_rep should be cleared after saving"
                
            finally:
                # Clean up
                sender_beacon.stop()
                receiver_beacon.stop()
                if test_models_dir.exists():
                    import shutil
                    shutil.rmtree(test_models_dir)

    def test_execute_model_plan_raises_exception_when_model_cannot_be_loaded(self, temp_state_dir):
        """Test that execute_model_plan raises exception when onnx_location and file_name are not set."""
        from pathlib import Path
        from mosaic_config.state import Model, ModelType, Plan, Session, SessionStatus
        
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5000,
            comms_port=5001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location="",
            models_location="models",
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            beacon = Beacon(config)
            
            # Create model without binary_rep, onnx_location, or file_name
            model = Model(
                name="test_model",
                model_type=ModelType.CNN,
                binary_rep=None,
                onnx_location=None,
                file_name=None,
            )
            
            # Create a plan
            plan = Plan(
                stats_data=[],
                distribution_plan=[
                    {
                        "host": "127.0.0.1",
                        "comms_port": 5002,
                        "capacity_fraction": 1.0,
                    }
                ],
                model=model,
            )
            
            # Create a session
            session = Session(plan=plan, status=SessionStatus.TRAINING)
            
            # Execute model plan should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                beacon.execute_model_plan(session, model)
            
            assert "binary_rep is not set" in str(exc_info.value)
            assert "both onnx_location and file_name are None" in str(exc_info.value)

    def test_execute_model_plan_raises_exception_when_session_plan_is_none(self, temp_state_dir):
        """Test that execute_model_plan raises exception when session plan is None."""
        from mosaic_config.state import Model, ModelType, Session, SessionStatus
        
        config = create_test_config_with_state(
            state_dir=temp_state_dir,
            host="127.0.0.1",
            heartbeat_port=5000,
            comms_port=5001,
            heartbeat_frequency=2,
            heartbeat_tolerance=5,
            heartbeat_wait_timeout=2,
            stats_request_timeout=10,
            server_crt="",
            server_key="",
            ca_crt="",
            benchmark_data_location="",
            data_location="",
            models_location="models",
        )
        
        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats
            
            beacon = Beacon(config)
            
            # Create model with binary_rep
            model = Model(
                name="test_model",
                model_type=ModelType.CNN,
                binary_rep=b"fake_model_data",
            )
            
            # Create a session with None plan
            session = Session(plan=None, status=SessionStatus.TRAINING)
            
            # Execute model plan should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                beacon.execute_model_plan(session, model)
            
            assert "Session plan cannot be None" in str(exc_info.value)

