"""Unit tests for mosaic.repl_commands module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from mosaic.repl_commands import (
    execute_calcd,
    execute_hb,
    execute_help,
    execute_rhb,
    execute_shb,
    initialize,
    process_command,
)
from tests.conftest import create_test_config_with_state


class TestExecuteShb:
    """Test execute_shb command."""

    def test_execute_shb_with_beacon(self):
        """Test execute_shb when beacon is initialized."""
        mock_beacon = MagicMock()
        mock_status = MagicMock()
        mock_status.host = "127.0.0.1"
        mock_status.heartbeat_port = 5000
        mock_status.connection_status = "online"
        mock_status.last_time_sent = 1234567890
        mock_beacon.send_heartbeat_statuses = [mock_status]

        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        # Initialize with mock beacon
        initialize(mock_beacon)
        try:
            execute_shb(output_fn)
        finally:
            # Clean up
            initialize(None)

        assert len(output_lines) > 0
        assert any("127.0.0.1" in line or "127.0.0.1" in "".join(output_lines) for line in output_lines)

    def test_execute_shb_without_beacon(self):
        """Test execute_shb when beacon is not initialized."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        # Ensure beacon is not initialized
        original_beacon = None
        from mosaic.repl_commands import _beacon as repl_beacon
        original_beacon = repl_beacon
        initialize(None)
        try:
            execute_shb(output_fn)
        finally:
            initialize(original_beacon)

        assert any("Error" in line or "not initialized" in line for line in output_lines)


class TestExecuteRhb:
    """Test execute_rhb command."""

    def test_execute_rhb_with_beacon(self):
        """Test execute_rhb when beacon is initialized."""
        mock_beacon = MagicMock()
        mock_status = MagicMock()
        mock_status.host = "127.0.0.1"
        mock_status.heartbeat_port = 5000
        mock_status.comms_port = 5001
        mock_status.connection_status = "online"
        mock_status.last_time_received = 1234567890
        mock_status.delay = 1000
        mock_beacon.receive_heartbeat_statuses = [mock_status]

        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        # Initialize with mock beacon
        initialize(mock_beacon)
        try:
            execute_rhb(output_fn)
        finally:
            # Clean up
            initialize(None)

        assert len(output_lines) > 0

    def test_execute_rhb_without_beacon(self):
        """Test execute_rhb when beacon is not initialized."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        # Ensure beacon is not initialized
        from mosaic.repl_commands import _beacon as repl_beacon
        original_beacon = repl_beacon
        initialize(None)
        try:
            execute_rhb(output_fn)
        finally:
            initialize(original_beacon)

        assert any("Error" in line or "not initialized" in line for line in output_lines)


class TestExecuteHb:
    """Test execute_hb command."""

    def test_execute_hb_calls_both(self):
        """Test that execute_hb calls both shb and rhb."""
        mock_beacon = MagicMock()
        mock_beacon.send_heartbeat_statuses = []
        mock_beacon.receive_heartbeat_statuses = []

        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        # Initialize with mock beacon
        initialize(mock_beacon)
        try:
            with patch("mosaic.repl_commands.execute_shb") as mock_shb:
                with patch("mosaic.repl_commands.execute_rhb") as mock_rhb:
                    execute_hb(output_fn)
                    mock_shb.assert_called_once_with(output_fn)
                    mock_rhb.assert_called_once_with(output_fn)
        finally:
            # Clean up
            initialize(None)


class TestExecuteCalcd:
    """Test execute_calcd command."""

    def test_execute_calcd_with_beacon(self):
        """Test execute_calcd when beacon is initialized."""
        mock_beacon = MagicMock()
        mock_stats_data = [{"host": "node1", "connection_status": "online"}]
        mock_beacon.collect_stats.return_value = mock_stats_data

        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        # Initialize with mock beacon
        initialize(mock_beacon)
        try:
            with patch("mosaic.mosaic.calculate_data_distribution") as mock_calc:
                execute_calcd(output_fn, method="weighted_shard")
                mock_calc.assert_called_once_with("weighted_shard")
        finally:
            # Clean up
            initialize(None)

    def test_execute_calcd_without_beacon(self):
        """Test execute_calcd when beacon is not initialized."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        # Ensure beacon is not initialized
        from mosaic.repl_commands import _beacon as repl_beacon
        original_beacon = repl_beacon
        initialize(None)
        try:
            execute_calcd(output_fn, method="weighted_shard")
            assert any("Error" in line or "not initialized" in line for line in output_lines)
        finally:
            initialize(original_beacon)


class TestExecuteHelp:
    """Test execute_help command."""

    def test_execute_help(self):
        """Test that execute_help outputs help text."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        execute_help(output_fn)

        output_text = "".join(output_lines)
        assert "shb" in output_text
        assert "rhb" in output_text
        assert "calcd" in output_text
        assert "help" in output_text


class TestProcessCommand:
    """Test process_command function."""

    def test_process_command_shb(self):
        """Test process_command with shb command."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        with patch("mosaic.repl_commands.execute_shb") as mock_execute:
            process_command("shb", output_fn)
            mock_execute.assert_called_once_with(output_fn)

    def test_process_command_rhb(self):
        """Test process_command with rhb command."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        with patch("mosaic.repl_commands.execute_rhb") as mock_execute:
            process_command("rhb", output_fn)
            mock_execute.assert_called_once_with(output_fn)

    def test_process_command_hb(self):
        """Test process_command with hb command."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        with patch("mosaic.repl_commands.execute_hb") as mock_execute:
            process_command("hb", output_fn)
            mock_execute.assert_called_once_with(output_fn)

    def test_process_command_calcd(self):
        """Test process_command with calcd command."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        with patch("mosaic.repl_commands.execute_calcd") as mock_execute:
            process_command("calcd weighted_shard", output_fn)
            mock_execute.assert_called_once_with(output_fn, "weighted_shard")

    def test_process_command_calcd_no_method(self):
        """Test process_command with calcd command without method."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        with patch("mosaic.repl_commands.execute_calcd") as mock_execute:
            process_command("calcd", output_fn)
            mock_execute.assert_called_once_with(output_fn, None)

    def test_process_command_help(self):
        """Test process_command with help command."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        with patch("mosaic.repl_commands.execute_help") as mock_execute:
            process_command("help", output_fn)
            mock_execute.assert_called_once_with(output_fn)

    def test_process_command_unknown(self):
        """Test process_command with unknown command."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        process_command("unknown_command", output_fn)
        assert any("Unknown" in line or "help" in line for line in output_lines)

    def test_process_command_empty(self):
        """Test process_command with empty command."""
        output_lines = []

        def output_fn(text: str) -> None:
            output_lines.append(text)

        process_command("", output_fn)
        # Should not produce any output for empty command
        assert len(output_lines) == 0


class TestReplCommandsWithTwoBeacons:
    """Test REPL commands with two real beacons set up."""

    def test_shb_rhb_hb_commands_with_two_beacons(self, temp_state_dir):
        """Test that shb, rhb, and hb commands work correctly with 2 beacons."""
        import mosaic.mosaic as mosaic_module
        from mosaic_comms.beacon import Beacon

        # Create configs for 2 beacons with different ports
        config1 = create_test_config_with_state(
            state_dir=temp_state_dir / "beacon1",
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
        )

        config2 = create_test_config_with_state(
            state_dir=temp_state_dir / "beacon2",
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
        )

        # Add peer to config1 so it knows about config2
        from mosaic_config.config import Peer
        config1.peers = [
            Peer(
                host=config2.host,
                heartbeat_port=config2.heartbeat_port,
                comms_port=config2.comms_port,
            )
        ]

        # Add peer to config2 so it knows about config1
        config2.peers = [
            Peer(
                host=config1.host,
                heartbeat_port=config1.heartbeat_port,
                comms_port=config1.comms_port,
            )
        ]

        with patch("mosaic_comms.beacon.StatsCollector") as mock_stats_class:
            mock_stats = MagicMock()
            mock_stats.get_last_stats_json.return_value = '{"cpu_percent": 45.3}'
            mock_stats_class.return_value = mock_stats

            # Create and start beacons
            beacon1 = Beacon(config1)
            beacon2 = Beacon(config2)

            try:
                # Initialize REPL commands with beacon1 (simulating what main() does)
                initialize(beacon1)

                # Start both beacons
                beacon1.start()
                beacon2.start()

                # Give them time to start listening
                time.sleep(0.5)

                # Send a heartbeat from beacon1 to beacon2 to establish connection
                payload = {
                    "host": config1.host,
                    "port": config1.heartbeat_port,
                    "comms_port": config1.comms_port,
                    "stats": {},
                }
                beacon1.send_heartbeat(
                    host=config2.host,
                    port=config2.heartbeat_port,
                    heartbeat_wait_timeout=config1.heartbeat_wait_timeout,
                    server_crt=config1.server_crt,
                    server_key=config1.server_key,
                    ca_crt=config1.ca_crt,
                    json_payload=payload,
                )

                # Wait a bit for heartbeat to be received
                time.sleep(0.5)

                # Collect output
                output_lines = []

                def output_fn(text: str) -> None:
                    output_lines.append(text)

                # Test shb command - should not raise an error
                output_lines.clear()
                execute_shb(output_fn)
                assert len(output_lines) > 0
                # Should not contain "Error: Beacon not initialized"
                output_text = "".join(output_lines)
                assert "Error: Beacon not initialized" not in output_text

                # Test rhb command - should not raise an error
                output_lines.clear()
                execute_rhb(output_fn)
                assert len(output_lines) > 0
                # Should not contain "Error: Beacon not initialized"
                output_text = "".join(output_lines)
                assert "Error: Beacon not initialized" not in output_text

                # Test hb command - should not raise an error
                output_lines.clear()
                execute_hb(output_fn)
                assert len(output_lines) > 0
                # Should not contain "Error: Beacon not initialized"
                output_text = "".join(output_lines)
                assert "Error: Beacon not initialized" not in output_text

                # Test process_command with shb
                output_lines.clear()
                process_command("shb", output_fn)
                assert len(output_lines) > 0
                output_text = "".join(output_lines)
                assert "Error: Beacon not initialized" not in output_text

                # Test process_command with rhb
                output_lines.clear()
                process_command("rhb", output_fn)
                assert len(output_lines) > 0
                output_text = "".join(output_lines)
                assert "Error: Beacon not initialized" not in output_text

                # Test process_command with hb
                output_lines.clear()
                process_command("hb", output_fn)
                assert len(output_lines) > 0
                output_text = "".join(output_lines)
                assert "Error: Beacon not initialized" not in output_text

            finally:
                # Restore original beacon state
                initialize(None)
                # Stop beacons
                beacon1.stop()
                beacon2.stop()

