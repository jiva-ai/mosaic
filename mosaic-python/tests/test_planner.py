"""Unit tests for mosaic_planner module."""

import time

from mosaic_planner import (
    capacity_score,
    eligibility_filter,
    live_load_factor,
    network_factor,
    plan_dynamic_weighted_batches,
    plan_static_weighted_shards,
)

def test_eligibility_filter_online_only():
    """Test that only nodes with connection_status='online' are included."""
    current_time = int(time.time() * 1000)
    
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,  # 1 second ago
        },
        {
            "host": "node2",
            "connection_status": "stale",
            "last_time_received": current_time - 1000,
        },
        {
            "host": "node3",
            "connection_status": "nascent",
            "last_time_received": current_time - 1000,
        },
        {
            "host": "node4",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
        },
    ]

    result = eligibility_filter(stats_data)
    
    assert len(result) == 2
    assert all(stat["connection_status"] == "online" for stat in result)
    assert result[0]["host"] == "node1"
    assert result[1]["host"] == "node4"


def test_eligibility_filter_stale_threshold():
    """Test that nodes older than stale_threshold are excluded."""
    current_time = int(time.time() * 1000)
    
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 30000,  # 30 seconds ago (fresh)
        },
        {
            "host": "node2",
            "connection_status": "online",
            "last_time_received": current_time - 90000,  # 90 seconds ago (stale with default 60s threshold)
        },
        {
            "host": "node3",
            "connection_status": "online",
            "last_time_received": current_time - 5000,  # 5 seconds ago (fresh)
        },
    ]

    result = eligibility_filter(stats_data, stale_threshold=60)
    
    assert len(result) == 2
    assert result[0]["host"] == "node1"
    assert result[1]["host"] == "node3"
    assert "node2" not in [stat["host"] for stat in result]


def test_eligibility_filter_custom_stale_threshold():
    """Test that custom stale_threshold works correctly."""
    current_time = int(time.time() * 1000)
    
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 30000,  # 30 seconds ago
        },
        {
            "host": "node2",
            "connection_status": "online",
            "last_time_received": current_time - 90000,  # 90 seconds ago
        },
    ]

    # With 120 second threshold, both should be included
    result = eligibility_filter(stats_data, stale_threshold=120)
    assert len(result) == 2

    # With 30 second threshold, only node1 should be included
    # Use 29999ms (just under 30 seconds) to pass the strict < check
    stats_data_30s = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 29999,  # Just under 30 seconds
        },
        {
            "host": "node2",
            "connection_status": "online",
            "last_time_received": current_time - 90000,  # 90 seconds ago
        },
    ]
    result = eligibility_filter(stats_data_30s, stale_threshold=30)
    assert len(result) == 1
    assert result[0]["host"] == "node1"


def test_eligibility_filter_never_received():
    """Test that nodes with last_time_received=0 are excluded."""
    current_time = int(time.time() * 1000)
    
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,  # 1 second ago
        },
        {
            "host": "node2",
            "connection_status": "online",
            "last_time_received": 0,  # Never received
        },
        {
            "host": "node3",
            "connection_status": "online",
            # Missing last_time_received field
        },
    ]

    result = eligibility_filter(stats_data)
    
    assert len(result) == 1
    assert result[0]["host"] == "node1"


def test_eligibility_filter_empty_list():
    """Test that empty input returns empty list."""
    result = eligibility_filter([])
    assert result == []


def test_eligibility_filter_no_eligible_nodes():
    """Test that when no nodes are eligible, empty list is returned."""
    current_time = int(time.time() * 1000)
    
    stats_data = [
        {
            "host": "node1",
            "connection_status": "stale",
            "last_time_received": current_time - 1000,
        },
        {
            "host": "node2",
            "connection_status": "online",
            "last_time_received": current_time - 120000,  # 120 seconds ago (stale)
        },
    ]

    result = eligibility_filter(stats_data, stale_threshold=60)
    assert result == []


def test_eligibility_filter_exact_threshold():
    """Test behavior at exact threshold boundary."""
    current_time = int(time.time() * 1000)
    threshold_seconds = 60
    threshold_millis = threshold_seconds * 1000
    
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - threshold_millis + 1,  # Just under threshold
        },
        {
            "host": "node2",
            "connection_status": "online",
            "last_time_received": current_time - threshold_millis,  # Exactly at threshold
        },
        {
            "host": "node3",
            "connection_status": "online",
            "last_time_received": current_time - threshold_millis - 1,  # Just over threshold
        },
    ]

    result = eligibility_filter(stats_data, stale_threshold=threshold_seconds)
    
    # Only node1 should be included (just under threshold)
    # node2 is exactly at threshold, so it's excluded (time_since_received < threshold_millis)
    # Actually, wait - let me check the logic: time_since_received = current_time - last_time_received
    # For node2: time_since_received = threshold_millis, which is NOT < threshold_millis, so excluded
    # For node1: time_since_received = threshold_millis - 1, which IS < threshold_millis, so included
    # For node3: time_since_received = threshold_millis + 1, which is NOT < threshold_millis, so excluded
    assert len(result) == 1
    assert result[0]["host"] == "node1"


def test_eligibility_filter_preserves_all_fields():
    """Test that all fields from original stats are preserved in filtered results."""
    current_time = int(time.time() * 1000)
    
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "stats_payload": {"cpu": 50, "memory": 60},
            "benchmark": {"score": 100},
        },
    ]

    result = eligibility_filter(stats_data)
    
    assert len(result) == 1
    assert result[0] == stats_data[0]


def test_capacity_score_basic():
    """Test basic capacity score calculation with all components."""
    stats_data = [
        {
            "host": "node1",
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [
                    {"gflops": 1000.0},
                    {"gflops": 500.0},
                ],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {
                    "read_speed_mbps": 200.0,
                    "write_speed_mbps": 150.0,
                },
            },
        },
    ]

    result = capacity_score(stats_data)

    assert len(result) == 1
    assert "capacity_score" in result[0]
    # With single node, all normalized scores are 0.5 (min == max)
    # Expected: 0.5 * (0.15 + 0.75 + 0.05 + 0.05) = 0.5 * 1.0 = 0.5
    expected_score = 0.5
    assert abs(result[0]["capacity_score"] - expected_score) < 0.01


def test_capacity_score_custom_weights():
    """Test capacity score with custom weights."""
    stats_data = [
        {
            "host": "node1",
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {
                    "read_speed_mbps": 200.0,
                    "write_speed_mbps": 150.0,
                },
            },
        },
    ]

    result = capacity_score(stats_data, w_cpu=0.5, w_gpu=0.3, w_ram=0.1, w_disk=0.1)

    assert len(result) == 1
    # With single node, all normalized scores are 0.5 (min == max)
    # Expected: 0.5 * (0.5 + 0.3 + 0.1 + 0.1) = 0.5 * 1.0 = 0.5
    expected_score = 0.5
    assert abs(result[0]["capacity_score"] - expected_score) < 0.01


def test_capacity_score_missing_components():
    """Test capacity score with missing benchmark components."""
    stats_data = [
        {
            "host": "node1",
            "benchmark": {
                "cpu": {"gflops": 100.0},
                # Missing gpus, ram, disk
            },
        },
    ]

    result = capacity_score(stats_data)

    assert len(result) == 1
    # With single node, all normalized scores are 0.5 (min == max)
    # CPU normalized = 0.5, others normalized = 0.5 (all zeros normalize to 0.5)
    # Expected: 0.5 * (0.15 + 0.75 + 0.05 + 0.05) = 0.5 * 1.0 = 0.5
    assert abs(result[0]["capacity_score"] - 0.5) < 0.01


def test_capacity_score_no_benchmark():
    """Test capacity score when benchmark is missing."""
    stats_data = [
        {
            "host": "node1",
            # Missing benchmark
        },
    ]

    result = capacity_score(stats_data)

    assert len(result) == 1
    assert result[0]["capacity_score"] == 0.0


def test_capacity_score_empty_gpus():
    """Test capacity score with empty GPU list."""
    stats_data = [
        {
            "host": "node1",
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [],  # Empty list
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {
                    "read_speed_mbps": 200.0,
                    "write_speed_mbps": 150.0,
                },
            },
        },
    ]

    result = capacity_score(stats_data)

    assert len(result) == 1
    # With single node, all normalized scores are 0.5 (min == max)
    # Expected: 0.5 * (0.15 + 0.75 + 0.05 + 0.05) = 0.5 * 1.0 = 0.5
    assert abs(result[0]["capacity_score"] - 0.5) < 0.01


def test_capacity_score_multiple_nodes():
    """Test capacity score with multiple nodes."""
    stats_data = [
        {
            "host": "node1",
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "benchmark": {
                "cpu": {"gflops": 200.0},
                "gpus": [{"gflops": 2000.0}],
                "ram": {"bandwidth_gbps": 100.0},
                "disk": {"read_speed_mbps": 400.0, "write_speed_mbps": 300.0},
            },
        },
    ]

    result = capacity_score(stats_data)

    assert len(result) == 2
    assert result[0]["host"] == "node1"
    assert result[1]["host"] == "node2"
    assert "capacity_score" in result[0]
    assert "capacity_score" in result[1]
    # Node2 should have a higher score (it has higher raw values, so normalized values will be higher)
    assert result[1]["capacity_score"] > result[0]["capacity_score"]
    # Node1 should have normalized score of 0.0 (min values)
    # Node2 should have normalized score of 1.0 (max values)
    # Expected for node1: 0.0 * 0.15 + 0.0 * 0.75 + 0.0 * 0.05 + 0.0 * 0.05 = 0.0
    # Expected for node2: 1.0 * 0.15 + 1.0 * 0.75 + 1.0 * 0.05 + 1.0 * 0.05 = 1.0
    assert abs(result[0]["capacity_score"] - 0.0) < 0.01
    assert abs(result[1]["capacity_score"] - 1.0) < 0.01


def test_capacity_score_preserves_original_fields():
    """Test that capacity_score preserves all original fields."""
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": 1234567890,
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = capacity_score(stats_data)

    assert len(result) == 1
    assert result[0]["host"] == "node1"
    assert result[0]["connection_status"] == "online"
    assert result[0]["last_time_received"] == 1234567890
    assert "capacity_score" in result[0]


def test_capacity_score_normalization_three_nodes():
    """Test that normalization works correctly with three nodes."""
    stats_data = [
        {
            "host": "node1",
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "benchmark": {
                "cpu": {"gflops": 150.0},
                "gpus": [{"gflops": 1500.0}],
                "ram": {"bandwidth_gbps": 75.0},
                "disk": {"read_speed_mbps": 300.0, "write_speed_mbps": 225.0},
            },
        },
        {
            "host": "node3",
            "benchmark": {
                "cpu": {"gflops": 200.0},
                "gpus": [{"gflops": 2000.0}],
                "ram": {"bandwidth_gbps": 100.0},
                "disk": {"read_speed_mbps": 400.0, "write_speed_mbps": 300.0},
            },
        },
    ]

    result = capacity_score(stats_data)

    assert len(result) == 3
    # Node1 should have lowest score (min values)
    assert abs(result[0]["capacity_score"] - 0.0) < 0.01
    # Node3 should have highest score (max values)
    assert abs(result[2]["capacity_score"] - 1.0) < 0.01
    # Node2 should have score between 0 and 1
    assert 0.0 < result[1]["capacity_score"] < 1.0
    # Scores should be in ascending order
    assert result[0]["capacity_score"] < result[1]["capacity_score"] < result[2]["capacity_score"]


def test_live_load_factor_basic():
    """Test basic live load factor calculation."""
    stats_data = [
        {
            "host": "node1",
            "stats_payload": {
                "cpu_percent": 50.0,
                "ram_percent": 30.0,
            },
        },
    ]

    result = live_load_factor(stats_data)

    assert len(result) == 1
    assert "live_load_factor" in result[0]
    # cpu_avail = max(0.0, 1.0 - 50.0/100) = 0.5
    # ram_avail = max(0.0, 1.0 - 30.0/100) = 0.7
    # load_factor = 0.7 * 0.5 + 0.3 * 0.7 = 0.35 + 0.21 = 0.56
    expected_factor = 0.7 * 0.5 + 0.3 * 0.7
    assert abs(result[0]["live_load_factor"] - expected_factor) < 0.01


def test_live_load_factor_custom_alpha():
    """Test live load factor with custom alpha."""
    stats_data = [
        {
            "host": "node1",
            "stats_payload": {
                "cpu_percent": 50.0,
                "ram_percent": 30.0,
            },
        },
    ]

    result = live_load_factor(stats_data, alpha=0.5)

    assert len(result) == 1
    # cpu_avail = 0.5, ram_avail = 0.7
    # load_factor = 0.5 * 0.5 + 0.5 * 0.7 = 0.25 + 0.35 = 0.6
    expected_factor = 0.5 * 0.5 + 0.5 * 0.7
    assert abs(result[0]["live_load_factor"] - expected_factor) < 0.01


def test_live_load_factor_fully_utilized():
    """Test live load factor when CPU and RAM are fully utilized."""
    stats_data = [
        {
            "host": "node1",
            "stats_payload": {
                "cpu_percent": 100.0,
                "ram_percent": 100.0,
            },
        },
    ]

    result = live_load_factor(stats_data)

    assert len(result) == 1
    # cpu_avail = max(0.0, 1.0 - 100.0/100) = 0.0
    # ram_avail = max(0.0, 1.0 - 100.0/100) = 0.0
    # load_factor = 0.7 * 0.0 + 0.3 * 0.0 = 0.0
    assert abs(result[0]["live_load_factor"] - 0.0) < 0.01


def test_live_load_factor_fully_available():
    """Test live load factor when CPU and RAM are fully available."""
    stats_data = [
        {
            "host": "node1",
            "stats_payload": {
                "cpu_percent": 0.0,
                "ram_percent": 0.0,
            },
        },
    ]

    result = live_load_factor(stats_data)

    assert len(result) == 1
    # cpu_avail = max(0.0, 1.0 - 0.0/100) = 1.0
    # ram_avail = max(0.0, 1.0 - 0.0/100) = 1.0
    # load_factor = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
    assert abs(result[0]["live_load_factor"] - 1.0) < 0.01


def test_live_load_factor_over_100_percent():
    """Test live load factor when utilization exceeds 100% (clamped)."""
    stats_data = [
        {
            "host": "node1",
            "stats_payload": {
                "cpu_percent": 150.0,  # Over 100%
                "ram_percent": 120.0,  # Over 100%
            },
        },
    ]

    result = live_load_factor(stats_data)

    assert len(result) == 1
    # cpu_avail = max(0.0, 1.0 - 150.0/100) = max(0.0, -0.5) = 0.0
    # ram_avail = max(0.0, 1.0 - 120.0/100) = max(0.0, -0.2) = 0.0
    # load_factor = 0.0
    assert abs(result[0]["live_load_factor"] - 0.0) < 0.01


def test_live_load_factor_no_stats_payload():
    """Test live load factor when stats_payload is missing."""
    stats_data = [
        {
            "host": "node1",
            # Missing stats_payload
        },
    ]

    result = live_load_factor(stats_data)

    assert len(result) == 1
    assert result[0]["live_load_factor"] == 0.0


def test_live_load_factor_missing_percentages():
    """Test live load factor when CPU or RAM percentages are missing."""
    stats_data = [
        {
            "host": "node1",
            "stats_payload": {
                "cpu_percent": 50.0,
                # Missing ram_percent
            },
        },
        {
            "host": "node2",
            "stats_payload": {
                # Missing cpu_percent
                "ram_percent": 30.0,
            },
        },
    ]

    result = live_load_factor(stats_data)

    assert len(result) == 2
    # Missing values default to 0.0
    # node1: cpu_avail = 0.5, ram_avail = 1.0, load_factor = 0.7 * 0.5 + 0.3 * 1.0 = 0.65
    # node2: cpu_avail = 1.0, ram_avail = 0.7, load_factor = 0.7 * 1.0 + 0.3 * 0.7 = 0.91
    assert abs(result[0]["live_load_factor"] - 0.65) < 0.01
    assert abs(result[1]["live_load_factor"] - 0.91) < 0.01


def test_live_load_factor_multiple_nodes():
    """Test live load factor with multiple nodes."""
    stats_data = [
        {
            "host": "node1",
            "stats_payload": {
                "cpu_percent": 80.0,
                "ram_percent": 60.0,
            },
        },
        {
            "host": "node2",
            "stats_payload": {
                "cpu_percent": 20.0,
                "ram_percent": 40.0,
            },
        },
    ]

    result = live_load_factor(stats_data)

    assert len(result) == 2
    assert result[0]["host"] == "node1"
    assert result[1]["host"] == "node2"
    assert "live_load_factor" in result[0]
    assert "live_load_factor" in result[1]
    # node1: cpu_avail = 0.2, ram_avail = 0.4, load_factor = 0.7 * 0.2 + 0.3 * 0.4 = 0.26
    # node2: cpu_avail = 0.8, ram_avail = 0.6, load_factor = 0.7 * 0.8 + 0.3 * 0.6 = 0.74
    assert abs(result[0]["live_load_factor"] - 0.26) < 0.01
    assert abs(result[1]["live_load_factor"] - 0.74) < 0.01
    # Node2 should have higher load factor (more available)
    assert result[1]["live_load_factor"] > result[0]["live_load_factor"]


def test_live_load_factor_preserves_original_fields():
    """Test that live_load_factor preserves all original fields."""
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": 1234567890,
            "stats_payload": {
                "cpu_percent": 50.0,
                "ram_percent": 30.0,
            },
        },
    ]

    result = live_load_factor(stats_data)

    assert len(result) == 1
    assert result[0]["host"] == "node1"
    assert result[0]["connection_status"] == "online"
    assert result[0]["last_time_received"] == 1234567890
    assert "live_load_factor" in result[0]


def test_network_factor_basic():
    """Test basic network factor calculation."""
    stats_data = [
        {
            "host": "node1",
            "delay": 10.0,  # Fastest
        },
        {
            "host": "node2",
            "delay": 50.0,  # Middle
        },
        {
            "host": "node3",
            "delay": 100.0,  # Slowest
        },
    ]

    result = network_factor(stats_data)

    assert len(result) == 3
    assert "network_factor" in result[0]
    assert "network_factor" in result[1]
    assert "network_factor" in result[2]
    
    # min_delay = 10, max_delay = 100
    # node1 (fastest): norm = (10-10)/(100-10) = 0, factor = 1.0 - 0 * 0.6 = 1.0
    # node2 (middle): norm = (50-10)/(100-10) = 40/90 ≈ 0.444, factor = 1.0 - 0.444 * 0.6 ≈ 0.733
    # node3 (slowest): norm = (100-10)/(100-10) = 1.0, factor = 1.0 - 1.0 * 0.6 = 0.4
    
    assert abs(result[0]["network_factor"] - 1.0) < 0.01  # Fastest
    assert abs(result[2]["network_factor"] - 0.4) < 0.01  # Slowest (MIN_NET_FACTOR)
    # Middle should be between 0.4 and 1.0
    assert 0.4 < result[1]["network_factor"] < 1.0


def test_network_factor_same_delays():
    """Test network factor when all delays are the same."""
    stats_data = [
        {
            "host": "node1",
            "delay": 50.0,
        },
        {
            "host": "node2",
            "delay": 50.0,
        },
        {
            "host": "node3",
            "delay": 50.0,
        },
    ]

    result = network_factor(stats_data)

    assert len(result) == 3
    # When max_delay == min_delay, all should get network_factor = 1.0
    assert abs(result[0]["network_factor"] - 1.0) < 0.01
    assert abs(result[1]["network_factor"] - 1.0) < 0.01
    assert abs(result[2]["network_factor"] - 1.0) < 0.01


def test_network_factor_custom_min_net_factor():
    """Test network factor with custom min_net_factor."""
    stats_data = [
        {
            "host": "node1",
            "delay": 10.0,  # Fastest
        },
        {
            "host": "node2",
            "delay": 100.0,  # Slowest
        },
    ]

    result = network_factor(stats_data, min_net_factor=0.2)

    assert len(result) == 2
    # Fastest should get ~1.0
    assert abs(result[0]["network_factor"] - 1.0) < 0.01
    # Slowest should get min_net_factor = 0.2
    assert abs(result[1]["network_factor"] - 0.2) < 0.01


def test_network_factor_no_delays():
    """Test network factor when no nodes have delay information."""
    stats_data = [
        {
            "host": "node1",
            # No delay field
        },
        {
            "host": "node2",
            # No delay field
        },
    ]

    result = network_factor(stats_data)

    assert len(result) == 2
    # All should get 0.0 when no delays available
    assert result[0]["network_factor"] == 0.0
    assert result[1]["network_factor"] == 0.0


def test_network_factor_mixed_delays():
    """Test network factor when some nodes have delays and some don't."""
    stats_data = [
        {
            "host": "node1",
            "delay": 10.0,
        },
        {
            "host": "node2",
            # No delay
        },
        {
            "host": "node3",
            "delay": 100.0,
        },
    ]

    result = network_factor(stats_data)

    assert len(result) == 3
    # node1 should get 1.0 (fastest)
    assert abs(result[0]["network_factor"] - 1.0) < 0.01
    # node2 should get 0.0 (no delay)
    assert result[1]["network_factor"] == 0.0
    # node3 should get 0.4 (slowest, MIN_NET_FACTOR)
    assert abs(result[2]["network_factor"] - 0.4) < 0.01


def test_network_factor_invalid_delay():
    """Test network factor with invalid delay values."""
    stats_data = [
        {
            "host": "node1",
            "delay": 10.0,
        },
        {
            "host": "node2",
            "delay": "invalid",  # Invalid delay
        },
        {
            "host": "node3",
            "delay": 100.0,
        },
    ]

    result = network_factor(stats_data)

    assert len(result) == 3
    # node1 should get 1.0 (fastest of valid delays)
    assert abs(result[0]["network_factor"] - 1.0) < 0.01
    # node2 should get 0.0 (invalid delay)
    assert result[1]["network_factor"] == 0.0
    # node3 should get 0.4 (slowest of valid delays)
    assert abs(result[2]["network_factor"] - 0.4) < 0.01


def test_network_factor_none_delay():
    """Test network factor when delay is None."""
    stats_data = [
        {
            "host": "node1",
            "delay": 10.0,
        },
        {
            "host": "node2",
            "delay": None,  # None delay
        },
        {
            "host": "node3",
            "delay": 100.0,
        },
    ]

    result = network_factor(stats_data)

    assert len(result) == 3
    # node1 should get 1.0 (fastest)
    assert abs(result[0]["network_factor"] - 1.0) < 0.01
    # node2 should get 0.0 (None delay)
    assert result[1]["network_factor"] == 0.0
    # node3 should get 0.4 (slowest)
    assert abs(result[2]["network_factor"] - 0.4) < 0.01


def test_network_factor_preserves_original_fields():
    """Test that network_factor preserves all original fields."""
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": 1234567890,
            "delay": 10.0,
        },
    ]

    result = network_factor(stats_data)

    assert len(result) == 1
    assert result[0]["host"] == "node1"
    assert result[0]["connection_status"] == "online"
    assert result[0]["last_time_received"] == 1234567890
    assert "network_factor" in result[0]


def test_network_factor_calculation_accuracy():
    """Test that network factor calculation is mathematically correct."""
    stats_data = [
        {
            "host": "node1",
            "delay": 0.0,  # Fastest
        },
        {
            "host": "node2",
            "delay": 50.0,  # Middle
        },
        {
            "host": "node3",
            "delay": 100.0,  # Slowest
        },
    ]

    result = network_factor(stats_data, min_net_factor=0.4)

    assert len(result) == 3
    
    # min_delay = 0, max_delay = 100
    # node1: norm = (0-0)/(100-0) = 0, factor = 1.0 - 0 * 0.6 = 1.0
    # node2: norm = (50-0)/(100-0) = 0.5, factor = 1.0 - 0.5 * 0.6 = 0.7
    # node3: norm = (100-0)/(100-0) = 1.0, factor = 1.0 - 1.0 * 0.6 = 0.4
    
    assert abs(result[0]["network_factor"] - 1.0) < 0.01
    assert abs(result[1]["network_factor"] - 0.7) < 0.01
    assert abs(result[2]["network_factor"] - 0.4) < 0.01


def test_plan_static_weighted_shards_basic():
    """Test basic shard allocation with multiple peers."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,  # Fastest
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "heartbeat_port": 5002,
            "comms_port": 5003,
            "connection_status": "online",
            "last_time_received": current_time - 2000,
            "delay": 50.0,  # Middle
            "benchmark": {
                "cpu": {"gflops": 200.0},
                "gpus": [{"gflops": 2000.0}],
                "ram": {"bandwidth_gbps": 100.0},
                "disk": {"read_speed_mbps": 400.0, "write_speed_mbps": 300.0},
            },
        },
    ]

    result = plan_static_weighted_shards(stats_data, total_samples=100)

    assert len(result) == 2
    # Check all required fields
    for allocation in result:
        assert "host" in allocation
        assert "heartbeat_port" in allocation
        assert "comms_port" in allocation
        assert "allocated_samples" in allocation
        assert "capacity_fraction" in allocation
        assert "compute_score" in allocation
        assert "network_factor" in allocation
        assert "effective_score" in allocation
        assert isinstance(allocation["allocated_samples"], int)
        assert allocation["allocated_samples"] >= 0
    
    # Total allocated should equal total_samples
    total_allocated = sum(alloc["allocated_samples"] for alloc in result)
    assert total_allocated == 100
    
    # Fractions should sum to approximately 1.0
    total_fraction = sum(alloc["capacity_fraction"] for alloc in result)
    assert abs(total_fraction - 1.0) < 0.01


def test_plan_static_weighted_shards_no_eligible_peers():
    """Test that empty list is returned when no eligible peers."""
    stats_data = [
        {
            "host": "node1",
            "connection_status": "stale",  # Not online
            "last_time_received": 1234567890,
        },
    ]

    result = plan_static_weighted_shards(stats_data, total_samples=100)
    assert result == []


def test_plan_static_weighted_shards_no_benchmark():
    """Test that empty list is returned when no benchmarks."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            # No benchmark
        },
    ]

    result = plan_static_weighted_shards(stats_data, total_samples=100)
    assert result == []


def test_plan_static_weighted_shards_zero_capacity_score():
    """Test that peers with zero capacity score are dropped."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "benchmark": {
                # All zeros - will result in capacity_score = 0.0
                "cpu": {"gflops": 0.0},
                "gpus": [],
                "ram": {"bandwidth_gbps": 0.0},
                "disk": {"read_speed_mbps": 0.0, "write_speed_mbps": 0.0},
            },
        },
    ]

    result = plan_static_weighted_shards(stats_data, total_samples=100)
    assert result == []


def test_plan_static_weighted_shards_remainder_allocation():
    """Test that remainder samples are allocated correctly."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "heartbeat_port": 5002,
            "comms_port": 5003,
            "connection_status": "online",
            "last_time_received": current_time - 2000,
            "delay": 10.0,  # Same delay
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node3",
            "heartbeat_port": 5004,
            "comms_port": 5005,
            "connection_status": "online",
            "last_time_received": current_time - 3000,
            "delay": 10.0,  # Same delay
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    # Use 10 samples with 3 peers - should have remainder of 1
    result = plan_static_weighted_shards(stats_data, total_samples=10)

    assert len(result) == 3
    total_allocated = sum(alloc["allocated_samples"] for alloc in result)
    assert total_allocated == 10
    
    # With equal scores, each should get 3 base, and one should get +1
    allocations = [alloc["allocated_samples"] for alloc in result]
    assert sum(allocations) == 10
    # One peer should have 4, others should have 3
    assert max(allocations) == 4
    assert min(allocations) == 3


def test_plan_static_weighted_shards_effective_score_zero():
    """Test that peers with zero effective score are dropped."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            # No delay - network_factor will be 0.0
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_static_weighted_shards(stats_data, total_samples=100)
    # With network_factor = 0.0, effective_score = 0.0, so peer is dropped
    assert result == []


def test_plan_static_weighted_shards_single_peer():
    """Test shard allocation with a single eligible peer."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_static_weighted_shards(stats_data, total_samples=50)

    assert len(result) == 1
    assert result[0]["host"] == "node1"
    assert result[0]["allocated_samples"] == 50
    assert abs(result[0]["capacity_fraction"] - 1.0) < 0.01
    assert result[0]["compute_score"] > 0.0
    assert result[0]["network_factor"] > 0.0
    assert result[0]["effective_score"] > 0.0


def test_plan_static_weighted_shards_proportional_allocation():
    """Test that allocation is proportional to effective scores."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,  # Fastest - network_factor = 1.0
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "heartbeat_port": 5002,
            "comms_port": 5003,
            "connection_status": "online",
            "last_time_received": current_time - 2000,
            "delay": 100.0,  # Slowest - network_factor = 0.4
            "benchmark": {
                "cpu": {"gflops": 200.0},
                "gpus": [{"gflops": 2000.0}],
                "ram": {"bandwidth_gbps": 100.0},
                "disk": {"read_speed_mbps": 400.0, "write_speed_mbps": 300.0},
            },
        },
    ]

    result = plan_static_weighted_shards(stats_data, total_samples=1000)

    assert len(result) == 2
    total_allocated = sum(alloc["allocated_samples"] for alloc in result)
    assert total_allocated == 1000
    
    # Node2 should have higher capacity_score (better hardware)
    # But node1 should have better network_factor (faster)
    # The allocation should reflect the effective_score (capacity * network)
    node1_alloc = next(alloc for alloc in result if alloc["host"] == "node1")
    node2_alloc = next(alloc for alloc in result if alloc["host"] == "node2")
    
    # Both should get some allocation
    assert node1_alloc["allocated_samples"] > 0
    assert node2_alloc["allocated_samples"] > 0
    # Fractions should be proportional to effective scores
    assert node1_alloc["capacity_fraction"] > 0.0
    assert node2_alloc["capacity_fraction"] > 0.0
    assert abs(node1_alloc["capacity_fraction"] + node2_alloc["capacity_fraction"] - 1.0) < 0.01


def test_plan_static_weighted_shards_does_not_mutate_input():
    """Test that the function does not mutate the input list."""
    import time
    
    current_time = int(time.time() * 1000)
    original_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]
    
    # Create a deep copy for comparison
    import copy
    original_copy = copy.deepcopy(original_data)
    
    result = plan_static_weighted_shards(original_data, total_samples=100)
    
    # Original data should be unchanged
    assert original_data == original_copy
    assert len(result) > 0


def test_plan_dynamic_weighted_batches_basic():
    """Test basic batch allocation with multiple peers."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {
                "cpu_percent": 50.0,  # Under MAX_CPU_UTIL (90.0)
                "ram_percent": 60.0,  # Under MAX_RAM_UTIL (95.0)
            },
            "benchmark": {
                "cpu": {"gflops": 100.0},  # Has positive CPU
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "heartbeat_port": 5002,
            "comms_port": 5003,
            "connection_status": "online",
            "last_time_received": current_time - 2000,
            "delay": 50.0,
            "stats_payload": {
                "cpu_percent": 40.0,
                "ram_percent": 50.0,
            },
            "benchmark": {
                "cpu": {"gflops": 200.0},
                "gpus": [{"gflops": 2000.0}],
                "ram": {"bandwidth_gbps": 100.0},
                "disk": {"read_speed_mbps": 400.0, "write_speed_mbps": 300.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100)

    assert len(result) == 2
    # Check all required fields
    for allocation in result:
        assert "host" in allocation
        assert "heartbeat_port" in allocation
        assert "comms_port" in allocation
        assert "allocated_batches" in allocation
        assert "effective_score" in allocation
        assert "compute_score" in allocation
        assert "load_factor" in allocation
        assert "freshness_factor" in allocation
        assert "network_factor" in allocation
        assert isinstance(allocation["allocated_batches"], int)
        assert allocation["allocated_batches"] >= 0
    
    # Total allocated should equal total_batches
    total_allocated = sum(alloc["allocated_batches"] for alloc in result)
    assert total_allocated == 100


def test_plan_dynamic_weighted_batches_no_eligible_peers():
    """Test that empty list is returned when no eligible peers."""
    stats_data = [
        {
            "host": "node1",
            "connection_status": "stale",  # Not online
            "last_time_received": 1234567890,
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100)
    assert result == []


def test_plan_dynamic_weighted_batches_no_benchmark():
    """Test that empty list is returned when no benchmarks."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            # No benchmark
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100)
    assert result == []


def test_plan_dynamic_weighted_batches_no_cpu_or_gpu():
    """Test that peers without CPU or GPU with positive values are dropped."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 0.0},  # No positive CPU
                "gpus": [],  # No GPUs
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100)
    assert result == []


def test_plan_dynamic_weighted_batches_high_cpu_utilization():
    """Test that peers with CPU utilization >= MAX_CPU_UTIL are dropped."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {
                "cpu_percent": 95.0,  # >= MAX_CPU_UTIL (90.0)
                "ram_percent": 60.0,
            },
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100)
    assert result == []


def test_plan_dynamic_weighted_batches_high_ram_utilization():
    """Test that peers with RAM utilization >= MAX_RAM_UTIL are dropped."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {
                "cpu_percent": 50.0,
                "ram_percent": 98.0,  # >= MAX_RAM_UTIL (95.0)
            },
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100)
    assert result == []


def test_plan_dynamic_weighted_batches_stale_peer():
    """Test that peers beyond stale_threshold_ms are dropped."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 70000,  # 70 seconds ago, > 60s threshold
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100, stale_threshold_ms=60000)
    assert result == []


def test_plan_dynamic_weighted_batches_zero_load_factor():
    """Test that peers with zero load factor are dropped."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {
                "cpu_percent": 100.0,  # Fully utilized - load_factor will be 0.0
                "ram_percent": 100.0,
            },
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100)
    # Note: This peer will be filtered out at CPU/RAM utilization check first,
    # but if it passed that, load_factor would be 0.0
    assert result == []


def test_plan_dynamic_weighted_batches_freshness_factor():
    """Test that freshness_factor is computed correctly."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 0,  # Very fresh
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "heartbeat_port": 5002,
            "comms_port": 5003,
            "connection_status": "online",
            "last_time_received": current_time - 30000,  # 30 seconds ago (half of 60s threshold)
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 40.0, "ram_percent": 50.0},
            "benchmark": {
                "cpu": {"gflops": 200.0},
                "gpus": [{"gflops": 2000.0}],
                "ram": {"bandwidth_gbps": 100.0},
                "disk": {"read_speed_mbps": 400.0, "write_speed_mbps": 300.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100, stale_threshold_ms=60000)

    assert len(result) == 2
    node1 = next(alloc for alloc in result if alloc["host"] == "node1")
    node2 = next(alloc for alloc in result if alloc["host"] == "node2")
    
    # Node1 should have higher freshness_factor (fresher)
    assert node1["freshness_factor"] > node2["freshness_factor"]
    # Node1 should have freshness_factor close to 1.0 (very fresh)
    assert abs(node1["freshness_factor"] - 1.0) < 0.1
    # Node2 should have freshness_factor between 0.1 and 1.0 (linear decay)
    assert 0.1 <= node2["freshness_factor"] < 1.0


def test_plan_dynamic_weighted_batches_single_peer():
    """Test batch allocation with a single eligible peer."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=50)

    assert len(result) == 1
    assert result[0]["host"] == "node1"
    assert result[0]["allocated_batches"] == 50
    assert result[0]["compute_score"] > 0.0
    assert result[0]["load_factor"] > 0.0
    assert result[0]["freshness_factor"] > 0.0
    assert result[0]["network_factor"] > 0.0
    assert result[0]["effective_score"] > 0.0


def test_plan_dynamic_weighted_batches_default_total_batches():
    """Test that total_batches defaults to current time in milliseconds."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=None)

    assert len(result) == 1
    # total_batches should be approximately current time (within reasonable range)
    total_allocated = result[0]["allocated_batches"]
    assert total_allocated > 1000000000000  # Should be a timestamp-like value
    assert abs(total_allocated - current_time) < 5000  # Within 5 seconds


def test_plan_dynamic_weighted_batches_proportional_allocation():
    """Test that allocation is proportional to effective scores."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,  # Fastest - network_factor = 1.0
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "heartbeat_port": 5002,
            "comms_port": 5003,
            "connection_status": "online",
            "last_time_received": current_time - 2000,
            "delay": 100.0,  # Slowest - network_factor = 0.4
            "stats_payload": {"cpu_percent": 40.0, "ram_percent": 50.0},
            "benchmark": {
                "cpu": {"gflops": 200.0},
                "gpus": [{"gflops": 2000.0}],
                "ram": {"bandwidth_gbps": 100.0},
                "disk": {"read_speed_mbps": 400.0, "write_speed_mbps": 300.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=1000)

    assert len(result) == 2
    total_allocated = sum(alloc["allocated_batches"] for alloc in result)
    assert total_allocated == 1000
    
    node1_alloc = next(alloc for alloc in result if alloc["host"] == "node1")
    node2_alloc = next(alloc for alloc in result if alloc["host"] == "node2")
    
    # Both should get some allocation
    assert node1_alloc["allocated_batches"] > 0
    assert node2_alloc["allocated_batches"] > 0
    # Effective scores should be positive
    assert node1_alloc["effective_score"] > 0.0
    assert node2_alloc["effective_score"] > 0.0


def test_plan_dynamic_weighted_batches_does_not_mutate_input():
    """Test that the function does not mutate the input list."""
    import time
    import copy
    
    current_time = int(time.time() * 1000)
    original_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]
    
    original_copy = copy.deepcopy(original_data)
    
    result = plan_dynamic_weighted_batches(original_data, total_batches=100)
    
    # Original data should be unchanged
    assert original_data == original_copy
    assert len(result) > 0


def test_plan_dynamic_weighted_batches_zero_effective_score():
    """Test that peers with zero effective score are dropped."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            # No delay - network_factor will be 0.0
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    result = plan_dynamic_weighted_batches(stats_data, total_batches=100)
    # With network_factor = 0.0, effective_score = 0.0, so peer is dropped
    assert result == []


def test_plan_dynamic_weighted_batches_remainder_allocation():
    """Test that remainder batches are allocated correctly."""
    import time
    
    current_time = int(time.time() * 1000)
    stats_data = [
        {
            "host": "node1",
            "heartbeat_port": 5000,
            "comms_port": 5001,
            "connection_status": "online",
            "last_time_received": current_time - 1000,
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 50.0, "ram_percent": 60.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node2",
            "heartbeat_port": 5002,
            "comms_port": 5003,
            "connection_status": "online",
            "last_time_received": current_time - 2000,
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 40.0, "ram_percent": 50.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
        {
            "host": "node3",
            "heartbeat_port": 5004,
            "comms_port": 5005,
            "connection_status": "online",
            "last_time_received": current_time - 3000,
            "delay": 10.0,
            "stats_payload": {"cpu_percent": 30.0, "ram_percent": 40.0},
            "benchmark": {
                "cpu": {"gflops": 100.0},
                "gpus": [{"gflops": 1000.0}],
                "ram": {"bandwidth_gbps": 50.0},
                "disk": {"read_speed_mbps": 200.0, "write_speed_mbps": 150.0},
            },
        },
    ]

    # Use 10 batches with 3 peers - should have remainder of 1
    result = plan_dynamic_weighted_batches(stats_data, total_batches=10)

    assert len(result) == 3
    total_allocated = sum(alloc["allocated_batches"] for alloc in result)
    assert total_allocated == 10
    
    # With similar scores, each should get some allocation
    allocations = [alloc["allocated_batches"] for alloc in result]
    assert sum(allocations) == 10
    assert all(alloc > 0 for alloc in allocations)


