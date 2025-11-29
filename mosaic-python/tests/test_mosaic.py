"""Unit tests for mosaic.mosaic module."""

from unittest.mock import MagicMock, patch

from mosaic.mosaic import calculate_data_distribution


def test_calculate_data_distribution_weighted_shard_default():
    """Test that calculate_data_distribution calls plan_static_weighted_shards when method is None."""
    mock_beacon = MagicMock()
    mock_stats_data = [
        {"host": "node1", "connection_status": "online"},
        {"host": "node2", "connection_status": "online"},
    ]
    mock_beacon.collect_stats.return_value = mock_stats_data
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            mock_plan_shards.return_value = [
                {"host": "node1", "allocated_samples": 1},
                {"host": "node2", "allocated_samples": 1},
            ]
            
            calculate_data_distribution(method=None)
            
            # Verify collect_stats was called
            mock_beacon.collect_stats.assert_called_once()
            
            # Verify plan_static_weighted_shards was called with correct args
            mock_plan_shards.assert_called_once_with(
                mock_stats_data,
                total_samples=len(mock_stats_data)
            )


def test_calculate_data_distribution_weighted_shard_explicit():
    """Test that calculate_data_distribution calls plan_static_weighted_shards when method is 'weighted_shard'."""
    mock_beacon = MagicMock()
    mock_stats_data = [
        {"host": "node1", "connection_status": "online"},
        {"host": "node2", "connection_status": "online"},
        {"host": "node3", "connection_status": "online"},
    ]
    mock_beacon.collect_stats.return_value = mock_stats_data
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            mock_plan_shards.return_value = [
                {"host": "node1", "allocated_samples": 1},
                {"host": "node2", "allocated_samples": 1},
                {"host": "node3", "allocated_samples": 1},
            ]
            
            calculate_data_distribution(method="weighted_shard")
            
            # Verify collect_stats was called
            mock_beacon.collect_stats.assert_called_once()
            
            # Verify plan_static_weighted_shards was called with correct args
            mock_plan_shards.assert_called_once_with(
                mock_stats_data,
                total_samples=len(mock_stats_data)
            )


def test_calculate_data_distribution_weighted_batches():
    """Test that calculate_data_distribution calls plan_dynamic_weighted_batches when method is 'weighted_batches'."""
    mock_beacon = MagicMock()
    mock_stats_data = [
        {"host": "node1", "connection_status": "online"},
        {"host": "node2", "connection_status": "online"},
        {"host": "node3", "connection_status": "online"},
        {"host": "node4", "connection_status": "online"},
    ]
    mock_beacon.collect_stats.return_value = mock_stats_data
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_dynamic_weighted_batches") as mock_plan_batches:
            mock_plan_batches.return_value = [
                {"host": "node1", "allocated_batches": 1},
                {"host": "node2", "allocated_batches": 1},
                {"host": "node3", "allocated_batches": 1},
                {"host": "node4", "allocated_batches": 1},
            ]
            
            calculate_data_distribution(method="weighted_batches")
            
            # Verify collect_stats was called
            mock_beacon.collect_stats.assert_called_once()
            
            # Verify plan_dynamic_weighted_batches was called with correct args
            mock_plan_batches.assert_called_once_with(
                mock_stats_data,
                total_batches=len(mock_stats_data)
            )


def test_calculate_data_distribution_no_stats_data():
    """Test that calculate_data_distribution handles empty stats data."""
    mock_beacon = MagicMock()
    mock_beacon.collect_stats.return_value = []
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            calculate_data_distribution(method=None)
            
            # Verify collect_stats was called
            mock_beacon.collect_stats.assert_called_once()
            
            # Verify planner function was not called (no stats data)
            mock_plan_shards.assert_not_called()


def test_calculate_data_distribution_invalid_method():
    """Test that calculate_data_distribution handles invalid method."""
    mock_beacon = MagicMock()
    mock_stats_data = [{"host": "node1", "connection_status": "online"}]
    mock_beacon.collect_stats.return_value = mock_stats_data
    
    with patch("mosaic.mosaic._beacon", mock_beacon):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            with patch("mosaic.mosaic.plan_dynamic_weighted_batches") as mock_plan_batches:
                calculate_data_distribution(method="invalid_method")
                
                # Verify collect_stats was called
                mock_beacon.collect_stats.assert_called_once()
                
                # Verify neither planner function was called
                mock_plan_shards.assert_not_called()
                mock_plan_batches.assert_not_called()


def test_calculate_data_distribution_no_beacon():
    """Test that calculate_data_distribution handles missing beacon."""
    with patch("mosaic.mosaic._beacon", None):
        with patch("mosaic.mosaic.plan_static_weighted_shards") as mock_plan_shards:
            calculate_data_distribution(method=None)
            
            # Verify planner function was not called
            mock_plan_shards.assert_not_called()

