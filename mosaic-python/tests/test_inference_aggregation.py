"""Unit tests for mosaic.inference_aggregation module."""

import numpy as np
import pytest

from mosaic.inference_aggregation import (
    fedavg_aggregate,
    fedprox_aggregate,
    majority_vote_aggregate,
    weighted_average_aggregate,
    max_aggregate,
    min_aggregate,
    get_aggregation_method,
    list_aggregation_methods,
)


class TestFedAvgAggregate:
    """Tests for fedavg_aggregate function."""
    
    def test_fedavg_uniform_weights(self):
        """Test FedAvg with uniform weights (default)."""
        predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0]),
        ]
        
        result = fedavg_aggregate(predictions)
        
        expected = np.array([2.0, 3.0, 4.0])  # Average of [1,2,3], [2,3,4], [3,4,5]
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fedavg_custom_weights(self):
        """Test FedAvg with custom weights."""
        predictions = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        weights = [0.7, 0.3]
        
        result = fedavg_aggregate(predictions, weights)
        
        expected = np.array([1.0 * 0.7 + 3.0 * 0.3, 2.0 * 0.7 + 4.0 * 0.3])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fedavg_weights_normalized(self):
        """Test that weights are normalized even if they don't sum to 1."""
        predictions = [
            np.array([1.0]),
            np.array([3.0]),
        ]
        weights = [2.0, 2.0]  # Should be normalized to [0.5, 0.5]
        
        result = fedavg_aggregate(predictions, weights)
        
        expected = np.array([2.0])  # Average of 1.0 and 3.0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fedavg_empty_predictions(self):
        """Test FedAvg raises error with empty predictions."""
        with pytest.raises(ValueError, match="No predictions provided"):
            fedavg_aggregate([])
    
    def test_fedavg_weight_mismatch(self):
        """Test FedAvg raises error when weights don't match predictions."""
        predictions = [np.array([1.0]), np.array([2.0])]
        weights = [0.5, 0.3, 0.2]  # Wrong length
        
        with pytest.raises(ValueError, match="must match"):
            fedavg_aggregate(predictions, weights)
    
    def test_fedavg_zero_total_weight(self):
        """Test FedAvg raises error when total weight is zero."""
        predictions = [np.array([1.0]), np.array([2.0])]
        weights = [0.0, 0.0]
        
        with pytest.raises(ValueError, match="Total weight cannot be zero"):
            fedavg_aggregate(predictions, weights)
    
    def test_fedavg_2d_array(self):
        """Test FedAvg with 2D arrays."""
        predictions = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]
        
        result = fedavg_aggregate(predictions)
        
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestFedProxAggregate:
    """Tests for fedprox_aggregate function."""
    
    def test_fedprox_single_prediction(self):
        """Test FedProx with single prediction (no regularization)."""
        predictions = [np.array([1.0, 2.0, 3.0])]
        
        result = fedprox_aggregate(predictions)
        
        # Should be same as input since no variance
        np.testing.assert_array_almost_equal(result, predictions[0])
    
    def test_fedprox_multiple_predictions(self):
        """Test FedProx with multiple predictions (adds regularization)."""
        predictions = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        
        result = fedprox_aggregate(predictions, mu=0.1)
        
        # Should be average plus regularization term
        base_avg = np.array([2.0, 3.0])
        variance = np.var([np.array([1.0, 2.0]), np.array([3.0, 4.0])], axis=0)
        expected = base_avg + 0.1 * variance
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fedprox_custom_mu(self):
        """Test FedProx with custom mu parameter."""
        predictions = [
            np.array([1.0]),
            np.array([3.0]),
        ]
        
        result_low_mu = fedprox_aggregate(predictions, mu=0.01)
        result_high_mu = fedprox_aggregate(predictions, mu=0.5)
        
        # Higher mu should produce higher result (more regularization)
        assert result_high_mu[0] > result_low_mu[0]


class TestMajorityVoteAggregate:
    """Tests for majority_vote_aggregate function."""
    
    def test_majority_vote_classification(self):
        """Test majority vote with classification predictions (integers)."""
        predictions = [
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([1, 1, 2]),
        ]
        
        result = majority_vote_aggregate(predictions)
        
        # First element: 0 appears twice, 1 appears once -> 0
        # Second element: 1 appears three times -> 1
        # Third element: 2 appears three times -> 2
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)
    
    def test_majority_vote_regression_fallback(self):
        """Test majority vote falls back to averaging for regression (floats)."""
        predictions = [
            np.array([1.5, 2.5]),
            np.array([2.5, 3.5]),
            np.array([3.5, 4.5]),
        ]
        
        result = majority_vote_aggregate(predictions)
        
        # Should fall back to averaging
        expected = np.array([2.5, 3.5])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_majority_vote_large_values(self):
        """Test majority vote treats large values as regression."""
        predictions = [
            np.array([1000.0, 2000.0]),
            np.array([2000.0, 3000.0]),
        ]
        
        result = majority_vote_aggregate(predictions)
        
        # Should fall back to averaging (not classification)
        expected = np.array([1500.0, 2500.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_majority_vote_empty_predictions(self):
        """Test majority vote raises error with empty predictions."""
        with pytest.raises(ValueError, match="No predictions provided"):
            majority_vote_aggregate([])


class TestWeightedAverageAggregate:
    """Tests for weighted_average_aggregate function."""
    
    def test_weighted_average(self):
        """Test weighted average aggregation."""
        predictions = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        weights = [0.8, 0.2]
        
        result = weighted_average_aggregate(predictions, weights)
        
        expected = np.array([1.0 * 0.8 + 3.0 * 0.2, 2.0 * 0.8 + 4.0 * 0.2])
        np.testing.assert_array_almost_equal(result, expected)


class TestMaxAggregate:
    """Tests for max_aggregate function."""
    
    def test_max_aggregate(self):
        """Test max aggregation takes element-wise maximum."""
        predictions = [
            np.array([1.0, 5.0, 3.0]),
            np.array([4.0, 2.0, 6.0]),
            np.array([2.0, 3.0, 1.0]),
        ]
        
        result = max_aggregate(predictions)
        
        expected = np.array([4.0, 5.0, 6.0])  # Max of each position
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_max_aggregate_empty_predictions(self):
        """Test max aggregate raises error with empty predictions."""
        with pytest.raises(ValueError, match="No predictions provided"):
            max_aggregate([])
    
    def test_max_aggregate_2d(self):
        """Test max aggregate with 2D arrays."""
        predictions = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 1.0], [2.0, 6.0]]),
        ]
        
        result = max_aggregate(predictions)
        
        expected = np.array([[5.0, 2.0], [3.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestMinAggregate:
    """Tests for min_aggregate function."""
    
    def test_min_aggregate(self):
        """Test min aggregation takes element-wise minimum."""
        predictions = [
            np.array([1.0, 5.0, 3.0]),
            np.array([4.0, 2.0, 6.0]),
            np.array([2.0, 3.0, 1.0]),
        ]
        
        result = min_aggregate(predictions)
        
        expected = np.array([1.0, 2.0, 1.0])  # Min of each position
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_min_aggregate_empty_predictions(self):
        """Test min aggregate raises error with empty predictions."""
        with pytest.raises(ValueError, match="No predictions provided"):
            min_aggregate([])


class TestGetAggregationMethod:
    """Tests for get_aggregation_method function."""
    
    def test_get_fedavg(self):
        """Test getting FedAvg method."""
        method = get_aggregation_method("fedavg")
        assert method == fedavg_aggregate
    
    def test_get_fedprox(self):
        """Test getting FedProx method."""
        method = get_aggregation_method("fedprox")
        assert method == fedprox_aggregate
    
    def test_get_case_insensitive(self):
        """Test method name is case insensitive."""
        method1 = get_aggregation_method("FEDAVG")
        method2 = get_aggregation_method("fedavg")
        assert method1 == method2
    
    def test_get_unknown_method(self):
        """Test getting unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            get_aggregation_method("unknown_method")


class TestListAggregationMethods:
    """Tests for list_aggregation_methods function."""
    
    def test_list_methods(self):
        """Test listing all aggregation methods."""
        methods = list_aggregation_methods()
        
        assert "fedavg" in methods
        assert "fedprox" in methods
        assert "majority_vote" in methods
        assert "weighted_average" in methods
        assert "max" in methods
        assert "min" in methods
        assert len(methods) == 6

