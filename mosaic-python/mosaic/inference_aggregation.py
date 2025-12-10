"""Federated learning inference aggregation methods for Mosaic."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def fedavg_aggregate(predictions: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Federated Averaging (FedAvg) aggregation method.
    
    Averages predictions from multiple models. If weights are provided, performs
    weighted averaging. Otherwise, uses uniform averaging.
    
    Args:
        predictions: List of prediction arrays from different nodes
        weights: Optional list of weights for each prediction (default: uniform)
        
    Returns:
        Aggregated prediction array
    """
    if not predictions:
        raise ValueError("No predictions provided for aggregation")
    
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    
    if len(weights) != len(predictions):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of predictions ({len(predictions)})")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    weights = [w / total_weight for w in weights]
    
    # Weighted average
    result = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        result += pred * weight
    
    return result


def fedprox_aggregate(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
    mu: float = 0.01,
) -> np.ndarray:
    """
    FedProx aggregation method.
    
    Similar to FedAvg but with a proximal term that adds regularization.
    For inference aggregation, this is approximated by adding a small regularization
    term to the averaging process.
    
    Args:
        predictions: List of prediction arrays from different nodes
        weights: Optional list of weights for each prediction (default: uniform)
        mu: Proximal parameter (default: 0.01)
        
    Returns:
        Aggregated prediction array
    """
    # For inference, FedProx is similar to FedAvg with slight regularization
    # In practice, we use weighted averaging with a small regularization term
    base_result = fedavg_aggregate(predictions, weights)
    
    # Add small regularization term (scaled by mu)
    # This helps stabilize predictions when there's high variance
    if len(predictions) > 1:
        variance = np.var([pred for pred in predictions], axis=0)
        regularization = mu * variance
        result = base_result + regularization
    else:
        result = base_result
    
    return result


def majority_vote_aggregate(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Majority voting aggregation method.
    
    For classification tasks, returns the most common prediction.
    For regression tasks, falls back to averaging.
    
    Args:
        predictions: List of prediction arrays from different nodes
        
    Returns:
        Aggregated prediction array
    """
    if not predictions:
        raise ValueError("No predictions provided for aggregation")
    
    # Check if predictions are discrete (classification) or continuous (regression)
    # If all predictions are integers and in a small range, treat as classification
    first_pred = predictions[0]
    is_classification = (
        np.all([np.allclose(p, p.astype(int)) for p in predictions]) and
        np.max(first_pred) < 1000  # Reasonable threshold for class indices
    )
    
    if is_classification:
        # For classification: majority vote
        # Stack predictions and take mode along axis 0
        stacked = np.stack(predictions, axis=0)
        result = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=stacked)
        return result.astype(first_pred.dtype)
    else:
        # For regression: fall back to averaging
        return fedavg_aggregate(predictions)


def weighted_average_aggregate(
    predictions: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """
    Weighted average aggregation method.
    
    Performs weighted averaging of predictions with explicit weights.
    
    Args:
        predictions: List of prediction arrays from different nodes
        weights: List of weights for each prediction (must sum to 1.0)
        
    Returns:
        Aggregated prediction array
    """
    return fedavg_aggregate(predictions, weights)


def max_aggregate(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Maximum aggregation method.
    
    Takes the maximum value across all predictions for each element.
    Useful for confidence scores or when you want the most optimistic prediction.
    
    Args:
        predictions: List of prediction arrays from different nodes
        
    Returns:
        Aggregated prediction array (element-wise maximum)
    """
    if not predictions:
        raise ValueError("No predictions provided for aggregation")
    
    stacked = np.stack(predictions, axis=0)
    return np.max(stacked, axis=0)


def min_aggregate(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Minimum aggregation method.
    
    Takes the minimum value across all predictions for each element.
    Useful for conservative predictions or when you want the most pessimistic estimate.
    
    Args:
        predictions: List of prediction arrays from different nodes
        
    Returns:
        Aggregated prediction array (element-wise minimum)
    """
    if not predictions:
        raise ValueError("No predictions provided for aggregation")
    
    stacked = np.stack(predictions, axis=0)
    return np.min(stacked, axis=0)


# Mapping of method names to aggregation functions
AGGREGATION_METHODS: Dict[str, Any] = {
    "fedavg": fedavg_aggregate,
    "fedprox": fedprox_aggregate,
    "majority_vote": majority_vote_aggregate,
    "weighted_average": weighted_average_aggregate,
    "max": max_aggregate,
    "min": min_aggregate,
}


def get_aggregation_method(method_name: str) -> Any:
    """
    Get an aggregation method by name.
    
    Args:
        method_name: Name of the aggregation method
        
    Returns:
        Aggregation function
        
    Raises:
        ValueError: If method name is not recognized
    """
    method_name_lower = method_name.lower()
    if method_name_lower not in AGGREGATION_METHODS:
        available = ", ".join(AGGREGATION_METHODS.keys())
        raise ValueError(
            f"Unknown aggregation method: {method_name}. "
            f"Available methods: {available}"
        )
    return AGGREGATION_METHODS[method_name_lower]


def list_aggregation_methods() -> List[str]:
    """
    List all available aggregation methods.
    
    Returns:
        List of method names
    """
    return list(AGGREGATION_METHODS.keys())

