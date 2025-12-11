"""Model planning functions for compressing ONNX models based on node capabilities."""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

from mosaic_config.config import MosaicConfig, read_config
from mosaic_config.state import Model, ModelType, Plan

logger = logging.getLogger(__name__)


def _load_onnx_model(model: Model, config: Optional[MosaicConfig] = None) -> onnx.ModelProto:
    """
    Load ONNX model from binary_rep or from file.
    
    Args:
        model: Model instance
        config: Optional MosaicConfig for resolving paths
        
    Returns:
        Loaded ONNX model
        
    Raises:
        ValueError: If model cannot be loaded
    """
    if model.binary_rep is not None:
        # binary_rep is bytes from reading an ONNX file, so we use BytesIO to load it
        return onnx.load(io.BytesIO(model.binary_rep))
    
    if model.onnx_location is None or model.file_name is None:
        raise ValueError(
            f"Cannot load model {model.name}: both binary_rep and "
            "(onnx_location, file_name) are None"
        )
    
    # Resolve model path
    if config is None:
        config = read_config()
    
    models_base = Path(config.models_location)
    if model.onnx_location:
        model_path = models_base / model.onnx_location / model.file_name
    else:
        model_path = models_base / model.file_name
    
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")
    
    return onnx.load(str(model_path))


def _get_node_capabilities(plan: Plan) -> List[Tuple[Dict[str, Any], float]]:
    """
    Extract node capabilities from plan, sorted by capability (highest first).
    
    Args:
        plan: Plan instance with distribution_plan
        
    Returns:
        List of (node_dict, capability_score) tuples, sorted by score descending
    """
    nodes_with_scores = []
    
    for node in plan.distribution_plan:
        # Use effective_score if available, otherwise capacity_score
        score = node.get("effective_score") or node.get("capacity_score", 0.0)
        nodes_with_scores.append((node, score))
    
    # Sort by capability (highest first)
    nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    return nodes_with_scores


def _calculate_compression_ratio(capability_score: float, min_score: float, max_score: float) -> float:
    """
    Calculate compression ratio based on capability score.
    
    Higher capability = less compression (higher ratio of normal layers).
    
    Args:
        capability_score: Node's capability score
        min_score: Minimum capability score in the plan
        max_score: Maximum capability score in the plan
        
    Returns:
        Compression ratio (0.0 = full compression, 1.0 = no compression)
    """
    if max_score == min_score:
        return 0.5  # Default to moderate compression
    
    # Normalize score to [0, 1] range
    normalized = (capability_score - min_score) / (max_score - min_score)
    
    # Map to compression ratio: higher score = less compression
    # Cap at reasonable bounds: 0.2 (80% compression) to 0.9 (10% compression)
    compression_ratio = 0.2 + (normalized * 0.7)
    
    return compression_ratio


def plan_cnn_model(model: Model, plan: Plan, config: Optional[MosaicConfig] = None) -> Dict[str, Any]:
    """
    Plan CNN model compression (ResNet-50, ResNet-101) based on node capabilities.
    
    Uses magnitude-based channel pruning for bottleneck layers.
    More capable nodes get less compression (more normal layers).
    
    Args:
        model: Model instance
        plan: Plan with distribution_plan containing node capabilities
        config: Optional MosaicConfig for loading models
        
    Returns:
        Dictionary with compression metadata for each node
    """
    onnx_model = _load_onnx_model(model, config)
    nodes_with_scores = _get_node_capabilities(plan)
    
    if not nodes_with_scores:
        logger.warning(f"No nodes in plan for model {model.name}")
        return {}
    
    # Get score range
    scores = [score for _, score in nodes_with_scores]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    
    # Find convolutional layers (especially bottleneck layers)
    conv_layers = []
    for i, node in enumerate(onnx_model.graph.node):
        if node.op_type in ["Conv", "ConvTranspose"]:
            conv_layers.append((i, node))
    
    if not conv_layers:
        logger.warning(f"No convolutional layers found in model {model.name}")
        return {}
    
    # Identify bottleneck layers (middle layers, not first/last)
    bottleneck_start = len(conv_layers) // 4
    bottleneck_end = 3 * len(conv_layers) // 4
    
    compression_metadata = {}
    
    for node_dict, capability_score in nodes_with_scores:
        compression_ratio = _calculate_compression_ratio(capability_score, min_score, max_score)
        
        # Calculate channel reduction: 0% for high capability, 40-50% for low capability
        channel_reduction = (1.0 - compression_ratio) * 0.5  # Max 50% reduction
        
        # Determine which layers to compress
        normal_layers = []
        compressed_layers = []
        
        for idx, (layer_idx, conv_node) in enumerate(conv_layers):
            if idx < bottleneck_start or idx >= bottleneck_end:
                # First/last layers: always normal
                normal_layers.append(layer_idx)
            elif compression_ratio > 0.6:
                # High capability: keep more layers normal
                if idx % 2 == 0:  # Keep every other layer
                    normal_layers.append(layer_idx)
                else:
                    compressed_layers.append((layer_idx, channel_reduction))
            else:
                # Low capability: compress bottleneck layers
                compressed_layers.append((layer_idx, channel_reduction))
        
        node_id = f"{node_dict.get('host')}:{node_dict.get('comms_port')}"
        compression_metadata[node_id] = {
            "compression_ratio": compression_ratio,
            "channel_reduction": channel_reduction,
            "normal_layers": normal_layers,
            "compressed_layers": compressed_layers,
            "compression_type": "magnitude_based_channel_pruning",
        }
    
    return compression_metadata


def plan_wav2vec_model(model: Model, plan: Plan, config: Optional[MosaicConfig] = None) -> Dict[str, Any]:
    """
    Plan Wav2Vec2 model compression based on node capabilities.
    
    Uses structured encoder layer dropping and hidden dimension compression.
    
    Args:
        model: Model instance
        plan: Plan with distribution_plan containing node capabilities
        config: Optional MosaicConfig for loading models
        
    Returns:
        Dictionary with compression metadata for each node
    """
    onnx_model = _load_onnx_model(model, config)
    nodes_with_scores = _get_node_capabilities(plan)
    
    if not nodes_with_scores:
        logger.warning(f"No nodes in plan for model {model.name}")
        return {}
    
    # Get score range
    scores = [score for _, score in nodes_with_scores]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    
    # Find transformer encoder layers
    encoder_layers = []
    for i, node in enumerate(onnx_model.graph.node):
        # Look for MultiHeadAttention, LayerNorm, or FeedForward patterns
        if any(op in node.op_type for op in ["Attention", "MatMul", "Add", "LayerNormalization"]):
            encoder_layers.append((i, node))
    
    compression_metadata = {}
    
    for node_dict, capability_score in nodes_with_scores:
        compression_ratio = _calculate_compression_ratio(capability_score, min_score, max_score)
        
        # Layer reduction: 0% for high capability, 30-40% for low capability
        layer_reduction = (1.0 - compression_ratio) * 0.4
        
        # Hidden dimension reduction: 0% for high capability, 30-40% for low capability
        hidden_dim_reduction = (1.0 - compression_ratio) * 0.4
        
        # Determine which layers to keep
        total_layers = len(encoder_layers)
        layers_to_keep = int(total_layers * (1.0 - layer_reduction))
        
        # Keep layers distributed across the network
        if compression_ratio > 0.6:
            # High capability: keep more layers
            kept_layers = list(range(0, total_layers, max(1, total_layers // layers_to_keep)))
        else:
            # Low capability: keep fewer layers, distributed
            step = max(1, total_layers // layers_to_keep)
            kept_layers = list(range(0, total_layers, step))[:layers_to_keep]
        
        dropped_layers = [i for i in range(total_layers) if i not in kept_layers]
        
        node_id = f"{node_dict.get('host')}:{node_dict.get('comms_port')}"
        compression_metadata[node_id] = {
            "compression_ratio": compression_ratio,
            "layer_reduction": layer_reduction,
            "hidden_dim_reduction": hidden_dim_reduction,
            "kept_layers": kept_layers,
            "dropped_layers": dropped_layers,
            "compression_type": "layer_dropping_and_dimension_compression",
        }
    
    return compression_metadata


def plan_transformer_model(model: Model, plan: Plan, config: Optional[MosaicConfig] = None) -> Dict[str, Any]:
    """
    Plan GPT-Neo transformer model compression based on node capabilities.
    
    Uses aggressive layer removal and width reduction.
    
    Args:
        model: Model instance
        plan: Plan with distribution_plan containing node capabilities
        config: Optional MosaicConfig for loading models
        
    Returns:
        Dictionary with compression metadata for each node
    """
    onnx_model = _load_onnx_model(model, config)
    nodes_with_scores = _get_node_capabilities(plan)
    
    if not nodes_with_scores:
        logger.warning(f"No nodes in plan for model {model.name}")
        return {}
    
    # Get score range
    scores = [score for _, score in nodes_with_scores]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    
    # Find transformer blocks
    transformer_blocks = []
    for i, node in enumerate(onnx_model.graph.node):
        if any(op in node.op_type for op in ["Attention", "MatMul", "LayerNormalization"]):
            transformer_blocks.append((i, node))
    
    compression_metadata = {}
    
    for node_dict, capability_score in nodes_with_scores:
        compression_ratio = _calculate_compression_ratio(capability_score, min_score, max_score)
        
        # Layer reduction: 0% for high capability, 50% for low capability
        layer_reduction = (1.0 - compression_ratio) * 0.5
        
        # Hidden dimension reduction: 0% for high capability, 30-40% for low capability
        hidden_dim_reduction = (1.0 - compression_ratio) * 0.4
        
        # Determine which layers to keep
        total_layers = len(transformer_blocks)
        layers_to_keep = int(total_layers * (1.0 - layer_reduction))
        
        # Keep every Nth layer based on capability
        if compression_ratio > 0.7:
            # High capability: keep every other layer
            kept_layers = list(range(0, total_layers, 2))
        elif compression_ratio > 0.4:
            # Medium capability: keep every 3rd layer
            kept_layers = list(range(0, total_layers, 3))
        else:
            # Low capability: keep every 4th layer
            kept_layers = list(range(0, total_layers, 4))
        
        kept_layers = kept_layers[:layers_to_keep]
        dropped_layers = [i for i in range(total_layers) if i not in kept_layers]
        
        node_id = f"{node_dict.get('host')}:{node_dict.get('comms_port')}"
        compression_metadata[node_id] = {
            "compression_ratio": compression_ratio,
            "layer_reduction": layer_reduction,
            "hidden_dim_reduction": hidden_dim_reduction,
            "kept_layers": kept_layers,
            "dropped_layers": dropped_layers,
            "compression_type": "aggressive_layer_removal_and_width_reduction",
        }
    
    return compression_metadata


def plan_gnn_model(model: Model, plan: Plan, config: Optional[MosaicConfig] = None) -> Dict[str, Any]:
    """
    Plan GCN model compression based on node capabilities.
    
    Uses feature dimension reduction (preferred over layer dropping for graph models).
    
    Args:
        model: Model instance
        plan: Plan with distribution_plan containing node capabilities
        config: Optional MosaicConfig for loading models
        
    Returns:
        Dictionary with compression metadata for each node
    """
    onnx_model = _load_onnx_model(model, config)
    nodes_with_scores = _get_node_capabilities(plan)
    
    if not nodes_with_scores:
        logger.warning(f"No nodes in plan for model {model.name}")
        return {}
    
    # Get score range
    scores = [score for _, score in nodes_with_scores]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    
    # Find GraphConv layers
    gcn_layers = []
    for i, node in enumerate(onnx_model.graph.node):
        if node.op_type in ["MatMul", "Gemm"]:  # GraphConv typically uses MatMul/Gemm
            gcn_layers.append((i, node))
    
    compression_metadata = {}
    
    for node_dict, capability_score in nodes_with_scores:
        compression_ratio = _calculate_compression_ratio(capability_score, min_score, max_score)
        
        # Hidden dimension reduction: 0% for high capability, 30-40% for low capability
        hidden_dim_reduction = (1.0 - compression_ratio) * 0.4
        
        # Keep first layer normal, compress others
        normal_layers = [0] if len(gcn_layers) > 0 else []
        compressed_layers = []
        
        for idx, (layer_idx, _) in enumerate(gcn_layers[1:], start=1):
            if compression_ratio > 0.6:
                # High capability: keep more layers normal
                if idx % 2 == 0:
                    normal_layers.append(layer_idx)
                else:
                    compressed_layers.append((layer_idx, hidden_dim_reduction))
            else:
                # Low capability: compress more layers
                compressed_layers.append((layer_idx, hidden_dim_reduction))
        
        node_id = f"{node_dict.get('host')}:{node_dict.get('comms_port')}"
        compression_metadata[node_id] = {
            "compression_ratio": compression_ratio,
            "hidden_dim_reduction": hidden_dim_reduction,
            "normal_layers": normal_layers,
            "compressed_layers": compressed_layers,
            "compression_type": "feature_dimension_reduction",
        }
    
    return compression_metadata


def plan_vae_model(model: Model, plan: Plan, config: Optional[MosaicConfig] = None) -> Dict[str, Any]:
    """
    Plan BigGAN (VAE) model compression based on node capabilities.
    
    Uses asymmetric compression: channel reduction for generator, different for discriminator.
    
    Args:
        model: Model instance
        plan: Plan with distribution_plan containing node capabilities
        config: Optional MosaicConfig for loading models
        
    Returns:
        Dictionary with compression metadata for each node
    """
    onnx_model = _load_onnx_model(model, config)
    nodes_with_scores = _get_node_capabilities(plan)
    
    if not nodes_with_scores:
        logger.warning(f"No nodes in plan for model {model.name}")
        return {}
    
    # Get score range
    scores = [score for _, score in nodes_with_scores]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    
    # Find generator layers (ConvTranspose) and discriminator layers (Conv)
    generator_layers = []
    discriminator_layers = []
    
    for i, node in enumerate(onnx_model.graph.node):
        if node.op_type == "ConvTranspose":
            generator_layers.append((i, node))
        elif node.op_type == "Conv":
            discriminator_layers.append((i, node))
    
    compression_metadata = {}
    
    for node_dict, capability_score in nodes_with_scores:
        compression_ratio = _calculate_compression_ratio(capability_score, min_score, max_score)
        
        # Generator: channel reduction in middle layers only
        gen_channel_reduction = (1.0 - compression_ratio) * 0.4
        gen_class_embed_reduction = (1.0 - compression_ratio) * 0.3
        
        # Discriminator: filter reduction
        disc_filter_reduction = (1.0 - compression_ratio) * 0.5
        
        # Determine which generator layers to compress (preserve first/last)
        gen_normal_layers = []
        gen_compressed_layers = []
        
        if generator_layers:
            gen_normal_layers.append(0)  # First layer
            if len(generator_layers) > 1:
                gen_normal_layers.append(len(generator_layers) - 1)  # Last layer
            
            # Compress middle layers
            for idx, (layer_idx, _) in enumerate(generator_layers[1:-1], start=1):
                gen_compressed_layers.append((layer_idx, gen_channel_reduction))
        
        # Discriminator layers
        disc_compressed_layers = []
        for idx, (layer_idx, _) in enumerate(discriminator_layers):
            if compression_ratio > 0.5:
                # High capability: compress every other layer
                if idx % 2 == 1:
                    disc_compressed_layers.append((layer_idx, disc_filter_reduction))
            else:
                # Low capability: compress all layers
                disc_compressed_layers.append((layer_idx, disc_filter_reduction))
        
        node_id = f"{node_dict.get('host')}:{node_dict.get('comms_port')}"
        compression_metadata[node_id] = {
            "compression_ratio": compression_ratio,
            "generator_channel_reduction": gen_channel_reduction,
            "generator_class_embed_reduction": gen_class_embed_reduction,
            "discriminator_filter_reduction": disc_filter_reduction,
            "generator_normal_layers": gen_normal_layers,
            "generator_compressed_layers": gen_compressed_layers,
            "discriminator_compressed_layers": disc_compressed_layers,
            "compression_type": "asymmetric_gan_compression",
        }
    
    return compression_metadata


def plan_rl_model(model: Model, plan: Plan, config: Optional[MosaicConfig] = None) -> Dict[str, Any]:
    """
    Plan PPO policy network compression based on node capabilities.
    
    Uses hidden layer neuron reduction while preserving input/output dimensions.
    
    Args:
        model: Model instance
        plan: Plan with distribution_plan containing node capabilities
        config: Optional MosaicConfig for loading models
        
    Returns:
        Dictionary with compression metadata for each node
    """
    onnx_model = _load_onnx_model(model, config)
    nodes_with_scores = _get_node_capabilities(plan)
    
    if not nodes_with_scores:
        logger.warning(f"No nodes in plan for model {model.name}")
        return {}
    
    # Get score range
    scores = [score for _, score in nodes_with_scores]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    
    # Find fully connected layers (Gemm/MatMul)
    fc_layers = []
    for i, node in enumerate(onnx_model.graph.node):
        if node.op_type in ["Gemm", "MatMul"]:
            fc_layers.append((i, node))
    
    compression_metadata = {}
    
    for node_dict, capability_score in nodes_with_scores:
        compression_ratio = _calculate_compression_ratio(capability_score, min_score, max_score)
        
        # Hidden dimension reduction: 0% for high capability, 30-40% for low capability
        hidden_dim_reduction = (1.0 - compression_ratio) * 0.4
        
        # Skip first and last layers (input/output compatibility)
        normal_layers = []
        compressed_layers = []
        
        if len(fc_layers) > 0:
            normal_layers.append(0)  # First layer (input)
        if len(fc_layers) > 1:
            normal_layers.append(len(fc_layers) - 1)  # Last layer (output)
        
        # Compress internal hidden layers
        for idx, (layer_idx, _) in enumerate(fc_layers[1:-1], start=1):
            compressed_layers.append((layer_idx, hidden_dim_reduction))
        
        node_id = f"{node_dict.get('host')}:{node_dict.get('comms_port')}"
        compression_metadata[node_id] = {
            "compression_ratio": compression_ratio,
            "hidden_dim_reduction": hidden_dim_reduction,
            "normal_layers": normal_layers,
            "compressed_layers": compressed_layers,
            "compression_type": "hidden_layer_neuron_reduction",
        }
    
    return compression_metadata


# Mapping from ModelType to planning function
MODEL_PLANNER_MAP = {
    ModelType.CNN: plan_cnn_model,
    ModelType.WAV2VEC: plan_wav2vec_model,
    ModelType.TRANSFORMER: plan_transformer_model,
    ModelType.GNN: plan_gnn_model,
    ModelType.VAE: plan_vae_model,
    ModelType.RL: plan_rl_model,
}


def plan_model(model: Model, plan: Plan, config: Optional[MosaicConfig] = None) -> Dict[str, Any]:
    """
    Plan model compression based on model type and node capabilities.
    
    Args:
        model: Model instance
        plan: Plan with distribution_plan containing node capabilities
        config: Optional MosaicConfig for loading models
        
    Returns:
        Dictionary with compression metadata for each node
        
    Raises:
        ValueError: If model type is not supported or model cannot be loaded
    """
    if model.model_type is None:
        raise ValueError(f"Model {model.name} has no model_type specified")
    
    if model.model_type not in MODEL_PLANNER_MAP:
        raise ValueError(
            f"Model type {model.model_type} is not supported for planning. "
            f"Supported types: {list(MODEL_PLANNER_MAP.keys())}"
        )
    
    planner_func = MODEL_PLANNER_MAP[model.model_type]
    return planner_func(model, plan, config)

