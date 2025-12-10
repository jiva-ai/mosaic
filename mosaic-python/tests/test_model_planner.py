"""Unit tests for mosaic_planner.model_planner module."""

from typing import List
from unittest.mock import MagicMock, patch

import onnx
import pytest
from onnx import helper

from mosaic_planner.model_planner import plan_model
from mosaic_config.state import Model, ModelType, Plan


def _create_mock_onnx_model_with_conv_layers(num_layers: int = 10) -> onnx.ModelProto:
    """Create a mock ONNX model with convolutional layers."""
    # Create input
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1000])
    
    # Create nodes (conv layers)
    nodes = []
    for i in range(num_layers):
        if i == 0:
            # First layer
            nodes.append(
                helper.make_node(
                    "Conv",
                    ["input", f"weight_{i}"],
                    [f"conv_{i}"],
                    name=f"conv_{i}",
                )
            )
        elif i == num_layers - 1:
            # Last layer
            nodes.append(
                helper.make_node(
                    "Conv",
                    [f"conv_{i-1}", f"weight_{i}"],
                    ["output"],
                    name=f"conv_{i}",
                )
            )
        else:
            # Middle layers
            nodes.append(
                helper.make_node(
                    "Conv",
                    [f"conv_{i-1}", f"weight_{i}"],
                    [f"conv_{i}"],
                    name=f"conv_{i}",
                )
            )
    
    # Create graph
    graph = helper.make_graph(nodes, "test_graph", [input_tensor], [output_tensor])
    
    # Create model
    model = helper.make_model(graph)
    return model


def _create_mock_onnx_model_with_transformer_layers(num_layers: int = 12) -> onnx.ModelProto:
    """Create a mock ONNX model with transformer layers."""
    input_tensor = helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [1, 10])
    output_tensor = helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 10, 50256])
    
    nodes = []
    for i in range(num_layers):
        nodes.append(
            helper.make_node(
                "MatMul",
                [f"hidden_{i}" if i > 0 else "input_ids", f"weight_{i}"],
                [f"hidden_{i+1}"],
                name=f"transformer_block_{i}",
            )
        )
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [f"hidden_{i+1}"],
                [f"norm_{i}"],
                name=f"layer_norm_{i}",
            )
        )
    
    # Connect last layer to output
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"norm_{num_layers-1}", "output_weight"],
            ["logits"],
            name="output_layer",
        )
    )
    
    graph = helper.make_graph(nodes, "test_transformer_graph", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    return model


def _create_mock_onnx_model_with_gcn_layers(num_layers: int = 3) -> onnx.ModelProto:
    """Create a mock ONNX model with GCN layers."""
    input_tensor = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [100, 128])
    edge_tensor = helper.make_tensor_value_info("edge_index", onnx.TensorProto.INT64, [2, 200])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [100, 40])
    
    nodes = []
    for i in range(num_layers):
        if i == 0:
            nodes.append(
                helper.make_node(
                    "Gemm",
                    ["x", f"weight_{i}"],
                    [f"gcn_{i}"],
                    name=f"gcn_layer_{i}",
                )
            )
        elif i == num_layers - 1:
            nodes.append(
                helper.make_node(
                    "Gemm",
                    [f"gcn_{i-1}", f"weight_{i}"],
                    ["output"],
                    name=f"gcn_layer_{i}",
                )
            )
        else:
            nodes.append(
                helper.make_node(
                    "Gemm",
                    [f"gcn_{i-1}", f"weight_{i}"],
                    [f"gcn_{i}"],
                    name=f"gcn_layer_{i}",
                )
            )
    
    graph = helper.make_graph(nodes, "test_gcn_graph", [input_tensor, edge_tensor], [output_tensor])
    model = helper.make_model(graph)
    return model


def _create_mock_onnx_model_with_gan_layers() -> onnx.ModelProto:
    """Create a mock ONNX model with GAN layers (generator and discriminator)."""
    noise_tensor = helper.make_tensor_value_info("noise", onnx.TensorProto.FLOAT, [1, 128])
    class_tensor = helper.make_tensor_value_info("class_labels", onnx.TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 128, 128])
    
    nodes = []
    # Generator layers (ConvTranspose)
    for i in range(4):
        nodes.append(
            helper.make_node(
                "ConvTranspose",
                [f"gen_{i}" if i > 0 else "noise", f"gen_weight_{i}"],
                [f"gen_{i+1}"],
                name=f"generator_layer_{i}",
            )
        )
    
    # Discriminator layers (Conv)
    for i in range(3):
        nodes.append(
            helper.make_node(
                "Conv",
                [f"disc_{i}" if i > 0 else "gen_4", f"disc_weight_{i}"],
                [f"disc_{i+1}"],
                name=f"discriminator_layer_{i}",
            )
        )
    
    nodes.append(
        helper.make_node(
            "Identity",
            ["disc_3"],
            ["output"],
            name="output_node",
        )
    )
    
    graph = helper.make_graph(nodes, "test_gan_graph", [noise_tensor, class_tensor], [output_tensor])
    model = helper.make_model(graph)
    return model


def _create_mock_onnx_model_with_fc_layers(num_layers: int = 3) -> onnx.ModelProto:
    """Create a mock ONNX model with fully connected layers (for RL)."""
    input_tensor = helper.make_tensor_value_info("observation", onnx.TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("action_logits", onnx.TensorProto.FLOAT, [1, 2])
    
    nodes = []
    for i in range(num_layers):
        if i == 0:
            nodes.append(
                helper.make_node(
                    "Gemm",
                    ["observation", f"weight_{i}"],
                    [f"fc_{i}"],
                    name=f"fc_layer_{i}",
                )
            )
        elif i == num_layers - 1:
            nodes.append(
                helper.make_node(
                    "Gemm",
                    [f"fc_{i-1}", f"weight_{i}"],
                    ["action_logits"],
                    name=f"fc_layer_{i}",
                )
            )
        else:
            nodes.append(
                helper.make_node(
                    "Gemm",
                    [f"fc_{i-1}", f"weight_{i}"],
                    [f"fc_{i}"],
                    name=f"fc_layer_{i}",
                )
            )
    
    graph = helper.make_graph(nodes, "test_fc_graph", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    return model


def _create_plan_with_nodes(capabilities: List[float]) -> Plan:
    """Create a Plan with distribution_plan containing nodes with different capabilities."""
    distribution_plan = []
    for i, capability in enumerate(capabilities):
        distribution_plan.append({
            "host": f"node{i}",
            "comms_port": 5000 + i,
            "heartbeat_port": 6000 + i,
            "effective_score": capability,
            "capacity_score": capability,
        })
    
    model = Model(name="test_model", model_type=ModelType.CNN)
    return Plan(
        stats_data=[],
        distribution_plan=distribution_plan,
        model=model,
    )


@patch("mosaic_planner.model_planner._load_onnx_model")
def test_plan_cnn_model(mock_load_onnx):
    """Test plan_model for CNN model type."""
    # Create mock ONNX model with conv layers
    mock_onnx_model = _create_mock_onnx_model_with_conv_layers(num_layers=10)
    mock_load_onnx.return_value = mock_onnx_model
    
    # Create plan with nodes of different capabilities
    plan = _create_plan_with_nodes([0.9, 0.5, 0.2])  # High, medium, low capability
    plan.model.model_type = ModelType.CNN
    
    # Run planning
    result = plan_model(plan.model, plan)
    
    # Verify structure
    assert isinstance(result, dict)
    assert len(result) == 3  # Three nodes
    
    # Verify each node has compression metadata
    for node_id, metadata in result.items():
        assert "compression_ratio" in metadata
        assert "channel_reduction" in metadata
        assert "normal_layers" in metadata
        assert "compressed_layers" in metadata
        assert "compression_type" in metadata
        assert metadata["compression_type"] == "magnitude_based_channel_pruning"
        
        # Verify compression_ratio is between 0.2 and 0.9
        assert 0.2 <= metadata["compression_ratio"] <= 0.9
        
        # Verify more capable nodes have higher compression_ratio (less compression)
        if "node0" in node_id:  # Highest capability
            assert metadata["compression_ratio"] > 0.6
        elif "node2" in node_id:  # Lowest capability
            assert metadata["compression_ratio"] < 0.5
    
    # Verify we can determine which layers are normal vs compressed for each node
    node0_metadata = result["node0:5000"]
    node2_metadata = result["node2:5002"]
    
    # Higher capability node should have more normal layers
    assert len(node0_metadata["normal_layers"]) >= len(node2_metadata["normal_layers"])


@patch("mosaic_planner.model_planner._load_onnx_model")
def test_plan_wav2vec_model(mock_load_onnx):
    """Test plan_model for Wav2Vec model type."""
    # Create mock ONNX model with transformer layers
    mock_onnx_model = _create_mock_onnx_model_with_transformer_layers(num_layers=12)
    mock_load_onnx.return_value = mock_onnx_model
    
    # Create plan with nodes of different capabilities
    plan = _create_plan_with_nodes([0.9, 0.5, 0.2])
    plan.model.model_type = ModelType.WAV2VEC
    
    # Run planning
    result = plan_model(plan.model, plan)
    
    # Verify structure
    assert isinstance(result, dict)
    assert len(result) == 3
    
    # Verify each node has compression metadata
    for node_id, metadata in result.items():
        assert "compression_ratio" in metadata
        assert "layer_reduction" in metadata
        assert "hidden_dim_reduction" in metadata
        assert "kept_layers" in metadata
        assert "dropped_layers" in metadata
        assert "compression_type" in metadata
        assert metadata["compression_type"] == "layer_dropping_and_dimension_compression"
        
        # Verify more capable nodes keep more layers
        if "node0" in node_id:
            assert len(metadata["kept_layers"]) >= len(result["node2:5002"]["kept_layers"])


@patch("mosaic_planner.model_planner._load_onnx_model")
def test_plan_transformer_model(mock_load_onnx):
    """Test plan_model for Transformer model type."""
    # Create mock ONNX model with transformer layers
    mock_onnx_model = _create_mock_onnx_model_with_transformer_layers(num_layers=36)
    mock_load_onnx.return_value = mock_onnx_model
    
    # Create plan with nodes of different capabilities
    plan = _create_plan_with_nodes([0.9, 0.5, 0.2])
    plan.model.model_type = ModelType.TRANSFORMER
    
    # Run planning
    result = plan_model(plan.model, plan)
    
    # Verify structure
    assert isinstance(result, dict)
    assert len(result) == 3
    
    # Verify each node has compression metadata
    for node_id, metadata in result.items():
        assert "compression_ratio" in metadata
        assert "layer_reduction" in metadata
        assert "hidden_dim_reduction" in metadata
        assert "kept_layers" in metadata
        assert "dropped_layers" in metadata
        assert "compression_type" in metadata
        assert metadata["compression_type"] == "aggressive_layer_removal_and_width_reduction"
        
        # Verify ordering information (kept_layers should be sorted)
        assert metadata["kept_layers"] == sorted(metadata["kept_layers"])


@patch("mosaic_planner.model_planner._load_onnx_model")
def test_plan_gnn_model(mock_load_onnx):
    """Test plan_model for GNN model type."""
    # Create mock ONNX model with GCN layers
    mock_onnx_model = _create_mock_onnx_model_with_gcn_layers(num_layers=3)
    mock_load_onnx.return_value = mock_onnx_model
    
    # Create plan with nodes of different capabilities
    plan = _create_plan_with_nodes([0.9, 0.5, 0.2])
    plan.model.model_type = ModelType.GNN
    
    # Run planning
    result = plan_model(plan.model, plan)
    
    # Verify structure
    assert isinstance(result, dict)
    assert len(result) == 3
    
    # Verify each node has compression metadata
    for node_id, metadata in result.items():
        assert "compression_ratio" in metadata
        assert "hidden_dim_reduction" in metadata
        assert "normal_layers" in metadata
        assert "compressed_layers" in metadata
        assert "compression_type" in metadata
        assert metadata["compression_type"] == "feature_dimension_reduction"
        
        # Verify first layer is always normal
        assert 0 in metadata["normal_layers"]


@patch("mosaic_planner.model_planner._load_onnx_model")
def test_plan_vae_model(mock_load_onnx):
    """Test plan_model for VAE (BigGAN) model type."""
    # Create mock ONNX model with GAN layers
    mock_onnx_model = _create_mock_onnx_model_with_gan_layers()
    mock_load_onnx.return_value = mock_onnx_model
    
    # Create plan with nodes of different capabilities
    plan = _create_plan_with_nodes([0.9, 0.5, 0.2])
    plan.model.model_type = ModelType.VAE
    
    # Run planning
    result = plan_model(plan.model, plan)
    
    # Verify structure
    assert isinstance(result, dict)
    assert len(result) == 3
    
    # Verify each node has compression metadata
    for node_id, metadata in result.items():
        assert "compression_ratio" in metadata
        assert "generator_channel_reduction" in metadata
        assert "generator_class_embed_reduction" in metadata
        assert "discriminator_filter_reduction" in metadata
        assert "generator_normal_layers" in metadata
        assert "generator_compressed_layers" in metadata
        assert "discriminator_compressed_layers" in metadata
        assert "compression_type" in metadata
        assert metadata["compression_type"] == "asymmetric_gan_compression"
        
        # Verify generator first/last layers are preserved
        if metadata["generator_normal_layers"]:
            assert 0 in metadata["generator_normal_layers"]


@patch("mosaic_planner.model_planner._load_onnx_model")
def test_plan_rl_model(mock_load_onnx):
    """Test plan_model for RL (PPO) model type."""
    # Create mock ONNX model with FC layers
    mock_onnx_model = _create_mock_onnx_model_with_fc_layers(num_layers=3)
    mock_load_onnx.return_value = mock_onnx_model
    
    # Create plan with nodes of different capabilities
    plan = _create_plan_with_nodes([0.9, 0.5, 0.2])
    plan.model.model_type = ModelType.RL
    
    # Run planning
    result = plan_model(plan.model, plan)
    
    # Verify structure
    assert isinstance(result, dict)
    assert len(result) == 3
    
    # Verify each node has compression metadata
    for node_id, metadata in result.items():
        assert "compression_ratio" in metadata
        assert "hidden_dim_reduction" in metadata
        assert "normal_layers" in metadata
        assert "compressed_layers" in metadata
        assert "compression_type" in metadata
        assert metadata["compression_type"] == "hidden_layer_neuron_reduction"
        
        # Verify input/output layers are preserved
        assert 0 in metadata["normal_layers"]  # First layer (input)
        assert len(metadata["normal_layers"]) >= 1  # At least input layer


def test_plan_model_invalid_model_type():
    """Test plan_model raises error for unsupported model type."""
    plan = _create_plan_with_nodes([0.5])
    plan.model.model_type = ModelType.BERT  # Not in MODEL_PLANNER_MAP
    
    with pytest.raises(ValueError, match="is not supported for planning"):
        plan_model(plan.model, plan)


def test_plan_model_no_model_type():
    """Test plan_model raises error when model_type is None."""
    plan = _create_plan_with_nodes([0.5])
    plan.model.model_type = None
    
    with pytest.raises(ValueError, match="has no model_type specified"):
        plan_model(plan.model, plan)


def test_plan_model_empty_distribution_plan():
    """Test plan_model handles empty distribution plan gracefully."""
    plan = Plan(
        stats_data=[],
        distribution_plan=[],
        model=Model(name="test_model", model_type=ModelType.CNN),
    )
    
    with patch("mosaic_planner.model_planner._load_onnx_model") as mock_load_onnx:
        mock_onnx_model = _create_mock_onnx_model_with_conv_layers()
        mock_load_onnx.return_value = mock_onnx_model
        
        result = plan_model(plan.model, plan)
        
        # Should return empty dict with warning logged
        assert isinstance(result, dict)
        assert len(result) == 0


@patch("mosaic_planner.model_planner._load_onnx_model")
def test_plan_model_capability_ordering(mock_load_onnx):
    """Test that more capable nodes get less compression."""
    mock_onnx_model = _create_mock_onnx_model_with_conv_layers(num_layers=10)
    mock_load_onnx.return_value = mock_onnx_model
    
    # Create plan with wide range of capabilities
    plan = _create_plan_with_nodes([1.0, 0.8, 0.5, 0.3, 0.1])
    plan.model.model_type = ModelType.CNN
    
    result = plan_model(plan.model, plan)
    
    # Extract compression ratios
    compression_ratios = {
        node_id: metadata["compression_ratio"]
        for node_id, metadata in result.items()
    }
    
    # Verify higher capability = higher compression_ratio (less compression)
    ratios = list(compression_ratios.values())
    assert ratios[0] > ratios[-1], "Highest capability node should have less compression"
    
    # Verify ratios are in descending order (matching capability order)
    for i in range(len(ratios) - 1):
        assert ratios[i] >= ratios[i + 1], f"Node {i} should have >= compression ratio than node {i+1}"


@patch("mosaic_planner.model_planner._load_onnx_model")
def test_plan_model_metadata_structure(mock_load_onnx):
    """Test that returned metadata has all required fields for reconstruction."""
    mock_onnx_model = _create_mock_onnx_model_with_conv_layers(num_layers=10)
    mock_load_onnx.return_value = mock_onnx_model
    
    plan = _create_plan_with_nodes([0.7])
    plan.model.model_type = ModelType.CNN
    
    result = plan_model(plan.model, plan)
    
    # Verify metadata structure
    assert len(result) == 1
    metadata = list(result.values())[0]
    
    # Required fields for CNN
    assert "compression_ratio" in metadata
    assert "channel_reduction" in metadata
    assert "normal_layers" in metadata
    assert "compressed_layers" in metadata
    assert "compression_type" in metadata
    
    # Verify normal_layers is a list of indices
    assert isinstance(metadata["normal_layers"], list)
    assert all(isinstance(layer, int) for layer in metadata["normal_layers"])
    
    # Verify compressed_layers has structure (layer_idx, reduction)
    assert isinstance(metadata["compressed_layers"], list)
    if metadata["compressed_layers"]:
        assert isinstance(metadata["compressed_layers"][0], tuple)
        assert len(metadata["compressed_layers"][0]) == 2

