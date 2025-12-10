"""Unit tests for mosaic_planner.model_execution module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import onnx
import pytest
import torch
import torch.nn as nn
from onnx import helper

from mosaic_config.config import MosaicConfig
from mosaic_planner.model_execution import (
    _load_onnx_model,
    _train_cnn_model,
    _train_gnn_model,
    _train_rl_model,
    _train_transformer_model,
    _train_vae_model,
    _train_wav2vec_model,
)
from mosaic_planner.model_planner import _calculate_compression_ratio, _get_node_capabilities
from mosaic_config.state import Data, DataType, FileDefinition, Model, ModelType, Plan


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary MosaicConfig for testing."""
    return MosaicConfig(
        models_location=str(tmp_path / "models"),
        data_location=str(tmp_path / "data"),
    )


@pytest.fixture
def small_cnn_model():
    """Create a very small CNN model for fast testing."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 10),
    )
    return model


@pytest.fixture
def small_wav2vec_model():
    """Create a very small Wav2Vec2-like model for fast testing."""
    model = nn.Sequential(
        nn.Conv1d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(10),
        nn.Flatten(),
        nn.Linear(40, 32),
    )
    return model


@pytest.fixture
def small_transformer_model():
    """Create a very small transformer model for fast testing."""
    class SmallTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 8)
            self.linear = nn.Linear(8, 100)  # vocab_size=100
        
        def forward(self, input_ids):
            # Simple embedding + linear projection
            # Input: (batch, seq_len), Output: (batch, seq_len, vocab_size)
            x = self.embedding(input_ids)  # (batch, seq_len, 8)
            # Apply linear layer to each position
            batch_size, seq_len, embed_dim = x.shape
            x = x.view(-1, embed_dim)  # (batch * seq_len, 8)
            x = self.linear(x)  # (batch * seq_len, vocab_size)
            x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, vocab_size)
            return x
    
    return SmallTransformer()


@pytest.fixture
def small_gnn_model():
    """Create a very small GCN model for fast testing."""
    from torch_geometric.nn import GCNConv

    class SmallGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(4, 8)
            self.conv2 = GCNConv(8, 5)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            return torch.log_softmax(x, dim=1)

    return SmallGCN()


@pytest.fixture
def small_vae_model():
    """Create a very small GAN/VAE model for fast testing."""
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 3 * 4 * 4),  # Small image
        nn.Tanh(),
    )

    class VAEWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.class_embed = nn.Embedding(10, 8)

        def forward(self, noise, class_labels):
            class_emb = self.class_embed(class_labels)
            combined = torch.cat([noise, class_emb], dim=1)
            return self.base(combined).view(-1, 3, 4, 4)

    return VAEWrapper(model)


@pytest.fixture
def small_rl_model():
    """Create a very small RL policy model for fast testing."""
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    return model


@pytest.fixture
def sample_data(tmp_path):
    """Create sample data for testing."""
    # Create a dummy data directory with multiple files to ensure dataset has samples
    data_dir = tmp_path / "data" / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple dummy files so dataset has samples
    for i in range(5):
        (data_dir / f"dummy_{i}.txt").write_text("test")
    
    file_def = FileDefinition(
        location="test_data",
        data_type=DataType.TEXT,
        input_shape=[3, 224, 224],
    )
    
    return Data(
        file_definitions=[file_def],
        batch_size_hint=2,
        data_loading_hints={"shuffle": False, "num_workers": 0},
    )


def test_train_cnn_model(small_cnn_model, sample_data, tmp_config):
    """Test _train_cnn_model completes successfully."""
    fast_hyperparams = {
        "learning_rate": 0.01,
        "batch_size": 2,
        "epochs": 1,
        "optimizer": "SGD",
        "loss_function": "CrossEntropyLoss",
        "num_workers": 0,
    }
    
    # Update input_shape to match model (smaller for faster testing)
    sample_data.file_definitions[0].input_shape = [3, 32, 32]
    
    trained_model = _train_cnn_model(
        small_cnn_model,
        sample_data,
        tmp_config,
        epochs=1,
        hyperparameters=fast_hyperparams,
    )
    
    assert trained_model is not None
    assert isinstance(trained_model, nn.Module)

@pytest.mark.skip(reason="TODO fix later")
def test_train_wav2vec_model(small_wav2vec_model, sample_data, tmp_config):
    """Test _train_wav2vec_model completes successfully."""
    fast_hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 2,
        "epochs": 1,
        "optimizer": "Adam",
        "num_workers": 0,
        "sample_rate": 16000,
        "audio_max_length": 100,  # Very short
    }
    
    trained_model = _train_wav2vec_model(
        small_wav2vec_model,
        sample_data,
        tmp_config,
        epochs=1,
        hyperparameters=fast_hyperparams,
    )
    
    assert trained_model is not None
    assert isinstance(trained_model, nn.Module)


def test_train_transformer_model(small_transformer_model, sample_data, tmp_config):
    """Test _train_transformer_model completes successfully."""
    fast_hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 2,
        "epochs": 1,
        "optimizer": "AdamW",
        "num_workers": 0,
        "max_sequence_length": 10,  # Very short
    }
    
    trained_model = _train_transformer_model(
        small_transformer_model,
        sample_data,
        tmp_config,
        epochs=1,
        hyperparameters=fast_hyperparams,
    )
    
    assert trained_model is not None
    assert isinstance(trained_model, nn.Module)


def test_train_gnn_model(small_gnn_model, sample_data, tmp_config):
    """Test _train_gnn_model completes successfully."""
    fast_hyperparams = {
        "learning_rate": 0.01,
        "batch_size": 1,
        "epochs": 1,
        "optimizer": "Adam",
        "num_workers": 0,
    }
    
    trained_model = _train_gnn_model(
        small_gnn_model,
        sample_data,
        tmp_config,
        epochs=1,
        hyperparameters=fast_hyperparams,
    )
    
    assert trained_model is not None
    assert isinstance(trained_model, nn.Module)


@pytest.mark.skip(reason="TODO fix later")
def test_train_vae_model(small_vae_model, sample_data, tmp_config):
    """Test _train_vae_model completes successfully."""
    fast_hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 2,
        "epochs": 1,
        "optimizer": "Adam",
        "num_workers": 0,
        "latent_dim": 4,  # Very small
        "num_classes": 10,
    }
    
    trained_model = _train_vae_model(
        small_vae_model,
        sample_data,
        tmp_config,
        epochs=1,
        hyperparameters=fast_hyperparams,
    )
    
    assert trained_model is not None
    assert isinstance(trained_model, nn.Module)


def test_train_rl_model(small_rl_model, sample_data, tmp_config):
    """Test _train_rl_model completes successfully."""
    fast_hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 2,
        "epochs": 1,
        "optimizer": "Adam",
        "num_workers": 0,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
    }
    
    trained_model = _train_rl_model(
        small_rl_model,
        sample_data,
        tmp_config,
        epochs=1,
        hyperparameters=fast_hyperparams,
    )
    
    assert trained_model is not None
    assert isinstance(trained_model, nn.Module)


def _create_simple_onnx_model() -> onnx.ModelProto:
    """Create a simple ONNX model for testing."""
    # Create input
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10])
    
    # Create a simple node
    node = helper.make_node(
        "Flatten",
        ["input"],
        ["flatten_output"],
        name="flatten",
    )
    
    # Create graph
    graph = helper.make_graph(
        [node],
        "test_graph",
        [input_tensor],
        [output_tensor],
    )
    
    # Create model
    model = helper.make_model(graph, producer_name="test")
    return model


def test_load_onnx_model_from_file(tmp_path, tmp_config):
    """Test _load_onnx_model loads correctly from file."""
    # Create a simple ONNX model
    onnx_model = _create_simple_onnx_model()
    
    # Save to file
    model_dir = tmp_path / "models" / "test_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "test.onnx"
    onnx.save(onnx_model, str(model_file))
    
    # Create model with file path
    model = Model(
        name="test_model",
        model_type=ModelType.CNN,
        onnx_location="test_model",
        file_name="test.onnx",
    )
    
    # Load model
    loaded_model = _load_onnx_model(model, tmp_config)
    
    assert loaded_model is not None
    assert isinstance(loaded_model, onnx.ModelProto)
    assert len(loaded_model.graph.node) == 1
    assert loaded_model.graph.node[0].name == "flatten"


def test_load_onnx_model_from_binary_rep(tmp_config):
    """Test _load_onnx_model loads correctly from binary_rep."""
    # Create a simple ONNX model
    onnx_model = _create_simple_onnx_model()
    
    # Convert to bytes (simulate reading from file)
    buffer = io.BytesIO()
    onnx.save(onnx_model, buffer)
    binary_data = buffer.getvalue()
    
    # Create model with binary_rep
    model = Model(
        name="test_model",
        model_type=ModelType.CNN,
        binary_rep=binary_data,
    )
    
    # Load model
    loaded_model = _load_onnx_model(model, tmp_config)
    
    assert loaded_model is not None
    assert isinstance(loaded_model, onnx.ModelProto)
    assert len(loaded_model.graph.node) == 1
    assert loaded_model.graph.node[0].name == "flatten"


def test_load_onnx_model_raises_when_missing():
    """Test _load_onnx_model raises ValueError when both binary_rep and file are missing."""
    model = Model(
        name="test_model",
        model_type=ModelType.CNN,
        # No binary_rep, onnx_location, or file_name
    )
    
    with pytest.raises(ValueError, match="Cannot load model"):
        _load_onnx_model(model)


def test_get_node_capabilities():
    """Test _get_node_capabilities extracts and sorts node capabilities correctly."""
    # Create a plan with distribution_plan
    distribution_plan = [
        {"node_id": "node1", "capacity_score": 0.3},
        {"node_id": "node2", "capacity_score": 0.8},
        {"node_id": "node3", "capacity_score": 0.5, "effective_score": 0.6},
        {"node_id": "node4", "capacity_score": 0.1},
    ]
    
    plan = Plan(
        stats_data=[],
        distribution_plan=distribution_plan,
        model=Model(name="test", model_type=ModelType.CNN),
    )
    
    nodes_with_scores = _get_node_capabilities(plan)
    
    # Should be sorted by score (highest first)
    assert len(nodes_with_scores) == 4
    
    # Extract scores and node IDs
    scores = [score for _, score in nodes_with_scores]
    node_ids = [node["node_id"] for node, _ in nodes_with_scores]
    
    # Should be sorted descending by score
    assert scores == sorted(scores, reverse=True)
    
    # Check specific scores
    assert 0.6 in scores  # node3 with effective_score
    assert 0.8 in scores  # node2
    assert 0.3 in scores  # node1
    assert 0.1 in scores  # node4
    
    # Check that node3 uses effective_score (0.6) not capacity_score (0.5)
    node3_idx = node_ids.index("node3")
    assert scores[node3_idx] == 0.6
    
    # Check that highest score is first
    assert scores[0] == 0.8  # node2
    assert node_ids[0] == "node2"


def test_get_node_capabilities_empty_plan():
    """Test _get_node_capabilities handles empty plan."""
    plan = Plan(
        stats_data=[],
        distribution_plan=[],
        model=Model(name="test", model_type=ModelType.CNN),
    )
    
    nodes_with_scores = _get_node_capabilities(plan)
    assert nodes_with_scores == []


def test_calculate_compression_ratio():
    """Test _calculate_compression_ratio calculates correctly."""
    # Test normal case
    ratio = _calculate_compression_ratio(capability_score=0.5, min_score=0.0, max_score=1.0)
    assert 0.2 <= ratio <= 0.9  # Should be in bounds
    assert ratio == pytest.approx(0.55)  # 0.2 + (0.5 * 0.7)
    
    # Test high capability (less compression)
    ratio_high = _calculate_compression_ratio(capability_score=1.0, min_score=0.0, max_score=1.0)
    assert ratio_high == pytest.approx(0.9)  # 0.2 + (1.0 * 0.7)
    
    # Test low capability (more compression)
    ratio_low = _calculate_compression_ratio(capability_score=0.0, min_score=0.0, max_score=1.0)
    assert ratio_low == pytest.approx(0.2)  # 0.2 + (0.0 * 0.7)
    
    # Test middle capability
    ratio_mid = _calculate_compression_ratio(capability_score=0.5, min_score=0.0, max_score=1.0)
    assert ratio_mid == pytest.approx(0.55)


def test_calculate_compression_ratio_same_scores():
    """Test _calculate_compression_ratio handles equal min/max scores."""
    ratio = _calculate_compression_ratio(capability_score=0.5, min_score=0.5, max_score=0.5)
    assert ratio == 0.5  # Default to moderate compression


def test_calculate_compression_ratio_custom_range():
    """Test _calculate_compression_ratio with custom score range."""
    # Scores from 10 to 100
    ratio = _calculate_compression_ratio(capability_score=55, min_score=10, max_score=100)
    # Normalized: (55-10)/(100-10) = 45/90 = 0.5
    # Ratio: 0.2 + (0.5 * 0.7) = 0.55
    assert ratio == pytest.approx(0.55)

