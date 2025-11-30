"""Unit tests for mosaic_model_runtime.model_factory module."""

from unittest.mock import patch

import pytest

from mosaic_planner.state import ModelType

from mosaic_model_runtime.model_factory import (
    create_biggan_model,
    create_gcn_model,
    create_gpt_neo_model,
    create_ppo_model,
    create_resnet50_model,
    create_resnet101_model,
    create_wav2vec2_model,
)


@pytest.fixture
def mock_onnx_bytes():
    """Mock ONNX model bytes."""
    return b"mock_onnx_model_bytes"


@patch("mosaic_model_runtime.model_factory.create_resnet50_onnx")
def test_create_resnet50_model_default_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_resnet50_model with default name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_resnet50_model()
    
    assert model.name == "resnet50"
    assert model.model_type == ModelType.CNN
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "resnet50.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_resnet50_onnx")
def test_create_resnet50_model_custom_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_resnet50_model with custom name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_resnet50_model(name="custom_resnet50")
    
    assert model.name == "custom_resnet50"
    assert model.model_type == ModelType.CNN
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "resnet50.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_resnet101_onnx")
def test_create_resnet101_model_default_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_resnet101_model with default name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_resnet101_model()
    
    assert model.name == "resnet101"
    assert model.model_type == ModelType.CNN
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "resnet101.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_resnet101_onnx")
def test_create_resnet101_model_custom_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_resnet101_model with custom name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_resnet101_model(name="custom_resnet101")
    
    assert model.name == "custom_resnet101"
    assert model.model_type == ModelType.CNN
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "resnet101.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_wav2vec2_onnx")
def test_create_wav2vec2_model_default_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_wav2vec2_model with default name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_wav2vec2_model()
    
    assert model.name == "wav2vec2"
    assert model.model_type == ModelType.WAV2VEC
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "wav2vec2.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_wav2vec2_onnx")
def test_create_wav2vec2_model_custom_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_wav2vec2_model with custom name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_wav2vec2_model(name="custom_wav2vec2")
    
    assert model.name == "custom_wav2vec2"
    assert model.model_type == ModelType.WAV2VEC
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "wav2vec2.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_gpt_neo_onnx")
def test_create_gpt_neo_model_default_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_gpt_neo_model with default name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_gpt_neo_model()
    
    assert model.name == "gpt-neo"
    assert model.model_type == ModelType.TRANSFORMER
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "gpt-neo.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_gpt_neo_onnx")
def test_create_gpt_neo_model_custom_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_gpt_neo_model with custom name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_gpt_neo_model(name="custom_gpt_neo")
    
    assert model.name == "custom_gpt_neo"
    assert model.model_type == ModelType.TRANSFORMER
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "gpt-neo.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_gcn_onnx")
def test_create_gcn_model_default_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_gcn_model with default name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_gcn_model()
    
    assert model.name == "gcn-ogbn-arxiv"
    assert model.model_type == ModelType.GNN
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "gcn-ogbn-arxiv.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_gcn_onnx")
def test_create_gcn_model_custom_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_gcn_model with custom name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_gcn_model(name="custom_gcn")
    
    assert model.name == "custom_gcn"
    assert model.model_type == ModelType.GNN
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "gcn-ogbn-arxiv.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_biggan_onnx")
def test_create_biggan_model_default_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_biggan_model with default name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_biggan_model()
    
    assert model.name == "biggan"
    assert model.model_type == ModelType.VAE
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "biggan.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_biggan_onnx")
def test_create_biggan_model_custom_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_biggan_model with custom name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_biggan_model(name="custom_biggan")
    
    assert model.name == "custom_biggan"
    assert model.model_type == ModelType.VAE
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "biggan.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_ppo_onnx")
def test_create_ppo_model_default_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_ppo_model with default name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_ppo_model()
    
    assert model.name == "ppo"
    assert model.model_type == ModelType.RL
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "ppo.onnx"
    mock_create_onnx.assert_called_once()


@patch("mosaic_model_runtime.model_factory.create_ppo_onnx")
def test_create_ppo_model_custom_name(mock_create_onnx, mock_onnx_bytes):
    """Test create_ppo_model with custom name."""
    mock_create_onnx.return_value = mock_onnx_bytes
    
    model = create_ppo_model(name="custom_ppo")
    
    assert model.name == "custom_ppo"
    assert model.model_type == ModelType.RL
    assert model.binary_rep == mock_onnx_bytes
    assert model.file_name == "ppo.onnx"
    mock_create_onnx.assert_called_once()

