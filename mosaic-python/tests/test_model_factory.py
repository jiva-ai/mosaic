"""Unit tests for mosaic_model_runtime.model_factory module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mosaic_config.state import ModelType

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
def mock_output_dir(tmp_path):
    """Mock output directory."""
    return tmp_path


@patch("mosaic_model_runtime.model_factory.create_resnet50_onnx")
def test_create_resnet50_model_default_name(mock_create_onnx, mock_output_dir):
    """Test create_resnet50_model with default name."""
    mock_create_onnx.return_value = "resnet50.onnx"
    
    model = create_resnet50_model(mock_output_dir)
    
    assert model.name == "resnet50"
    assert model.model_type == ModelType.CNN
    assert model.onnx_location == "resnet50.onnx"
    assert model.file_name == "resnet50.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_resnet50_onnx")
def test_create_resnet50_model_custom_name(mock_create_onnx, mock_output_dir):
    """Test create_resnet50_model with custom name."""
    mock_create_onnx.return_value = "resnet50.onnx"
    
    model = create_resnet50_model(mock_output_dir, name="custom_resnet50")
    
    assert model.name == "custom_resnet50"
    assert model.model_type == ModelType.CNN
    assert model.onnx_location == "resnet50.onnx"
    assert model.file_name == "resnet50.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_resnet101_onnx")
def test_create_resnet101_model_default_name(mock_create_onnx, mock_output_dir):
    """Test create_resnet101_model with default name."""
    mock_create_onnx.return_value = "resnet101.onnx"
    
    model = create_resnet101_model(mock_output_dir)
    
    assert model.name == "resnet101"
    assert model.model_type == ModelType.CNN
    assert model.onnx_location == "resnet101.onnx"
    assert model.file_name == "resnet101.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_resnet101_onnx")
def test_create_resnet101_model_custom_name(mock_create_onnx, mock_output_dir):
    """Test create_resnet101_model with custom name."""
    mock_create_onnx.return_value = "resnet101.onnx"
    
    model = create_resnet101_model(mock_output_dir, name="custom_resnet101")
    
    assert model.name == "custom_resnet101"
    assert model.model_type == ModelType.CNN
    assert model.onnx_location == "resnet101.onnx"
    assert model.file_name == "resnet101.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_wav2vec2_onnx")
def test_create_wav2vec2_model_default_name(mock_create_onnx, mock_output_dir):
    """Test create_wav2vec2_model with default name."""
    mock_create_onnx.return_value = "wav2vec2.onnx"
    
    model = create_wav2vec2_model(mock_output_dir)
    
    assert model.name == "wav2vec2"
    assert model.model_type == ModelType.WAV2VEC
    assert model.onnx_location == "wav2vec2.onnx"
    assert model.file_name == "wav2vec2.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_wav2vec2_onnx")
def test_create_wav2vec2_model_custom_name(mock_create_onnx, mock_output_dir):
    """Test create_wav2vec2_model with custom name."""
    mock_create_onnx.return_value = "wav2vec2.onnx"
    
    model = create_wav2vec2_model(mock_output_dir, name="custom_wav2vec2")
    
    assert model.name == "custom_wav2vec2"
    assert model.model_type == ModelType.WAV2VEC
    assert model.onnx_location == "wav2vec2.onnx"
    assert model.file_name == "wav2vec2.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_gpt_neo_onnx")
def test_create_gpt_neo_model_default_name(mock_create_onnx, mock_output_dir):
    """Test create_gpt_neo_model with default name."""
    mock_create_onnx.return_value = "gpt-neo.onnx"
    
    model = create_gpt_neo_model(mock_output_dir)
    
    assert model.name == "gpt-neo"
    assert model.model_type == ModelType.TRANSFORMER
    assert model.onnx_location == "gpt-neo.onnx"
    assert model.file_name == "gpt-neo.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_gpt_neo_onnx")
def test_create_gpt_neo_model_custom_name(mock_create_onnx, mock_output_dir):
    """Test create_gpt_neo_model with custom name."""
    mock_create_onnx.return_value = "gpt-neo.onnx"
    
    model = create_gpt_neo_model(mock_output_dir, name="custom_gpt_neo")
    
    assert model.name == "custom_gpt_neo"
    assert model.model_type == ModelType.TRANSFORMER
    assert model.onnx_location == "gpt-neo.onnx"
    assert model.file_name == "gpt-neo.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_gcn_onnx")
def test_create_gcn_model_default_name(mock_create_onnx, mock_output_dir):
    """Test create_gcn_model with default name."""
    mock_create_onnx.return_value = "gcn-ogbn-arxiv.onnx"
    
    model = create_gcn_model(mock_output_dir)
    
    assert model.name == "gcn-ogbn-arxiv"
    assert model.model_type == ModelType.GNN
    assert model.onnx_location == "gcn-ogbn-arxiv.onnx"
    assert model.file_name == "gcn-ogbn-arxiv.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_gcn_onnx")
def test_create_gcn_model_custom_name(mock_create_onnx, mock_output_dir):
    """Test create_gcn_model with custom name."""
    mock_create_onnx.return_value = "gcn-ogbn-arxiv.onnx"
    
    model = create_gcn_model(mock_output_dir, name="custom_gcn")
    
    assert model.name == "custom_gcn"
    assert model.model_type == ModelType.GNN
    assert model.onnx_location == "gcn-ogbn-arxiv.onnx"
    assert model.file_name == "gcn-ogbn-arxiv.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_biggan_onnx")
def test_create_biggan_model_default_name(mock_create_onnx, mock_output_dir):
    """Test create_biggan_model with default name."""
    mock_create_onnx.return_value = "biggan.onnx"
    
    model = create_biggan_model(mock_output_dir)
    
    assert model.name == "biggan"
    assert model.model_type == ModelType.VAE
    assert model.onnx_location == "biggan.onnx"
    assert model.file_name == "biggan.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_biggan_onnx")
def test_create_biggan_model_custom_name(mock_create_onnx, mock_output_dir):
    """Test create_biggan_model with custom name."""
    mock_create_onnx.return_value = "biggan.onnx"
    
    model = create_biggan_model(mock_output_dir, name="custom_biggan")
    
    assert model.name == "custom_biggan"
    assert model.model_type == ModelType.VAE
    assert model.onnx_location == "biggan.onnx"
    assert model.file_name == "biggan.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_ppo_onnx")
def test_create_ppo_model_default_name(mock_create_onnx, mock_output_dir):
    """Test create_ppo_model with default name."""
    mock_create_onnx.return_value = "ppo.onnx"
    
    model = create_ppo_model(mock_output_dir)
    
    assert model.name == "ppo"
    assert model.model_type == ModelType.RL
    assert model.onnx_location == "ppo.onnx"
    assert model.file_name == "ppo.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)


@patch("mosaic_model_runtime.model_factory.create_ppo_onnx")
def test_create_ppo_model_custom_name(mock_create_onnx, mock_output_dir):
    """Test create_ppo_model with custom name."""
    mock_create_onnx.return_value = "ppo.onnx"
    
    model = create_ppo_model(mock_output_dir, name="custom_ppo")
    
    assert model.name == "custom_ppo"
    assert model.model_type == ModelType.RL
    assert model.onnx_location == "ppo.onnx"
    assert model.file_name == "ppo.onnx"
    mock_create_onnx.assert_called_once_with(mock_output_dir)

