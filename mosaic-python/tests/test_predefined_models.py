"""Unit tests for mosaic_model_runtime.predefined_models module."""

import pytest

from mosaic_model_runtime.predefined_models import (
    create_biggan_onnx,
    create_gcn_onnx,
    create_gpt_neo_onnx,
    create_ppo_onnx,
    create_resnet50_onnx,
    create_resnet101_onnx,
    create_wav2vec2_onnx,
)

@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_resnet50_onnx():
    """Test that create_resnet50_onnx returns a non-None value."""
    result = create_resnet50_onnx()
    assert result is not None

@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_resnet101_onnx():
    """Test that create_resnet101_onnx returns a non-None value."""
    result = create_resnet101_onnx()
    assert result is not None

@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_wav2vec2_onnx():
    """Test that create_wav2vec2_onnx returns a non-None value."""
    result = create_wav2vec2_onnx()
    assert result is not None

@pytest.mark.skip(reason="ntest takes too long; temp skip")
def test_create_gpt_neo_onnx():
    """Test that create_gpt_neo_onnx returns a non-None value."""
    result = create_gpt_neo_onnx()
    assert result is not None

@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_gcn_onnx():
    """Test that create_gcn_onnx returns a non-None value."""
    result = create_gcn_onnx()
    assert result is not None

@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_biggan_onnx():
    """Test that create_biggan_onnx returns a non-None value."""
    result = create_biggan_onnx()
    assert result is not None

@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_ppo_onnx():
    """Test that create_ppo_onnx returns a non-None value."""
    result = create_ppo_onnx()
    assert result is not None

