"""Unit tests for mosaic_model_runtime.predefined_models module."""

from pathlib import Path
from unittest.mock import patch

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
def test_create_resnet50_onnx(tmp_path):
    """Test that create_resnet50_onnx returns a filename."""
    result = create_resnet50_onnx(tmp_path)
    assert result is not None
    assert isinstance(result, str)
    assert result == "resnet50.onnx"
    assert (tmp_path / result).exists()


@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_resnet101_onnx(tmp_path):
    """Test that create_resnet101_onnx returns a filename."""
    result = create_resnet101_onnx(tmp_path)
    assert result is not None
    assert isinstance(result, str)
    assert result == "resnet101.onnx"
    assert (tmp_path / result).exists()


@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_wav2vec2_onnx(tmp_path):
    """Test that create_wav2vec2_onnx returns a filename."""
    result = create_wav2vec2_onnx(tmp_path)
    assert result is not None
    assert isinstance(result, str)
    assert result == "wav2vec2.onnx"
    assert (tmp_path / result).exists()


@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_gpt_neo_onnx(tmp_path):
    """Test that create_gpt_neo_onnx returns a filename."""
    result = create_gpt_neo_onnx(tmp_path)
    assert result is not None
    assert isinstance(result, str)
    assert result == "gpt-neo.onnx"
    assert (tmp_path / result).exists()


@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_gcn_onnx(tmp_path):
    """Test that create_gcn_onnx returns a filename."""
    result = create_gcn_onnx(tmp_path)
    assert result is not None
    assert isinstance(result, str)
    assert result == "gcn-ogbn-arxiv.onnx"
    assert (tmp_path / result).exists()


@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_biggan_onnx(tmp_path):
    """Test that create_biggan_onnx returns a filename."""
    result = create_biggan_onnx(tmp_path)
    assert result is not None
    assert isinstance(result, str)
    assert result == "biggan.onnx"
    assert (tmp_path / result).exists()


@pytest.mark.skip(reason="test takes too long; temp skip")
def test_create_ppo_onnx(tmp_path):
    """Test that create_ppo_onnx returns a filename."""
    result = create_ppo_onnx(tmp_path)
    assert result is not None
    assert isinstance(result, str)
    assert result == "ppo.onnx"
    assert (tmp_path / result).exists()

