"""Factory functions to create Model instances from predefined models."""

from typing import Optional

from mosaic_planner.state import Model, ModelType

from mosaic_model_runtime.predefined_models import (
    create_biggan_onnx,
    create_gcn_onnx,
    create_gpt_neo_onnx,
    create_ppo_onnx,
    create_resnet50_onnx,
    create_resnet101_onnx,
    create_wav2vec2_onnx,
)


def create_resnet50_model(name: Optional[str] = None) -> Model:
    """
    Create a ResNet-50 Model instance with ONNX representation.
    
    Args:
        name: Optional name for the model. Defaults to "resnet50".
        
    Returns:
        Model instance with CNN model type and ONNX binary representation
    """
    onnx_bytes = create_resnet50_onnx()
    return Model(
        name=name or "resnet50",
        model_type=ModelType.CNN,
        binary_rep=onnx_bytes,
        file_name="resnet50.onnx",
    )


def create_resnet101_model(name: Optional[str] = None) -> Model:
    """
    Create a ResNet-101 Model instance with ONNX representation.
    
    Args:
        name: Optional name for the model. Defaults to "resnet101".
        
    Returns:
        Model instance with CNN model type and ONNX binary representation
    """
    onnx_bytes = create_resnet101_onnx()
    return Model(
        name=name or "resnet101",
        model_type=ModelType.CNN,
        binary_rep=onnx_bytes,
        file_name="resnet101.onnx",
    )


def create_wav2vec2_model(name: Optional[str] = None) -> Model:
    """
    Create a Wav2Vec2 Model instance with ONNX representation.
    
    Args:
        name: Optional name for the model. Defaults to "wav2vec2".
        
    Returns:
        Model instance with WAV2VEC model type and ONNX binary representation
    """
    onnx_bytes = create_wav2vec2_onnx()
    return Model(
        name=name or "wav2vec2",
        model_type=ModelType.WAV2VEC,
        binary_rep=onnx_bytes,
        file_name="wav2vec2.onnx",
    )


def create_gpt_neo_model(name: Optional[str] = None) -> Model:
    """
    Create a GPT-Neo Model instance with ONNX representation.
    
    Args:
        name: Optional name for the model. Defaults to "gpt-neo".
        
    Returns:
        Model instance with TRANSFORMER model type and ONNX binary representation
    """
    onnx_bytes = create_gpt_neo_onnx()
    return Model(
        name=name or "gpt-neo",
        model_type=ModelType.TRANSFORMER,
        binary_rep=onnx_bytes,
        file_name="gpt-neo.onnx",
    )


def create_gcn_model(name: Optional[str] = None) -> Model:
    """
    Create a GCN (Graph Convolutional Network) Model instance with ONNX representation.
    
    Args:
        name: Optional name for the model. Defaults to "gcn-ogbn-arxiv".
        
    Returns:
        Model instance with GNN model type and ONNX binary representation
    """
    onnx_bytes = create_gcn_onnx()
    return Model(
        name=name or "gcn-ogbn-arxiv",
        model_type=ModelType.GNN,
        binary_rep=onnx_bytes,
        file_name="gcn-ogbn-arxiv.onnx",
    )


def create_biggan_model(name: Optional[str] = None) -> Model:
    """
    Create a BigGAN Model instance with ONNX representation.
    
    Args:
        name: Optional name for the model. Defaults to "biggan".
        
    Returns:
        Model instance with VAE model type and ONNX binary representation
    """
    onnx_bytes = create_biggan_onnx()
    return Model(
        name=name or "biggan",
        model_type=ModelType.VAE,
        binary_rep=onnx_bytes,
        file_name="biggan.onnx",
    )


def create_ppo_model(name: Optional[str] = None) -> Model:
    """
    Create a PPO (Proximal Policy Optimization) Model instance with ONNX representation.
    
    Args:
        name: Optional name for the model. Defaults to "ppo".
        
    Returns:
        Model instance with RL model type and ONNX binary representation
    """
    onnx_bytes = create_ppo_onnx()
    return Model(
        name=name or "ppo",
        model_type=ModelType.RL,
        binary_rep=onnx_bytes,
        file_name="ppo.onnx",
    )

