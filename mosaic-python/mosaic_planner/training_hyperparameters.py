"""Default hyperparameters for training different model types."""

from typing import Dict, Any

from mosaic_config.state import ModelType

# Default hyperparameters for CNN models (ResNet-50, ResNet-101)
DEFAULT_CNN_HYPERPARAMETERS: Dict[str, Any] = {
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 90,
    "optimizer": "SGD",
    "scheduler": "StepLR",
    "scheduler_step_size": 30,
    "scheduler_gamma": 0.1,
    "loss_function": "CrossEntropyLoss",
    "num_workers": 4,
    "pin_memory": True,
    "dropout": 0.0,  # ResNet doesn't use dropout in main layers
}

# Default hyperparameters for Wav2Vec2 models
DEFAULT_WAV2VEC_HYPERPARAMETERS: Dict[str, Any] = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "Adam",
    "optimizer_betas": (0.9, 0.999),
    "optimizer_eps": 1e-8,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "loss_function": "CTCLoss",
    "num_workers": 4,
    "sample_rate": 16000,
    "audio_max_length": 160000,  # 10 seconds at 16kHz
}

# Default hyperparameters for Transformer models (GPT-Neo)
DEFAULT_TRANSFORMER_HYPERPARAMETERS: Dict[str, Any] = {
    "learning_rate": 1e-4,
    "batch_size": 512,
    "epochs": 3,
    "optimizer": "AdamW",
    "optimizer_betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "max_grad_norm": 1.0,
    "loss_function": "CrossEntropyLoss",
    "num_workers": 4,
    "max_sequence_length": 1024,
    "gradient_accumulation_steps": 1,
    "fp16": False,  # Mixed precision training
}

# Default hyperparameters for GCN models (ogbn-arxiv)
DEFAULT_GNN_HYPERPARAMETERS: Dict[str, Any] = {
    "learning_rate": 0.01,
    "batch_size": 1,  # Often 1 graph per batch
    "epochs": 200,
    "optimizer": "Adam",
    "weight_decay": 5e-4,
    "dropout": 0.5,
    "hidden_dim": 64,
    "num_layers": 2,
    "loss_function": "NLLLoss",
    "num_workers": 0,  # Graph data often doesn't parallelize well
    "early_stopping_patience": 50,
}

# Default hyperparameters for VAE/GAN models (BigGAN)
DEFAULT_VAE_HYPERPARAMETERS: Dict[str, Any] = {
    "learning_rate": 2e-4,
    "batch_size": 2048,
    "epochs": 100,
    "optimizer": "Adam",
    "optimizer_betas": (0.0, 0.999),
    "weight_decay": 0.0,
    "loss_function": "hinge",  # Hinge loss for GAN
    "num_workers": 4,
    "latent_dim": 128,
    "num_classes": 1000,
    "generator_lr_multiplier": 1.0,
    "discriminator_lr_multiplier": 1.0,
    "gradient_penalty_weight": 10.0,
}

# Default hyperparameters for RL models (PPO)
DEFAULT_RL_HYPERPARAMETERS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "epochs": 10,  # PPO epochs per update
    "optimizer": "Adam",
    "optimizer_eps": 1e-5,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "max_grad_norm": 0.5,
    "num_workers": 1,  # RL typically single-threaded
    "rollout_length": 2048,
    "num_minibatches": 32,
}

# Mapping from ModelType to default hyperparameters
DEFAULT_HYPERPARAMETERS_MAP: Dict[ModelType, Dict[str, Any]] = {
    ModelType.CNN: DEFAULT_CNN_HYPERPARAMETERS,
    ModelType.WAV2VEC: DEFAULT_WAV2VEC_HYPERPARAMETERS,
    ModelType.TRANSFORMER: DEFAULT_TRANSFORMER_HYPERPARAMETERS,
    ModelType.GNN: DEFAULT_GNN_HYPERPARAMETERS,
    ModelType.VAE: DEFAULT_VAE_HYPERPARAMETERS,
    ModelType.RL: DEFAULT_RL_HYPERPARAMETERS,
}


def get_default_hyperparameters(model_type: ModelType) -> Dict[str, Any]:
    """
    Get default hyperparameters for a given model type.
    
    Args:
        model_type: ModelType enum value
        
    Returns:
        Dictionary of default hyperparameters (a copy, so modifications don't affect defaults)
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in DEFAULT_HYPERPARAMETERS_MAP:
        raise ValueError(
            f"Model type {model_type} does not have default hyperparameters. "
            f"Supported types: {list(DEFAULT_HYPERPARAMETERS_MAP.keys())}"
        )
    
    # Return a copy so modifications don't affect the defaults
    return DEFAULT_HYPERPARAMETERS_MAP[model_type].copy()

