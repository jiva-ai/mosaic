"""Functions to create various models and convert them to ONNX format."""

from pathlib import Path
from typing import Union

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_resnet50_onnx(output_dir: Union[str, Path]) -> str:
    """
    Create a ResNet-50 model and convert it to ONNX format.
    
    Args:
        output_dir: Directory path where the ONNX file will be saved
        
    Returns:
        str: The filename of the created ONNX file (relative to output_dir)
        
    Raises:
        Exception: If model creation or validation fails
    """
    try:
        import torchvision.models as models
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load the pre-trained ResNet-50 model
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.eval()
        
        # Create dummy input matching the model's input dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Create ONNX file path
        filename = "resnet50.onnx"
        onnx_path = output_path / filename
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        
        # Load and validate the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        return filename
    except Exception as e:
        raise Exception(f"Failed to create ResNet-50 ONNX model: {e}") from e


def create_resnet101_onnx(output_dir: Union[str, Path]) -> str:
    """
    Create a ResNet-101 model and convert it to ONNX format.
    
    Note: ResNet-102 doesn't exist in standard torchvision. This function creates
    ResNet-101, which is the closest standard variant.
    
    Args:
        output_dir: Directory path where the ONNX file will be saved
        
    Returns:
        str: The filename of the created ONNX file (relative to output_dir)
        
    Raises:
        Exception: If model creation or validation fails
    """
    try:
        import torchvision.models as models
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load the pre-trained ResNet-101 model
        model = models.resnet101(weights="IMAGENET1K_V1")
        model.eval()
        
        # Create dummy input matching the model's input dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Create ONNX file path
        filename = "resnet101.onnx"
        onnx_path = output_path / filename
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        
        # Load and validate the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        return filename
    except Exception as e:
        raise Exception(f"Failed to create ResNet-101 ONNX model: {e}") from e


def create_wav2vec2_onnx(output_dir: Union[str, Path]) -> str:
    """
    Create a Wav2Vec2 model and convert it to ONNX format.
    
    Args:
        output_dir: Directory path where the ONNX file will be saved
        
    Returns:
        str: The filename of the created ONNX file (relative to output_dir)
        
    Raises:
        Exception: If model creation or validation fails
    """
    try:
        from transformers import Wav2Vec2Model
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load the pre-trained Wav2Vec2 model
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        model.eval()
        
        # Create dummy input matching the model's input dimensions
        # Wav2Vec2 expects audio waveform input
        dummy_input = torch.randn(1, 16000)
        
        # Create ONNX file path
        filename = "wav2vec2.onnx"
        onnx_path = output_path / filename
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=14,
            input_names=["input_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            },
        )
        
        # Load and validate the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        return filename
    except Exception as e:
        raise Exception(f"Failed to create Wav2Vec2 ONNX model: {e}") from e


def create_gpt_neo_onnx(output_dir: Union[str, Path]) -> str:
    """
    Create a GPT-Neo model and convert it to ONNX format.
    
    Args:
        output_dir: Directory path where the ONNX file will be saved
        
    Returns:
        str: The filename of the created ONNX file (relative to output_dir)
        
    Raises:
        Exception: If model creation or validation fails
    """
    try:
        from transformers import GPTNeoForCausalLM
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load a smaller GPT-Neo model for easier conversion
        # Using 125M version for faster download and conversion
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
        model.eval()
        
        # Create a wrapper that only returns logits (not the full output with cache)
        class GPTNeoWrapper(nn.Module):
            """Wrapper to extract only logits from GPT-Neo output."""
            
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                """Forward pass returning only logits."""
                outputs = self.model(input_ids)
                return outputs.logits
        
        wrapped_model = GPTNeoWrapper(model)
        wrapped_model.eval()
        
        # Create dummy input matching the model's input dimensions
        # GPT-Neo expects token IDs
        dummy_input = torch.randint(0, model.config.vocab_size, (1, 10))
        
        # Create ONNX file path
        filename = "gpt-neo.onnx"
        onnx_path = output_path / filename
        
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(onnx_path),
            opset_version=18,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )
        
        # Load and validate the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        return filename
    except Exception as e:
        raise Exception(f"Failed to create GPT-Neo ONNX model: {e}") from e


def create_gcn_onnx(output_dir: Union[str, Path]) -> str:
    """
    Create a Graph Convolutional Network (GCN) model for ogbn-arxiv
    and convert it to ONNX format.
    
    Args:
        output_dir: Directory path where the ONNX file will be saved
        
    Returns:
        str: The filename of the created ONNX file (relative to output_dir)
        
    Raises:
        Exception: If model creation or validation fails
    """
    try:
        from torch_geometric.nn import GCNConv
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        class GCN(nn.Module):
            """Graph Convolutional Network for node classification."""
            
            def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64):
                super(GCN, self).__init__()
                self.conv1 = GCNConv(num_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, num_classes)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                """
                Forward pass of the GCN.
                
                Args:
                    x: Node feature matrix [num_nodes, num_features]
                    edge_index: Graph connectivity in COO format [2, num_edges]
                    
                Returns:
                    Node embeddings [num_nodes, num_classes]
                """
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return F.log_softmax(x, dim=1)
        
        # ogbn-arxiv has 128 features and 40 classes
        num_features = 128
        num_classes = 40
        model = GCN(num_features, num_classes)
        model.eval()
        
        # Create dummy input matching ogbn-arxiv structure
        # For GCN, we need node features and edge indices
        num_nodes = 100  # Example number of nodes
        num_edges = 200  # Example number of edges
        dummy_x = torch.randn(num_nodes, num_features)
        dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        
        # Create ONNX file path
        filename = "gcn-ogbn-arxiv.onnx"
        onnx_path = output_path / filename
        
        torch.onnx.export(
            model,
            (dummy_x, dummy_edge_index),
            str(onnx_path),
            opset_version=14,
            input_names=["x", "edge_index"],
            output_names=["output"],
            dynamic_axes={
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "output": {0: "num_nodes"},
            },
        )
        
        # Load and validate the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        return filename
    except Exception as e:
        raise Exception(f"Failed to create GCN ONNX model: {e}") from e


def create_biggan_onnx(output_dir: Union[str, Path]) -> str:
    """
    Create a BigGAN generator model and convert it to ONNX format.
    
    This implements a simplified BigGAN generator architecture suitable for ONNX export.
    
    Args:
        output_dir: Directory path where the ONNX file will be saved
        
    Returns:
        str: The filename of the created ONNX file (relative to output_dir)
        
    Raises:
        Exception: If model creation or validation fails
    """
    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        class BigGANGenerator(nn.Module):
            """
            Simplified BigGAN Generator architecture.
            
            BigGAN uses class-conditional batch normalization and self-attention
            for high-quality image generation.
            """
            
            def __init__(self, z_dim: int = 128, num_classes: int = 1000, img_size: int = 128):
                super(BigGANGenerator, self).__init__()
                self.z_dim = z_dim
                self.num_classes = num_classes
                self.img_size = img_size
                
                # Class embedding
                self.class_embed = nn.Embedding(num_classes, z_dim)
                
                # Initial linear layer
                self.linear = nn.Linear(z_dim * 2, 4 * 4 * 512)
                
                # Generator blocks (simplified)
                self.conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
                self.bn1 = nn.BatchNorm2d(256)
                self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
                self.bn2 = nn.BatchNorm2d(128)
                self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
                self.bn3 = nn.BatchNorm2d(64)
                self.conv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
            
            def forward(self, noise: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
                """
                Forward pass of the BigGAN generator.
                
                Args:
                    noise: Random noise vector [batch_size, z_dim]
                    class_labels: Class labels [batch_size]
                    
                Returns:
                    Generated images [batch_size, 3, img_size, img_size]
                """
                # Combine noise with class embedding
                class_emb = self.class_embed(class_labels)
                combined = torch.cat([noise, class_emb], dim=1)
                
                # Initial projection
                x = self.linear(combined)
                x = x.view(-1, 512, 4, 4)
                
                # Generator blocks
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = torch.tanh(self.conv4(x))
                
                return x
        
        # Create BigGAN generator
        z_dim = 128
        num_classes = 1000
        img_size = 128
        model = BigGANGenerator(z_dim, num_classes, img_size)
        model.eval()
        
        # Create dummy inputs: noise vector and class labels
        batch_size = 1
        noise = torch.randn(batch_size, z_dim)
        class_labels = torch.randint(0, num_classes, (batch_size,))
        
        # Create ONNX file path
        filename = "biggan.onnx"
        onnx_path = output_path / filename
        
        torch.onnx.export(
            model,
            (noise, class_labels),
            str(onnx_path),
            opset_version=14,
            input_names=["noise", "class_labels"],
            output_names=["output"],
            dynamic_axes={
                "noise": {0: "batch_size"},
                "class_labels": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        
        # Load and validate the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        return filename
    except Exception as e:
        raise Exception(f"Failed to create BigGAN ONNX model: {e}") from e


def create_ppo_onnx(output_dir: Union[str, Path]) -> str:
    """
    Create a PPO (Proximal Policy Optimization) policy network
    and convert it to ONNX format.
    
    Args:
        output_dir: Directory path where the ONNX file will be saved
        
    Returns:
        str: The filename of the created ONNX file (relative to output_dir)
        
    Raises:
        Exception: If model creation or validation fails
    """
    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        class PPOPolicy(nn.Module):
            """
            PPO Policy Network.
            
            This represents a typical policy network used in PPO algorithms.
            """
            
            def __init__(self, input_dim: int = 4, output_dim: int = 2, hidden_dim: int = 64):
                super(PPOPolicy, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass of the policy network.
                
                Args:
                    x: Observation tensor [batch_size, input_dim]
                    
                Returns:
                    Action logits [batch_size, output_dim]
                """
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        
        # Create a PPO policy network
        # Default dimensions suitable for common RL environments like CartPole
        input_dim = 4
        output_dim = 2
        model = PPOPolicy(input_dim, output_dim)
        model.eval()
        
        # Create dummy input matching typical observation space
        dummy_input = torch.randn(1, input_dim)
        
        # Create ONNX file path
        filename = "ppo.onnx"
        onnx_path = output_path / filename
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=14,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
        )
        
        # Load and validate the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        return filename
    except Exception as e:
        raise Exception(f"Failed to create PPO ONNX model: {e}") from e

