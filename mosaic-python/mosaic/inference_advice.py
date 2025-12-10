"""Pre-canned advice messages for inference commands based on model and data types."""

from typing import Dict, Optional

from mosaic_config.state import DataType, ModelType


# Advice messages for different model type and data type combinations
INFERENCE_ADVICE: Dict[str, Dict[str, str]] = {
    # CNN + Image
    "cnn_image": (
        "For CNN image models, provide an image file path or image data.\n"
        "Example: infer /path/to/image.jpg\n"
        "Or: infer <base64_encoded_image>\n"
        "Expected input: Image file (JPEG, PNG, etc.) or image tensor\n"
        "Input shape: [batch_size, channels, height, width] (e.g., [1, 3, 224, 224])"
    ),
    
    # CNN + DIR (directory of images)
    "cnn_dir": (
        "For CNN models with directory input, provide a directory path containing images.\n"
        "Example: infer /path/to/images/\n"
        "Expected input: Directory path containing image files\n"
        "The model will process all images in the directory"
    ),
    
    # Wav2Vec + Audio
    "wav2vec_audio": (
        "For Wav2Vec audio models, provide an audio file path or audio waveform data.\n"
        "Example: infer /path/to/audio.wav\n"
        "Or: infer <audio_waveform_array>\n"
        "Expected input: Audio file (WAV, MP3, etc.) or audio waveform\n"
        "Input shape: [batch_size, samples] (e.g., [1, 16000] for 1 second at 16kHz)"
    ),
    
    # Transformer + Text
    "transformer_text": (
        "For Transformer text models, provide text input or token IDs.\n"
        "Example: infer 'Hello, how are you?'\n"
        "Or: infer <token_ids_array>\n"
        "Expected input: Text string or token IDs\n"
        "Input shape: [batch_size, sequence_length] (e.g., [1, 512])"
    ),
    
    # BERT + Text
    "bert_text": (
        "For BERT text models, provide text input or token IDs.\n"
        "Example: infer 'The quick brown fox jumps over the lazy dog'\n"
        "Or: infer <token_ids_array>\n"
        "Expected input: Text string or token IDs\n"
        "Input shape: [batch_size, sequence_length] (e.g., [1, 128])"
    ),
    
    # GNN + Graph
    "gnn_graph": (
        "For GNN graph models, provide graph data (nodes and edges).\n"
        "Example: infer /path/to/graph.json\n"
        "Or: infer <graph_data_dict>\n"
        "Expected input: Graph file (JSON, GraphML, etc.) or graph data structure\n"
        "Input format: {nodes: [...], edges: [...], features: [...]}"
    ),
    
    # RL + RL
    "rl_rl": (
        "For RL models, provide an observation state.\n"
        "Example: infer [0.1, 0.2, 0.3, 0.4]\n"
        "Or: infer /path/to/observation.json\n"
        "Expected input: Observation array or observation file\n"
        "Input shape: [batch_size, observation_dim] (e.g., [1, 4])"
    ),
    
    # VAE + Image
    "vae_image": (
        "For VAE image models, provide an image file path or image data.\n"
        "Example: infer /path/to/image.jpg\n"
        "Or: infer <image_tensor>\n"
        "Expected input: Image file or image tensor\n"
        "Input shape: [batch_size, channels, height, width] (e.g., [1, 3, 128, 128])"
    ),
    
    # RNN + Text
    "rnn_text": (
        "For RNN text models, provide text input or sequence data.\n"
        "Example: infer 'The cat sat on the mat'\n"
        "Or: infer <sequence_array>\n"
        "Expected input: Text string or sequence array\n"
        "Input shape: [batch_size, sequence_length, features]"
    ),
    
    # LSTM + Text
    "lstm_text": (
        "For LSTM text models, provide text input or sequence data.\n"
        "Example: infer 'Once upon a time'\n"
        "Or: infer <sequence_array>\n"
        "Expected input: Text string or sequence array\n"
        "Input shape: [batch_size, sequence_length, features]"
    ),
    
    # ViT + Image
    "vit_image": (
        "For Vision Transformer (ViT) models, provide an image file path or image data.\n"
        "Example: infer /path/to/image.jpg\n"
        "Or: infer <image_tensor>\n"
        "Expected input: Image file or image tensor\n"
        "Input shape: [batch_size, channels, height, width] (e.g., [1, 3, 224, 224])"
    ),
    
    # Diffusion + Image
    "diffusion_image": (
        "For Diffusion image models, provide an image file path, text prompt, or noise seed.\n"
        "Example: infer 'a beautiful sunset over mountains'\n"
        "Or: infer /path/to/image.jpg\n"
        "Expected input: Text prompt, image file, or noise tensor\n"
        "Input shape: Varies based on generation or conditioning mode"
    ),
    
    # Transformer + CSV
    "transformer_csv": (
        "For Transformer models with CSV input, provide a CSV file path or CSV data.\n"
        "Example: infer /path/to/data.csv\n"
        "Or: infer <csv_data_string>\n"
        "Expected input: CSV file path or CSV-formatted string\n"
        "The model will process tabular data from the CSV"
    ),
    
    # CNN + CSV
    "cnn_csv": (
        "For CNN models with CSV input, provide a CSV file path.\n"
        "Example: infer /path/to/data.csv\n"
        "Expected input: CSV file path\n"
        "Note: CSV data will be converted to appropriate format for CNN processing"
    ),
}


def get_inference_advice(model_type: Optional[ModelType], data_type: Optional[DataType]) -> str:
    """
    Get inference advice message based on model type and data type.
    
    Args:
        model_type: Model type enum
        data_type: Data type enum
        
    Returns:
        Advice message string
    """
    if model_type is None or data_type is None:
        return (
            "Please provide input for inference.\n"
            "Input format depends on the model type and data type.\n"
            "Example: infer /path/to/input\n"
            "Or: infer <input_data>"
        )
    
    # Create key from model type and data type
    model_key = model_type.value.lower() if hasattr(model_type, 'value') else str(model_type).lower()
    data_key = data_type.value.lower() if hasattr(data_type, 'value') else str(data_type).lower()
    advice_key = f"{model_key}_{data_key}"
    
    # Try exact match first
    if advice_key in INFERENCE_ADVICE:
        return INFERENCE_ADVICE[advice_key]
    
    # Try model type only (for common patterns)
    model_only_key = f"{model_key}_*"
    # Check if there's a pattern match
    for key, advice in INFERENCE_ADVICE.items():
        if key.startswith(model_key + "_"):
            return advice
    
    # Try data type only
    for key, advice in INFERENCE_ADVICE.items():
        if key.endswith("_" + data_key):
            return advice
    
    # Default generic advice
    return (
        f"For {model_type.value if hasattr(model_type, 'value') else model_type} model "
        f"with {data_type.value if hasattr(data_type, 'value') else data_type} data type:\n"
        f"Provide input data appropriate for the model.\n"
        f"Example: infer /path/to/input\n"
        f"Or: infer <input_data>\n"
        f"Consult model documentation for specific input format requirements."
    )

