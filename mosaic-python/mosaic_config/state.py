"""Planning state management classes for Mosaic network."""

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mosaic_config.config import MosaicConfig


class DataType(Enum):
    """Data type enumeration."""

    DIR = "dir"
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    GRAPH = "graph"
    RL = "rl"
    CSV = "csv"


class ModelType(Enum):
    """Model type enumeration."""

    CNN = "cnn"
    WAV2VEC = "wav2vec"
    TRANSFORMER = "transformer"
    GNN = "gnn"
    RL = "rl"
    VAE = "vae"
    RNN = "rnn"
    LSTM = "lstm"
    BERT = "bert"
    VIT = "vit"
    DIFFUSION = "diffusion"


class SessionStatus(Enum):
    """Session status enumeration."""

    IDLE = "idle"
    TRAINING = "training"
    INFERRING = "inferring"
    ERROR_CORRECTION = "error_correction"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class FileDefinition:
    """
    Represents a file definition in a Data instance.
    
    Training hints fields help understand how to prepare data for model training:
    
    Examples:
        # Image classification (ResNet-50/101):
        input_shape=[3, 224, 224]
        preprocessing_hints={"normalize": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "resize": [224, 224]}
        batch_format="image_tensor"
        label_format="class_index"
        
        # Audio speech recognition (Wav2Vec2):
        input_shape=[16000]  # 1 second at 16kHz
        preprocessing_hints={"sample_rate": 16000, "normalize": True}
        batch_format="audio_waveform"
        label_format="text_sequence"
        
        # Text language modeling (GPT-Neo):
        input_shape=[None]  # Variable length
        preprocessing_hints={"tokenizer": "gpt2", "max_length": 1024, "padding": True}
        batch_format="text_tokens"
        label_format="text_sequence"  # Next token prediction
        
        # Graph node classification (GCN):
        input_shape=[None, 128]  # [num_nodes, num_features]
        preprocessing_hints={"edge_format": "coo", "normalize_adjacency": True}
        batch_format="graph_data"
        label_format="node_labels"
        
        # Image generation (BigGAN):
        input_shape=[3, 128, 128]
        preprocessing_hints={"normalize": True, "range": [-1, 1]}
        batch_format="image_tensor"
        label_format="class_index"  # Class-conditional generation
        
        # Reinforcement learning (PPO):
        input_shape=[4]  # Observation space
        preprocessing_hints={"normalize_obs": True, "clip_rewards": True}
        batch_format="observation_tensor"
        label_format="action_logits"  # Policy output
    """

    location: str  # Relative to MosaicConfig's data_folder
    data_type: DataType
    is_segmentable: bool = True
    binary_data: Optional[bytes] = None  # Optional, filled in afterwards
    # Training hints for data preparation
    input_shape: Optional[List[int]] = None  # Expected input shape, e.g., [3, 224, 224] for images
    preprocessing_hints: Optional[Dict[str, Any]] = None  # Hints for preprocessing (normalization, resize, etc.)
    batch_format: Optional[str] = None  # Batch format hint, e.g., "image_tensor", "audio_waveform", "text_tokens"
    label_format: Optional[str] = None  # Label format hint, e.g., "class_index", "one_hot", "text_sequence"


@dataclass
class Data:
    """
    Represents data to be processed in the Mosaic network.
    
    Training hints fields help understand how to load and batch data for training:
    
    Examples:
        # Image classification:
        training_task_type="classification"
        batch_size_hint=32
        data_loading_hints={"shuffle": True, "num_workers": 4, "pin_memory": True}
        target_format="class_index"
        
        # Language modeling:
        training_task_type="generation"
        batch_size_hint=8
        data_loading_hints={"shuffle": True, "num_workers": 2}
        target_format="next_token_ids"
        
        # Graph learning:
        training_task_type="node_classification"
        batch_size_hint=1  # Often 1 graph per batch
        data_loading_hints={"shuffle": True}
        target_format="node_labels"
        
        # Reinforcement learning:
        training_task_type="reinforcement_learning"
        batch_size_hint=64  # Rollout batch size
        data_loading_hints={"collect_rollouts": True, "gamma": 0.99}
        target_format="action_distribution"
    """

    file_definitions: List[FileDefinition] = field(default_factory=list)
    # Training hints for data loading and batching
    training_task_type: Optional[str] = None  # e.g., "classification", "generation", "regression", "reinforcement_learning"
    batch_size_hint: Optional[int] = None  # Suggested batch size for training
    data_loading_hints: Optional[Dict[str, Any]] = None  # Hints for data loading (shuffle, num_workers, etc.)
    target_format: Optional[str] = None  # Expected target/label format for the model


@dataclass
class Model:
    """Represents a model in the Mosaic network."""

    name: str  # Must be unique
    model_type: Optional[ModelType] = None
    onnx_location: Optional[str] = None  # Relative to MosaicConfig's models_location
    binary_rep: Optional[bytes] = None  # ONNX model loaded from onnx_location
    file_name: Optional[str] = None  # File name of the model (automatically sanitized)
    trained: bool = False  # Whether the model has been trained
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier
    
    def __post_init__(self):
        """Sanitize file_name after initialization if it's set."""
        if self.file_name is not None:
            object.__setattr__(self, 'file_name', self._sanitize_filename(self.file_name))
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Sanitize file_name when it's set after initialization."""
        if name == "file_name" and value is not None:
            value = self._sanitize_filename(value)
        object.__setattr__(self, name, value)
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Sanitize a filename for filesystem compatibility.
        
        Replaces invalid characters (spaces, symbols) with underscores.
        Ensures the filename is valid for filesystems.
        
        Args:
            name: Original filename/name
        
        Returns:
            Sanitized filename safe for filesystems
        """
        # Replace spaces and invalid filename characters with underscore
        # Invalid characters for Unix: / \0 and any control characters
        # Also replace common problematic characters: spaces, < > : " | ? * and symbols
        # Keep only alphanumeric, underscores, hyphens, and dots (dots will be stripped later)
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
        # Remove leading/trailing dots and spaces (converted to underscores)
        sanitized = sanitized.strip('._')
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed"
        # Limit length to reasonable size (255 chars is typical max)
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        return sanitized


@dataclass
class Plan:
    """Represents a distribution plan for a model and data."""

    stats_data: List[Dict[str, Any]]  # Result of beacon.collect_stats()
    distribution_plan: List[Dict[str, Any]]  # Result of calculate_data_distribution call
    model_id: Optional[str] = None  # ID of the model (Model objects are not persisted, loaded lazily)
    data_segmentation_plan: Optional[List[Dict[str, Any]]] = None  # Plan for data sharding across machines
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier
    _model: Optional[Model] = field(default=None, init=False, repr=False)  # Lazy-loaded model
    _model_loader: Optional[Any] = field(default=None, init=False, repr=False)  # Function to load model by ID
    
    def __init__(
        self,
        stats_data: List[Dict[str, Any]],
        distribution_plan: List[Dict[str, Any]],
        model: Optional[Model] = None,
        model_id: Optional[str] = None,
        data_segmentation_plan: Optional[List[Dict[str, Any]]] = None,
        id: Optional[str] = None,
    ):
        """
        Initialize Plan instance.
        
        Args:
            stats_data: Result of beacon.collect_stats()
            distribution_plan: Result of calculate_data_distribution call
            model: Model instance (if provided, model_id will be extracted from it)
            model_id: Model ID (required if model is not provided)
            data_segmentation_plan: Optional plan for data sharding
            id: Optional unique identifier
        """
        self.stats_data = stats_data
        self.distribution_plan = distribution_plan
        self.data_segmentation_plan = data_segmentation_plan
        self.id = id if id is not None else str(uuid.uuid4())
        self._model = None
        self._model_loader = None
        
        if model is not None:
            self._model = model
            self.model_id = model.id
        elif model_id is not None:
            self.model_id = model_id
        else:
            raise ValueError("Either model or model_id must be provided")
    
    @property
    def model(self) -> Optional[Model]:
        """
        Get the model, loading it lazily if necessary.
        
        Returns:
            Model instance if available, None if model_id is invalid
        """
        if self._model is not None:
            return self._model
        
        # Try to load model lazily if loader is available
        if self._model_loader is not None:
            try:
                self._model = self._model_loader(self.model_id)
                return self._model
            except Exception:
                return None
        
        return None
    
    @model.setter
    def model(self, value: Optional[Model]) -> None:
        """Set the model and update model_id."""
        self._model = value
        if value is not None:
            self.model_id = value.id
        else:
            self.model_id = ""
    
    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickle state - exclude Model objects."""
        # Ensure model_id is set from model if model is present but model_id is not
        model_id = self.model_id
        if model_id is None and self._model is not None:
            model_id = self._model.id
        
        state = {
            'stats_data': self.stats_data,
            'distribution_plan': self.distribution_plan,
            'model_id': model_id,
            'data_segmentation_plan': self.data_segmentation_plan,
            'id': self.id,
        }
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom unpickle state - Model objects are not loaded."""
        self.stats_data = state['stats_data']
        self.distribution_plan = state['distribution_plan']
        self.model_id = state['model_id']
        self.data_segmentation_plan = state.get('data_segmentation_plan')
        self.id = state['id']
        self._model = None
        self._model_loader = None


@dataclass
class Session:
    """Represents a session executing a plan."""

    plan: Plan
    data: Optional["Data"] = None
    model_id: Optional[str] = None  # ID of the model (Model objects are not persisted, loaded lazily)
    time_started: int = field(init=False)  # Millis since epoch
    time_ended: int = -1  # Millis since epoch, -1 means not finished
    status: SessionStatus = SessionStatus.IDLE  # Session status
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier
    _model: Optional["Model"] = field(default=None, init=False, repr=False)  # Lazy-loaded model
    _model_loader: Optional[Any] = field(default=None, init=False, repr=False)  # Function to load model by ID

    def __init__(
        self,
        plan: Plan,
        data: Optional["Data"] = None,
        model: Optional["Model"] = None,
        model_id: Optional[str] = None,
        time_started: Optional[int] = None,
        time_ended: int = -1,
        status: SessionStatus = SessionStatus.IDLE,
        id: Optional[str] = None,
    ):
        """
        Initialize Session instance.

        Args:
            plan: Plan instance
            data: Optional Data instance associated with this session
            model: Optional Model instance (if provided, model_id will be extracted from it)
            model_id: Optional Model ID (used if model is not provided)
            time_started: Optional start time in millis since epoch. Defaults to current time.
            time_ended: End time in millis since epoch. Defaults to -1 (not finished).
            status: Session status. Defaults to SessionStatus.IDLE.
            id: Optional unique identifier. If not provided, generates a new UUID.
        """
        self.plan = plan
        self.data = data
        self.time_started = time_started if time_started is not None else int(time.time() * 1000)
        self.time_ended = time_ended
        self.status = status
        self.id = id if id is not None else str(uuid.uuid4())
        self._model = None
        self._model_loader = None
        
        if model is not None:
            self._model = model
            self.model_id = model.id
        elif model_id is not None:
            self.model_id = model_id
        else:
            # If no model provided, try to get model_id from plan
            self.model_id = plan.model_id if hasattr(plan, 'model_id') else None
    
    @property
    def model(self) -> Optional["Model"]:
        """
        Get the model, loading it lazily if necessary.
        
        Returns:
            Model instance if available, None if model_id is invalid
        """
        if self._model is not None:
            return self._model
        
        # Try to load model lazily if loader is available
        if self._model_loader is not None and self.model_id is not None:
            try:
                self._model = self._model_loader(self.model_id)
                return self._model
            except Exception:
                return None
        
        return None
    
    @model.setter
    def model(self, value: Optional["Model"]) -> None:
        """Set the model and update model_id."""
        self._model = value
        if value is not None:
            self.model_id = value.id
        else:
            self.model_id = None
    
    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickle state - exclude Model objects."""
        # Ensure model_id is set from model if model is present but model_id is not
        model_id = self.model_id
        if model_id is None and self._model is not None:
            model_id = self._model.id
        
        state = {
            'plan': self.plan,
            'data': self.data,
            'model_id': model_id,
            'time_started': self.time_started,
            'time_ended': self.time_ended,
            'status': self.status.value if isinstance(self.status, SessionStatus) else self.status,
            'id': self.id,
        }
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom unpickle state - Model objects are not loaded."""
        self.plan = state['plan']
        self.data = state.get('data')
        self.model_id = state.get('model_id')
        self.time_started = state['time_started']
        self.time_ended = state.get('time_ended', -1)
        # Handle status - could be enum value or enum
        status_val = state.get('status', SessionStatus.IDLE.value)
        if isinstance(status_val, str):
            self.status = SessionStatus(status_val)
        else:
            self.status = status_val
        self.id = state['id']
        self._model = None
        self._model_loader = None


@dataclass
class Project:
    """Represents a project in the Mosaic network."""

    name: str  # No spaces allowed, used in commands
    config: MosaicConfig  # Mandatory
    data: Optional["Data"] = None  # Data instance for the project
    plans: List[Plan] = field(default_factory=list)
    _creation_time: int = field(init=False, repr=False)

    def __init__(
        self,
        name: str,
        config: MosaicConfig,
        data: Optional["Data"] = None,
        plans: Optional[List[Plan]] = None,
    ):
        """
        Initialize Project instance.

        Args:
            name: Project name (no spaces allowed)
            config: MosaicConfig instance (mandatory)
            data: Optional Data instance for the project
            plans: Optional list of Plan instances. Defaults to None/empty list.

        Raises:
            ValueError: If name contains spaces
        """
        if " " in name:
            raise ValueError("Project name cannot contain spaces")
        self.name = name
        self.config = config
        self.data = data
        self.plans = plans if plans is not None else []
        # Fix creation time as immutable, millis since epoch
        self._creation_time = int(time.time() * 1000)

    @property
    def creation_time(self) -> int:
        """
        Get the creation time in milliseconds since epoch.

        Returns:
            Creation time in milliseconds since epoch (immutable)
        """
        return self._creation_time


@dataclass
class SendHeartbeatStatus:
    """Status of heartbeats being sent to a peer."""

    host: str
    heartbeat_port: int
    last_time_sent: int = 0  # milliseconds since epoch, 0 means never sent
    connection_status: str = "nascent"  # cert_error, nascent, ok, timeout

    def __post_init__(self):
        """Validate connection status."""
        valid_statuses = {"cert_error", "nascent", "ok", "timeout"}
        if self.connection_status not in valid_statuses:
            raise ValueError(
                f"Invalid connection_status: {self.connection_status}. "
                f"Must be one of {valid_statuses}"
            )


@dataclass
class ReceiveHeartbeatStatus:
    """Status of heartbeats being received from a peer."""

    host: str
    heartbeat_port: int
    comms_port: int = 0  # Comms port for the peer, 0 means not set
    last_time_received: int = 0  # milliseconds since epoch, 0 means never received
    connection_status: str = "nascent"  # nascent, online, stale
    stats_payload: Optional[Dict[str, Any]] = None
    delay: Optional[int] = None  # Delay in nanoseconds (receive_time_ns - send_time_ns)

    def __post_init__(self):
        """Validate connection status."""
        valid_statuses = {"nascent", "online", "stale"}
        if self.connection_status not in valid_statuses:
            raise ValueError(
                f"Invalid connection_status: {self.connection_status}. "
                f"Must be one of {valid_statuses}"
            )
