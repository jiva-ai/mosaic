"""Planning state management classes for Mosaic network."""

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
    RUNNING = "running"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class FileDefinition:
    """Represents a file definition in a Data instance."""

    location: str  # Relative to MosaicConfig's data_folder
    data_type: DataType
    is_segmentable: bool = True
    binary_data: Optional[bytes] = None  # Optional, filled in afterwards


@dataclass
class Data:
    """Represents data to be processed in the Mosaic network."""

    file_definitions: List[FileDefinition] = field(default_factory=list)


@dataclass
class Model:
    """Represents a model in the Mosaic network."""

    name: str  # Must be unique
    model_type: Optional[ModelType] = None
    onnx_location: Optional[str] = None  # Relative to MosaicConfig's models_location
    metadata_location: Optional[str] = None  # Relative to MosaicConfig's models_location


@dataclass
class Plan:
    """Represents a distribution plan for a model and data."""

    stats_data: List[Dict[str, Any]]  # Result of beacon.collect_stats()
    distribution_plan: List[Dict[str, Any]]  # Result of calculate_data_distribution call
    model: Model
    data_segmentation_plan: Optional[List[Dict[str, Any]]] = None  # Plan for data sharding across machines
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier


@dataclass
class Session:
    """Represents a session executing a plan."""

    plan: Plan
    time_started: int = field(init=False)  # Millis since epoch
    time_ended: int = -1  # Millis since epoch, -1 means not finished
    status: str = "idle"  # idle, running, error, complete
    id: str = field(init=False)  # Unique identifier

    def __init__(
        self,
        plan: Plan,
        time_started: Optional[int] = None,
        time_ended: int = -1,
        status: str = "idle",
        id: Optional[str] = None,
    ):
        """
        Initialize Session instance.

        Args:
            plan: Plan instance
            time_started: Optional start time in millis since epoch. Defaults to current time.
            time_ended: End time in millis since epoch. Defaults to -1 (not finished).
            status: Status string. Defaults to "idle".
            id: Optional unique identifier. If not provided, generates a new UUID.
        """
        self.plan = plan
        self.time_started = time_started if time_started is not None else int(time.time() * 1000)
        self.time_ended = time_ended
        self.status = status
        self.id = id if id is not None else str(uuid.uuid4())


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

