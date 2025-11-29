"""Unit tests for data segmentation in plan_data_distribution."""

import csv
import json
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np

from mosaic_config.config import MosaicConfig
from mosaic_planner.planner import plan_data_distribution
from mosaic_planner.state import Data, DataType, FileDefinition, Model, Project

from tests.create_dummy_data import (
    create_dummy_csv,
    create_dummy_images,
    create_dummy_librispeech,
    create_dummy_c4_pile,
    create_dummy_ogbn_arxiv,
    create_dummy_mujoco_halfcheetah,
)


def _count_csv_rows(csv_path: Path) -> int:
    """Count rows in a CSV file (excluding header)."""
    if not csv_path.exists():
        return 0
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        return sum(1 for _ in reader)


def _count_images_in_dir(dir_path: Path) -> int:
    """Count image files in a directory (recursively)."""
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    count = 0
    for item in dir_path.rglob('*.png'):
        if item.is_file():
            count += 1
    return count


def _count_audio_files(dir_path: Path) -> int:
    """Count audio files in a directory (recursively)."""
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    count = 0
    for ext in ['*.flac', '*.wav', '*.mp3']:
        for item in dir_path.rglob(ext):
            if item.is_file():
                count += 1
    return count


def _count_text_chars(file_path: Path) -> int:
    """Count characters in a text file or sum across JSONL files."""
    if not file_path.exists():
        return 0
    if file_path.is_file():
        if file_path.suffix == '.jsonl':
            # Count characters across all lines in JSONL
            total = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    total += len(line)
            return total
        else:
            # Regular text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.read())
    elif file_path.is_dir():
        # Sum characters across all JSONL files in directory
        total = 0
        for jsonl_file in file_path.rglob('*.jsonl'):
            if jsonl_file.is_file():
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        total += len(line)
        return total
    return 0


def _count_graph_nodes(ogbn_dir: Path) -> int:
    """Count nodes in ogbn-arxiv graph (from node-label.csv)."""
    node_label_file = ogbn_dir / 'node-label.csv'
    if not node_label_file.exists():
        return 0
    with open(node_label_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        return sum(1 for _ in reader)


def _count_rl_episodes(mujoco_dir: Path) -> int:
    """Count episodes in Mujoco trajectory file."""
    trajectories_file = mujoco_dir / 'trajectories.npz'
    if not trajectories_file.exists():
        return 0
    data = np.load(trajectories_file)
    if 'episode_starts' in data:
        return len(data['episode_starts']) - 1  # episode_starts has one extra entry
    return 0


def _count_dir_files(dir_path: Path) -> int:
    """Count files in a directory (non-recursive)."""
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    files = [f for f in dir_path.iterdir() if f.is_file()]
    # If no files, count subdirectories instead
    if len(files) == 0:
        return len([d for d in dir_path.iterdir() if d.is_dir()])
    return len(files)


def _get_file_metadata_for_test(file_def: FileDefinition, data_folder: str) -> dict:
    """Get actual metadata for test files."""
    full_path = Path(data_folder) / file_def.location
    metadata = {}
    
    if file_def.data_type == DataType.CSV:
        metadata["total_rows"] = _count_csv_rows(full_path)
    elif file_def.data_type == DataType.IMAGE:
        metadata["total_images"] = _count_images_in_dir(full_path)
    elif file_def.data_type == DataType.AUDIO:
        metadata["total_files"] = _count_audio_files(full_path)
    elif file_def.data_type == DataType.TEXT:
        metadata["total_chars"] = _count_text_chars(full_path)
    elif file_def.data_type == DataType.GRAPH:
        metadata["total_nodes"] = _count_graph_nodes(full_path)
    elif file_def.data_type == DataType.RL:
        metadata["total_episodes"] = _count_rl_episodes(full_path)
    elif file_def.data_type == DataType.DIR:
        metadata["total_files"] = _count_dir_files(full_path)
    
    return metadata


def _create_mock_distribution_plan(fractions: List[float]) -> List[dict]:
    """Create a mock distribution plan with specified fractions.
    
    Args:
        fractions: List of fractions for each machine (e.g., [0.0, 0.1, 0.5, 0.2, 0.2])
    
    Returns:
        List of distribution plan dictionaries
    """
    plan = []
    base_port = 5001
    for i, fraction in enumerate(fractions):
        plan.append({
            "host": f"node{i+1}",
            "heartbeat_port": 5000 + i,
            "comms_port": base_port + i,
            "capacity_fraction": fraction,
            "effective_score": fraction * 10.0,  # Dummy effective score
        })
    return plan


def test_segment_csv_data():
    """Test CSV data segmentation with 5 machines: 0%, 10%, 50%, 20%, 20%."""
    # Create dummy CSV data
    create_dummy_csv()
    
    # Get test data directory
    test_data_dir = Path(__file__).parent / 'test_data'
    csv_path = test_data_dir / 'dummy_tabular.csv'
    
    # Count actual rows
    total_rows = _count_csv_rows(csv_path)
    assert total_rows > 0, "CSV file should have rows"
    
    # Create project with CSV data
    config = MosaicConfig(data_location=str(test_data_dir))
    file_def = FileDefinition(
        location='dummy_tabular.csv',
        data_type=DataType.CSV,
        is_segmentable=True,
    )
    data = Data(file_definitions=[file_def])
    project = Project(name="test_csv", config=config, data=data)
    
    # Create mock distribution plan: 0%, 10%, 50%, 20%, 20%
    distribution_plan = _create_mock_distribution_plan([0.0, 0.1, 0.5, 0.2, 0.2])
    stats_data = []  # Not used in segmentation
    model = Model(name="test_model")
    
    # Patch _get_file_metadata to use our test version
    with patch('mosaic_planner.planner._get_file_metadata', _get_file_metadata_for_test):
        plan = plan_data_distribution(
            distribution_plan=distribution_plan,
            project=project,
            stats_data=stats_data,
            model=model,
        )
    
    # Verify segmentation plan
    assert plan.data_segmentation_plan is not None
    assert len(plan.data_segmentation_plan) == 4  # 0% machine excluded
    
    # Check allocations
    allocations = {}
    for machine_plan in plan.data_segmentation_plan:
        host = machine_plan["host"]
        fraction = machine_plan["fraction"]
        segments = machine_plan["segments"]
        assert len(segments) == 1
        
        segment = segments[0]
        assert segment["data_type"] == "csv"
        assert segment["is_segmentable"] is True
        
        start_row = segment["start_row"]
        end_row = segment["end_row"]
        allocated_rows = end_row - start_row
        
        allocations[host] = {
            "fraction": fraction,
            "allocated_rows": allocated_rows,
            "expected_rows": int(total_rows * fraction),
        }
    
    # Verify 0% machine got nothing
    node1_plan = next((m for m in plan.data_segmentation_plan if m["host"] == "node1"), None)
    if node1_plan:
        assert node1_plan["fraction"] == 0.0
    
    # Verify other machines got correct allocations (within rounding tolerance)
    node2 = allocations.get("node2", {})
    node3 = allocations.get("node3", {})
    node4 = allocations.get("node4", {})
    node5 = allocations.get("node5", {})
    
    # Allow ±1 row tolerance for rounding
    assert abs(node2["allocated_rows"] - node2["expected_rows"]) <= 1
    assert abs(node3["allocated_rows"] - node3["expected_rows"]) <= 1
    assert abs(node4["allocated_rows"] - node4["expected_rows"]) <= 1
    assert abs(node5["allocated_rows"] - node5["expected_rows"]) <= 1
    
    # Verify total allocated rows equals total rows
    total_allocated = sum(a["allocated_rows"] for a in allocations.values())
    assert abs(total_allocated - total_rows) <= 2  # Allow small rounding error


def test_segment_image_data():
    """Test image data segmentation with 5 machines: 0%, 10%, 50%, 20%, 20%."""
    # Create dummy image data
    create_dummy_images()
    
    # Get test data directory
    test_data_dir = Path(__file__).parent / 'test_data'
    images_dir = test_data_dir / 'dummy_images'
    
    # Count actual images
    total_images = _count_images_in_dir(images_dir)
    assert total_images > 0, "Should have images"
    
    # Create project with image data
    config = MosaicConfig(data_location=str(test_data_dir))
    file_def = FileDefinition(
        location='dummy_images',
        data_type=DataType.IMAGE,
        is_segmentable=True,
    )
    data = Data(file_definitions=[file_def])
    project = Project(name="test_images", config=config, data=data)
    
    # Create mock distribution plan: 0%, 10%, 50%, 20%, 20%
    distribution_plan = _create_mock_distribution_plan([0.0, 0.1, 0.5, 0.2, 0.2])
    stats_data = []
    model = Model(name="test_model")
    
    with patch('mosaic_planner.planner._get_file_metadata', _get_file_metadata_for_test):
        plan = plan_data_distribution(
            distribution_plan=distribution_plan,
            project=project,
            stats_data=stats_data,
            model=model,
        )
    
    # Verify segmentation plan
    assert plan.data_segmentation_plan is not None
    assert len(plan.data_segmentation_plan) == 4  # 0% machine excluded
    
    # Check allocations
    allocations = {}
    for machine_plan in plan.data_segmentation_plan:
        host = machine_plan["host"]
        fraction = machine_plan["fraction"]
        segments = machine_plan["segments"]
        assert len(segments) == 1
        
        segment = segments[0]
        assert segment["data_type"] == "image"
        assert segment["is_segmentable"] is True
        
        image_indices = segment["image_indices"]
        allocated_count = len(image_indices)
        
        allocations[host] = {
            "fraction": fraction,
            "allocated_count": allocated_count,
            "expected_count": int(total_images * fraction),
        }
    
    # Verify allocations (allow ±1 for rounding)
    node2 = allocations.get("node2", {})
    node3 = allocations.get("node3", {})
    node4 = allocations.get("node4", {})
    node5 = allocations.get("node5", {})
    
    assert abs(node2["allocated_count"] - node2["expected_count"]) <= 1
    assert abs(node3["allocated_count"] - node3["expected_count"]) <= 1
    assert abs(node4["allocated_count"] - node4["expected_count"]) <= 1
    assert abs(node5["allocated_count"] - node5["expected_count"]) <= 1
    
    # Verify total allocated
    total_allocated = sum(a["allocated_count"] for a in allocations.values())
    assert abs(total_allocated - total_images) <= 2


def test_segment_audio_data():
    """Test audio data segmentation with 5 machines: 0%, 10%, 50%, 20%, 20%."""
    # Create dummy audio data
    create_dummy_librispeech()
    
    # Get test data directory
    test_data_dir = Path(__file__).parent / 'test_data'
    audio_dir = test_data_dir / 'LibriSpeech'
    
    # Count actual audio files
    total_files = _count_audio_files(audio_dir)
    assert total_files > 0, "Should have audio files"
    
    # Create project with audio data
    config = MosaicConfig(data_location=str(test_data_dir))
    file_def = FileDefinition(
        location='LibriSpeech',
        data_type=DataType.AUDIO,
        is_segmentable=True,
    )
    data = Data(file_definitions=[file_def])
    project = Project(name="test_audio", config=config, data=data)
    
    # Create mock distribution plan: 0%, 10%, 50%, 20%, 20%
    distribution_plan = _create_mock_distribution_plan([0.0, 0.1, 0.5, 0.2, 0.2])
    stats_data = []
    model = Model(name="test_model")
    
    with patch('mosaic_planner.planner._get_file_metadata', _get_file_metadata_for_test):
        plan = plan_data_distribution(
            distribution_plan=distribution_plan,
            project=project,
            stats_data=stats_data,
            model=model,
        )
    
    # Verify segmentation plan
    assert plan.data_segmentation_plan is not None
    assert len(plan.data_segmentation_plan) == 4  # 0% machine excluded
    
    # Check allocations
    allocations = {}
    for machine_plan in plan.data_segmentation_plan:
        host = machine_plan["host"]
        fraction = machine_plan["fraction"]
        segments = machine_plan["segments"]
        assert len(segments) == 1
        
        segment = segments[0]
        assert segment["data_type"] == "audio"
        assert segment["is_segmentable"] is True
        
        file_indices = segment["file_indices"]
        allocated_count = len(file_indices)
        
        allocations[host] = {
            "fraction": fraction,
            "allocated_count": allocated_count,
            "expected_count": int(total_files * fraction),
        }
    
    # Verify allocations (allow ±1 for rounding)
    node2 = allocations.get("node2", {})
    node3 = allocations.get("node3", {})
    node4 = allocations.get("node4", {})
    node5 = allocations.get("node5", {})
    
    assert abs(node2["allocated_count"] - node2["expected_count"]) <= 1
    assert abs(node3["allocated_count"] - node3["expected_count"]) <= 1
    assert abs(node4["allocated_count"] - node4["expected_count"]) <= 1
    assert abs(node5["allocated_count"] - node5["expected_count"]) <= 1
    
    # Verify total allocated
    total_allocated = sum(a["allocated_count"] for a in allocations.values())
    assert abs(total_allocated - total_files) <= 2


def test_segment_text_data():
    """Test text data segmentation with 5 machines: 0%, 10%, 50%, 20%, 20%."""
    # Create dummy text data
    create_dummy_c4_pile()
    
    # Get test data directory
    test_data_dir = Path(__file__).parent / 'test_data'
    text_dir = test_data_dir / 'c4_pile'
    
    # Count actual characters
    total_chars = _count_text_chars(text_dir)
    assert total_chars > 0, "Should have text characters"
    
    # Create project with text data
    config = MosaicConfig(data_location=str(test_data_dir))
    file_def = FileDefinition(
        location='c4_pile',
        data_type=DataType.TEXT,
        is_segmentable=True,
    )
    data = Data(file_definitions=[file_def])
    project = Project(name="test_text", config=config, data=data)
    
    # Create mock distribution plan: 0%, 10%, 50%, 20%, 20%
    distribution_plan = _create_mock_distribution_plan([0.0, 0.1, 0.5, 0.2, 0.2])
    stats_data = []
    model = Model(name="test_model")
    
    with patch('mosaic_planner.planner._get_file_metadata', _get_file_metadata_for_test):
        plan = plan_data_distribution(
            distribution_plan=distribution_plan,
            project=project,
            stats_data=stats_data,
            model=model,
        )
    
    # Verify segmentation plan
    assert plan.data_segmentation_plan is not None
    assert len(plan.data_segmentation_plan) == 4  # 0% machine excluded
    
    # Check allocations
    allocations = {}
    for machine_plan in plan.data_segmentation_plan:
        host = machine_plan["host"]
        fraction = machine_plan["fraction"]
        segments = machine_plan["segments"]
        assert len(segments) == 1
        
        segment = segments[0]
        assert segment["data_type"] == "text"
        assert segment["is_segmentable"] is True
        
        start_char = segment["start_char"]
        end_char = segment["end_char"]
        allocated_chars = end_char - start_char
        
        allocations[host] = {
            "fraction": fraction,
            "allocated_chars": allocated_chars,
            "expected_chars": int(total_chars * fraction),
        }
    
    # Verify allocations (allow small tolerance for rounding)
    node2 = allocations.get("node2", {})
    node3 = allocations.get("node3", {})
    node4 = allocations.get("node4", {})
    node5 = allocations.get("node5", {})
    
    tolerance = max(1, int(total_chars * 0.01))  # 1% tolerance
    assert abs(node2["allocated_chars"] - node2["expected_chars"]) <= tolerance
    assert abs(node3["allocated_chars"] - node3["expected_chars"]) <= tolerance
    assert abs(node4["allocated_chars"] - node4["expected_chars"]) <= tolerance
    assert abs(node5["allocated_chars"] - node5["expected_chars"]) <= tolerance
    
    # Verify total allocated
    total_allocated = sum(a["allocated_chars"] for a in allocations.values())
    assert abs(total_allocated - total_chars) <= tolerance * 2


def test_segment_graph_data():
    """Test graph data segmentation with 5 machines: 0%, 10%, 50%, 20%, 20%."""
    # Create dummy graph data
    create_dummy_ogbn_arxiv()
    
    # Get test data directory
    test_data_dir = Path(__file__).parent / 'test_data'
    graph_dir = test_data_dir / 'ogbn_arxiv'
    
    # Count actual nodes
    total_nodes = _count_graph_nodes(graph_dir)
    assert total_nodes > 0, "Should have graph nodes"
    
    # Create project with graph data
    config = MosaicConfig(data_location=str(test_data_dir))
    file_def = FileDefinition(
        location='ogbn_arxiv',
        data_type=DataType.GRAPH,
        is_segmentable=True,
    )
    data = Data(file_definitions=[file_def])
    project = Project(name="test_graph", config=config, data=data)
    
    # Create mock distribution plan: 0%, 10%, 50%, 20%, 20%
    distribution_plan = _create_mock_distribution_plan([0.0, 0.1, 0.5, 0.2, 0.2])
    stats_data = []
    model = Model(name="test_model")
    
    with patch('mosaic_planner.planner._get_file_metadata', _get_file_metadata_for_test):
        plan = plan_data_distribution(
            distribution_plan=distribution_plan,
            project=project,
            stats_data=stats_data,
            model=model,
        )
    
    # Verify segmentation plan
    assert plan.data_segmentation_plan is not None
    assert len(plan.data_segmentation_plan) == 4  # 0% machine excluded
    
    # Check allocations
    allocations = {}
    for machine_plan in plan.data_segmentation_plan:
        host = machine_plan["host"]
        fraction = machine_plan["fraction"]
        segments = machine_plan["segments"]
        assert len(segments) == 1
        
        segment = segments[0]
        assert segment["data_type"] == "graph"
        assert segment["is_segmentable"] is True
        
        node_range = segment["node_range"]
        start_node, end_node = node_range
        allocated_nodes = end_node - start_node
        
        allocations[host] = {
            "fraction": fraction,
            "allocated_nodes": allocated_nodes,
            "expected_nodes": int(total_nodes * fraction),
        }
    
    # Verify allocations (allow ±1 for rounding)
    node2 = allocations.get("node2", {})
    node3 = allocations.get("node3", {})
    node4 = allocations.get("node4", {})
    node5 = allocations.get("node5", {})
    
    assert abs(node2["allocated_nodes"] - node2["expected_nodes"]) <= 1
    assert abs(node3["allocated_nodes"] - node3["expected_nodes"]) <= 1
    assert abs(node4["allocated_nodes"] - node4["expected_nodes"]) <= 1
    assert abs(node5["allocated_nodes"] - node5["expected_nodes"]) <= 1
    
    # Verify total allocated
    total_allocated = sum(a["allocated_nodes"] for a in allocations.values())
    assert abs(total_allocated - total_nodes) <= 2


def test_segment_rl_data():
    """Test RL data segmentation with 5 machines: 0%, 10%, 50%, 20%, 20%."""
    # Create dummy RL data
    create_dummy_mujoco_halfcheetah()
    
    # Get test data directory
    test_data_dir = Path(__file__).parent / 'test_data'
    rl_dir = test_data_dir / 'mujoco_halfcheetah'
    
    # Count actual episodes
    total_episodes = _count_rl_episodes(rl_dir)
    assert total_episodes > 0, "Should have RL episodes"
    
    # Create project with RL data
    config = MosaicConfig(data_location=str(test_data_dir))
    file_def = FileDefinition(
        location='mujoco_halfcheetah',
        data_type=DataType.RL,
        is_segmentable=True,
    )
    data = Data(file_definitions=[file_def])
    project = Project(name="test_rl", config=config, data=data)
    
    # Create mock distribution plan: 0%, 10%, 50%, 20%, 20%
    distribution_plan = _create_mock_distribution_plan([0.0, 0.1, 0.5, 0.2, 0.2])
    stats_data = []
    model = Model(name="test_model")
    
    with patch('mosaic_planner.planner._get_file_metadata', _get_file_metadata_for_test):
        plan = plan_data_distribution(
            distribution_plan=distribution_plan,
            project=project,
            stats_data=stats_data,
            model=model,
        )
    
    # Verify segmentation plan
    assert plan.data_segmentation_plan is not None
    assert len(plan.data_segmentation_plan) == 4  # 0% machine excluded
    
    # Check allocations
    allocations = {}
    for machine_plan in plan.data_segmentation_plan:
        host = machine_plan["host"]
        fraction = machine_plan["fraction"]
        segments = machine_plan["segments"]
        assert len(segments) == 1
        
        segment = segments[0]
        assert segment["data_type"] == "rl"
        assert segment["is_segmentable"] is True
        
        episode_range = segment["episode_range"]
        start_episode, end_episode = episode_range
        allocated_episodes = end_episode - start_episode
        
        allocations[host] = {
            "fraction": fraction,
            "allocated_episodes": allocated_episodes,
            "expected_episodes": int(total_episodes * fraction),
        }
    
    # Verify allocations (allow ±1 for rounding)
    node2 = allocations.get("node2", {})
    node3 = allocations.get("node3", {})
    node4 = allocations.get("node4", {})
    node5 = allocations.get("node5", {})
    
    assert abs(node2["allocated_episodes"] - node2["expected_episodes"]) <= 1
    assert abs(node3["allocated_episodes"] - node3["expected_episodes"]) <= 1
    assert abs(node4["allocated_episodes"] - node4["expected_episodes"]) <= 1
    assert abs(node5["allocated_episodes"] - node5["expected_episodes"]) <= 1
    
    # Verify total allocated
    total_allocated = sum(a["allocated_episodes"] for a in allocations.values())
    assert abs(total_allocated - total_episodes) <= 2


def test_segment_dir_data():
    """Test directory data segmentation with 5 machines: 0%, 10%, 50%, 20%, 20%."""
    # Create dummy image directory (can use any directory)
    create_dummy_images()
    
    # Get test data directory
    test_data_dir = Path(__file__).parent / 'test_data'
    dir_path = test_data_dir / 'dummy_images'
    
    # Count actual files in directory (non-recursive, or subdirectories if no files)
    total_files = _count_dir_files(dir_path)
    assert total_files > 0, "Should have files or subdirectories in directory"
    
    # Create project with directory data
    config = MosaicConfig(data_location=str(test_data_dir))
    file_def = FileDefinition(
        location='dummy_images',
        data_type=DataType.DIR,
        is_segmentable=True,
    )
    data = Data(file_definitions=[file_def])
    project = Project(name="test_dir", config=config, data=data)
    
    # Create mock distribution plan: 0%, 10%, 50%, 20%, 20%
    distribution_plan = _create_mock_distribution_plan([0.0, 0.1, 0.5, 0.2, 0.2])
    stats_data = []
    model = Model(name="test_model")
    
    with patch('mosaic_planner.planner._get_file_metadata', _get_file_metadata_for_test):
        plan = plan_data_distribution(
            distribution_plan=distribution_plan,
            project=project,
            stats_data=stats_data,
            model=model,
        )
    
    # Verify segmentation plan
    assert plan.data_segmentation_plan is not None
    assert len(plan.data_segmentation_plan) == 4  # 0% machine excluded
    
    # Check allocations
    allocations = {}
    for machine_plan in plan.data_segmentation_plan:
        host = machine_plan["host"]
        fraction = machine_plan["fraction"]
        segments = machine_plan["segments"]
        assert len(segments) == 1
        
        segment = segments[0]
        assert segment["data_type"] == "dir"
        assert segment["is_segmentable"] is True
        
        file_indices = segment["file_indices"]
        allocated_count = len(file_indices)
        
        allocations[host] = {
            "fraction": fraction,
            "allocated_count": allocated_count,
            "expected_count": int(total_files * fraction),
        }
    
    # Verify allocations (allow ±1 for rounding, or ±2 for very small numbers)
    node2 = allocations.get("node2", {})
    node3 = allocations.get("node3", {})
    node4 = allocations.get("node4", {})
    node5 = allocations.get("node5", {})
    
    # For small numbers, allow larger tolerance due to rounding with min(1, ...) in segmentation
    tolerance = 2 if total_files <= 5 else 1
    
    assert abs(node2["allocated_count"] - node2["expected_count"]) <= tolerance
    assert abs(node3["allocated_count"] - node3["expected_count"]) <= tolerance
    assert abs(node4["allocated_count"] - node4["expected_count"]) <= tolerance
    assert abs(node5["allocated_count"] - node5["expected_count"]) <= tolerance
    
    # Verify total allocated (allow larger tolerance for small numbers due to rounding)
    total_allocated = sum(a["allocated_count"] for a in allocations.values())
    # For small numbers, allow up to 20% difference or at least 2
    tolerance = max(2, int(total_files * 0.2)) if total_files <= 10 else max(2, int(total_files * 0.1))
    assert abs(total_allocated - total_files) <= tolerance

