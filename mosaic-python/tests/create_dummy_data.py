import csv
import json
import random
from pathlib import Path

from PIL import Image
import numpy as np
import soundfile as sf


def create_dummy_csv():
    """
    Generate dummy CSV file in test_data directory.
    
    Creates a CSV file with 100 rows and 3 columns (id, feature1, feature2).
    If the file already exists, the function skips the writing process.
    """
    # Get the test_data directory path (relative to this file)
    test_data_dir = Path(__file__).parent / 'test_data'
    test_data_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    
    csv_path = test_data_dir / 'dummy_tabular.csv'
    
    # Check if file already exists
    if csv_path.exists():
        return
    
    # Generate dummy CSV: 100 rows, 3 columns (id, feature1, feature2)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'feature1', 'feature2'])  # header
        
        for i in range(100):
            row = [
                f'row_{i}',
                round(random.uniform(-2.0, 2.0), 4),  # float feature
                random.randint(0, 100)                 # int feature
            ]
            writer.writerow(row)


def create_dummy_images():
    """
    Generate dummy image files in test_data/dummy_images directory.
    
    Creates 100 small images (64x64 pixels) and distributes them across
    5 subdirectories (20 images per directory). If the images already exist,
    the function skips the creation process.
    """
    # Get the dummy_images directory path (relative to this file)
    test_data_dir = Path(__file__).parent / 'test_data'
    dummy_images_dir = test_data_dir / 'dummy_images'
    
    # Create the main directory
    dummy_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 subdirectories
    num_dirs = 5
    images_per_dir = 20
    
    # Check if all images already exist
    all_exist = True
    for dir_idx in range(num_dirs):
        subdir = dummy_images_dir / f'dir_{dir_idx}'
        for img_idx in range(images_per_dir):
            img_path = subdir / f'image_{img_idx}.png'
            if not img_path.exists():
                all_exist = False
                break
        if not all_exist:
            break
    
    if all_exist:
        return
    
    # Generate dummy images
    for dir_idx in range(num_dirs):
        subdir = dummy_images_dir / f'dir_{dir_idx}'
        subdir.mkdir(exist_ok=True)
        
        for img_idx in range(images_per_dir):
            img_path = subdir / f'image_{img_idx}.png'
            
            # Skip if image already exists
            if img_path.exists():
                continue
            
            # Create a small image (64x64 pixels) with random colors
            img = Image.new('RGB', (64, 64), color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))
            
            # Save the image
            img.save(img_path, 'PNG')


def create_dummy_librispeech():
    """
    Generate dummy LibriSpeech-style dataset in test_data directory.
    
    Creates a LibriSpeech-like directory structure with:
    - Split directories (train-clean-100, dev-clean)
    - Speaker directories (numeric IDs)
    - Chapter directories (numeric IDs)
    - Audio files (.flac format) with dummy audio data
    - Transcription files (.trans.txt) with dummy transcriptions
    
    If the data already exists, the function skips the creation process.
    """
    # Get the test_data directory path
    test_data_dir = Path(__file__).parent / 'test_data'
    librispeech_dir = test_data_dir / 'LibriSpeech'
    
    # LibriSpeech structure: split -> speaker -> chapter -> audio files + trans.txt
    splits = [
        ('train-clean-100', 2, 2, 3),  # (split_name, num_speakers, num_chapters_per_speaker, num_utterances_per_chapter)
        ('dev-clean', 1, 1, 2),
    ]
    
    # Dummy transcriptions
    dummy_transcriptions = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test transcription",
        "the cat sat on the mat",
        "once upon a time in a far away land",
        "to be or not to be that is the question",
        "all work and no play makes jack a dull boy",
        "the early bird catches the worm",
        "practice makes perfect",
        "a picture is worth a thousand words",
        "actions speak louder than words",
    ]
    
    # Check if all data already exists
    all_exist = True
    for split_name, num_speakers, num_chapters, num_utterances in splits:
        split_dir = librispeech_dir / split_name
        for speaker_idx in range(num_speakers):
            speaker_id = speaker_idx + 1
            speaker_dir = split_dir / str(speaker_id)
            for chapter_idx in range(num_chapters):
                chapter_id = chapter_idx + 1
                chapter_dir = speaker_dir / str(chapter_id)
                
                # Check transcription file
                trans_file = chapter_dir / f'{speaker_id}-{chapter_id}.trans.txt'
                if not trans_file.exists():
                    all_exist = False
                    break
                
                # Check audio files
                for utt_idx in range(num_utterances):
                    audio_file = chapter_dir / f'{speaker_id}-{chapter_id}-{utt_idx:04d}.flac'
                    if not audio_file.exists():
                        all_exist = False
                        break
                
                if not all_exist:
                    break
            if not all_exist:
                break
        if not all_exist:
            break
    
    if all_exist:
        return
    
    # Generate dummy LibriSpeech data
    for split_name, num_speakers, num_chapters, num_utterances in splits:
        split_dir = librispeech_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for speaker_idx in range(num_speakers):
            speaker_id = speaker_idx + 1
            speaker_dir = split_dir / str(speaker_id)
            speaker_dir.mkdir(exist_ok=True)
            
            for chapter_idx in range(num_chapters):
                chapter_id = chapter_idx + 1
                chapter_dir = speaker_dir / str(chapter_id)
                chapter_dir.mkdir(exist_ok=True)
                
                # Create transcription file
                trans_file = chapter_dir / f'{speaker_id}-{chapter_id}.trans.txt'
                
                # Read existing transcriptions if file exists
                existing_transcriptions = {}
                if trans_file.exists():
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split(' ', 1)
                                if len(parts) == 2:
                                    existing_transcriptions[parts[0]] = parts[1]
                
                new_transcriptions = []
                
                # Generate audio files and collect transcriptions
                for utt_idx in range(num_utterances):
                    utterance_id = f'{speaker_id}-{chapter_id}-{utt_idx:04d}'
                    audio_file = chapter_dir / f'{utterance_id}.flac'
                    
                    # Skip if audio file already exists
                    if audio_file.exists():
                        continue
                    
                    # Generate dummy audio (1 second of simple tone with slight noise)
                    sample_rate = 16000  # LibriSpeech standard sample rate
                    duration = 1.0  # 1 second
                    samples = int(sample_rate * duration)
                    
                    # Generate simple audio: sine wave with random frequency + noise
                    freq = random.uniform(200, 800)
                    t = np.linspace(0, duration, samples, False)
                    audio_data = np.sin(2 * np.pi * freq * t) * 0.3
                    audio_data += np.random.normal(0, 0.01, samples)  # Add slight noise
                    audio_data = audio_data.astype(np.float32)
                    
                    # Save as FLAC
                    sf.write(str(audio_file), audio_data, sample_rate, format='FLAC')
                    
                    # Get transcription
                    transcription = random.choice(dummy_transcriptions)
                    existing_transcriptions[utterance_id] = transcription
                    new_transcriptions.append(utterance_id)
                
                # Write transcription file (combine existing and new)
                if not trans_file.exists() or new_transcriptions:
                    all_transcriptions = []
                    for utt_idx in range(num_utterances):
                        utterance_id = f'{speaker_id}-{chapter_id}-{utt_idx:04d}'
                        if utterance_id in existing_transcriptions:
                            all_transcriptions.append(f'{utterance_id} {existing_transcriptions[utterance_id]}')
                        else:
                            # Generate transcription for any missing entries
                            transcription = random.choice(dummy_transcriptions)
                            all_transcriptions.append(f'{utterance_id} {transcription}')
                    
                    with open(trans_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(all_transcriptions))


def create_dummy_c4_pile():
    """
    Generate dummy C4/The Pile-style dataset in test_data directory.
    
    Creates a JSONL (JSON Lines) dataset similar to C4 and The Pile with:
    - Multiple shard files (c4-train.00000.jsonl, c4-train.00001.jsonl, etc.)
    - Each line is a JSON object containing text and metadata
    - Fields include: text, url (for C4-style), meta (for The Pile-style)
    
    If the data already exists, the function skips the creation process.
    """
    # Get the test_data directory path
    test_data_dir = Path(__file__).parent / 'test_data'
    c4_pile_dir = test_data_dir / 'c4_pile'
    
    # Create the directory
    c4_pile_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration: (split_name, num_shards, examples_per_shard)
    splits = [
        ('train', 3, 10),  # 3 shards, 10 examples each
        ('validation', 1, 5),  # 1 shard, 5 examples
    ]
    
    # Dummy text templates for generating varied content
    text_templates = [
        "This is a sample document about {topic}. It contains multiple sentences that form a coherent paragraph. The content discusses various aspects of the subject matter.",
        "In this article, we explore {topic} from different perspectives. The analysis covers historical context, current trends, and future implications.",
        "The following text discusses {topic} in detail. We examine the key concepts, provide examples, and draw conclusions based on the evidence presented.",
        "A comprehensive overview of {topic} is provided here. This document serves as an introduction to the subject and covers fundamental principles.",
        "This passage describes {topic} and its significance. The text includes background information, main points, and supporting details.",
    ]
    
    topics = [
        "artificial intelligence",
        "machine learning",
        "natural language processing",
        "computer science",
        "data science",
        "software engineering",
        "web development",
        "cloud computing",
        "cybersecurity",
        "quantum computing",
    ]
    
    # Check if all files already exist
    all_exist = True
    for split_name, num_shards, examples_per_shard in splits:
        for shard_idx in range(num_shards):
            shard_file = c4_pile_dir / f'c4-{split_name}.{shard_idx:05d}.jsonl'
            if not shard_file.exists():
                all_exist = False
                break
        if not all_exist:
            break
    
    if all_exist:
        return
    
    # Generate dummy C4/The Pile data
    for split_name, num_shards, examples_per_shard in splits:
        for shard_idx in range(num_shards):
            shard_file = c4_pile_dir / f'c4-{split_name}.{shard_idx:05d}.jsonl'
            
            # Skip if file already exists
            if shard_file.exists():
                continue
            
            # Generate examples for this shard
            examples = []
            for example_idx in range(examples_per_shard):
                topic = random.choice(topics)
                text_template = random.choice(text_templates)
                text = text_template.format(topic=topic)
                
                # Create a JSON object similar to C4/The Pile format
                # Mix of C4-style (with url) and Pile-style (with meta) entries
                if random.random() < 0.5:
                    # C4-style entry
                    example = {
                        "text": text,
                        "url": f"https://example.com/article/{split_name}/{shard_idx}/{example_idx}",
                        "timestamp": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                    }
                else:
                    # The Pile-style entry
                    example = {
                        "text": text,
                        "meta": {
                            "source": random.choice(["wikipedia", "books", "arxiv", "github", "stackexchange"]),
                            "split": split_name,
                            "shard": shard_idx,
                            "index": example_idx,
                        }
                    }
                
                examples.append(example)
            
            # Write JSONL file (one JSON object per line)
            with open(shard_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')


def create_dummy_ogbn_arxiv():
    """
    Generate dummy ogbn-arxiv-style graph dataset in test_data directory.
    
    Creates a citation network graph similar to ogbn-arxiv with:
    - Edge list (edges.csv) - source and target node IDs
    - Node features (node-feat.csv) - feature vectors for each node
    - Node labels (node-label.csv) - class labels for each node
    - Split information (split.csv) - train/val/test split assignments
    
    If the data already exists, the function skips the creation process.
    """
    # Get the test_data directory path
    test_data_dir = Path(__file__).parent / 'test_data'
    ogbn_dir = test_data_dir / 'ogbn_arxiv'
    
    # Create the directory
    ogbn_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    num_nodes = 50  # Small graph for testing
    num_features = 128  # Feature dimension (typical for ogbn-arxiv)
    num_classes = 5  # Number of label categories
    
    # File paths
    edges_file = ogbn_dir / 'edges.csv'
    node_feat_file = ogbn_dir / 'node-feat.csv'
    node_label_file = ogbn_dir / 'node-label.csv'
    split_file = ogbn_dir / 'split.csv'
    
    # Check if all files already exist
    if all(f.exists() for f in [edges_file, node_feat_file, node_label_file, split_file]):
        return
    
    # Generate edges (citation network - directed graph)
    if not edges_file.exists():
        edges = []
        # Create a sparse graph with some structure
        # Add some forward citations (newer papers cite older ones)
        for source in range(num_nodes):
            # Each node cites a few random nodes (with bias toward earlier nodes)
            num_citations = random.randint(1, 5)
            possible_targets = list(range(max(0, source - 10), source))
            if not possible_targets:
                possible_targets = list(range(num_nodes))
            
            cited = random.sample(possible_targets, min(num_citations, len(possible_targets)))
            for target in cited:
                edges.append((source, target))
        
        # Add some random edges for connectivity
        for _ in range(num_nodes // 2):
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            if source != target and (source, target) not in edges:
                edges.append((source, target))
        
        # Write edges.csv
        with open(edges_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target'])  # header
            for source, target in edges:
                writer.writerow([source, target])
    
    # Generate node features (feature vectors)
    if not node_feat_file.exists():
        # Generate random feature vectors (simulating word embeddings)
        with open(node_feat_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            header = ['node_id'] + [f'feat_{i}' for i in range(num_features)]
            writer.writerow(header)
            
            for node_id in range(num_nodes):
                # Generate feature vector (normalized random values)
                features = np.random.normal(0, 1, num_features).astype(np.float32)
                # Normalize
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
                
                row = [node_id] + [f'{feat:.6f}' for feat in features]
                writer.writerow(row)
    
    # Generate node labels (paper categories)
    if not node_label_file.exists():
        with open(node_label_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id', 'label'])  # header
            
            for node_id in range(num_nodes):
                label = random.randint(0, num_classes - 1)
                writer.writerow([node_id, label])
    
    # Generate split information (train/val/test)
    if not split_file.exists():
        # Split: 60% train, 20% val, 20% test
        node_ids = list(range(num_nodes))
        random.shuffle(node_ids)
        
        train_size = int(num_nodes * 0.6)
        val_size = int(num_nodes * 0.2)
        
        train_nodes = set(node_ids[:train_size])
        val_nodes = set(node_ids[train_size:train_size + val_size])
        test_nodes = set(node_ids[train_size + val_size:])
        
        with open(split_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id', 'split'])  # header
            
            for node_id in train_nodes:
                writer.writerow([node_id, 'train'])
            for node_id in val_nodes:
                writer.writerow([node_id, 'val'])
            for node_id in test_nodes:
                writer.writerow([node_id, 'test'])


def create_dummy_mujoco_halfcheetah():
    """
    Generate dummy Mujoco Half-Cheetah-style dataset in test_data directory.
    
    Creates a reinforcement learning trajectory dataset similar to Mujoco Half-Cheetah with:
    - Trajectory files (trajectories.npz) containing:
      - observations: state vectors (17-dim for Half-Cheetah)
      - actions: action vectors (6-dim for Half-Cheetah)
      - rewards: scalar rewards
      - dones: episode termination flags
      - next_observations: next state vectors (optional)
    - Metadata file (metadata.json) with environment info
    
    If the data already exists, the function skips the creation process.
    """
    # Get the test_data directory path
    test_data_dir = Path(__file__).parent / 'test_data'
    mujoco_dir = test_data_dir / 'mujoco_halfcheetah'
    
    # Create the directory
    mujoco_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    num_episodes = 10  # Number of episodes
    max_steps_per_episode = 100  # Maximum steps per episode
    obs_dim = 17  # Observation dimension for Half-Cheetah
    action_dim = 6  # Action dimension for Half-Cheetah
    
    # File paths
    trajectories_file = mujoco_dir / 'trajectories.npz'
    metadata_file = mujoco_dir / 'metadata.json'
    
    # Check if files already exist
    if trajectories_file.exists() and metadata_file.exists():
        return
    
    # Track total steps for metadata
    total_steps = 0
    
    # Generate trajectories
    if not trajectories_file.exists():
        all_observations = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_next_observations = []
        episode_starts = [0]  # Track where each episode starts
        
        for episode in range(num_episodes):
            episode_length = random.randint(50, max_steps_per_episode)
            
            # Generate episode trajectory
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_next_obs = []
            
            # Initial observation
            obs = np.random.normal(0, 1, obs_dim).astype(np.float32)
            
            for step in range(episode_length):
                # Action (bounded between -1 and 1, typical for continuous control)
                action = np.clip(np.random.normal(0, 0.5, action_dim), -1.0, 1.0).astype(np.float32)
                
                # Next observation (simulate dynamics - simple random walk with some structure)
                next_obs = obs + np.random.normal(0, 0.1, obs_dim).astype(np.float32)
                next_obs = np.clip(next_obs, -10, 10)  # Bound observations
                
                # Reward (simulate forward velocity reward - typical for Half-Cheetah)
                # Reward is based on forward velocity (first few obs dimensions)
                forward_velocity = np.sum(next_obs[:3]) * 0.1
                control_cost = -0.1 * np.sum(action ** 2)
                reward = float(forward_velocity + control_cost + random.uniform(-0.1, 0.1))
                
                # Done flag (episode ends at max length or randomly)
                done = (step == episode_length - 1) or (random.random() < 0.02)
                
                episode_obs.append(obs)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_dones.append(done)
                episode_next_obs.append(next_obs)
                
                # Update observation for next step
                obs = next_obs.copy()
                
                if done:
                    break
            
            # Append episode data
            all_observations.extend(episode_obs)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            all_dones.extend(episode_dones)
            all_next_observations.extend(episode_next_obs)
            
            # Track episode start for next episode
            episode_starts.append(len(all_observations))
        
        # Convert to numpy arrays
        observations = np.array(all_observations, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.float32)
        rewards = np.array(all_rewards, dtype=np.float32)
        dones = np.array(all_dones, dtype=bool)
        next_observations = np.array(all_next_observations, dtype=np.float32)
        episode_starts = np.array(episode_starts, dtype=np.int64)
        
        # Track total steps
        total_steps = len(all_observations)
        
        # Save as NPZ file
        np.savez_compressed(
            str(trajectories_file),
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_observations=next_observations,
            episode_starts=episode_starts,
        )
    
    # Generate metadata file
    if not metadata_file.exists():
        
        metadata = {
            "env_name": "HalfCheetah-v4",
            "env_type": "mujoco",
            "observation_space": {
                "shape": [obs_dim],
                "dtype": "float32",
                "low": -10.0,
                "high": 10.0,
            },
            "action_space": {
                "shape": [action_dim],
                "dtype": "float32",
                "low": -1.0,
                "high": 1.0,
            },
            "num_episodes": num_episodes,
            "total_steps": total_steps if total_steps > 0 else "unknown",
            "max_episode_steps": max_steps_per_episode,
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
