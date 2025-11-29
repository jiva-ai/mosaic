"""Utility functions for planning data and ML workload distribution."""

import time
from typing import Any, Dict, List, Optional

# Constants for dynamic batch planning
MAX_CPU_UTIL = 90.0
MAX_RAM_UTIL = 95.0


def eligibility_filter(
    stats_data: List[Dict[str, Any]], stale_threshold: int = 60
) -> List[Dict[str, Any]]:
    """
    Filter stats data for eligible nodes based on connection status and staleness.

    Filters nodes that are:
    1. Online (connection_status == "online")
    2. Not stale (current_time - last_time_received < stale_threshold in seconds)

    Args:
        stats_data: List of stat dictionaries from beacon.collect_stats()
        stale_threshold: Maximum age in seconds for a node to be considered fresh.
                         Defaults to 60 seconds.

    Returns:
        Filtered list of stat dictionaries containing only eligible nodes.
    """
    current_time_millis = int(time.time() * 1000)
    stale_threshold_millis = stale_threshold * 1000

    filtered_data = []
    for stat in stats_data:
        # Check connection status
        if stat.get("connection_status") != "online":
            continue

        # Check staleness
        last_time_received = stat.get("last_time_received", 0)
        if last_time_received == 0:
            # Never received, consider stale
            continue

        time_since_received = current_time_millis - last_time_received
        if time_since_received < stale_threshold_millis:
            filtered_data.append(stat)

    return filtered_data


def _min_max_normalize(values: List[float]) -> List[float]:
    """
    Normalize a list of values using min-max normalization.
    
    Formula: normalized = (value - min) / (max - min)
    If all values are the same (max == min), returns 0.5 for all values.
    
    Args:
        values: List of numeric values to normalize
        
    Returns:
        List of normalized values in the same order
    """
    if not values:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    # Handle case where all values are the same
    if max_val == min_val:
        return [0.5] * len(values)
    
    # Normalize: (value - min) / (max - min)
    return [(val - min_val) / (max_val - min_val) for val in values]


def capacity_score(
    stats_data: List[Dict[str, Any]],
    w_cpu: float = 0.15,
    w_gpu: float = 0.75,
    w_ram: float = 0.05,
    w_disk: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Compute capacity scores for nodes based on CPU, GPU, RAM, and disk metrics.

    The capacity score is a weighted combination of min-max normalized scores:
    - CPU score: cpu.gflops (normalized)
    - GPU score: sum of gpu.gflops for each GPU in benchmark.gpus (normalized)
    - RAM score: ram.bandwidth_gbps (normalized)
    - Disk score: min(disk.read_speed_mbps, disk.write_speed_mbps) (normalized)

    All scores are normalized using min-max normalization across all nodes before
    computing the weighted combination.

    Args:
        stats_data: List of stat dictionaries from beacon.collect_stats()
        w_cpu: Weight for CPU score (default: 0.15)
        w_gpu: Weight for GPU score (default: 0.75)
        w_ram: Weight for RAM score (default: 0.05)
        w_disk: Weight for disk score (default: 0.05)

    Returns:
        List of dictionaries with added 'capacity_score' field for each node.
        Nodes with missing benchmark data will have capacity_score set to 0.0.
    """
    # First pass: extract raw scores for all nodes
    raw_cpu_scores = []
    raw_gpu_scores = []
    raw_ram_scores = []
    raw_disk_scores = []
    node_data = []
    
    for stat in stats_data:
        benchmark = stat.get("benchmark", {})
        if not isinstance(benchmark, dict) or not benchmark:
            # No benchmark or empty benchmark - treat as no benchmark
            raw_cpu_scores.append(0.0)
            raw_gpu_scores.append(0.0)
            raw_ram_scores.append(0.0)
            raw_disk_scores.append(0.0)
            node_data.append({"stat": stat, "has_benchmark": False})
            continue
        
        # Extract CPU score
        cpu_data = benchmark.get("cpu", {})
        cpu_score = cpu_data.get("gflops", 0.0) if isinstance(cpu_data, dict) else 0.0
        
        # Extract GPU score (sum of all GPUs)
        gpus_data = benchmark.get("gpus", [])
        gpu_score = 0.0
        if isinstance(gpus_data, list):
            for gpu in gpus_data:
                if isinstance(gpu, dict):
                    gpu_score += gpu.get("gflops", 0.0)
        
        # Extract RAM score
        ram_data = benchmark.get("ram", {})
        ram_score = ram_data.get("bandwidth_gbps", 0.0) if isinstance(ram_data, dict) else 0.0
        
        # Extract disk score (min of read and write speeds)
        disk_data = benchmark.get("disk", {})
        disk_score = 0.0
        if isinstance(disk_data, dict):
            read_speed = disk_data.get("read_speed_mbps", 0.0)
            write_speed = disk_data.get("write_speed_mbps", 0.0)
            disk_score = min(read_speed, write_speed)
        
        raw_cpu_scores.append(cpu_score)
        raw_gpu_scores.append(gpu_score)
        raw_ram_scores.append(ram_score)
        raw_disk_scores.append(disk_score)
        node_data.append({"stat": stat, "has_benchmark": True})
    
    # Normalize all scores
    normalized_cpu_scores = _min_max_normalize(raw_cpu_scores)
    normalized_gpu_scores = _min_max_normalize(raw_gpu_scores)
    normalized_ram_scores = _min_max_normalize(raw_ram_scores)
    normalized_disk_scores = _min_max_normalize(raw_disk_scores)
    
    # Second pass: compute weighted capacity scores using normalized values
    result = []
    for i, node_info in enumerate(node_data):
        stat = node_info["stat"]
        stat_with_score = stat.copy()
        
        if not node_info["has_benchmark"]:
            stat_with_score["capacity_score"] = 0.0
        else:
            # Check if all raw scores are 0.0 (no valid data)
            # In this case, return 0.0 instead of normalized 0.5
            if (raw_cpu_scores[i] == 0.0 and raw_gpu_scores[i] == 0.0 and 
                raw_ram_scores[i] == 0.0 and raw_disk_scores[i] == 0.0):
                stat_with_score["capacity_score"] = 0.0
            else:
                # Compute weighted capacity score using normalized values
                compute_score = (
                    w_cpu * normalized_cpu_scores[i] +
                    w_gpu * normalized_gpu_scores[i] +
                    w_ram * normalized_ram_scores[i] +
                    w_disk * normalized_disk_scores[i]
                )
                stat_with_score["capacity_score"] = compute_score
        
        result.append(stat_with_score)
    
    return result


def live_load_factor(
    stats_data: List[Dict[str, Any]],
    alpha: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Compute live load factors for nodes based on current CPU and RAM utilization.

    The load factor represents how much capacity is available (1.0 = fully available,
    0.0 = fully utilized). It's computed as a weighted combination of CPU and RAM
    availability.

    Args:
        stats_data: List of stat dictionaries from beacon.collect_stats()
        alpha: Weight for CPU availability vs RAM availability (default: 0.7).
               Higher alpha means more weight on CPU. Range should be [0.0, 1.0].

    Returns:
        List of dictionaries with added 'live_load_factor' field for each node.
        Nodes with missing stats_payload will have live_load_factor set to 0.0.
    """
    result = []
    
    for stat in stats_data:
        # Create a copy to avoid modifying the original
        stat_with_factor = stat.copy()
        
        stats_payload = stat.get("stats_payload")
        if not isinstance(stats_payload, dict):
            stat_with_factor["live_load_factor"] = 0.0
            result.append(stat_with_factor)
            continue
        
        # Extract CPU and RAM percentages
        cpu_percent = stats_payload.get("cpu_percent", 0.0)
        ram_percent = stats_payload.get("ram_percent", 0.0)
        
        # Convert to float if needed
        try:
            cpu_percent = float(cpu_percent)
            ram_percent = float(ram_percent)
        except (ValueError, TypeError):
            stat_with_factor["live_load_factor"] = 0.0
            result.append(stat_with_factor)
            continue
        
        # Compute availability (1.0 - utilization, clamped to [0.0, 1.0])
        cpu_avail = max(0.0, 1.0 - cpu_percent / 100.0)
        ram_avail = max(0.0, 1.0 - ram_percent / 100.0)
        
        # Compute weighted load factor
        load_factor = alpha * cpu_avail + (1.0 - alpha) * ram_avail
        
        stat_with_factor["live_load_factor"] = load_factor
        result.append(stat_with_factor)
    
    return result


def network_factor(
    stats_data: List[Dict[str, Any]],
    min_net_factor: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Compute network factors for nodes based on network delay.

    The network factor represents how network latency affects a node's desirability.
    Faster nodes (lower delay) get a factor closer to 1.0, while slower nodes get
    a factor closer to min_net_factor.

    Args:
        stats_data: List of stat dictionaries from beacon.collect_stats()
        min_net_factor: Minimum network factor for the slowest node (default: 0.4).
                        The worst node still gets this percentage of its compute score.

    Returns:
        List of dictionaries with added 'network_factor' field for each node.
        Nodes without delay information will have network_factor set to 0.0.
    """
    # Collect delays for eligible peers (peers that have "delay" field)
    delays = []
    for stat in stats_data:
        if "delay" in stat and stat["delay"] is not None:
            try:
                delay = float(stat["delay"])
                delays.append(delay)
            except (ValueError, TypeError):
                # Invalid delay value, skip
                continue
    
    # If no eligible peers with delays, return 0.0 for all
    if not delays:
        result = []
        for stat in stats_data:
            stat_with_factor = stat.copy()
            stat_with_factor["network_factor"] = 0.0
            result.append(stat_with_factor)
        return result
    
    # Compute min and max delays
    min_delay = min(delays)
    max_delay = max(delays)
    
    # Build result list
    result = []
    
    for stat in stats_data:
        stat_with_factor = stat.copy()
        
        if "delay" in stat and stat["delay"] is not None:
            try:
                delay = float(stat["delay"])
                
                if max_delay == min_delay:
                    # All delays are the same
                    network_factor = 1.0
                else:
                    # Normalize delay: 0 = fastest, 1 = slowest
                    norm = (delay - min_delay) / (max_delay - min_delay)
                    # fastest -> ~1.0, slowest -> ~min_net_factor
                    network_factor = 1.0 - norm * (1.0 - min_net_factor)
                
                stat_with_factor["network_factor"] = network_factor
            except (ValueError, TypeError):
                # Invalid delay value
                stat_with_factor["network_factor"] = 0.0
        else:
            # No delay information
            stat_with_factor["network_factor"] = 0.0
        
        result.append(stat_with_factor)
    
    return result


def plan_static_weighted_shards(
    stats_data: List[Dict[str, Any]],
    total_samples: int,
    stale_threshold: int = 60,
    w_cpu: float = 0.15,
    w_gpu: float = 0.75,
    w_ram: float = 0.05,
    w_disk: float = 0.05,
    min_net_factor: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Plan static weighted shard allocation across eligible peers.

    This function filters eligible peers, computes capacity and network factors,
    and allocates integer shard sizes based on weighted effective scores.

    Args:
        stats_data: List of stat dictionaries from beacon.collect_stats()
        total_samples: Total number of samples to allocate across peers
        stale_threshold: Maximum age in seconds for a node to be considered fresh
        w_cpu: Weight for CPU score in capacity calculation
        w_gpu: Weight for GPU score in capacity calculation
        w_ram: Weight for RAM score in capacity calculation
        w_disk: Weight for disk score in capacity calculation
        min_net_factor: Minimum network factor for the slowest node

    Returns:
        List of dictionaries with allocation information for each peer:
        - host: Peer hostname
        - heartbeat_port: Peer heartbeat port
        - comms_port: Peer comms port
        - allocated_samples: Integer number of samples allocated
        - capacity_fraction: Fraction of total capacity (frac_i)
        - compute_score: Capacity score (normalized)
        - network_factor: Network factor
        - effective_score: compute_score * network_factor
    """
    # Step 1: Filter for eligibility
    eligible_peers = eligibility_filter(stats_data, stale_threshold=stale_threshold)
    
    if not eligible_peers:
        return []
    
    # Check that benchmark exists in each peer
    peers_with_benchmark = []
    for peer in eligible_peers:
        benchmark = peer.get("benchmark")
        if isinstance(benchmark, dict) and benchmark:
            peers_with_benchmark.append(peer)
    
    if not peers_with_benchmark:
        return []
    
    # Step 2: Call capacity_score
    peers_with_capacity = capacity_score(
        peers_with_benchmark,
        w_cpu=w_cpu,
        w_gpu=w_gpu,
        w_ram=w_ram,
        w_disk=w_disk,
    )
    
    # Drop peers where capacity_score <= 0
    # Note: After normalization, the minimum peer can get capacity_score = 0.0,
    # but that's still a valid peer. We need to check if the peer has any non-zero
    # raw benchmark data to distinguish "no data" from "minimum after normalization".
    peers_with_positive_capacity = []
    for peer in peers_with_capacity:
        capacity_score_val = peer.get("capacity_score", 0.0)
        
        # Check if peer has any non-zero raw benchmark data
        benchmark = peer.get("benchmark", {})
        has_nonzero_data = False
        if isinstance(benchmark, dict):
            cpu_data = benchmark.get("cpu", {})
            if isinstance(cpu_data, dict) and cpu_data.get("gflops", 0.0) > 0.0:
                has_nonzero_data = True
            gpus = benchmark.get("gpus", [])
            if isinstance(gpus, list):
                for gpu in gpus:
                    if isinstance(gpu, dict) and gpu.get("gflops", 0.0) > 0.0:
                        has_nonzero_data = True
                        break
            ram_data = benchmark.get("ram", {})
            if isinstance(ram_data, dict) and ram_data.get("bandwidth_gbps", 0.0) > 0.0:
                has_nonzero_data = True
            disk_data = benchmark.get("disk", {})
            if isinstance(disk_data, dict):
                read_speed = disk_data.get("read_speed_mbps", 0.0)
                write_speed = disk_data.get("write_speed_mbps", 0.0)
                if read_speed > 0.0 or write_speed > 0.0:
                    has_nonzero_data = True
        
        # Keep peer if capacity_score > 0 OR if it has non-zero benchmark data
        # (the latter handles the case where normalization results in 0.0 for minimum peer)
        if capacity_score_val > 0.0 or has_nonzero_data:
            peers_with_positive_capacity.append(peer)
    
    if not peers_with_positive_capacity:
        return []
    
    # Step 3: Compute network_factor
    peers_with_network = network_factor(
        peers_with_positive_capacity,
        min_net_factor=min_net_factor,
    )
    
    # Step 4: Compute effective_score and drop peers with effective_score <= 0
    peers_with_effective = []
    for peer in peers_with_network:
        compute_score = peer.get("capacity_score", 0.0)
        network_factor_val = peer.get("network_factor", 0.0)
        effective_score = compute_score * network_factor_val
        
        # Check if peer has valid benchmark data (to handle case where capacity_score = 0.0
        # after normalization but peer still has valid data)
        benchmark = peer.get("benchmark", {})
        has_nonzero_data = False
        if isinstance(benchmark, dict):
            cpu_data = benchmark.get("cpu", {})
            if isinstance(cpu_data, dict) and cpu_data.get("gflops", 0.0) > 0.0:
                has_nonzero_data = True
            gpus = benchmark.get("gpus", [])
            if isinstance(gpus, list):
                for gpu in gpus:
                    if isinstance(gpu, dict) and gpu.get("gflops", 0.0) > 0.0:
                        has_nonzero_data = True
                        break
            ram_data = benchmark.get("ram", {})
            if isinstance(ram_data, dict) and ram_data.get("bandwidth_gbps", 0.0) > 0.0:
                has_nonzero_data = True
            disk_data = benchmark.get("disk", {})
            if isinstance(disk_data, dict):
                read_speed = disk_data.get("read_speed_mbps", 0.0)
                write_speed = disk_data.get("write_speed_mbps", 0.0)
                if read_speed > 0.0 or write_speed > 0.0:
                    has_nonzero_data = True
        
        # Keep peer if effective_score > 0 OR if it has valid data and network_factor > 0
        # (the latter handles the case where normalization results in capacity_score = 0.0
        # for the minimum peer, but it still has valid benchmark data)
        if effective_score > 0.0 or (has_nonzero_data and network_factor_val > 0.0):
            # If effective_score is 0.0 but we're keeping it due to valid data,
            # use a small epsilon to ensure it gets some allocation
            if effective_score == 0.0 and has_nonzero_data:
                effective_score = 0.0001  # Small epsilon to ensure it's included
            
            peer_with_effective = peer.copy()
            peer_with_effective["effective_score"] = effective_score
            peers_with_effective.append(peer_with_effective)
    
    if not peers_with_effective:
        return []
    
    # Step 5: Normalize effective scores
    total_effective = sum(peer.get("effective_score", 0.0) for peer in peers_with_effective)
    
    if total_effective <= 0.0:
        return []
    
    # Compute fractions
    for peer in peers_with_effective:
        effective_score = peer.get("effective_score", 0.0)
        frac = effective_score / total_effective
        peer["capacity_fraction"] = frac
    
    # Step 6: Convert fractions into integer shard sizes
    # Calculate raw allocations and remainders
    allocations = []
    for peer in peers_with_effective:
        frac = peer.get("capacity_fraction", 0.0)
        raw = frac * total_samples
        base = int(raw)  # floor
        remainder = raw - base
        allocations.append({
            "peer": peer,
            "base": base,
            "remainder": remainder,
        })
    
    # Ensure all peers with valid data get at least 1 sample
    # (handles case where minimum peer after normalization gets 0 base allocation)
    for alloc in allocations:
        peer = alloc["peer"]
        benchmark = peer.get("benchmark", {})
        has_nonzero_data = False
        if isinstance(benchmark, dict):
            cpu_data = benchmark.get("cpu", {})
            if isinstance(cpu_data, dict) and cpu_data.get("gflops", 0.0) > 0.0:
                has_nonzero_data = True
            gpus = benchmark.get("gpus", [])
            if isinstance(gpus, list):
                for gpu in gpus:
                    if isinstance(gpu, dict) and gpu.get("gflops", 0.0) > 0.0:
                        has_nonzero_data = True
                        break
            ram_data = benchmark.get("ram", {})
            if isinstance(ram_data, dict) and ram_data.get("bandwidth_gbps", 0.0) > 0.0:
                has_nonzero_data = True
            disk_data = benchmark.get("disk", {})
            if isinstance(disk_data, dict):
                read_speed = disk_data.get("read_speed_mbps", 0.0)
                write_speed = disk_data.get("write_speed_mbps", 0.0)
                if read_speed > 0.0 or write_speed > 0.0:
                    has_nonzero_data = True
        
        # Ensure peers with valid data get at least 1 sample
        if has_nonzero_data and alloc["base"] == 0:
            alloc["base"] = 1
    
    # Calculate remaining samples
    total_base = sum(alloc["base"] for alloc in allocations)
    remaining = total_samples - total_base
    
    # If we've overallocated due to minimum guarantees, adjust
    if remaining < 0:
        # Sort by base descending, then reduce from highest
        allocations.sort(key=lambda x: (x["base"], x["remainder"]), reverse=True)
        for alloc in allocations:
            if remaining >= 0:
                break
            if alloc["base"] > 1:  # Don't reduce below 1 for peers with valid data
                alloc["base"] -= 1
                remaining += 1
        # If still negative, we have more peers than samples - give 1 to each
        if remaining < 0:
            for alloc in allocations:
                alloc["base"] = 1
            remaining = total_samples - len(allocations)
    
    # Sort by remainder descending and assign +1 to top "remaining" peers
    allocations.sort(key=lambda x: x["remainder"], reverse=True)
    
    for i in range(min(remaining, len(allocations))):
        allocations[i]["base"] += 1
    
    # Step 7: Build result list
    result = []
    for alloc in allocations:
        peer = alloc["peer"]
        allocated_samples = alloc["base"]
        
        result.append({
            "host": peer.get("host", ""),
            "heartbeat_port": peer.get("heartbeat_port", 0),
            "comms_port": peer.get("comms_port", 0),
            "allocated_samples": allocated_samples,
            "capacity_fraction": peer.get("capacity_fraction", 0.0),
            "compute_score": peer.get("capacity_score", 0.0),
            "network_factor": peer.get("network_factor", 0.0),
            "effective_score": peer.get("effective_score", 0.0),
        })
    
    return result


def plan_dynamic_weighted_batches(
    stats_data: List[Dict[str, Any]],
    total_batches: Optional[int] = None,
    stale_threshold_ms: int = 60_000,
    w_cpu: float = 0.15,
    w_gpu: float = 0.75,
    w_ram: float = 0.05,
    w_disk: float = 0.05,
    min_net_factor: float = 0.4,
    alpha: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Plan dynamic weighted batch allocation across eligible peers.

    This function filters eligible peers based on utilization limits, computes
    capacity, load, freshness, and network factors, and allocates integer batch
    sizes based on weighted effective scores.

    Args:
        stats_data: List of stat dictionaries from beacon.collect_stats()
        total_batches: Total number of batches to allocate. Defaults to current
                      time in milliseconds since epoch if None.
        stale_threshold_ms: Maximum age in milliseconds for a node to be considered
                           fresh (default: 60000 = 60 seconds)
        w_cpu: Weight for CPU score in capacity calculation
        w_gpu: Weight for GPU score in capacity calculation
        w_ram: Weight for RAM score in capacity calculation
        w_disk: Weight for disk score in capacity calculation
        min_net_factor: Minimum network factor for the slowest node
        alpha: Weight for CPU vs RAM in load factor calculation

    Returns:
        List of dictionaries with allocation information for each peer:
        - host: Peer hostname
        - heartbeat_port: Peer heartbeat port
        - comms_port: Peer comms port
        - allocated_batches: Integer number of batches allocated
        - effective_score: Product of all factors
        - compute_score: Capacity score
        - load_factor: Live load factor
        - freshness_factor: Freshness factor based on heartbeat age
        - network_factor: Network factor

    Raises:
        ValueError: If no eligible peers with positive capacity remain after filtering
    """
    # Set default total_batches to current time in milliseconds
    if total_batches is None:
        total_batches = int(time.time() * 1000)
    
    now_ms = int(time.time() * 1000)
    
    # Step 1: Filter for eligible peers with stricter criteria
    # First apply basic eligibility filter
    eligible_peers = eligibility_filter(
        stats_data,
        stale_threshold=stale_threshold_ms // 1000,  # Convert ms to seconds
    )
    
    if not eligible_peers:
        return []
    
    # Apply additional filters
    filtered_peers = []
    for peer in eligible_peers:
        # Check benchmark exists and has CPU or GPUs with positive values
        benchmark = peer.get("benchmark", {})
        if not isinstance(benchmark, dict) or not benchmark:
            continue
        
        has_cpu_or_gpu = False
        cpu_data = benchmark.get("cpu", {})
        if isinstance(cpu_data, dict) and cpu_data.get("gflops", 0.0) > 0.0:
            has_cpu_or_gpu = True
        
        if not has_cpu_or_gpu:
            gpus = benchmark.get("gpus", [])
            if isinstance(gpus, list):
                for gpu in gpus:
                    if isinstance(gpu, dict) and gpu.get("gflops", 0.0) > 0.0:
                        has_cpu_or_gpu = True
                        break
        
        if not has_cpu_or_gpu:
            continue
        
        # Check CPU and RAM utilization limits
        stats_payload = peer.get("stats_payload", {})
        if not isinstance(stats_payload, dict):
            continue
        
        cpu_percent = stats_payload.get("cpu_percent", 0.0)
        ram_percent = stats_payload.get("ram_percent", 0.0)
        
        try:
            cpu_percent = float(cpu_percent)
            ram_percent = float(ram_percent)
        except (ValueError, TypeError):
            continue
        
        if cpu_percent >= MAX_CPU_UTIL or ram_percent >= MAX_RAM_UTIL:
            continue
        
        # Check freshness: now_ms - last_time_received <= stale_threshold_ms
        last_time_received = peer.get("last_time_received", 0)
        age_ms = now_ms - last_time_received
        if age_ms > stale_threshold_ms:
            continue
        
        filtered_peers.append(peer)
    
    if not filtered_peers:
        return []
    
    # Step 2: Call capacity_score
    peers_with_capacity = capacity_score(
        filtered_peers,
        w_cpu=w_cpu,
        w_gpu=w_gpu,
        w_ram=w_ram,
        w_disk=w_disk,
    )
    
    # Drop peers where capacity_score <= 0
    # (similar logic to plan_static_weighted_shards - keep peers with valid data)
    peers_with_positive_capacity = []
    for peer in peers_with_capacity:
        capacity_score_val = peer.get("capacity_score", 0.0)
        
        # Check if peer has any non-zero raw benchmark data
        benchmark = peer.get("benchmark", {})
        has_nonzero_data = False
        if isinstance(benchmark, dict):
            cpu_data = benchmark.get("cpu", {})
            if isinstance(cpu_data, dict) and cpu_data.get("gflops", 0.0) > 0.0:
                has_nonzero_data = True
            gpus = benchmark.get("gpus", [])
            if isinstance(gpus, list):
                for gpu in gpus:
                    if isinstance(gpu, dict) and gpu.get("gflops", 0.0) > 0.0:
                        has_nonzero_data = True
                        break
            ram_data = benchmark.get("ram", {})
            if isinstance(ram_data, dict) and ram_data.get("bandwidth_gbps", 0.0) > 0.0:
                has_nonzero_data = True
            disk_data = benchmark.get("disk", {})
            if isinstance(disk_data, dict):
                read_speed = disk_data.get("read_speed_mbps", 0.0)
                write_speed = disk_data.get("write_speed_mbps", 0.0)
                if read_speed > 0.0 or write_speed > 0.0:
                    has_nonzero_data = True
        
        if capacity_score_val > 0.0 or has_nonzero_data:
            peers_with_positive_capacity.append(peer)
    
    if not peers_with_positive_capacity:
        return []
    
    # Step 3: Compute load_factor
    peers_with_load = live_load_factor(peers_with_positive_capacity, alpha=alpha)
    
    # Drop peers where load_factor <= 0.0
    peers_with_positive_load = []
    for peer in peers_with_load:
        load_factor_val = peer.get("live_load_factor", 0.0)
        if load_factor_val > 0.0:
            peers_with_positive_load.append(peer)
    
    if not peers_with_positive_load:
        return []
    
    # Step 4: Compute freshness_factor
    peers_with_freshness = []
    for peer in peers_with_positive_load:
        peer_with_freshness = peer.copy()
        last_time_received = peer.get("last_time_received", 0)
        age_ms = now_ms - last_time_received
        
        if age_ms <= 0:
            freshness_factor = 1.0
        elif age_ms >= stale_threshold_ms:
            freshness_factor = 0.0
        else:
            # Linear decay from 1.0 to 0.1
            freshness_factor = max(0.1, 1.0 - 0.9 * age_ms / stale_threshold_ms)
        
        peer_with_freshness["freshness_factor"] = freshness_factor
        
        # Drop peers where freshness_factor <= 0.0
        if freshness_factor > 0.0:
            peers_with_freshness.append(peer_with_freshness)
    
    if not peers_with_freshness:
        return []
    
    # Step 5: Compute network_factor
    peers_with_network = network_factor(
        peers_with_freshness,
        min_net_factor=min_net_factor,
    )
    
    # Step 6: Compute effective_score
    peers_with_effective = []
    for peer in peers_with_network:
        compute_score = peer.get("capacity_score", 0.0)
        load_factor_val = peer.get("live_load_factor", 0.0)
        freshness_factor_val = peer.get("freshness_factor", 0.0)
        network_factor_val = peer.get("network_factor", 0.0)
        
        effective_score = (
            compute_score *
            load_factor_val *
            freshness_factor_val *
            network_factor_val
        )
        
        # Check if peer has valid benchmark data (to handle case where capacity_score = 0.0
        # after normalization but peer still has valid data)
        benchmark = peer.get("benchmark", {})
        has_nonzero_data = False
        if isinstance(benchmark, dict):
            cpu_data = benchmark.get("cpu", {})
            if isinstance(cpu_data, dict) and cpu_data.get("gflops", 0.0) > 0.0:
                has_nonzero_data = True
            gpus = benchmark.get("gpus", [])
            if isinstance(gpus, list):
                for gpu in gpus:
                    if isinstance(gpu, dict) and gpu.get("gflops", 0.0) > 0.0:
                        has_nonzero_data = True
                        break
            ram_data = benchmark.get("ram", {})
            if isinstance(ram_data, dict) and ram_data.get("bandwidth_gbps", 0.0) > 0.0:
                has_nonzero_data = True
            disk_data = benchmark.get("disk", {})
            if isinstance(disk_data, dict):
                read_speed = disk_data.get("read_speed_mbps", 0.0)
                write_speed = disk_data.get("write_speed_mbps", 0.0)
                if read_speed > 0.0 or write_speed > 0.0:
                    has_nonzero_data = True
        
        # Keep peer if effective_score > 0 OR if it has valid data and all factors > 0
        # (the latter handles the case where normalization results in capacity_score = 0.0
        # for the minimum peer, but it still has valid benchmark data)
        if effective_score > 0.0 or (has_nonzero_data and load_factor_val > 0.0 and freshness_factor_val > 0.0 and network_factor_val > 0.0):
            # If effective_score is 0.0 but we're keeping it due to valid data,
            # use a small epsilon to ensure it gets some allocation
            if effective_score == 0.0 and has_nonzero_data:
                effective_score = 0.0001  # Small epsilon to ensure it's included
            
            peer_with_effective = peer.copy()
            peer_with_effective["effective_score"] = effective_score
            peers_with_effective.append(peer_with_effective)
    
    if not peers_with_effective:
        return []
    
    # Step 7: Normalize effective scores
    total_effective = sum(peer.get("effective_score", 0.0) for peer in peers_with_effective)
    
    if total_effective <= 0.0:
        raise ValueError("No eligible peers with positive capacity")
    
    # Compute fractions
    for peer in peers_with_effective:
        effective_score = peer.get("effective_score", 0.0)
        frac = effective_score / total_effective
        peer["capacity_fraction"] = frac  # Store for reference, not in final output
    
    # Step 8: Allocate total_batches
    allocations = []
    for peer in peers_with_effective:
        frac = peer.get("capacity_fraction", 0.0)
        raw = frac * total_batches
        base = int(raw)  # floor
        remainder = raw - base
        allocations.append({
            "peer": peer,
            "base": base,
            "remainder": remainder,
        })
    
    # Ensure all peers with valid data get at least 1 batch
    # (handles case where minimum peer after normalization gets 0 base allocation)
    for alloc in allocations:
        peer = alloc["peer"]
        benchmark = peer.get("benchmark", {})
        has_nonzero_data = False
        if isinstance(benchmark, dict):
            cpu_data = benchmark.get("cpu", {})
            if isinstance(cpu_data, dict) and cpu_data.get("gflops", 0.0) > 0.0:
                has_nonzero_data = True
            gpus = benchmark.get("gpus", [])
            if isinstance(gpus, list):
                for gpu in gpus:
                    if isinstance(gpu, dict) and gpu.get("gflops", 0.0) > 0.0:
                        has_nonzero_data = True
                        break
            ram_data = benchmark.get("ram", {})
            if isinstance(ram_data, dict) and ram_data.get("bandwidth_gbps", 0.0) > 0.0:
                has_nonzero_data = True
            disk_data = benchmark.get("disk", {})
            if isinstance(disk_data, dict):
                read_speed = disk_data.get("read_speed_mbps", 0.0)
                write_speed = disk_data.get("write_speed_mbps", 0.0)
                if read_speed > 0.0 or write_speed > 0.0:
                    has_nonzero_data = True
        
        # Ensure peers with valid data get at least 1 batch
        if has_nonzero_data and alloc["base"] == 0:
            alloc["base"] = 1
    
    # Calculate remaining batches
    total_base = sum(alloc["base"] for alloc in allocations)
    remaining = total_batches - total_base
    
    # If we've overallocated due to minimum guarantees, adjust
    if remaining < 0:
        # Sort by base descending, then reduce from highest
        allocations.sort(key=lambda x: (x["base"], x["remainder"]), reverse=True)
        for alloc in allocations:
            if remaining >= 0:
                break
            if alloc["base"] > 1:  # Don't reduce below 1 for peers with valid data
                alloc["base"] -= 1
                remaining += 1
        # If still negative, we have more peers than batches - give 1 to each
        if remaining < 0:
            for alloc in allocations:
                alloc["base"] = 1
            remaining = total_batches - len(allocations)
    
    # Sort by remainder descending and assign +1 to top "remaining" peers
    allocations.sort(key=lambda x: x["remainder"], reverse=True)
    
    for i in range(min(remaining, len(allocations))):
        allocations[i]["base"] += 1
    
    # Step 9: Build result list
    result = []
    for alloc in allocations:
        peer = alloc["peer"]
        allocated_batches = alloc["base"]
        
        result.append({
            "host": peer.get("host", ""),
            "heartbeat_port": peer.get("heartbeat_port", 0),
            "comms_port": peer.get("comms_port", 0),
            "allocated_batches": allocated_batches,
            "effective_score": peer.get("effective_score", 0.0),
            "compute_score": peer.get("capacity_score", 0.0),
            "load_factor": peer.get("live_load_factor", 0.0),
            "freshness_factor": peer.get("freshness_factor", 0.0),
            "network_factor": peer.get("network_factor", 0.0),
        })
    
    return result

