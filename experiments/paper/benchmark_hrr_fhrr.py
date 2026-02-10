"""
Benchmark script for comparing HRR and FHRR backends on 2D spatial costmaps.

This script generates all experimental data for the paper's Results section,
measuring stage-by-stage latency and reconstruction quality.

Usage:
    python benchmark_hrr_fhrr.py --output results.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt, zoom
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hyperspace.backends.hrr import HRRBackend
from hyperspace.backends.fhrr import FHRRBackend


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP for regression decoding (FHRR)."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def generate_smooth_gaussian_terrain(
    resolution: int,
    num_gaussians: int = 8,
    sigma: float = 0.15,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate smooth Gaussian terrain costmap."""
    np.random.seed(seed)
    
    # Create grid
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten for processing
    positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (N, 2)
    
    # Generate Gaussian hills
    costmap = torch.zeros(resolution * resolution)
    for _ in range(num_gaussians):
        amplitude = np.random.uniform(0.3, 1.0)
        center_x = np.random.uniform(0, 1)
        center_y = np.random.uniform(0, 1)
        
        dist_sq = (positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2
        costmap += amplitude * torch.exp(-dist_sq / (2 * sigma**2))
    
    # Normalize to [1, resolution] (scaled to match spatial extent)
    costmap = (costmap - costmap.min()) / (costmap.max() - costmap.min() + 1e-8)
    costmap = costmap * (resolution - 1) + 1.0  # Scale from [0, 1] to [1, resolution]
    
    return positions, costmap


def generate_obstacle_distance_field(
    resolution: int,
    num_obstacles: int = 5,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate obstacle distance field costmap."""
    np.random.seed(seed)
    
    # Create grid
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Generate obstacles
    min_dist = torch.ones(resolution * resolution) * float('inf')
    for _ in range(num_obstacles):
        center_x = np.random.uniform(0.1, 0.9)
        center_y = np.random.uniform(0.1, 0.9)
        radius = np.random.uniform(0.05, 0.15)
        
        dist = torch.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
        obstacle_dist = torch.clamp(dist - radius, min=0)
        min_dist = torch.minimum(min_dist, obstacle_dist)
    
    # Normalize to [1, resolution] (scaled to match spatial extent)
    costmap = min_dist / (min_dist.max() + 1e-8)
    costmap = costmap * (resolution - 1) + 1.0  # Scale from [0, 1] to [1, resolution]
    
    return positions, costmap


def generate_mixed_terrain(
    resolution: int,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate mixed complexity terrain."""
    # Generate components
    pos1, gauss = generate_smooth_gaussian_terrain(resolution, seed=seed)
    pos2, dist = generate_obstacle_distance_field(resolution, seed=seed + 1)
    
    # Weighted combination with noise
    np.random.seed(seed)
    noise = torch.randn_like(gauss) * 0.05
    costmap = 0.6 * gauss + 0.3 * dist + 0.1 * noise
    
    # Normalize to [1, resolution] (scaled to match spatial extent)
    costmap = torch.clamp(costmap, 0, 1)
    costmap = (costmap - costmap.min()) / (costmap.max() - costmap.min() + 1e-8)
    costmap = costmap * (resolution - 1) + 1.0  # Scale from [0, 1] to [1, resolution]
    
    return pos1, costmap


def generate_x_pattern_terrain(
    resolution: int,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate simple X-pattern terrain: high cost everywhere except X from corner to corner (low cost)."""
    np.random.seed(seed)
    
    # Create grid
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten for processing
    positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (N, 2)
    
    # Initialize with high cost values
    costmap = torch.ones(resolution * resolution) * resolution
    
    # Create X pattern with low cost
    # For each point, check if it's near the diagonal or anti-diagonal
    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    
    # Distance from main diagonal (y = x)
    dist_main_diag = torch.abs(y_vals - x_vals)
    
    # Distance from anti-diagonal (y = 1 - x)
    dist_anti_diag = torch.abs(y_vals - (1 - x_vals))
    
    # Width of the X lines (as fraction of grid)
    line_width = 2.0 / resolution  # About 2 pixels wide
    
    # Points on the X have low cost
    on_x = (dist_main_diag < line_width) | (dist_anti_diag < line_width)
    costmap[on_x] = 1.0  # Low cost on the X
    
    return positions, costmap


def create_spline(points: List[Tuple[int, int]], num_samples: int) -> np.ndarray:
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]

    cs = CubicSpline(
        np.linspace(0, 1, len(x_points)),
        np.vstack([x_points, y_points]), axis=1
    )
    t = np.linspace(0, 1, num_samples)
    spline_points = cs(t).T
    return spline_points

def create_costmap(world: np.ndarray, spline_points: np.ndarray, path_cost: float) -> np.ndarray:
    costmap = np.zeros(world.shape)
    for point in spline_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < world.shape[0] and 0 <= y < world.shape[1]:
            costmap[x, y] = path_cost

    path_mask = (costmap == path_cost)
    distances = distance_transform_edt(~path_mask)
    costmap = 1 + distances
    return costmap

def create_discretized_costmaps(cost_maps: List[np.ndarray], num_samples: int) -> List[np.ndarray]:
    """
    convert the continuous valued cost maps into a discretized version
    """
    discretized_maps = []
    for cost_map in cost_maps:
        normalized_map = (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min())
        # Create a new map with the same shape but discretized values
        discretized_map = np.digitize(normalized_map, bins=np.linspace(0, 1, num_samples))
        discretized_maps.append(discretized_map)
    return discretized_maps


def split_train_test(
    positions: torch.Tensor,
    values: torch.Tensor,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split data into train/test sets."""
    torch.manual_seed(seed)
    n = len(positions)
    indices = torch.randperm(n)
    
    n_train = int(n * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return (positions[train_idx], values[train_idx],
            positions[test_idx], values[test_idx])


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function call and return (result, elapsed_time)."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    result = func(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark_encoding(
    backend,
    positions: torch.Tensor,
    values: torch.Tensor,
    device: str
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Benchmark memory storage (encoding) phase - BATCHED for fair comparison."""
    positions = positions.to(device)
    values = values.to(device)
    
    timings = {}
    
    # Time full encoding process using batch operations
    total_start = time.perf_counter()
    
    # Batch positional encoding: (N, 2) → (N, D)
    pos_vecs, _ = backend.positional_encoding(positions)
    
    # Batch value encoding: (N,) → (N, 1) → (N, D)
    val_vecs, _ = backend.value_encoding(values.unsqueeze(1))
    
    # Batch bind (pairwise): (N, D) × (N, D) → (N, D)
    bound_vecs, _ = backend.bind(pos_vecs, val_vecs)
    
    # List bundle (reduce batch): (N, D) → (D)
    memory, _ = backend.bundle(bound_vecs, None)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - total_start
    
    timings['total_encoding'] = total_time
    timings['per_sample'] = total_time / len(positions)
    
    # Normalize memory
    memory, _ = backend.normalize(memory)
    
    return memory, timings

def convert_costmap_to_points(costmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a costmap to a set of points sampled according to the cost distribution.
    """
    # Get coordinates
    x_indices, y_indices = np.meshgrid(
        np.arange(costmap.shape[1]),  # x = columns
        np.arange(costmap.shape[0])   # y = rows
    )

    # Flatten to a list of (x, y) points
    points = np.stack([x_indices.ravel(), y_indices.ravel()], axis=-1)

    # Flatten the costmap values
    labels = costmap.ravel()

    # Subtract 1 from all labels to ensure they are zero-indexed
    # labels -= 1
    return points, labels

def generate_spline_terrain(downscale_factor: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    WORLD_SIZE: Tuple[int, int] = (1000, 1000)
    START_POSITION: Tuple[int, int] = (1, 1)
    END_POSITION: Tuple[int, int] = (999, 999)
    PATH_COST: float = 1.0
    NUM_SPLINE_SAMPLES: int = 10000
    SPLINE_POINTS_BEGINNING: List[Tuple[int, int]] = [
        START_POSITION,
        (200, 200),
        (200, 400),
        (500, 500),
        (800, 600),
        (800, 800),
        END_POSITION
    ]

    cost_map = create_costmap(
        world=np.zeros(WORLD_SIZE),
        spline_points=create_spline(SPLINE_POINTS_BEGINNING, NUM_SPLINE_SAMPLES),
        path_cost=PATH_COST
    )

    cost_map = zoom(cost_map, downscale_factor, order=1)

    X, y = convert_costmap_to_points(cost_map)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    y_norm = (y - y.min()) / (y.max() - y.min())
    y_norm = y_norm * 10  # Scale to [0, 10] range
    y_norm += 1
    # y_norm = y_norm.unsqueeze(-1)

    return X.numpy(), y_norm.numpy()

def benchmark_query(
    backend,
    memory: torch.Tensor,
    query_positions: torch.Tensor,
    device: str,
    num_queries: int = 100
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Benchmark query (positional inversion) phase - BATCHED for fair comparison."""
    query_positions = query_positions[:num_queries].to(device)
    memory = memory.to(device)
    
    timings = {}
    
    # Time batched query operations
    start = time.perf_counter()
    
    # Batch positional encoding: (Q, 2) → (Q, D)
    pos_vecs, _ = backend.positional_encoding(query_positions)
    
    # Batch invert: (Q, D) → (Q, D)
    pos_inv, _ = backend.invert(pos_vecs)
    
    # Single-to-batch bind: (D) with (Q, D) → (Q, D)
    decoded_batch, _ = backend.bind(memory, pos_inv)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - start
    
    timings['total_query'] = total_time
    timings['per_query'] = total_time / num_queries
    timings['std_query'] = 0.0  # No per-query variation in batched mode
    
    return decoded_batch, timings


def build_codebook(backend, k: int = 64, device: str = 'cpu', max_value: float = 256.0) -> torch.Tensor:
    """Build uniform codebook for cleanup."""
    values = torch.linspace(1, max_value, k).unsqueeze(1).to(device)  # Scale from 1 to max_value (resolution)
    codebook_vecs = []
    
    for v in values:
        vec, _ = backend.value_encoding(v)
        codebook_vecs.append(vec)
    
    codebook = torch.stack(codebook_vecs).to(device)
    return codebook


def benchmark_cleanup(
    backend,
    decoded_vecs: torch.Tensor,
    codebook: torch.Tensor,
    num_iterations: int = 4,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Benchmark resonator cleanup phase."""
    decoded_vecs = decoded_vecs.to(device)
    codebook = codebook.to(device)
    
    timings = {}
    
    # Time cleanup iterations
    start = time.perf_counter()
    
    cleaned = decoded_vecs.clone()
    for _ in range(num_iterations):
        cleaned, _ = backend._resonator_cleanup(cleaned, codebook, num_iters=1)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - start
    
    timings['total_cleanup'] = total_time
    timings['per_query'] = total_time / len(decoded_vecs)
    timings['per_iteration'] = total_time / (len(decoded_vecs) * num_iterations)
    
    return cleaned, timings


def train_mlp_decoder(
    backend,
    train_pos: torch.Tensor,
    train_vals: torch.Tensor,
    vector_dim: int,
    device: str,
    epochs: int = 50
) -> SimpleMLP:
    """Train MLP decoder for FHRR."""
    model = SimpleMLP(vector_dim * 2).to(device)  # *2 for real+imag parts
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Encode training data
    train_pos = train_pos.to(device)
    train_vals = train_vals.to(device)
    
    # Encode once and convert complex to real
    with torch.no_grad():
        train_vecs, _ = backend.value_encoding(train_vals.unsqueeze(1))
        # Convert complex to real by concatenating real and imaginary parts
        if train_vecs.is_complex():
            train_vecs = torch.cat([train_vecs.real, train_vecs.imag], dim=-1)
        train_vecs = train_vecs.clone().detach().requires_grad_(False)
    
    # Train
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(train_vecs)
        loss = criterion(pred, train_vals)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


def benchmark_regression_codebook(
    backend,
    cleaned_vecs: torch.Tensor,
    codebook: torch.Tensor,
    device: str,
    temperature: float = 1.0,
    max_value: float = 256.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Benchmark codebook-based regression."""
    cleaned_vecs = cleaned_vecs.to(device)
    codebook = codebook.to(device)
    
    timings = {}
    
    start = time.perf_counter()
    
    # Compute similarities
    sims, _ = backend.similarity(cleaned_vecs, codebook, mode='all_pairs')
    
    # Softmax
    weights = torch.softmax(sims / temperature, dim=1)
    
    # Expectation
    codebook_values = torch.linspace(1, max_value, len(codebook)).to(device)  # Scale from 1 to max_value (resolution)
    predictions = (weights * codebook_values).sum(dim=1)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - start
    
    timings['total_regression'] = total_time
    timings['per_query'] = total_time / len(cleaned_vecs)
    
    return predictions, timings


def benchmark_regression_mlp(
    model: SimpleMLP,
    cleaned_vecs: torch.Tensor,
    device: str
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Benchmark MLP-based regression."""
    cleaned_vecs = cleaned_vecs.to(device)
    
    # Convert complex to real if needed
    if cleaned_vecs.is_complex():
        cleaned_vecs = torch.cat([cleaned_vecs.real, cleaned_vecs.imag], dim=-1)
    
    timings = {}
    
    start = time.perf_counter()
    
    with torch.no_grad():
        predictions = model(cleaned_vecs)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - start
    
    timings['total_regression'] = total_time
    timings['per_query'] = total_time / len(cleaned_vecs)
    
    return predictions, timings


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute RMSE between predictions and targets."""
    mse = ((predictions - targets) ** 2).mean()
    return torch.sqrt(mse).item()


def get_memory_footprint(backend, memory: torch.Tensor, codebook: torch.Tensor = None) -> Dict[str, float]:
    """Compute memory footprint in MB."""
    footprint = {}
    
    # Memory vector
    mem_bytes = memory.element_size() * memory.nelement()
    footprint['memory_vector_mb'] = mem_bytes / (1024 ** 2)
    
    # Codebook (if provided)
    if codebook is not None:
        cb_bytes = codebook.element_size() * codebook.nelement()
        footprint['codebook_mb'] = cb_bytes / (1024 ** 2)
        footprint['total_mb'] = footprint['memory_vector_mb'] + footprint['codebook_mb']
    else:
        footprint['total_mb'] = footprint['memory_vector_mb']
    
    return footprint


def run_full_benchmark(
    backend_name: str,
    resolution: int,
    vector_dim: int,
    vector_length_scale: float = 1.0,
    device: str = 'cpu',
    terrain_type: str = 'mixed',
    codebook_size: int = 64,
    cleanup_iterations: int = 4,
    num_test_queries: int = 100,
    seed: int = 42
) -> Dict:
    """Run complete benchmark for one configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend_name} | Resolution: {1000 * resolution}x{1000 * resolution} | D: {vector_dim}")
    print(f"{'='*60}")
    
    # Initialize backend
    if backend_name == 'HRR':
        backend = HRRBackend(
            vector_dim=vector_dim,
            length_scale=vector_length_scale,
            device=device,
            env_dim=2,
            value_dim=1,
            seed=seed
        )
    else:  # FHRR
        backend = FHRRBackend(
            vector_dim=vector_dim,
            length_scale=vector_length_scale,
            device=device,
            env_dim=2,
            value_dim=1,
            seed=seed
        )
    
    # Generate terrain
    print(f"Generating {terrain_type} terrain...")
    if terrain_type == 'gaussian':
        positions, values = generate_smooth_gaussian_terrain(resolution, seed=seed)
    elif terrain_type == 'obstacle':
        positions, values = generate_obstacle_distance_field(resolution, seed=seed)
    elif terrain_type == 'xpattern':
        positions, values = generate_x_pattern_terrain(resolution, seed=seed)
    elif terrain_type == 'spline':
        positions, values = generate_spline_terrain(resolution)
    else:
        positions, values = generate_mixed_terrain(resolution, seed=seed)

    positions = torch.from_numpy(positions).to(device)
    values = torch.from_numpy(values).to(device)
    
    # Split train/test
    train_pos, train_vals, test_pos, test_vals = split_train_test(
        positions, values, train_ratio=0.8, seed=seed
    )
    
    print(f"Train samples: {len(train_pos)}, Test samples: {len(test_pos)}")
    
    results = {
        'backend': backend_name,
        'resolution': resolution,
        'vector_dim': vector_dim,
        'terrain_type': terrain_type,
        'n_train': len(train_pos),
        'n_test': len(test_pos),
    }
    
    # 1. Encoding Phase
    print("1. Encoding phase...")
    memory, encoding_times = benchmark_encoding(backend, train_pos, train_vals, device)
    results['encoding'] = encoding_times
    
    # 2. Query Phase
    print("2. Query phase...")
    decoded_vecs, query_times = benchmark_query(
        backend, memory, test_pos, device, num_queries=num_test_queries
    )
    results['query'] = query_times
    
    # 3. Build Codebook
    print("3. Building codebook...")
    codebook = build_codebook(backend, k=codebook_size, device=device, max_value=float(resolution))
    
    # 4. Cleanup Phase
    print("4. Cleanup phase...")
    cleaned_vecs, cleanup_times = benchmark_cleanup(
        backend, decoded_vecs, codebook, num_iterations=cleanup_iterations, device=device
    )
    results['cleanup'] = cleanup_times
    
    # 5. Regression Phase
    print("5. Regression phase...")
    if backend_name == 'HRR':
        # Use codebook regression
        predictions, regress_times = benchmark_regression_codebook(
            backend, cleaned_vecs, codebook, device, max_value=float(resolution)
        )
        results['regression'] = regress_times
        results['regression']['method'] = 'codebook'
    else:
        # Train and use MLP
        print("   Training MLP decoder...")
        mlp = train_mlp_decoder(backend, train_pos, train_vals, vector_dim, device)
        predictions, regress_times = benchmark_regression_mlp(mlp, cleaned_vecs, device)
        results['regression'] = regress_times
        results['regression']['method'] = 'mlp'
    
    # 6. Compute RMSE
    test_vals_subset = test_vals[:num_test_queries].to(device)
    rmse = compute_rmse(predictions, test_vals_subset)
    results['rmse'] = rmse
    
    # 7. Memory Footprint
    footprint = get_memory_footprint(backend, memory, codebook)
    results['memory_footprint'] = footprint
    
    # 8. Total time
    total_time = (
        encoding_times['total_encoding'] +
        query_times['total_query'] +
        cleanup_times['total_cleanup'] +
        regress_times['total_regression']
    )
    results['total_time'] = total_time
    
    print(f"\nResults Summary:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Encoding: {encoding_times['total_encoding']:.3f}s")
    print(f"  Query: {query_times['total_query']*1000:.2f}ms total, {query_times['per_query']*1000:.2f}ms per query")
    print(f"  Cleanup: {cleanup_times['total_cleanup']*1000:.2f}ms total, {cleanup_times['per_query']*1000:.2f}ms per query")
    print(f"  Regression: {regress_times['total_regression']*1000:.2f}ms total, {regress_times['per_query']*1000:.2f}ms per query")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Memory: {footprint['total_mb']:.2f} MB")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark HRR vs FHRR backends')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu/cuda)')
    parser.add_argument('--resolutions', type=float, nargs='+',
                        default=[0.01, 0.02, 0.03, 0.04, 0.05],
                        help='Grid resolutions to test')
    parser.add_argument('--vector-dims', type=int, nargs='+',
                        default=[1024, 2048, 4096, 8192],
                        help='Vector dimensions to test')
    parser.add_argument('--terrain-type', type=str, default='xpattern',
                        choices=['gaussian', 'obstacle', 'mixed', 'xpattern', 'spline', 'spline_end'],
                        help='Terrain type to generate')
    parser.add_argument('--codebook-size', type=int, default=64,
                        help='Codebook size for cleanup')
    parser.add_argument('--cleanup-iters', type=int, default=4,
                        help='Number of cleanup iterations')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='Number of test queries')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Run benchmarks
    all_results = []
    
    for resolution in args.resolutions:
        for vector_dim in args.vector_dims:
            # Benchmark HRR
            hrr_results = run_full_benchmark(
                'HRR',
                resolution=resolution,
                vector_dim=vector_dim,
                vector_length_scale=2.0,  # Longer vectors for HRR to improve capacity
                device=args.device,
                terrain_type=args.terrain_type,
                codebook_size=args.codebook_size,
                cleanup_iterations=args.cleanup_iters,
                num_test_queries=args.num_queries,
                seed=args.seed
            )
            all_results.append(hrr_results)
            
            # Benchmark FHRR
            fhrr_results = run_full_benchmark(
                'FHRR',
                resolution=resolution,
                vector_dim=vector_dim,
                vector_length_scale=2.0,
                device=args.device,
                terrain_type=args.terrain_type,
                codebook_size=args.codebook_size,
                cleanup_iterations=args.cleanup_iters,
                num_test_queries=args.num_queries,
                seed=args.seed
            )
            all_results.append(fhrr_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print comparison table
    print("\n\nCOMPARISON SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'Backend':<8} {'Encode(s)':<12} {'Query(ms)':<12} {'Cleanup(ms)':<12} {'Regress(ms)':<12} {'RMSE':<8} {'Memory(MB)':<12}")
    print("="*80)
    
    for i in range(0, len(all_results), 2):
        hrr = all_results[i]
        fhrr = all_results[i+1]
        
        config = f"{hrr['resolution']}² D={hrr['vector_dim']}"
        
        print(f"{config:<20} {'HRR':<8} {hrr['encoding']['total_encoding']:<12.3f} {hrr['query']['per_query']*1000:<12.2f} {hrr['cleanup']['per_query']*1000:<12.2f} {hrr['regression']['per_query']*1000:<12.2f} {hrr['rmse']:<8.4f} {hrr['memory_footprint']['total_mb']:<12.2f}")
        print(f"{'':<20} {'FHRR':<8} {fhrr['encoding']['total_encoding']:<12.3f} {fhrr['query']['per_query']*1000:<12.2f} {fhrr['cleanup']['per_query']*1000:<12.2f} {fhrr['regression']['per_query']*1000:<12.2f} {fhrr['rmse']:<8.4f} {fhrr['memory_footprint']['total_mb']:<12.2f}")
        
        speedup_encode = hrr['encoding']['total_encoding'] / fhrr['encoding']['total_encoding']
        speedup_query = hrr['query']['per_query'] / fhrr['query']['per_query']
        speedup_total = hrr['total_time'] / fhrr['total_time']
        
        print(f"{'':<20} {'Speedup':<8} {speedup_encode:<12.2f}× {speedup_query:<12.2f}× {'':<12} {'':<12} {'':<8} {'':<12}")
        print("-"*80)


if __name__ == '__main__':
    main()
