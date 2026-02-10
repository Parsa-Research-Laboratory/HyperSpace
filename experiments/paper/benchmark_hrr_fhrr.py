"""
Benchmark script for comparing HRR and FHRR backends on 2D spatial costmaps.

This script generates all experimental data for the paper's Results section,
measuring stage-by-stage latency and reconstruction quality.

Usage:
    python benchmark_hrr_fhrr.py --output results.json
"""

import argparse
import json
import os
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

from hyperspace.core.value_encoder import value_encoder_module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hyperspace.backends.hrr import HRRBackend
from hyperspace.backends.fhrr import FHRRBackend
from hyperspace.core import (
    PositionalEncoderModule,
    ValueEncoderModule,
    PositionalInversionModule,
    CleanupModule,
    MemoryStorageModule,
    RegressionModule
)

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





def benchmark_query(
    backend,
    memory: torch.Tensor,
    query_positions: torch.Tensor,
    device: str,
    num_queries: int = 100
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Benchmark query (positional inversion) phase - BATCHED for fair comparison."""
    query_positions = query_positions.to(device)
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
    timings['per_query'] = total_time / query_positions.shape[0]
    timings['std_query'] = 0.0  # No per-query variation in batched mode
    
    return decoded_batch, timings


def build_codebook(backend, k: int = 64, device: str = 'cpu', min_value: float = 1.0, max_value: float = 11.0, spacing: float = 0.1) -> torch.Tensor:
    """Build uniform codebook for cleanup using arange with fixed spacing (matching notebook).
    
    Args:
        backend: The backend instance
        k: Number of codebook entries (deprecated - spacing is used instead)
        device: Device to use
        min_value: Minimum value in the data range
        max_value: Maximum value in the data range
        spacing: Spacing between codebook entries (default: 0.1 to match notebook)
    """
    # Use arange with 0.1 spacing to match notebook behavior
    values = torch.arange(min_value, max_value, spacing).unsqueeze(1).to(device)
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
    method: str = 'resonator',
    num_iterations: int = 4,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Benchmark cleanup phase with configurable method.
    
    Args:
        backend: The backend instance
        decoded_vecs: Vectors to clean up
        codebook: Codebook for cleanup
        method: Cleanup method - 'none', 'resonator', or 'modern_hopfield'
        num_iterations: Number of cleanup iterations
        device: Device to run on
    """
    decoded_vecs = decoded_vecs.to(device)
    codebook = codebook.to(device)
    
    timings = {}
    
    if method == 'none':
        # No cleanup - return vectors as-is
        timings['total_cleanup'] = 0.0
        timings['per_query'] = 0.0
        timings['per_iteration'] = 0.0
        return decoded_vecs, timings
    
    # Time cleanup iterations
    start = time.perf_counter()
    
    cleaned = decoded_vecs.clone()
    if method == 'resonator':
        cleaned, _ = backend._resonator_cleanup(cleaned, codebook, num_iters=num_iterations)
    elif method == 'modern_hopfield':  
        cleaned, _ = backend._hopfield_cleanup(cleaned, codebook, num_iters=num_iterations)
    else:
        raise ValueError(f"Unknown cleanup method: {method}")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - start
    
    timings['total_cleanup'] = total_time
    timings['per_query'] = total_time / len(decoded_vecs)
    timings['per_iteration'] = total_time / (len(decoded_vecs) * num_iterations)
    
    return cleaned, timings


def benchmark_regression_codebook(
    backend,
    cleaned_vecs: torch.Tensor,
    codebook: torch.Tensor,
    device: str,
    temperature: float = 1.0,
    min_value: float = 1.0,
    max_value: float = 11.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Benchmark codebook-based regression.
    
    Args:
        backend: The backend instance
        cleaned_vecs: Cleaned vectors to regress
        codebook: Codebook vectors
        device: Device to use
        temperature: Temperature for softmax
        min_value: Minimum value in data range
        max_value: Maximum value in data range
    """
    cleaned_vecs = cleaned_vecs.to(device)
    codebook = codebook.to(device)
    
    timings = {}
    
    start = time.perf_counter()
    
    # Compute similarities
    sims, _ = backend.similarity(cleaned_vecs, codebook, mode='all_pairs')
    
    # Softmax
    weights = torch.softmax(sims / temperature, dim=1)
    
    # Expectation over codebook values
    codebook_values = torch.linspace(min_value, max_value, len(codebook)).to(device)
    predictions = (weights * codebook_values).sum(dim=1)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - start
    
    timings['total_regression'] = total_time
    timings['per_query'] = total_time / len(cleaned_vecs)
    
    return predictions, timings


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute MSE between predictions and targets."""
    mse = ((predictions - targets) ** 2).mean()
    return mse.item()


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
    resolution: float = 0.01,
    vector_dim: int = 1024,
    vector_length_scale: float = 2.0,
    device: str = 'cpu',
    cleanup_method: str = 'resonator',
    cleanup_iterations: int = 2,
    regression_method: str = 'codebook',
    seed: int = 42,
    scratch_dir: str = './scratch',
) -> Dict:
    """Run complete benchmark for one configuration."""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {backend_name}")
    print(f"  Resolution: {int(1000 * resolution)}x{int(1000 * resolution)} | Vector Dimension: {vector_dim}")
    print(f"  Cleanup: {cleanup_method} ({cleanup_iterations} iters) | Regression: {regression_method}")
    print(f"{'='*80}")
    
    # Initialize backend
    if backend_name == 'HRR':
        backend = HRRBackend(
            vector_dim=vector_dim,
            length_scale=vector_length_scale,
            device=device,
            env_dim=2,
            seed=seed
        )
    elif backend_name == 'FHRR':
        backend = FHRRBackend(
            vector_dim=vector_dim,
            length_scale=vector_length_scale,
            device=device,
            env_dim=2,
            seed=seed
        )
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

    # create scratch directory
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    positions, values = generate_spline_terrain(resolution)
    positions = torch.from_numpy(positions).to(device)
    values = torch.from_numpy(values).to(device)  # Dummy values for testing

    # plot the terrain for visualization
    import matplotlib.pyplot as plt
    plt.scatter(positions[:, 0].cpu(), positions[:, 1].cpu(), c=values.cpu(), cmap='viridis')
    plt.colorbar(label='Cost')
    plt.title(f'Terrain')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(scratch_dir, f'terrain_res{resolution}.png'))
    plt.close()
    
    # Split train/test
    train_pos, train_vals, test_pos, test_vals = split_train_test(
        positions, values, train_ratio=0.8, seed=0
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"     ├─ Train samples: {len(train_pos):,}")
    print(f"     ├─ Test samples:  {len(test_pos):,}")
    print(f"     ├─ Train value range: [{train_vals.min():.2f}, {train_vals.max():.2f}]")
    print(f"     └─ Test value range:  [{test_vals.min():.2f}, {test_vals.max():.2f}]")
    
    results = {
        'backend': backend_name,
        'resolution': resolution,
        'vector_dim': vector_dim,
        'cleanup_method': cleanup_method,
        'regression_method': regression_method,
        'n_train': len(train_pos),
        'n_test': len(test_pos),
    }

    # ------------------------------
    # Initialize HyperSpace modules
    # ------------------------------
    pe_module = PositionalEncoderModule(backend)
    ve_module = ValueEncoderModule(backend)
    global_pi_module = PositionalInversionModule(backend, positions)

    value_axis = torch.linspace(0.0, 12.0, steps=200).to(device).unsqueeze(1)
    value_codebook, _ = ve_module(value_axis)

    cleanup_module = CleanupModule(
        backend,
        codebook=value_codebook,
        method=cleanup_method
    )
    memory_module = MemoryStorageModule(backend)

    regression_module = RegressionModule(backend, value_codebook, value_axis)

    # -------------------------------
    # 1. Encoding Phase
    # -------------------------------
    print(f"\n{'─'*80}")
    print("⚙️  [1/5] ENCODING PHASE")
    print(f"{'─'*80}")
    print(f"     Encoding {len(train_pos):,} training samples...")
    
    encoding_start = time.perf_counter()

    pe_train, _ = pe_module(train_pos)
    ve_train, _ = ve_module(train_vals.unsqueeze(1))
    start_memory, _ = memory_module(
        p_vectors=pe_train,
        v_vectors=ve_train
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    encoding_end = time.perf_counter()
    encoding_time = encoding_end - encoding_start
    encoding_time_per_sample = encoding_time / train_vals.shape[0]
    
    results['encoding'] = {
        'total_encoding': encoding_time,
        'per_sample': encoding_time_per_sample
    }
    
    print(f"     ├─ Total time:   {encoding_time:.3f}s")
    print(f"     ├─ Per sample:   {encoding_time_per_sample*1000:.2f}ms")
    print(f"     └─ ✓ Complete")

    # ------------------------------
    # 2. Global Decoding
    # -------------------------------
    print(f"\n{'─'*80}")
    print("🔍 [2/5] GLOBAL DECODING PHASE")
    print(f"{'─'*80}")
    print(f"     Decoding all {len(positions):,} positions...")
    
    decoding_start = time.perf_counter()

    global_decoded_vecs, _ = global_pi_module(start_memory)
    global_decoded_vecs, _ = cleanup_module(
        global_decoded_vecs,
        num_iters=cleanup_iterations
    )
    global_predictions, _ = regression_module(global_decoded_vecs)
    global_predictions = global_predictions.squeeze(-1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    decoding_end = time.perf_counter()
    decoding_time = decoding_end - decoding_start
    global_mse = compute_mse(global_predictions, values)

    results['global_decoding'] = {
        'total_decoding': decoding_time,
        'per_sample': decoding_time / positions.shape[0],
        'global_mse': global_mse
    }
    results['global_rmse'] = global_mse

    print(f"     ├─ Total time:   {decoding_time:.3f}s")
    print(f"     ├─ Per sample:   {(decoding_time / positions.shape[0])*1000:.2f}ms")
    print(f"     ├─ Global MSE:   {global_mse:.4f}")
    print(f"     └─ ✓ Complete")

    # --------------------------------
    # 3. Saving Visualizations
    # --------------------------------
    print(f"\n{'─'*80}")
    print("📊 [3/5] SAVING VISUALIZATIONS")
    print(f"{'─'*80}")
    print(f"     Generating visualization plots...")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # plot the ground truth for visualization as image
    axs[0].imshow(values.reshape(10, 10).cpu().detach(), cmap='viridis', origin='lower')
    axs[0].set_title(f'Ground Truth Values')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    # plot the global predictions for visualization as image
    axs[1].imshow(global_predictions.reshape(10, 10).cpu().detach(), cmap='viridis', origin='lower')
    axs[1].set_title(f'Global Decoding Predictions')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    
    plot_filename = f'global_decoding_res{resolution}_D{vector_dim}_{backend_name}.png'
    plt.savefig(os.path.join(scratch_dir, plot_filename))
    plt.close()
    
    print(f"     ├─ Saved: {plot_filename}")
    print(f"     └─ ✓ Complete")

    # ------------------------------
    # 4. Training Set Decoding
    # -------------------------------
    print(f"\n{'─'*80}")
    print("📝 [4/5] TRAINING SET DECODING")
    print(f"{'─'*80}")
    print(f"     Decoding {len(train_pos):,} training positions...")

    train_pi_module = PositionalInversionModule(backend, train_pos)
    
    decoding_start = time.perf_counter()

    train_decoded_vecs, _ = train_pi_module(start_memory)
    train_decoded_vecs, _ = cleanup_module(
        train_decoded_vecs,
        num_iters=cleanup_iterations
    )
    train_predictions, _ = regression_module(train_decoded_vecs)
    train_predictions = train_predictions.squeeze(-1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    decoding_end = time.perf_counter()
    decoding_time = decoding_end - decoding_start
    train_mse = compute_mse(train_predictions, train_vals)

    results['train_decoding'] = {
        'total_decoding': decoding_time,
        'per_sample': decoding_time / train_pos.shape[0],
        'train_mse': train_mse
    }
    results['train_mse'] = train_mse

    print(f"     ├─ Total time:   {decoding_time:.3f}s")
    print(f"     ├─ Per sample:   {(decoding_time / train_pos.shape[0])*1000:.2f}ms")
    print(f"     ├─ Train MSE:    {train_mse:.4f}")
    print(f"     └─ ✓ Complete")

    # ------------------------------
    # 5. Testing Set Decoding
    # -------------------------------
    print(f"\n{'─'*80}")
    print("🧪 [5/5] TESTING SET DECODING")
    print(f"{'─'*80}")
    print(f"     Decoding {len(test_pos):,} testing positions...")

    test_pi_module = PositionalInversionModule(backend, test_pos)
    
    decoding_start = time.perf_counter()

    test_decoded_vecs, _ = test_pi_module(start_memory)
    test_decoded_vecs, _ = cleanup_module(
        test_decoded_vecs,
        num_iters=cleanup_iterations
    )
    test_predictions, _ = regression_module(test_decoded_vecs)
    test_predictions = test_predictions.squeeze(-1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    decoding_end = time.perf_counter()
    decoding_time = decoding_end - decoding_start
    test_mse = compute_mse(test_predictions, test_vals)

    results['test_decoding'] = {
        'total_decoding': decoding_time,
        'per_sample': decoding_time / test_pos.shape[0],
        'test_mse': test_mse
    }
    results['test_mse'] = test_mse

    print(f"     ├─ Total time:   {decoding_time:.3f}s")
    print(f"     ├─ Per sample:   {(decoding_time / test_pos.shape[0])*1000:.2f}ms")
    print(f"     ├─ Test MSE:     {test_mse:.4f}")
    print(f"     └─ ✓ Complete")

    print(f"\n{'='*80}")
    print(f"✅ BENCHMARK COMPLETE: {backend_name}")
    print(f"   Total Encoding Time: {encoding_time:.3f}s")
    print(f"   Global MSE:  {global_mse:.4f}")
    print(f"   Train MSE:   {train_mse:.4f}")
    print(f"   Test MSE:    {test_mse:.4f}")
    print(f"{'='*80}\n")

    return {**results}


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
    parser.add_argument('--cleanup-method', type=str, default='resonator',
                        choices=['none', 'resonator', 'modern_hopfield'],
                        help='Cleanup method to use')
    parser.add_argument('--cleanup-iters', type=int, default=2,
                        help='Number of cleanup iterations (notebook uses 2)')
    parser.add_argument('--regression-method', type=str, default='codebook',
                        choices=['codebook', 'neural'],
                        help='Regression method to use')
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
                vector_length_scale=2.0,
                device=args.device,
                cleanup_method=args.cleanup_method,
                cleanup_iterations=args.cleanup_iters,
                regression_method=args.regression_method,
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
                cleanup_method=args.cleanup_method,
                cleanup_iterations=args.cleanup_iters,
                regression_method=args.regression_method,
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
    print("\n\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    print(f"Cleanup: {all_results[0]['cleanup_method']} | Regression: {all_results[0]['regression_method']}")
    print("="*100)
    print(f"{'Configuration':<25} {'Backend':<10} {'Encode(s)':<15} {'Decode(s)':<15} {'Global MSE':<15}")
    print("="*100)
    
    for i in range(0, len(all_results), 2):
        hrr = all_results[i]
        fhrr = all_results[i+1]
        
        config = f"Res:{int(1000*hrr['resolution'])}x{int(1000*hrr['resolution'])} D:{hrr['vector_dim']}"
        
        # HRR results
        print(f"{config:<25} {'HRR':<10} {hrr['encoding']['total_encoding']:<15.3f} {hrr['global_decoding']['total_decoding']:<15.3f} {hrr['global_rmse']:<15.4f}")
        
        # FHRR results
        print(f"{'':<25} {'FHRR':<10} {fhrr['encoding']['total_encoding']:<15.3f} {fhrr['global_decoding']['total_decoding']:<15.3f} {fhrr['global_rmse']:<15.4f}")
        
        # Speedup comparison
        speedup_encode = hrr['encoding']['total_encoding'] / fhrr['encoding']['total_encoding']
        speedup_decode = hrr['global_decoding']['total_decoding'] / fhrr['global_decoding']['total_decoding']
        
        print(f"{'':<25} {'Speedup':<10} {speedup_encode:<15.2f}× {speedup_decode:<15.2f}× {'':<15}")
        print("-"*100)
    
    print("="*100)


if __name__ == '__main__':
    main()
