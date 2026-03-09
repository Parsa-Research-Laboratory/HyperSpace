"""
Benchmark script for comparing HRR and FHRR backends on 2D spatial costmaps.

This script generates all experimental data for the paper's Results section,
measuring stage-by-stage latency and reconstruction quality.

Usage:
    python benchmark_hrr_fhrr.py --output results.json
"""

import json
import os
import math
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import time
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
    RegressionModule,
    cleanup
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

    # ========================================================================
    # CRITICAL FIX: Shift positions to avoid identity vectors at origin (0,0)
    # ========================================================================
    # When position = 0, fractional power encoding gives: basis^(0/length_scale) = basis^0 = 1 (identity)
    # This causes complete loss of positional information at the origin
    # Shifting by 1.0 gives minimum power of 1.0/2.0 = 0.5, which is well-defined
    POSITION_OFFSET = 2.0
    X = X + POSITION_OFFSET
    # Now positions range from [1, 1000] instead of [0, 999]
    # This eliminates identity vector artifacts in lower-left corner
    # ========================================================================

    y_norm = (y - y.min()) / (y.max() - y.min())
    y_norm = y_norm * 10  # Scale to [0, 10] range
    y_norm += 2
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


def generate_pipeline_noise_samples(
    backend,
    clean_vectors: torch.Tensor,
    clean_values: torch.Tensor,
    num_superposition: int = 5,
    device: str = 'cpu',
    cleanup_module = None,
    cleanup_iterations: int = 3,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Generate training samples with realistic pipeline noise by running vectors
    through the full bind-bundle-unbind-cleanup cycle.
    
    This simulates the actual noise that occurs during memory operations,
    which is especially important for FHRR where complex arithmetic creates
    structured noise patterns.
    
    Args:
        backend: The backend instance (HRR or FHRR)
        clean_vectors: Clean encoded vectors (B, D)
        clean_values: Corresponding scalar values (B, 1)
        num_superposition: Number of other memories to bundle with (simulates interference)
        device: Device to use
        cleanup_module: Cleanup module to apply (must match inference cleanup)
        cleanup_iterations: Number of cleanup iterations to apply
        temperature: Temperature for Hopfield cleanup (if using modern_hopfield method)
    
    Returns:
        Noisy vectors after pipeline operations (B, D)
    """
    print(f"     Generating realistic pipeline noise samples...")
    print(f"     ├─ Simulating {num_superposition} interfering memories per sample")
    if cleanup_module is not None:
        print(f"     ├─ Applying cleanup ({cleanup_module.method}, {cleanup_iterations} iters)")
    
    noisy_vectors = []
    
    for i in range(len(clean_vectors)):
        # 1. Start with clean value encoding
        v = clean_vectors[i]
        
        # 2. Bind with a random position
        p, _ = backend.positional_encoding(torch.randn(1, 2).to(device) * 50 + 50)
        pv, _ = backend.bind(v, p.squeeze(0))
        
        # 3. Simulate superposition by bundling with other random memories
        for _ in range(num_superposition):
            # Create random interfering memory
            random_val = torch.rand(1, 1).to(device) * 10 + 1  # Random value in [1, 11]
            random_pos = torch.randn(1, 2).to(device) * 50 + 50  # Random position
            
            other_v, _ = backend.value_encoding(random_val)
            other_p, _ = backend.positional_encoding(random_pos)
            other_pv, _ = backend.bind(other_v.squeeze(0), other_p.squeeze(0))
            
            # Bundle interfering memory
            pv, _ = backend.bundle(pv, other_pv)
        
        # 4. Unbind to recover (noisy) value vector
        p_inv, _ = backend.invert(p.squeeze(0))
        v_noisy, _ = backend.bind(pv, p_inv)
        
        # 5. Apply cleanup (CRITICAL: must match inference pipeline)
        if cleanup_module is not None:
            v_noisy_batch = v_noisy.unsqueeze(0)  # Add batch dimension for cleanup
            if "hopfield" in cleanup_module.method:
                v_noisy_cleaned, _ = cleanup_module(
                    v_noisy_batch,
                    num_iters=cleanup_iterations,
                    temperature=temperature
                )
            else:
                v_noisy_cleaned, _ = cleanup_module(
                    v_noisy_batch,
                    num_iters=cleanup_iterations
                )
            v_noisy = v_noisy_cleaned.squeeze(0)  # Remove batch dimension
        
        noisy_vectors.append(v_noisy)
    
    result = torch.stack(noisy_vectors)
    print(f"     └─ Generated {len(result)} noisy training samples")
    return result


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
    global_positions: torch.Tensor = None,
    global_values: torch.Tensor = None,
    train_positions: torch.Tensor = None,
    train_values: torch.Tensor = None,
    test_positions: torch.Tensor = None,
    test_values: torch.Tensor = None,
    temperature: float = 0.1,
    use_pipeline_noise: bool = None,
    num_superposition: int = 5
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
    else:
        raise FileExistsError(f"Scratch directory already exists: {scratch_dir}")
    
    results = {
        'backend': backend_name,
        'resolution': resolution,
        'vector_dim': vector_dim,
        'cleanup_method': cleanup_method,
        'regression_method': regression_method,
        'n_train': len(train_positions),
        'n_test': len(test_positions),
        'training_time': 0.0
    }

    # ------------------------------
    # Initialize HyperSpace modules
    # ------------------------------
    pe_module = PositionalEncoderModule(backend)
    ve_module = ValueEncoderModule(backend)
    global_pi_module = PositionalInversionModule(backend, global_positions)
    train_pi_module = PositionalInversionModule(backend, train_positions)
    test_pi_module = PositionalInversionModule(backend, test_positions)

    value_axis = torch.linspace(0.0, 12.0, steps=200).to(device).unsqueeze(1)
    value_codebook, _ = ve_module(value_axis)

    if cleanup_method in ['resonator', 'modern_hopfield']:
        cleanup_module = CleanupModule(
            backend,
            codebook=value_codebook,
            method=cleanup_method
        )
    else:
        cleanup_module = None

    memory_module = MemoryStorageModule(backend)

    regression_module = RegressionModule(backend, value_codebook, value_axis, method=regression_method)

    if regression_method == 'neural':
        # Determine noise type based on backend if not specified
        # Use pipeline noise for both HRR and FHRR when cleanup is enabled for fair comparison
        if use_pipeline_noise is None:
            use_pipeline_noise = (cleanup_module is not None)
        
        noise_type = "pipeline" if use_pipeline_noise else "gaussian"
        print(f"\n[Network Training] Using {noise_type} noise for {backend_name}")
        if use_pipeline_noise and cleanup_module is not None:
            print(f"[Network Training] Pipeline includes cleanup: {cleanup_method} ({cleanup_iterations} iters)")
        
        num_samples = value_axis.shape[0]
        sample_idxs: np.ndarray = np.arange(0, num_samples - 1)
        np.random.shuffle(sample_idxs)

        num_test_samples: int = int(num_samples * 0.2)  # Fixed: was 0.8, should be 0.2 for 20% test
        network_Train_vectors = value_codebook[sample_idxs[num_test_samples:]]
        network_Test_vectors = value_codebook[sample_idxs[:num_test_samples]]
        network_Train_outputs = value_axis[sample_idxs[num_test_samples:]]
        network_Test_outputs = value_axis[sample_idxs[:num_test_samples]]

        # Generate realistic pipeline noise samples if using FHRR
        if use_pipeline_noise:
            print(f"\n[Network Training] Generating pipeline-realistic training data...")
            network_Train_vectors = generate_pipeline_noise_samples(
                backend=backend,
                clean_vectors=network_Train_vectors,
                clean_values=network_Train_outputs,
                num_superposition=num_superposition,
                device=device,
                cleanup_module=cleanup_module,
                cleanup_iterations=cleanup_iterations,
                temperature=temperature
            )
            network_Test_vectors = generate_pipeline_noise_samples(
                backend=backend,
                clean_vectors=network_Test_vectors,
                clean_values=network_Test_outputs,
                num_superposition=num_superposition,
                device=device,
                cleanup_module=cleanup_module,
                cleanup_iterations=cleanup_iterations,
                temperature=temperature
            )
            print(f"[Network Training] Pipeline noise samples generated")

        print(f"\n[Network Training] Num Samples: {num_samples}")
        print(f"[Network Training] Num Train Vectors: {network_Train_vectors.shape[0]}")
        print(f"[Network Training] Num Test Vectors: {network_Test_vectors.shape[0]}")
        print(f"[Network Training] Num Train Outputs: {network_Train_outputs.shape[0]}")
        print(f"[Network Training] Num Test Outputs: {network_Test_outputs.shape[0]}")
        print()
        print(f"[Network Training] Train Output Range: {torch.min(network_Train_outputs):.4f} to {torch.max(network_Train_outputs):.4f}.")
        print(f"[Network Training] Test Output Range: {torch.min(network_Test_outputs):.4f} to {torch.max(network_Test_outputs):.4f}.")

        # ----------------------------------------
        # normalize the training and testing data
        # ----------------------------------------
        y_train = network_Train_outputs.float()
        y_test  = network_Test_outputs.float()

        # compute stats on TRAIN ONLY
        y_mean = y_train.mean()
        y_std  = y_train.std(unbiased=False).clamp_min(1e-8)  # avoid divide-by-zero

        # normalize
        y_train_norm = (y_train - y_mean) / y_std
        y_test_norm  = (y_test  - y_mean) / y_std

        # helpers for later
        def denorm(y_norm: torch.Tensor) -> torch.Tensor:
            return y_norm * y_std + y_mean

        print("[Network Training] Train norm range:", y_train_norm.min().item(), "to", y_train_norm.max().item())
        print("[Network Training] Test  norm range:", y_test_norm.min().item(),  "to", y_test_norm.max().item())

        from hyperspace.core.regression.models import DenseLinearModel

        # ========================================================================
        # FAIR COMPARISON: Equalize parameter counts between HRR and FHRR
        # ========================================================================
        # FHRR has 2x input features (real + imag), so we reduce hidden size by ~2x
        # to keep total parameter count similar to HRR
        
        feature_dim = vector_dim if not torch.is_complex(network_Train_vectors) else 2 * vector_dim
        
        if backend_name == 'HRR':
            hidden_size = 512
        elif backend_name == 'FHRR':
            # Reduce hidden size to compensate for 2x input features
            hidden_size = 256
        else:
            hidden_size = 512  # default
        
        # Calculate actual parameter counts
        layer1_params = feature_dim * hidden_size + hidden_size
        layer2_params = hidden_size * 1 + 1
        total_params = layer1_params + layer2_params
        
        print(f"\n[Network Architecture] Backend: {backend_name}")
        print(f"[Network Architecture] Input features: {feature_dim:,}")
        print(f"[Network Architecture] Hidden size: {hidden_size}")
        print(f"[Network Architecture] Total parameters: {total_params:,}")
        print(f"[Network Architecture]   Layer 1: {layer1_params:,}")
        print(f"[Network Architecture]   Layer 2: {layer2_params:,}")

        train_losses: list = []
        test_losses: list = []

        num_epochs: int = 1000
        # Use smaller Gaussian noise when training with pipeline noise (since it's already noisy)
        noise_std = 0.05 if use_pipeline_noise else 0.1
        print(f"[Network Training] Additional Gaussian noise std: {noise_std}")

        def prepare_complex_input(v: torch.Tensor) -> torch.Tensor:
            """Flatten complex vectors for neural network input."""
            real_part = torch.real(v)
            imag_part = torch.imag(v)
            # return imag_part.float()
            return torch.cat([real_part, imag_part], dim=-1)
        
        # check if the vectors are complex and prepare input accordingly
        if torch.is_complex(network_Train_vectors):
            inputs_flat = prepare_complex_input(network_Train_vectors)
        else:
            inputs_flat = network_Train_vectors

        training_time_total = 0.0

        timer = time.perf_counter()

        regression_model = DenseLinearModel(
            feature_dim=feature_dim,
            value_dim=1,
            num_layers=2,
            hidden_size=hidden_size,
            hidden_act=nn.ReLU(),
        )

        # ------------------------------------------
        # train and validate the regression network
        # ------------------------------------------
        optimizer = torch.optim.Adam(
            regression_model.parameters(),
            lr=1e-3
        )
        loss_func = nn.MSELoss()

        training_time_total += time.perf_counter() - timer

        for i in range(num_epochs):

            timer = time.perf_counter()

            # ---------
            # Training
            # ---------
            regression_model.train()

            noise = noise_std * torch.randn_like(inputs_flat)
            noisy_inputs = inputs_flat + noise

            outputs = regression_model(noisy_inputs)
            train_loss = loss_func(outputs, y_train_norm)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            training_time_total += time.perf_counter() - timer

            train_losses.append(train_loss.item())

            # -----------
            # Evaluation
            # -----------
            regression_model.eval()

            with torch.no_grad():

                if torch.is_complex(network_Test_vectors):
                    test_inputs_flat = prepare_complex_input(network_Test_vectors)
                else:
                    test_inputs_flat = network_Test_vectors

                outputs = regression_model(test_inputs_flat)
                val_loss = loss_func(outputs, y_test_norm)
                test_losses.append(val_loss.item())

        plt.figure(figsize=(5, 3))
        plt.plot(train_losses, label="Train")
        plt.plot(test_losses, label="Test")
        plt.title("Training and Testing Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(scratch_dir, "regression_training.png"), bbox_inches="tight")

        regression_module.load_neural_network(regression_model)

        results['training_time'] = training_time_total

    # ------------------------------
    # 1. Positional Encoding Phase
    # ------------------------------
    print(f"\n{'─'*80}")
    print("⚙️  [1/12] ENCODING PHASE")
    print(f"{'─'*80}")
    print(f"     Encoding {len(train_positions):,} training samples...")

    pe_start = time.perf_counter()

    pe_train, _ = pe_module(train_positions)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pe_end = time.perf_counter()
    pe_time = pe_end - pe_start
    pe_time_per_sample = pe_time / train_positions.shape[0]

    print(f"     ├─ Total time:   {pe_time:.3f}s")
    print(f"     ├─ Per sample:   {pe_time_per_sample*1000:.2f}ms")
    print(f"     └─ ✓ Complete")

    results['positional_encoding'] = {
        'total_time': pe_time,
        'per_sample': pe_time_per_sample
    }

    # ------------------------------
    # 2. Value Encoding Phase
    # ------------------------------
    print(f"\n{'─'*80}")
    print("⚙️  [2/12] VALUE ENCODING PHASE")
    print(f"{'─'*80}")
    print(f"     Encoding {len(train_values):,} training values...")

    ve_start = time.perf_counter()

    ve_train, _ = ve_module(train_values.unsqueeze(1))

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ve_end = time.perf_counter()
    ve_time = ve_end - ve_start
    ve_time_per_sample = ve_time / train_values.shape[0]

    print(f"     ├─ Total time:   {ve_time:.3f}s")
    print(f"     ├─ Per sample:   {ve_time_per_sample*1000:.2f}ms")
    print(f"     └─ ✓ Complete")

    results['value_encoding'] = {
        'total_time': ve_time,
        'per_sample': ve_time_per_sample
    }

    # ------------------------------
    # 3. Memory Storage Phase
    # ------------------------------
    print(f"\n{'─'*80}")
    print("⚙️  [3/12] MEMORY STORAGE PHASE")
    print(f"{'─'*80}")
    print(f"     Binding and bundling {len(train_positions):,} samples into memory...")

    mem_start = time.perf_counter()

    memory, _ = memory_module(pe_train, ve_train)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_end = time.perf_counter()
    mem_time = mem_end - mem_start
    mem_time_per_sample = mem_time / train_positions.shape[0]

    print(f"     ├─ Total time:   {mem_time:.3f}s")
    print(f"     ├─ Per sample:   {mem_time_per_sample*1000:.2f}ms")
    print(f"     └─ ✓ Complete")

    results['memory_storage'] = {
        'total_time': mem_time,
        'per_sample': mem_time_per_sample
    }

    # -------------------------------------
    # 4. Global Positional Inversion Phase
    # -------------------------------------
    print(f"\n{'─'*80}")
    print("🔍 [4/12] GLOBAL POSITIONAL INVERSION PHASE")
    print(f"{'─'*80}")
    print(f"     Decoding all {len(global_positions):,} positions...")

    global_pi_start = time.perf_counter()

    global_decoded_vecs, _ = global_pi_module(memory)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    global_pi_end = time.perf_counter()
    global_pi_time = global_pi_end - global_pi_start
    global_pi_time_per_sample = global_pi_time / global_positions.shape[0]

    print(f"     ├─ Total time:   {global_pi_time:.3f}s")
    print(f"     ├─ Per sample:   {global_pi_time_per_sample*1000:.2f}ms")
    print(f"     └─ ✓ Complete")

    results['global_positional_inversion'] = {
        'total_time': global_pi_time,
        'per_sample': global_pi_time_per_sample
    }

    # ---------------------------------------
    # 5. Training Positional Inversion Phase
    # ---------------------------------------
    print(f"\n{'─'*80}")
    print("🔍 [5/12] TRAINING POSITIONAL INVERSION PHASE")
    print(f"{'─'*80}")
    print(f"     Decoding {len(train_positions):,} training positions...")

    train_pi_start = time.perf_counter()
    train_decoded_vecs, _ = train_pi_module(memory)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    train_pi_end = time.perf_counter()
    train_pi_time = train_pi_end - train_pi_start
    train_pi_time_per_sample = train_pi_time / train_positions.shape[0]
    print(f"     ├─ Total time:   {train_pi_time:.3f}s")
    print(f"     ├─ Per sample:   {train_pi_time_per_sample*1000:.2f}ms")
    print(f"     └─ ✓ Complete")

    results['train_positional_inversion'] = {
        'total_time': train_pi_time,
        'per_sample': train_pi_time_per_sample
    }

    # ---------------------------------------
    # 6. Testing Positional Inversion Phase
    # ---------------------------------------
    print(f"\n{'─'*80}")
    print("🔍 [6/12] TESTING POSITIONAL INVERSION PHASE")
    print(f"{'─'*80}")
    print(f"     Decoding {len(test_positions):,} testing positions...")

    test_pi_start = time.perf_counter()
    test_decoded_vecs, _ = test_pi_module(memory)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    test_pi_end = time.perf_counter()
    test_pi_time = test_pi_end - test_pi_start
    test_pi_time_per_sample = test_pi_time / test_positions.shape[0]
    print(f"     ├─ Total time:   {test_pi_time:.3f}s")
    print(f"     ├─ Per sample:   {test_pi_time_per_sample*1000:.2f}ms")
    print(f"     └─ ✓ Complete")
    
    results['test_positional_inversion'] = {
        'total_time': test_pi_time,
        'per_sample': test_pi_time_per_sample
    }

    # ---------------------------------------
    # Global Cleanup
    # ---------------------------------------
    print(f"\n{'─'*80}")
    print(f"🧹 [7/12] GLOBAL CLEANUP PHASE")
    print(f"{'─'*80}")

    if cleanup_module is None:
        print(f"     Skipping cleanup phase (method: none)...")

        global_decoded_vecs_cleaned = global_decoded_vecs

    else:
        print(f"     Cleaning up decoded vectors using {cleanup_method} method with {cleanup_iterations} iterations...")

        cleanup_start = time.perf_counter()

        if "hopfield" in cleanup_method:
            global_decoded_vecs_cleaned, _ = cleanup_module(
                global_decoded_vecs,
                num_iters=cleanup_iterations,
                temperature=temperature
            )
        else:
            global_decoded_vecs_cleaned, _ = cleanup_module(
                global_decoded_vecs,
                num_iters=cleanup_iterations
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cleanup_end = time.perf_counter()
        cleanup_time = cleanup_end - cleanup_start
        cleanup_time_per_sample = cleanup_time / global_decoded_vecs.shape[0]

        print(f"     ├─ Total time:   {cleanup_time:.3f}s")
        print(f"     ├─ Per sample:   {cleanup_time_per_sample*1000:.2f}ms")
        print(f"     └─ ✓ Complete")

    results['global_cleanup'] = {
        'total_time': cleanup_time if cleanup_module is not None else 0.0,
        'per_sample': cleanup_time_per_sample if cleanup_module is not None else 0.0
    }

    # ---------------------------------------
    # Training Cleanup
    # ---------------------------------------
    print(f"\n{'─'*80}")
    print(f"🧹 [8/12] TRAINING CLEANUP PHASE")
    print(f"{'─'*80}")

    if cleanup_module is None:
        print(f"     Skipping cleanup phase (method: none)...")

        train_decoded_vecs_cleaned = train_decoded_vecs
    else:
        print(f"     Cleaning up decoded vectors using {cleanup_method} method with {cleanup_iterations} iterations...")

        cleanup_start = time.perf_counter()

        if "hopfield" in cleanup_method:
            train_decoded_vecs_cleaned, _ = cleanup_module(
                train_decoded_vecs,
                num_iters=cleanup_iterations,
                temperature=temperature
            )
        else:
            train_decoded_vecs_cleaned, _ = cleanup_module(
                train_decoded_vecs,
                num_iters=cleanup_iterations
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cleanup_end = time.perf_counter()
        cleanup_time = cleanup_end - cleanup_start
        cleanup_time_per_sample = cleanup_time / train_decoded_vecs.shape[0]

        print(f"     ├─ Total time:   {cleanup_time:.3f}s")
        print(f"     ├─ Per sample:   {cleanup_time_per_sample*1000:.2f}ms")
        print(f"     └─ ✓ Complete")

    results['train_cleanup'] = {
        'total_time': cleanup_time if cleanup_module is not None else 0.0,
        'per_sample': cleanup_time_per_sample if cleanup_module is not None else 0.0
    }

    # ---------------------------------------
    # Testing Cleanup
    # ---------------------------------------
    print(f"\n{'─'*80}")
    print(f"🧹 [9/12] TESTING CLEANUP PHASE")
    print(f"{'─'*80}")

    if cleanup_module is None:
        print(f"     Skipping cleanup phase (method: none)...")

        test_decoded_vecs_cleaned = test_decoded_vecs
    else:
        print(f"     Cleaning up decoded vectors using {cleanup_method} method with {cleanup_iterations} iterations...")

        cleanup_start = time.perf_counter()

        if "hopfield" in cleanup_method:
            test_decoded_vecs_cleaned, _ = cleanup_module(
                test_decoded_vecs,
                num_iters=cleanup_iterations,
                temperature=temperature
            )
        else:
            test_decoded_vecs_cleaned, _ = cleanup_module(
                test_decoded_vecs,
                num_iters=cleanup_iterations
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cleanup_end = time.perf_counter()
        cleanup_time = cleanup_end - cleanup_start
        cleanup_time_per_sample = cleanup_time / test_decoded_vecs.shape[0]

        print(f"     ├─ Total time:   {cleanup_time:.3f}s")
        print(f"     ├─ Per sample:   {cleanup_time_per_sample*1000:.2f}ms")
        print(f"     └─ ✓ Complete")

    results['test_cleanup'] = {
        'total_time': cleanup_time if cleanup_module is not None else 0.0,
        'per_sample': cleanup_time_per_sample if cleanup_module is not None else 0.0
    }

    # ---------------------------------------
    # Global Regression
    # ---------------------------------------
    print(f"\n{'─'*80}")
    print(f"📈 [10/12] GLOBAL REGRESSION PHASE")
    print(f"{'─'*80}")
    print(f"     Regressing cleaned vectors to values using {regression_method} method...")

    regression_start = time.perf_counter()

    if torch.is_complex(global_decoded_vecs_cleaned) and regression_method == 'neural':
        global_decoded_vecs_cleaned = prepare_complex_input(global_decoded_vecs_cleaned)

    global_predictions, _ = regression_module(global_decoded_vecs_cleaned)
    global_predictions = global_predictions.squeeze(-1)

    if regression_method == 'neural':
        print(f"Before Denorm: {torch.min(global_predictions)} - {torch.max(global_predictions)}")
        global_predictions = denorm(global_predictions).detach()
        print(f"After Denorm: {torch.min(global_predictions)} - {torch.max(global_predictions)}")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    regression_end = time.perf_counter()
    regression_time = regression_end - regression_start
    regression_time_per_sample = regression_time / global_decoded_vecs_cleaned.shape[0]


    global_mse = compute_mse(global_predictions, global_values)

    results['global_regression'] = {
        'total_time': regression_time,
        'per_sample': regression_time_per_sample,
        'global_mse': global_mse
    }

    print(f"     ├─ Total time:   {regression_time:.3f}s")
    print(f"     ├─ Per sample:   {regression_time_per_sample*1000:.2f}ms")
    print(f"     ├─ Global MSE:   {global_mse:.4f}")
    print(f"     └─ ✓ Complete")

    # ---------------------------------------
    # Training Regression
    # ---------------------------------------
    print(f"\n{'─'*80}")
    print(f"📈 [11/12] TRAINING REGRESSION PHASE")
    print(f"{'─'*80}")
    print(f"     Regressing {len(train_decoded_vecs_cleaned):,} training vectors to values using {regression_method} method...")

    training_regression_start = time.perf_counter()

    if torch.is_complex(train_decoded_vecs_cleaned) and regression_method == 'neural':
        train_decoded_vecs_cleaned = prepare_complex_input(train_decoded_vecs_cleaned)

    train_predictions, _ = regression_module(train_decoded_vecs_cleaned)
    train_predictions = train_predictions.squeeze(-1)

    print(f"     [Debug] Train predictions before denorm: min={train_predictions.min().item():.4f}, max={train_predictions.max().item():.4f}")

    if regression_method == 'neural':
        train_predictions = denorm(train_predictions).detach()

    print(f"     [Debug] Train predictions after denorm: min={train_predictions.min().item():.4f}, max={train_predictions.max().item():.4f}")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    training_regression_end = time.perf_counter()
    training_regression_time = training_regression_end - training_regression_start
    training_regression_time_per_sample = training_regression_time / train_decoded_vecs_cleaned.shape[0]

    train_mse = compute_mse(train_predictions, train_values)

    results['training_regression'] = {
        'total_time': training_regression_time,
        'per_sample': training_regression_time_per_sample,
        'train_mse': train_mse
    }

    print(f"     ├─ Total time:   {training_regression_time:.3f}s")
    print(f"     ├─ Per sample:   {training_regression_time_per_sample*1000:.2f}ms")
    print(f"     ├─ Train MSE:    {train_mse:.4f}")
    print(f"     └─ ✓ Complete")

    # ---------------------------------------
    # Testing Regression
    # ---------------------------------------
    print(f"\n{'─'*80}")
    print(f"📈 [12/12] TESTING REGRESSION PHASE")
    print(f"{'─'*80}")
    print(f"     Regressing {len(test_decoded_vecs_cleaned):,} test vectors to values using {regression_method} method...")

    test_regression_start = time.perf_counter()

    if torch.is_complex(test_decoded_vecs_cleaned) and regression_method == 'neural':
        test_decoded_vecs_cleaned = prepare_complex_input(test_decoded_vecs_cleaned)

    test_predictions, _ = regression_module(test_decoded_vecs_cleaned)
    test_predictions = test_predictions.squeeze(-1)

    if regression_method == 'neural':
        test_predictions = denorm(test_predictions)
        test_predictions = test_predictions.detach()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    test_regression_end = time.perf_counter()
    test_regression_time = test_regression_end - test_regression_start
    test_regression_time_per_sample = test_regression_time / test_decoded_vecs_cleaned.shape[0]

    test_mse = compute_mse(test_predictions, test_values)

    results['test_regression'] = {
        'total_time': test_regression_time,
        'per_sample': test_regression_time_per_sample,
        'test_mse': test_mse
    }

    print(f"     ├─ Total time:   {test_regression_time:.3f}s")
    print(f"     ├─ Per sample:   {test_regression_time_per_sample*1000:.2f}ms")
    print(f"     ├─ Test MSE:     {test_mse:.4f}")
    print(f"     └─ ✓ Complete")

    print(f"{'='*80}\n")

    # Save results to JSON
    results_path = os.path.join(scratch_dir, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Save predictions and targets for test set
    test_results_path = os.path.join(scratch_dir, "test_predictions.npz")
    np.savez(test_results_path, predictions=test_predictions.cpu().numpy(), targets=test_values.cpu().numpy())

    # Save predictions and targets for training set
    train_results_path = os.path.join(scratch_dir, "train_predictions.npz")
    np.savez(train_results_path, predictions=train_predictions.cpu().numpy(), targets=train_values.cpu().numpy())

    # Save global predictions and targets
    global_results_path = os.path.join(scratch_dir, "global_predictions.npz")
    np.savez(global_results_path, predictions=global_predictions.cpu().numpy(), targets=global_values.cpu().numpy())

    # Plot the global predictions vs targets as images
    N = int(math.sqrt(global_positions.shape[0]))
    pred_grid = global_predictions.cpu().numpy().reshape(N, N)
    target_grid = global_values.cpu().numpy().reshape(N, N)
    plt.figure(figsize=(6, 6))
    plt.imshow(pred_grid, cmap="viridis", origin="lower")
    plt.colorbar(label="Predicted Value")
    plt.title("Global Predictions")
    plt.savefig(os.path.join(scratch_dir, "global_predictions.png"))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(target_grid, cmap="viridis", origin="lower")
    plt.colorbar(label="Target Value")
    plt.title("Global Targets")
    plt.savefig(os.path.join(scratch_dir, "global_targets.png"))
    plt.close()

    return {**results}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vector_dim: int = 8096
    vector_length_scale: float = 2.0
    resolution: float = 0.028
    cleanup_iterations: int = 3

    print(f"🚀 Starting benchmark on device: {device}")

    # ---------------------------
    # Generate results directory
    # ---------------------------
    results_dir = Path("./results")

    if results_dir.exists():
        shutil.rmtree(results_dir)  # delete entire directory tree

    results_dir.mkdir(parents=True, exist_ok=False)

    # -----------------------------------
    # Generate terrain and save as image
    # -----------------------------------
    positions, values = generate_spline_terrain(resolution)
    positions = torch.from_numpy(positions).to(device)
    values = torch.from_numpy(values).to(device)  # Dummy values for testing

    # plot the terrain for visualization
    n = values.shape[0]
    resolution = int(math.sqrt(n))
    grid = values.cpu().reshape(resolution, resolution)

    plt.figure()
    plt.imshow(
        grid,
        cmap="viridis",
        origin="lower",   # important so (0,0) is bottom-left
        interpolation="nearest"
    )
    plt.colorbar(label="Cost")
    plt.title("Terrain")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(results_dir / f"terrain_res{resolution}.png")
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
    
    backend_options: List[str] = ['HRR', 'FHRR']
    cleanup_methods: List[str] = ['none', 'resonator', 'modern_hopfield']
    regression_methods: List[str] = ['codebook', 'neural']
    seed_options: List[int] = [0, 1, 2, 3, 4, 5]

    # backend_options: List[str] = ['HRR']
    # cleanup_methods: List[str] = ['resonator']
    # regression_methods: List[str] = ['codebook']

    all_results = []

    for backend_name in backend_options:
        for cleanup_method in cleanup_methods:
            for regression_method in regression_methods:
                for seed in seed_options:

                    print(f"\n{'#'*80}")
                    print(f"🚀 Running benchmark for {backend_name} | Cleanup: {cleanup_method} | Regression: {regression_method} | Seed: {seed}")
                    print(f"{'#'*80}\n")

                    result = run_full_benchmark(
                        backend_name=backend_name,
                        resolution=resolution,
                        vector_dim=vector_dim,
                        vector_length_scale=vector_length_scale,
                        device=device,
                        cleanup_method=cleanup_method,
                        cleanup_iterations=cleanup_iterations,
                        regression_method=regression_method,
                        seed=42,
                        scratch_dir=str(results_dir / f"{backend_name}_{cleanup_method}_{regression_method}_seed={seed}"),
                        global_positions=positions,
                        global_values=values,
                        train_positions=train_pos,
                        train_values=train_vals,
                        test_positions=test_pos,
                        test_values=test_vals
                    )
                    all_results.append(result)
    


if __name__ == '__main__':
    main()
