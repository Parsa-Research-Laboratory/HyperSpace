"""
Quick test script to verify the benchmark pipeline works and compare HRR vs FHRR.
Tests multiple vector dimensions and plots trade-off spaces.
"""

import sys
import math
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from benchmark_hrr_fhrr import run_full_benchmark, generate_spline_terrain, split_train_test

# Test configurations
VECTOR_DIMS = [8096]
VECTOR_LENGTH_SCALE = 2.0
RESOLUTION = 0.028
CLEANUP_METHOD = 'modern_hopfield'
CLEANUP_ITERS = 3


# -----------------------------------
# Generate terrain and save as image
# -----------------------------------
positions, values = generate_spline_terrain(RESOLUTION)
positions = torch.from_numpy(positions).to('cpu')  # Dummy positions for testing
values = torch.from_numpy(values).to('cpu')  # Dummy values for testing

# plot the terrain for visualization
n = values.shape[0]
resolution = int(math.sqrt(n))
grid = values.cpu().reshape(resolution, resolution)

train_pos, train_vals, test_pos, test_vals = split_train_test(
    positions, values, train_ratio=0.8, seed=0
)

# Collect results for both backends
all_results = {'HRR': [], 'FHRR': []}

for vector_dim in VECTOR_DIMS:
    print(f"\n{'='*80}")
    print(f"Testing Vector Dimension: {vector_dim}")
    print(f"{'='*80}")
    
    # Run HRR
    results_hrr = run_full_benchmark(
        backend_name='HRR',
        resolution=RESOLUTION,
        vector_length_scale=VECTOR_LENGTH_SCALE,
        vector_dim=vector_dim,
        device='cpu',
        cleanup_method=CLEANUP_METHOD,
        cleanup_iterations=CLEANUP_ITERS,
        seed=42,
        global_positions=positions,
        global_values=values,
        train_positions=train_pos,
        train_values=train_vals,
        test_positions=test_pos,
        test_values=test_vals,
        scratch_dir="scratch/hrr_test"  # Save intermediate results for debugging
    )
    all_results['HRR'].append(results_hrr)
    
    # Run FHRR
    results_fhrr = run_full_benchmark(
        backend_name='FHRR',
        resolution=RESOLUTION,
        vector_length_scale=VECTOR_LENGTH_SCALE,
        vector_dim=vector_dim,
        device='cpu',
        cleanup_method=CLEANUP_METHOD,
        cleanup_iterations=CLEANUP_ITERS,
        seed=42,
        global_positions=positions,
        global_values=values,
        train_positions=train_pos,
        train_values=train_vals,
        test_positions=test_pos,
        test_values=test_vals,
        scratch_dir="scratch/fhrr_test"  # Save intermediate results for debugging
    )
    all_results['FHRR'].append(results_fhrr)

print("\nBenchmarking complete.")