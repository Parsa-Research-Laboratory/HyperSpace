"""
Visualization script for HRR and FHRR benchmark results.

This script visualizes:
- Input costmap
- Training/test split
- Decoded predictions at test points
- Full-field reconstruction across entire grid

Usage:
    python visualize_benchmark.py --backend HRR --resolution 32 --vector-dim 1024
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from hyperspace.backends.hrr import HRRBackend
from hyperspace.backends.fhrr import FHRRBackend
from benchmark_hrr_fhrr import (
    generate_mixed_terrain,
    generate_smooth_gaussian_terrain,
    generate_obstacle_distance_field,
    generate_x_pattern_terrain,
    split_train_test,
    build_codebook,
    train_mlp_decoder,
    SimpleMLP
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


def encode_memory(backend, positions, values, device='cpu'):
    """Encode training data into memory."""
    positions = positions.to(device)
    values = values.to(device)
    
    memory = backend.create_empty_vector().to(device)
    
    for i in range(len(positions)):
        pos_vec, _ = backend.positional_encoding(positions[i:i+1])
        val_vec, _ = backend.value_encoding(values[i:i+1].unsqueeze(0))
        bound, _ = backend.bind(pos_vec.squeeze(0), val_vec.squeeze(0))
        memory, _ = backend.bundle(memory, bound)
    
    memory, _ = backend.normalize(memory)
    return memory


def query_and_decode(backend, memory, positions, codebook, cleanup_iters=4, device='cpu'):
    """Query memory at given positions and decode values."""
    positions = positions.to(device)
    memory = memory.to(device)
    codebook = codebook.to(device)
    
    decoded_vecs = []
    for i in range(len(positions)):
        pos_vec, _ = backend.positional_encoding(positions[i:i+1])
        pos_inv, _ = backend.invert(pos_vec.squeeze(0))
        decoded, _ = backend.bind(memory, pos_inv)
        decoded_vecs.append(decoded)
    
    decoded_batch = torch.stack(decoded_vecs)
    
    # Cleanup
    cleaned = decoded_batch.clone()
    for _ in range(cleanup_iters):
        cleaned, _ = backend._resonator_cleanup(cleaned, codebook, num_iters=1)
    
    return cleaned


def decode_with_codebook(backend, cleaned_vecs, codebook, max_value, device='cpu'):
    """Decode using codebook regression."""
    sims, _ = backend.similarity(cleaned_vecs, codebook, mode='all_pairs')
    weights = torch.softmax(sims, dim=1)
    codebook_values = torch.linspace(1, max_value, len(codebook)).to(device)
    predictions = (weights * codebook_values).sum(dim=1)
    return predictions


def decode_with_mlp(mlp, cleaned_vecs, device='cpu'):
    """Decode using MLP."""
    cleaned_vecs = cleaned_vecs.to(device)
    if cleaned_vecs.is_complex():
        cleaned_vecs = torch.cat([cleaned_vecs.real, cleaned_vecs.imag], dim=-1)
    
    with torch.no_grad():
        predictions = mlp(cleaned_vecs)
    return predictions


def visualize_full_pipeline(
    backend_name='HRR',
    resolution=32,
    vector_dim=1024,
    terrain_type='mixed',
    codebook_size=64,
    cleanup_iters=4,
    device='cpu',
    seed=42,
    save_path=None
):
    """Visualize complete pipeline with all stages."""
    
    print(f"Visualizing {backend_name} pipeline...")
    print(f"Resolution: {resolution}×{resolution}, D={vector_dim}")
    
    # Initialize backend
    if backend_name == 'HRR':
        backend = HRRBackend(
            vector_dim=vector_dim,
            device=device,
            env_dim=2,
            value_dim=1,
            seed=seed
        )
    else:
        backend = FHRRBackend(
            vector_dim=vector_dim,
            device=device,
            env_dim=2,
            value_dim=1,
            seed=seed
        )
    
    # Generate terrain
    print("Generating terrain...")
    if terrain_type == 'gaussian':
        positions, values = generate_smooth_gaussian_terrain(resolution, seed=seed)
    elif terrain_type == 'obstacle':
        positions, values = generate_obstacle_distance_field(resolution, seed=seed)
    elif terrain_type == 'xpattern':
        positions, values = generate_x_pattern_terrain(resolution, seed=seed)
    else:
        positions, values = generate_mixed_terrain(resolution, seed=seed)
    
    # Split train/test
    train_pos, train_vals, test_pos, test_vals = split_train_test(
        positions, values, train_ratio=0.8, seed=seed
    )
    
    # Encode memory
    print("Encoding memory...")
    memory = encode_memory(backend, train_pos, train_vals, device)
    
    # Build codebook
    print("Building codebook...")
    codebook = build_codebook(backend, k=codebook_size, device=device, max_value=float(resolution))
    
    # Query all positions (full field reconstruction)
    print("Reconstructing full field...")
    full_cleaned = query_and_decode(backend, memory, positions, codebook, cleanup_iters, device)
    
    # Decode
    print("Decoding values...")
    if backend_name == 'HRR':
        full_predictions = decode_with_codebook(backend, full_cleaned, codebook, float(resolution), device)
    else:
        # Train MLP on training data
        mlp = train_mlp_decoder(backend, train_pos, train_vals, vector_dim, device, epochs=100)
        full_predictions = decode_with_mlp(mlp, full_cleaned, device)
    
    # Also get test predictions
    test_cleaned = query_and_decode(backend, memory, test_pos, codebook, cleanup_iters, device)
    if backend_name == 'HRR':
        test_predictions = decode_with_codebook(backend, test_cleaned, codebook, float(resolution), device)
    else:
        test_predictions = decode_with_mlp(mlp, test_cleaned, device)
    
    # Reshape for visualization
    costmap_2d = values.reshape(resolution, resolution).cpu().numpy()
    full_pred_2d = full_predictions.reshape(resolution, resolution).cpu().numpy()
    error_map = np.abs(costmap_2d - full_pred_2d)
    
    # Create split map
    split_map = np.zeros((resolution, resolution))
    train_indices = []
    test_indices = []
    for i, pos in enumerate(positions):
        idx = int(pos[0] * (resolution-1)), int(pos[1] * (resolution-1))
        if i < len(train_pos):
            if torch.any(torch.all(train_pos == pos, dim=1)):
                split_map[idx] = 1
                train_indices.append(idx)
        if torch.any(torch.all(test_pos == pos, dim=1)):
            split_map[idx] = 2
            test_indices.append(idx)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Custom colormap
    cmap = 'viridis'
    
    # Row 1: Input data
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(costmap_2d, cmap=cmap, origin='lower', interpolation='nearest')
    ax1.set_title(f'Ground Truth Costmap\n({resolution}×{resolution})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    split_cmap = LinearSegmentedColormap.from_list('split', ['white', 'blue', 'red'])
    im2 = ax2.imshow(split_map, cmap=split_cmap, origin='lower', interpolation='nearest', vmin=0, vmax=2)
    ax2.set_title(f'Train/Test Split\n(Train={len(train_pos)}, Test={len(test_pos)})', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, ticks=[0.5, 1.5])
    cbar2.ax.set_yticklabels(['Train', 'Test'])
    
    ax3 = fig.add_subplot(gs[0, 2])
    # Show training data as scatter
    train_costmap = np.full((resolution, resolution), np.nan)
    for i, pos in enumerate(train_pos):
        idx = int(pos[0].item() * (resolution-1)), int(pos[1].item() * (resolution-1))
        train_costmap[idx] = train_vals[i].item()
    im3 = ax3.imshow(train_costmap, cmap=cmap, origin='lower', interpolation='nearest')
    ax3.set_title(f'Training Data Only\n({len(train_pos)} points)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[0, 3])
    # Show test data as scatter
    test_costmap = np.full((resolution, resolution), np.nan)
    for i, pos in enumerate(test_pos):
        idx = int(pos[0].item() * (resolution-1)), int(pos[1].item() * (resolution-1))
        test_costmap[idx] = test_vals[i].item()
    im4 = ax4.imshow(test_costmap, cmap=cmap, origin='lower', interpolation='nearest')
    ax4.set_title(f'Test Data Only\n({len(test_pos)} points)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 2: Predictions
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(full_pred_2d, cmap=cmap, origin='lower', interpolation='nearest')
    ax5.set_title(f'{backend_name} Full Reconstruction\n(All {resolution}×{resolution} points)', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(error_map, cmap='Reds', origin='lower', interpolation='nearest')
    ax6.set_title(f'Absolute Error\n(RMSE={np.sqrt(np.mean(error_map**2)):.4f})', 
                  fontsize=12, fontweight='bold')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Test predictions only
    ax7 = fig.add_subplot(gs[1, 2])
    test_pred_map = np.full((resolution, resolution), np.nan)
    for i, pos in enumerate(test_pos):
        idx = int(pos[0].item() * (resolution-1)), int(pos[1].item() * (resolution-1))
        test_pred_map[idx] = test_predictions[i].item()
    im7 = ax7.imshow(test_pred_map, cmap=cmap, origin='lower', interpolation='nearest')
    ax7.set_title(f'Predictions at Test Points\n({len(test_pos)} points)', 
                  fontsize=12, fontweight='bold')
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    # Test error
    ax8 = fig.add_subplot(gs[1, 3])
    test_error_map = np.full((resolution, resolution), np.nan)
    for i, pos in enumerate(test_pos):
        idx = int(pos[0].item() * (resolution-1)), int(pos[1].item() * (resolution-1))
        test_error_map[idx] = abs(test_vals[i].item() - test_predictions[i].item())
    im8 = ax8.imshow(test_error_map, cmap='Reds', origin='lower', interpolation='nearest')
    test_rmse = torch.sqrt(((test_predictions - test_vals)**2).mean()).item()
    ax8.set_title(f'Test Error\n(RMSE={test_rmse:.4f})', fontsize=12, fontweight='bold')
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Row 3: Analysis plots
    ax9 = fig.add_subplot(gs[2, 0:2])
    # Scatter: ground truth vs prediction (test points)
    ax9.scatter(test_vals.cpu().numpy(), test_predictions.cpu().numpy(), 
                alpha=0.6, s=20, label='Test Points')
    ax9.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction', linewidth=2)
    ax9.set_xlabel('Ground Truth Value', fontsize=11)
    ax9.set_ylabel('Predicted Value', fontsize=11)
    ax9.set_title(f'Prediction Accuracy (Test Set)\nR²={1 - ((test_vals - test_predictions)**2).sum()/((test_vals - test_vals.mean())**2).sum():.4f}',
                  fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(-0.05, 1.05)
    ax9.set_ylim(-0.05, 1.05)
    
    ax10 = fig.add_subplot(gs[2, 2:4])
    # Error histogram
    errors = (test_vals - test_predictions).cpu().numpy()
    ax10.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax10.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax10.set_xlabel('Prediction Error', fontsize=11)
    ax10.set_ylabel('Frequency', fontsize=11)
    ax10.set_title(f'Error Distribution (Test Set)\nMean={errors.mean():.4f}, Std={errors.std():.4f}',
                   fontsize=12, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    plt.suptitle(f'{backend_name} Backend Visualization - {terrain_type.title()} Terrain\n'
                 f'Resolution: {resolution}×{resolution}, Vector Dim: {vector_dim}, '
                 f'Codebook: {codebook_size}, Cleanup Iters: {cleanup_iters}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return {
        'full_rmse': np.sqrt(np.mean(error_map**2)),
        'test_rmse': test_rmse,
        'full_predictions': full_pred_2d,
        'test_predictions': test_predictions.cpu().numpy(),
        'costmap': costmap_2d
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize HRR/FHRR benchmark results')
    parser.add_argument('--backend', type=str, default='HRR', choices=['HRR', 'FHRR'],
                        help='Backend to visualize')
    parser.add_argument('--resolution', type=int, default=32,
                        help='Grid resolution')
    parser.add_argument('--vector-dim', type=int, default=1024,
                        help='Vector dimension')
    parser.add_argument('--terrain-type', type=str, default='xpattern',
                        choices=['gaussian', 'obstacle', 'mixed', 'xpattern'],
                        help='Terrain type')
    parser.add_argument('--codebook-size', type=int, default=64,
                        help='Codebook size')
    parser.add_argument('--cleanup-iters', type=int, default=4,
                        help='Number of cleanup iterations')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save', type=str, default=None,
                        help='Save path for figure (e.g., figures/hrr_vis.png)')
    
    args = parser.parse_args()
    
    results = visualize_full_pipeline(
        backend_name=args.backend,
        resolution=args.resolution,
        vector_dim=args.vector_dim,
        terrain_type=args.terrain_type,
        codebook_size=args.codebook_size,
        cleanup_iters=args.cleanup_iters,
        device=args.device,
        seed=args.seed,
        save_path=args.save
    )
    
    print(f"\n{'='*60}")
    print(f"Visualization Complete!")
    print(f"Full Field RMSE: {results['full_rmse']:.4f}")
    print(f"Test Set RMSE: {results['test_rmse']:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
