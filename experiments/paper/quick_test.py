"""
Quick test script to verify the benchmark pipeline works and compare HRR vs FHRR.
Tests multiple vector dimensions and plots trade-off spaces.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from benchmark_hrr_fhrr import run_full_benchmark

# Test configurations
VECTOR_DIMS = [8096]
VECTOR_LENGTH_SCALE = 2.0
RESOLUTION = 0.028
CLEANUP_METHOD = 'resonator'
CLEANUP_ITERS = 3


print("="*80)
print("QUICK TEST: HRR vs FHRR Comparison")
print("="*80)
print(f"Testing vector dimensions: {VECTOR_DIMS}")
print(f"Resolution: {int(1000*RESOLUTION)}x{int(1000*RESOLUTION)}")
print(f"Cleanup: {CLEANUP_METHOD} ({CLEANUP_ITERS} iterations)")
print("="*80 + "\n")

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
        seed=42
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
        seed=42
    )
    all_results['FHRR'].append(results_fhrr)

# ============================================================================
# Print Summary Comparison
# ============================================================================
print("\n\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(f"{'Dim':<10} {'Backend':<10} {'Encode(s)':<12} {'Decode(s)':<12} {'Global MSE':<12} {'Train MSE':<12} {'Test MSE':<12}")
print("="*80)

for i, dim in enumerate(VECTOR_DIMS):
    hrr = all_results['HRR'][i]
    fhrr = all_results['FHRR'][i]
    
    print(f"{dim:<10} {'HRR':<10} {hrr['encoding']['total_encoding']:<12.3f} {hrr['global_decoding']['total_decoding']:<12.3f} {hrr['global_rmse']:<12.4f} {hrr['train_mse']:<12.4f} {hrr['test_mse']:<12.4f}")
    print(f"{'':<10} {'FHRR':<10} {fhrr['encoding']['total_encoding']:<12.3f} {fhrr['global_decoding']['total_decoding']:<12.3f} {fhrr['global_rmse']:<12.4f} {fhrr['train_mse']:<12.4f} {fhrr['test_mse']:<12.4f}")
    
    speedup_encode = hrr['encoding']['total_encoding'] / fhrr['encoding']['total_encoding']
    speedup_decode = hrr['global_decoding']['total_decoding'] / fhrr['global_decoding']['total_decoding']
    print(f"{'':<10} {'Speedup':<10} {speedup_encode:<.2f}×        {speedup_decode:<.2f}×")
    print("-"*80)

print("="*80)

# ============================================================================
# Plot Trade-off Spaces
# ============================================================================
print("\n📊 Generating trade-off space plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('HRR vs FHRR Trade-off Analysis', fontsize=16, fontweight='bold')

# Extract data for plotting
dims = np.array(VECTOR_DIMS)
hrr_results = all_results['HRR']
fhrr_results = all_results['FHRR']

hrr_encode_time = np.array([r['encoding']['total_encoding'] for r in hrr_results])
fhrr_encode_time = np.array([r['encoding']['total_encoding'] for r in fhrr_results])

hrr_decode_time = np.array([r['global_decoding']['total_decoding'] for r in hrr_results])
fhrr_decode_time = np.array([r['global_decoding']['total_decoding'] for r in fhrr_results])

hrr_global_mse = np.array([r['global_rmse'] for r in hrr_results])
fhrr_global_mse = np.array([r['global_rmse'] for r in fhrr_results])

hrr_test_mse = np.array([r['test_mse'] for r in hrr_results])
fhrr_test_mse = np.array([r['test_mse'] for r in fhrr_results])

# Plot 1: Accuracy vs Encoding Time
ax1 = axes[0, 0]
ax1.scatter(hrr_encode_time, hrr_global_mse, s=100, marker='o', label='HRR', alpha=0.7)
ax1.scatter(fhrr_encode_time, fhrr_global_mse, s=100, marker='s', label='FHRR', alpha=0.7)
for i, dim in enumerate(dims):
    ax1.annotate(f'D={dim}', (hrr_encode_time[i], hrr_global_mse[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax1.annotate(f'D={dim}', (fhrr_encode_time[i], fhrr_global_mse[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax1.set_xlabel('Encoding Time (seconds)', fontsize=11)
ax1.set_ylabel('Global MSE', fontsize=11)
ax1.set_title('Accuracy vs Encoding Speed', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy vs Decoding Time
ax2 = axes[0, 1]
ax2.scatter(hrr_decode_time, hrr_global_mse, s=100, marker='o', label='HRR', alpha=0.7)
ax2.scatter(fhrr_decode_time, fhrr_global_mse, s=100, marker='s', label='FHRR', alpha=0.7)
for i, dim in enumerate(dims):
    ax2.annotate(f'D={dim}', (hrr_decode_time[i], hrr_global_mse[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.annotate(f'D={dim}', (fhrr_decode_time[i], fhrr_global_mse[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax2.set_xlabel('Decoding Time (seconds)', fontsize=11)
ax2.set_ylabel('Global MSE', fontsize=11)
ax2.set_title('Accuracy vs Decoding Speed', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Test Set Accuracy vs Vector Dimension
ax3 = axes[1, 0]
ax3.plot(dims, hrr_test_mse, 'o-', linewidth=2, markersize=8, label='HRR', alpha=0.7)
ax3.plot(dims, fhrr_test_mse, 's-', linewidth=2, markersize=8, label='FHRR', alpha=0.7)
ax3.set_xlabel('Vector Dimension', fontsize=11)
ax3.set_ylabel('Test Set MSE', fontsize=11)
ax3.set_title('Test Accuracy vs Vector Dimension', fontsize=12, fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.set_xticks(dims)
ax3.set_xticklabels(dims)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Speedup vs Vector Dimension
ax4 = axes[1, 1]
speedup_encode = hrr_encode_time / fhrr_encode_time
speedup_decode = hrr_decode_time / fhrr_decode_time
ax4.plot(dims, speedup_encode, 'o-', linewidth=2, markersize=8, label='Encoding Speedup', alpha=0.7)
ax4.plot(dims, speedup_decode, 's-', linewidth=2, markersize=8, label='Decoding Speedup', alpha=0.7)
ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')
ax4.set_xlabel('Vector Dimension', fontsize=11)
ax4.set_ylabel('Speedup (HRR time / FHRR time)', fontsize=11)
ax4.set_title('FHRR Speedup vs Vector Dimension', fontsize=12, fontweight='bold')
ax4.set_xscale('log', base=2)
ax4.set_xticks(dims)
ax4.set_xticklabels(dims)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scratch/quick_test_tradeoff_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved trade-off plots to: scratch/quick_test_tradeoff_analysis.png")

print("\n" + "="*80)
print("✅ QUICK TEST COMPLETE!")
print("="*80)
