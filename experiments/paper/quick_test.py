"""
Quick test script to verify the benchmark pipeline works before running full experiments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark_hrr_fhrr import run_full_benchmark

# Quick test with small configuration
print("Running quick test with 16x16 map, D=256...")
print("This should complete in ~30 seconds\n")

results_hrr = run_full_benchmark(
    backend_name='HRR',
    resolution=16,
    vector_dim=256,
    device='cpu',
    terrain_type='mixed',
    codebook_size=32,
    cleanup_iterations=2,
    num_test_queries=20,
    seed=42
)

print("\n" + "="*60)

results_fhrr = run_full_benchmark(
    backend_name='FHRR',
    resolution=16,
    vector_dim=256,
    device='cpu',
    terrain_type='mixed',
    codebook_size=32,
    cleanup_iterations=2,
    num_test_queries=20,
    seed=42
)

print("\n" + "="*60)
print("QUICK COMPARISON")
print("="*60)
print(f"Encoding speedup: {results_hrr['encoding']['total_encoding'] / results_fhrr['encoding']['total_encoding']:.2f}×")
print(f"Query speedup: {results_hrr['query']['per_query'] / results_fhrr['query']['per_query']:.2f}×")
print(f"Memory ratio: {results_fhrr['memory_footprint']['total_mb'] / results_hrr['memory_footprint']['total_mb']:.2f}×")
print(f"RMSE comparison: HRR={results_hrr['rmse']:.4f}, FHRR={results_fhrr['rmse']:.4f}")
print("\n✅ Pipeline test successful! Ready for full benchmarks.")
