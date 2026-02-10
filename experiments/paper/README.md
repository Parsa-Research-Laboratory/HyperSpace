# HRR vs FHRR Benchmark Suite

This directory contains scripts for generating experimental data comparing HRR and FHRR backends on 2D spatial costmap encoding tasks.

## Files

- `benchmark_hrr_fhrr.py` - Main benchmark script for comprehensive experiments
- `quick_test.py` - Quick validation test (completes in ~30 seconds)
- `visualize_benchmark.py` - Visualization script for costmaps, predictions, and errors
- `README.md` - This file

## Quick Start

### 1. Verify Setup

First, run the quick test to ensure everything works:

```bash
cd experiments/paper
python quick_test.py
```

This will test a small 16×16 map with D=256 and should complete in ~30 seconds.

### 2. Run Full Benchmarks

#### Basic Run (CPU, default settings)
```bash
python benchmark_hrr_fhrr.py --output results/baseline_cpu.json
```

This will benchmark:
- Resolutions: 32×32, 64×64, 128×128, 256×256
- Vector dimensions: 512, 1024, 2048
- Mixed terrain type
- 4 cleanup iterations, 64 codebook size

**Estimated time:** ~2-4 hours on CPU

#### GPU Accelerated
```bash
python benchmark_hrr_fhrr.py --output results/baseline_gpu.json --device cuda
```

**Estimated time:** ~15-30 minutes on GPU

#### Custom Configuration
```bash
python benchmark_hrr_fhrr.py \
    --output results/custom.json \
    --device cpu \
    --resolutions 32 64 128 \
    --vector-dims 1024 2048 \
    --terrain-type mixed \
    --codebook-size 64 \
    --cleanup-iters 4 \
    --num-queries 100 \
    --seed 42
```

### 3. Command-Line Options

```
--output PATH           Output JSON file for results (default: benchmark_results.json)
--device {cpu,cuda}     Device to run on (default: cpu)
--resolutions N [N...]  Grid resolutions to test (default: 32 64 128 256)
--vector-dims N [N...]  Vector dimensions to test (default: 512 1024 2048)
--terrain-type TYPE     Terrain type: gaussian, obstacle, mixed (default: mixed)
--codebook-size N       Codebook size for cleanup (default: 64)
--cleanup-iters N       Number of cleanup iterations (default: 4)
--num-queries N         Number of test queries (default: 100)
--seed N                Random seed (default: 42)
```

## Output Format

The benchmark script produces a JSON file with the following structure:

```json
[
  {
    "backend": "HRR",
    "resolution": 32,
    "vector_dim": 1024,
    "terrain_type": "mixed",
    "n_train": 819,
    "n_test": 205,
    "encoding": {
      "total_encoding": 0.182,
      "per_sample": 0.000222
    },
    "query": {
      "total_query": 0.320,
      "per_query": 0.0032,
      "std_query": 0.0001
    },
    "cleanup": {
      "total_cleanup": 1.280,
      "per_query": 0.0128,
      "per_iteration": 0.0032
    },
    "regression": {
      "total_regression": 0.210,
      "per_query": 0.0021,
      "method": "codebook"
    },
    "rmse": 0.0482,
    "memory_footprint": {
      "memory_vector_mb": 0.004,
      "codebook_mb": 0.256,
      "total_mb": 0.260
    },
    "total_time": 1.992
  },
  ...
]
```

## Recommended Benchmark Configurations

### For Paper Tables (Balanced)
```bash
python benchmark_hrr_fhrr.py \
    --output results/paper_main.json \
    --resolutions 32 64 128 256 \
    --vector-dims 1024 \
    --terrain-type mixed \
    --device cpu
```

### For Dimension Scaling Study
```bash
python benchmark_hrr_fhrr.py \
    --output results/dimension_scaling.json \
    --resolutions 128 \
    --vector-dims 256 512 1024 2048 4096 \
    --device cpu
```

### For Resolution Scaling Study
```bash
python benchmark_hrr_fhrr.py \
    --output results/resolution_scaling.json \
    --resolutions 16 32 64 128 256 512 \
    --vector-dims 1024 \
    --device cpu
```

### For Different Terrain Types
```bash
# Smooth Gaussian
python benchmark_hrr_fhrr.py \
    --output results/terrain_gaussian.json \
    --terrain-type gaussian

# Obstacle Distance Fields
python benchmark_hrr_fhrr.py \
    --output results/terrain_obstacle.json \
    --terrain-type obstacle

# Mixed (default)
python benchmark_hrr_fhrr.py \
    --output results/terrain_mixed.json \
    --terrain-type mixed
```

## Analysis

After running benchmarks, you can analyze results using Python:

```python
import json

# Load results
with open('results/paper_main.json', 'r') as f:
    results = json.load(f)

# Extract HRR and FHRR results
hrr_results = [r for r in results if r['backend'] == 'HRR']
fhrr_results = [r for r in results if r['backend'] == 'FHRR']

# Compute speedups
for hrr, fhrr in zip(hrr_results, fhrr_results):
    encode_speedup = hrr['encoding']['total_encoding'] / fhrr['encoding']['total_encoding']
    query_speedup = hrr['query']['per_query'] / fhrr['query']['per_query']
    total_speedup = hrr['total_time'] / fhrr['total_time']
    
    print(f"Resolution: {hrr['resolution']}², D: {hrr['vector_dim']}")
    print(f"  Encoding speedup: {encode_speedup:.2f}×")
    print(f"  Query speedup: {query_speedup:.2f}×")
    print(f"  Total speedup: {total_speedup:.2f}×")
    print(f"  Memory ratio: {fhrr['memory_footprint']['total_mb'] / hrr['memory_footprint']['total_mb']:.2f}×")
    print()
```

## Visualization

### Visualize Pipeline Results

Visualize the complete pipeline including input costmaps, train/test splits, and full-field reconstructions:

```bash
# Visualize HRR backend
python visualize_benchmark.py --backend HRR --resolution 32 --vector-dim 1024

# Visualize FHRR backend
python visualize_benchmark.py --backend FHRR --resolution 32 --vector-dim 1024

# Save to file
python visualize_benchmark.py --backend HRR --resolution 64 --vector-dim 1024 \
    --save figures/hrr_reconstruction.png
```

The visualization includes:
- **Row 1**: Ground truth costmap, train/test split, training data only, test data only
- **Row 2**: Full-field reconstruction, absolute error map, test predictions, test error
- **Row 3**: Prediction accuracy scatter plot, error distribution histogram

### Visualization Options

```
--backend {HRR,FHRR}    Backend to visualize (required)
--resolution N          Grid resolution (default: 32)
--vector-dim N          Vector dimension (default: 1024)
--terrain-type TYPE     Terrain type: gaussian, obstacle, mixed (default: mixed)
--codebook-size N       Codebook size (default: 64)
--cleanup-iters N       Cleanup iterations (default: 4)
--device {cpu,cuda}     Device (default: cpu)
--seed N                Random seed (default: 42)
--save PATH             Save figure to file (e.g., figures/result.png)
```

### Example Visualizations

Compare HRR vs FHRR on the same terrain:

```bash
# Generate HRR visualization
python visualize_benchmark.py --backend HRR --resolution 32 --vector-dim 1024 \
    --seed 42 --save figures/hrr_32.png

# Generate FHRR visualization (same seed for comparison)
python visualize_benchmark.py --backend FHRR --resolution 32 --vector-dim 1024 \
    --seed 42 --save figures/fhrr_32.png
```

## Next Steps

1. **Run quick test** to verify setup
2. **Run visualizations** to understand the data and predictions
3. **Run main benchmarks** with desired configurations
4. **Generate LaTeX tables** from results for paper

## Troubleshooting

### Out of Memory (CPU)
- Reduce `--resolutions` (try smaller maps first)
- Reduce `--vector-dims` 
- Reduce `--codebook-size`

### Out of Memory (GPU)
- Same as CPU, or switch to `--device cpu`
- Use smaller `--num-queries`

### Slow Performance
- Use `--device cuda` if available
- Reduce number of configurations
- Run subsets separately and combine results

### Import Errors
- Ensure you're in the correct directory: `experiments/paper/`
- Ensure HyperSpace is installed: `pip install -e .` from repo root
- Check Python version (requires Python 3.8+)

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@article{snyder2026hyperspace,
  title={Efficient Encoding and Decoding of Continuous Spatial Information in Hyperdimensional Systems},
  author={Snyder, Shay and Capodieci, Andrew and Gorsich, David and Parsa, Maryam},
  journal={NICE},
  year={2026}
}
```
