<p align="center">
  <img src="assets/hyperspace_logo_v1.png" alt="HyperSpace Logo" width="420"/>
</p>

<p align="center">A Unified Framework for Evaluating Vector Symbolic Architectures on Continuous Spatial Domains</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%20-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.9%20-ee4c2c" />
</p>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#basic-usage">Basic Usage</a> •
  <a href="#installation">Installation</a> •
  <a href="#credits">Credits</a>
</p>

## Introduction

**HyperSpace** is a unified Python framework for building and evaluating Vector Symbolic Architectures (VSAs) on continuous spatial domains. Built on PyTorch, HyperSpace enables researchers and practitioners to leverage hyperdimensional computing principles to encode, store, and retrieve continuous-valued data in high-dimensional vector spaces.

The framework implements Holographic Reduced Representations (HRR) using Fourier-domain operations and fractional power encoding (FPE), allowing smooth interpolation and association of continuous spatial coordinates with arbitrary values. HyperSpace is designed to be modular, extensible, and performance-optimized with `torch.compile` support for efficient computation.

## Key Features

- **🔬 Holographic Reduced Representations (HRR) Backend**: Implements circular convolution (binding) and superposition (bundling) operations via FFT for efficient hypervector manipulation
- **📍 Continuous Encoding**: Fractional power encoding (FPE) enables smooth representation of continuous spatial positions and scalar values
- **🧩 Modular Architecture**: Six core modules for flexible VSA workflows:
  - `PositionalEncoderModule` - Encodes continuous positions into hypervectors
  - `ValueEncoderModule` - Encodes scalar values into hypervectors
  - `MemoryStorageModule` - Maintains associative key-value memory through binding and bundling
  - `PositionalInversionModule` - Decodes positions from memory vectors
  - `CleanupModule` - Refines noisy hypervectors using resonator or Hopfield networks
  - `RegressionModule` - Recovers continuous values from hypervectors
- **⚡ Performance Optimized**: Leverages `torch.compile` for accelerated computation and supports both CPU and GPU execution
- **🔄 Batched Operations**: Efficient batched processing for encoding, binding, bundling, and similarity computations
- **🧪 Research-Ready**: Built for experimentation with VSAs on continuous spatial tasks like function approximation and spatial memory

## Basic Usage

Here's a minimal example demonstrating how to use HyperSpace to encode and retrieve continuous position-value pairs:

```python
import torch
from hyperspace.backends import HRRBackend
from hyperspace.core import (
    PositionalEncoderModule,
    ValueEncoderModule,
    MemoryStorageModule,
    RegressionModule
)

# Initialize the HRR backend
backend = HRRBackend(vector_dim=256, device="cpu")

# Create modules
pos_encoder = PositionalEncoderModule(backend=backend)
val_encoder = ValueEncoderModule(backend=backend)
memory = MemoryStorageModule(backend=backend)

# Sample data: positions (X) and values (Y)
X = torch.tensor([[0.2], [0.5], [0.8]])  # 3 positions
Y = torch.tensor([[1.0], [2.0], [1.5]])  # 3 corresponding values

# Encode positions and values into hypervectors
pos_vectors, _ = pos_encoder(X)
val_vectors, _ = val_encoder(Y)

# Store position-value associations in memory
prev_memory = backend.create_empty_vector()
memory_vector, _ = memory(
    p_vectors=pos_vectors,
    v_vectors=val_vectors,
    prev_memory=prev_memory
)

# Create regression module for decoding
codebook, _ = backend.value_encoding(Y)
regressor = RegressionModule(backend=backend, codebook=codebook, values=Y)

# Query the memory at a position and retrieve the value
query_pos = torch.tensor([[0.5]])  # Query at x=0.5
query_vector, _ = pos_encoder(query_pos)
predicted_value, _ = regressor(query_vector)

print(f"Predicted value at x=0.5: {predicted_value.item():.2f}")
```

For more detailed examples including function approximation and spatial memory tasks, see the `experiments/` directory.

## Installation

HyperSpace leverages Python 3.12, Pytorch, and Conda to enable the development and exploration of vector symbolic architectures applies to continuous spatial domains. Assuming you have conda or miniconda installed, you can create the dedicated hyperspace environment with the following commands:

```bash
conda create -n hyperspace_env python=3.12 -y
conda activate hyperspace_env
pip install -e .
```

Once the environment is created, you can activate it with the following command:

```bash
conda activate hyperspace_env
```

If desired, you can also test the functionality of HyperSpace with the following command:

```bash
pytest
```

You are now ready to begin leveraging the HyperSpace framework. If you have any questions or concerns, please reach out the maintainer listed in `setup.py`.

## Future Works and Features

- multi-factor resonator support
- automation of regression network training

## Credits

```latex
TODO: Add ArXiv citation
```

The logo for this repository was created with assistance from ChatGPT (OpenAI GPT-5).
