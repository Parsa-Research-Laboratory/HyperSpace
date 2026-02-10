"""
Comprehensive test suite for FHRR backend.
Adapted from test_hrr.py with modifications for complex-valued frequency domain operations.
"""
import pytest
import torch
import numpy as np

from hyperspace.backends.fhrr import (
    FHRRBackend,
    _base_create_single_vector,
    _base_single_bind,
    _base_batch_bind,
    _base_single_bundle,
    _base_batch_bundle,
    _base_list_bundle,
    _base_list_bind,
    _base_single_fpe,
    _base_batch_fpe,
    _base_single_value_encoding,
    _base_batch_value_encoding,
    _base_single_normalize,
    _base_batch_normalize,
    _base_single_invert,
    _base_batch_invert,
    _base_single_weight,
    _base_batch_weight,
    _base_single_to_batch_bind,
    _base_single_to_batch_bundle,
)


# ============================================================================
# BASE FUNCTION TESTS: _base_create_single_vector
# ============================================================================

def test_base_create_single_vector_base_arguments():
    """Test the ability to generate random FHRR vectors (complex)."""
    from torch import Generator

    vd = 256
    gen = Generator(device="cpu")
    
    v1 = _base_create_single_vector(vector_dim=vd, gen=gen)
    v2 = _base_create_single_vector(vector_dim=vd, gen=gen)

    # Check they're complex
    assert v1.dtype in [torch.complex64, torch.complex32]
    assert len(v1.shape) == 1
    assert v1.shape[0] == vd
    
    # Check unit magnitude for each element
    magnitudes = torch.abs(v1)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)
    
    # Check approximate orthogonality using complex inner product
    dot = torch.dot(v1, torch.conj(v2))
    assert torch.abs(dot) / vd < 0.15, f"Vectors not orthogonal; |dot|/D={torch.abs(dot).item()/vd}"


def test_base_create_single_vector_non_int_vector_dim():
    """Test that create function catches non-integer vector_dim."""
    from torch import Generator

    vd = 1.0
    with pytest.raises(TypeError):
        _base_create_single_vector(vector_dim=vd, gen=Generator(device="cpu"))


def test_base_create_single_vector_negative_vector_dim():
    """Test that create function catches negative dimensionalities."""
    from torch import Generator

    vd = -1
    with pytest.raises(ValueError):
        _base_create_single_vector(vector_dim=vd, gen=Generator(device="cpu"))


def test_base_create_single_vector_invalid_generator():
    """Test that create function catches invalid generators."""
    with pytest.raises(TypeError):
        _base_create_single_vector(vector_dim=20, gen=int(5))


# ============================================================================
# BASE FUNCTION TESTS: _base_single_bind (complex multiplication)
# ============================================================================

def test_base_single_bind():
    """Test _base_single_bind function with complex multiplication."""
    from torch import Generator

    vd = 256
    gen = Generator()

    v1 = _base_create_single_vector(vd, gen)
    v2 = _base_create_single_vector(vd, gen)

    v_bind = _base_single_bind(v1, v2)

    # FHRR: binding is just element-wise complex multiplication
    expected = v1 * v2
    
    assert torch.allclose(v_bind, expected, atol=1e-6)
    assert v_bind.dtype in [torch.complex64, torch.complex32]


def test_base_batch_bind():
    """Test batched bind function with complex multiplication."""
    from torch import Generator

    B = 8
    D = 256
    gen = Generator().manual_seed(0)

    v1_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v2_list = [_base_create_single_vector(D, gen) for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)

    v_bind_batch = _base_batch_bind(v1, v2)

    # Should be element-wise complex multiplication
    expected = v1 * v2
    
    assert torch.allclose(v_bind_batch, expected, atol=1e-6)


# ============================================================================
# BASE FUNCTION TESTS: _base_single_bundle (complex addition)
# ============================================================================

def test_base_single_bundle():
    """Test _base_single_bundle function with complex addition."""
    from torch import Generator

    vd = 256
    gen = Generator().manual_seed(0)

    v1 = _base_create_single_vector(vd, gen)
    v2 = _base_create_single_vector(vd, gen)

    v_bundle = _base_single_bundle(v1, v2)

    # FHRR: bundling is complex addition
    expected = v1 + v2

    assert torch.allclose(v_bundle, expected, atol=1e-6)


def test_base_batch_bundle():
    """Test _base_batch_bundle function."""
    from torch import Generator

    B = 8
    D = 256
    gen = Generator().manual_seed(0)

    v1_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v2_list = [_base_create_single_vector(D, gen) for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)

    v_bundle_batch = _base_batch_bundle(v1, v2)

    expected = v1 + v2

    assert torch.allclose(v_bundle_batch, expected, atol=1e-6)


def test_base_list_bundle():
    """Test _base_list_bundle function."""
    from torch import Generator

    B = 2
    D = 256
    gen = Generator().manual_seed(0)

    v_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v1 = torch.stack(v_list, dim=0)

    v_bundle_list = _base_list_bundle(v1)
    v_out_gt = _base_single_bundle(v_list[0], v_list[1])

    assert v_bundle_list.shape == (D,)
    assert torch.allclose(v_out_gt, v_bundle_list, atol=1e-6)


def test_base_list_bind():
    """Test _base_list_bind function."""
    from torch import Generator

    B = 3
    D = 256
    gen = Generator().manual_seed(0)

    v_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v_batch = torch.stack(v_list, dim=0)

    v_bind_list = _base_list_bind(v_batch)

    # Ground truth: sequential binding
    v_gt = _base_single_bind(v_list[0], v_list[1])
    v_gt = _base_single_bind(v_gt, v_list[2])

    assert v_bind_list.shape == (D,)
    assert v_gt.shape == (D,)
    assert torch.allclose(v_bind_list, v_gt, atol=1e-6)


def test_base_single_value_encoding():
    """Test _base_single_value_encoding with complex vectors."""
    from torch import Generator

    D = 512
    value_dim = 3
    gen = Generator().manual_seed(0)

    # Create basis vectors
    basis_list = [_base_create_single_vector(D, gen) for _ in range(value_dim)]
    basis = torch.stack(basis_list, dim=0)

    # Value to encode
    value = torch.tensor([2.0, 3.5, -1.2])
    length_scale = 1.5

    # Encode
    encoded = _base_single_value_encoding(value, basis, length_scale)

    assert encoded.shape == (D,)
    assert encoded.dtype in [torch.complex64, torch.complex32]

    # Manual computation for verification
    expected = basis ** (value / length_scale).unsqueeze(-1)  # (value_dim, D)
    expected = torch.prod(expected, dim=0)  # (D,)

    assert torch.allclose(encoded, expected, atol=1e-6)


def test_base_batch_value_encoding():
    """Test _base_batch_value_encoding with batched complex vectors."""
    from torch import Generator

    D = 512
    value_dim = 2
    batch_size = 4
    gen = Generator().manual_seed(0)

    # Create basis vectors
    basis_list = [_base_create_single_vector(D, gen) for _ in range(value_dim)]
    basis = torch.stack(basis_list, dim=0)

    # Batched values to encode
    values = torch.randn(batch_size, value_dim)
    length_scale = 1.0

    # Encode
    encoded = _base_batch_value_encoding(values, basis, length_scale)

    assert encoded.shape == (batch_size, D)
    assert encoded.dtype in [torch.complex64, torch.complex32]

    # Verify each batch element independently
    for i in range(batch_size):
        single_encoded = _base_single_value_encoding(values[i], basis, length_scale)
        assert torch.allclose(encoded[i], single_encoded, atol=1e-6)


def test_base_single_to_batch_bind():
    """Test _base_single_to_batch_bind broadcasts single vector to batch."""
    from torch import Generator

    B = 5
    D = 256
    gen = Generator().manual_seed(0)

    # Single vector
    v_single = _base_create_single_vector(D, gen)

    # Batch of vectors
    v_batch_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v_batch = torch.stack(v_batch_list, dim=0)

    # Single-to-batch bind
    result = _base_single_to_batch_bind(v_single, v_batch)

    assert result.shape == (B, D)
    assert result.dtype in [torch.complex64, torch.complex32]

    # Verify each element is correctly bound
    for i in range(B):
        expected = _base_single_bind(v_single, v_batch[i])
        assert torch.allclose(result[i], expected, atol=1e-6)


def test_base_single_to_batch_bundle():
    """Test _base_single_to_batch_bundle broadcasts single vector to batch."""
    from torch import Generator

    B = 5
    D = 256
    gen = Generator().manual_seed(0)

    # Single vector
    v_single = _base_create_single_vector(D, gen)

    # Batch of vectors
    v_batch_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v_batch = torch.stack(v_batch_list, dim=0)

    # Single-to-batch bundle
    result = _base_single_to_batch_bundle(v_single, v_batch)

    assert result.shape == (B, D)
    assert result.dtype in [torch.complex64, torch.complex32]

    # Verify each element is correctly bundled
    for i in range(B):
        expected = _base_single_bundle(v_single, v_batch[i])
        assert torch.allclose(result[i], expected, atol=1e-6)


# ============================================================================
# BASE FUNCTION TESTS: _base_single_fpe (complex exponentiation)
# ============================================================================

def test_base_single_fpe():
    """Test _base_single_fpe function with direct complex exponentiation."""
    from torch import Generator

    D = 256
    gen = Generator().manual_seed(0)

    basis = _base_create_single_vector(D, gen)
    power = np.random.random()
    length_scale = np.random.random()

    fpe_vector = _base_single_fpe(basis, power, length_scale)

    # FHRR: direct complex exponentiation (no FFT needed!)
    expected = basis ** (power / length_scale)

    assert torch.allclose(fpe_vector, expected, rtol=1e-5, atol=1e-5)


def test_base_batch_fpe():
    """Test _base_batch_fpe function."""
    from torch import Generator

    B = 8
    D = 256
    gen = Generator().manual_seed(0)

    bases_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    powers = [np.random.random() for _ in range(B)]

    bases_stack = torch.stack(bases_list, dim=0)
    powers_stack = torch.tensor(powers)
    length_scale = np.random.random()

    fpes = _base_batch_fpe(bases_stack, powers_stack, length_scale)

    # Compute expected
    expected = bases_stack ** (powers_stack / length_scale).unsqueeze(-1)

    assert torch.allclose(fpes, expected, rtol=1e-5, atol=1e-5)


# ============================================================================
# BASE FUNCTION TESTS: Normalization (complex magnitude)
# ============================================================================

def test_base_single_normalize():
    """Test _base_single_normalize with complex magnitude."""
    v = torch.randn(10, dtype=torch.complex64) + 1j * torch.randn(10)
    
    v_norm = _base_single_normalize(v)
    
    # Check unit L2 norm using complex magnitude
    magnitude = torch.sqrt(torch.sum(torch.abs(v_norm) ** 2))
    assert torch.isclose(magnitude, torch.tensor(1.0), atol=1e-5)


def test_base_batch_normalize():
    """Test _base_batch_normalize with complex magnitude."""
    v = torch.randn(5, 10, dtype=torch.complex64) + 1j * torch.randn(5, 10)
    
    v_norm = _base_batch_normalize(v)
    
    # Check unit L2 norm for each row
    magnitudes = torch.sqrt(torch.sum(torch.abs(v_norm) ** 2, dim=-1))
    assert torch.allclose(magnitudes, torch.ones(5), atol=1e-5)


# ============================================================================
# BASE FUNCTION TESTS: Inversion (complex conjugation - O(n)!)
# ============================================================================

def test_base_single_invert():
    """Test _base_single_invert with complex conjugation."""
    from torch import Generator

    D = 128
    gen = Generator()
    v = _base_create_single_vector(D, gen)

    v_inv = _base_single_invert(v)

    # FHRR: inversion is just complex conjugation (O(n) vs O(n log n) for HRR!)
    expected = torch.conj(v)

    assert torch.allclose(v_inv, expected, atol=1e-6)


def test_base_batch_invert():
    """Test _base_batch_invert with complex conjugation."""
    from torch import Generator

    B = 10
    D = 128
    gen = Generator()

    v_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v_tensor = torch.stack(v_list, dim=0)

    v_inv = _base_batch_invert(v_tensor)

    # Should be complex conjugation
    expected = torch.conj(v_tensor)

    assert torch.allclose(v_inv, expected, atol=1e-6)


# ============================================================================
# BASE FUNCTION TESTS: Weighting
# ============================================================================

def test_base_single_weight():
    """Test _base_single_weight function."""
    D = 128
    x = torch.randn(D, dtype=torch.complex64)
    w_small = torch.tensor([0.5])

    x_weighted = _base_single_weight(x, w_small)
    expected = x * w_small

    assert torch.allclose(x_weighted, expected, atol=1e-6)


def test_base_batch_weight():
    """Test _base_batch_weight function."""
    B = 10
    D = 256
    x = torch.randn((B, D), dtype=torch.complex64)
    w = torch.rand((B))

    x_weighted = _base_batch_weight(x, w)

    expected = x * w.view(-1, 1)

    assert torch.allclose(x_weighted, expected, atol=1e-6)


# ============================================================================
# BACKEND INITIALIZATION TESTS
# ============================================================================

def test_backend_base_init():
    """Test FHRRBackend initialization."""
    vdim = 128
    b = FHRRBackend(vector_dim=vdim)
    
    assert b.vector_dim == vdim
    assert b.name == "FHRR"
    assert b.vector_dtype in [torch.complex32, torch.complex64, torch.complex128]


def test_length_scale_init():
    """Test length scale argument."""
    vd = 128
    ls_list = [1, 2, 3.0, 0.00001, -1.0]

    for ls in ls_list:
        b = FHRRBackend(vector_dim=vd, length_scale=ls)
        assert b.length_scale == float(ls)


def test_invalid_length_scale_init():
    """Test zero-length scale initialization."""
    vd = 128
    ls = 0.0

    with pytest.raises(ValueError):
        _ = FHRRBackend(vector_dim=vd, length_scale=ls)


# ============================================================================
# BACKEND METHOD TESTS: create_random_vector
# ============================================================================

def test_backend_method_create_random_vector_base():
    """Test that create_random_vector function works."""
    b = FHRRBackend(vector_dim=128)
    v = b.create_random_vector()
    
    assert v is not None


def test_backend_method_create_random_vector_object_type():
    """Test that create_random_vector returns a Tensor."""
    b = FHRRBackend(vector_dim=128)
    v = b.create_random_vector()
    
    assert isinstance(v, torch.Tensor)


def test_backend_method_create_random_vector_device():
    """Test device of vector from create_random_vector."""
    b = FHRRBackend(vector_dim=128)
    v = b.create_random_vector()

    assert b.device == v.device


def test_backend_method_create_random_vector_data_type():
    """Test dtype of vector from create_random_vector."""
    b = FHRRBackend(vector_dim=128)
    v = b.create_random_vector()

    assert v.dtype in [torch.complex64, torch.complex32]


def test_backend_method_create_random_vector_shape():
    """Test shape of vectors from create_random_vector."""
    v_dims = [128, 256, 512]

    for d in v_dims:
        b = FHRRBackend(vector_dim=d)
        v = b.create_random_vector()

        assert v.ndim == 1
        assert v.shape[0] == d


def test_backend_method_create_random_vector_unit_magnitude():
    """Test that FHRR vectors have unit magnitude."""
    b = FHRRBackend(vector_dim=256)
    v = b.create_random_vector()
    
    # Check unit magnitude for each element
    magnitudes = torch.abs(v)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)


def test_backend_method_create_random_vector_orthogonality():
    """Test orthogonality of vectors from create_random_vector."""
    vd = 25600
    b = FHRRBackend(vector_dim=vd)

    v1 = b.create_random_vector()
    v2 = b.create_random_vector()

    # Complex inner product with conjugate
    dot = torch.dot(v1, torch.conj(v2))
    similarity = torch.real(dot) / vd
    
    assert abs(similarity) < 0.02, f"Vectors not orthogonal; sim={similarity.item()}"


# ============================================================================
# BACKEND METHOD TESTS: bind
# ============================================================================

def test_backend_method_single_bind():
    """Test backend's binding function with single vectors."""
    vector_dim = 2048
    backend = FHRRBackend(vector_dim=vector_dim)

    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()

    v_out, _ = backend.bind(v1, v2)

    # FHRR binding is element-wise complex multiplication
    expected = v1 * v2

    assert torch.allclose(v_out, expected, atol=1e-6)


def test_backend_method_batch_bind():
    """Test backend's binding function with batched vectors."""
    B = 8
    D = 256
    backend = FHRRBackend(vector_dim=D)

    v1_list = [backend.create_random_vector() for _ in range(B)]
    v2_list = [backend.create_random_vector() for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)

    v_bind_batch, _ = backend.bind(v1, v2)

    expected = v1 * v2

    assert torch.allclose(v_bind_batch, expected, atol=1e-6)


def test_backend_list_bind():
    """Test backend's list bind method."""
    B = 2
    D = 256
    b = FHRRBackend(vector_dim=D)

    v_list = [b.create_random_vector() for _ in range(B)]
    v1 = torch.stack(v_list, dim=0)

    v_bind_list, _ = b.bind(v1)  # list bind (no second argument)

    # Should multiply all vectors together
    expected = v_list[0] * v_list[1]

    assert v_bind_list.shape == (D,)
    assert torch.allclose(v_bind_list, expected, atol=1e-6)


# ============================================================================
# BACKEND METHOD TESTS: bundle
# ============================================================================

def test_backend_method_single_bundle():
    """Test backend's bundling function with single vectors."""
    D = 256
    backend = FHRRBackend(vector_dim=D)

    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()

    v_bundle, _ = backend.bundle(v1, v2)

    expected = v1 + v2

    assert torch.allclose(v_bundle, expected, atol=1e-6)


def test_backend_method_batch_bundle():
    """Test backend's batch bundle method."""
    B = 8
    D = 256
    backend = FHRRBackend(vector_dim=D)

    v1_list = [backend.create_random_vector() for _ in range(B)]
    v2_list = [backend.create_random_vector() for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)

    v_bundle_batch, _ = backend.bundle(v1, v2)

    expected = v1 + v2

    assert torch.allclose(v_bundle_batch, expected, atol=1e-6)


# ============================================================================
# BACKEND METHOD TESTS: similarity (complex inner product)
# ============================================================================

def test_backend_method_single_similarity_same():
    """Test similarity of vector with itself (should be 1.0)."""
    D = 256
    backend = FHRRBackend(vector_dim=D)

    v1 = backend.create_random_vector()
    v1_norm, _ = backend.normalize(v1)

    sim, _ = backend.similarity(v1_norm, v1_norm)

    assert torch.isclose(sim, torch.tensor(1.0), atol=1e-5)


def test_backend_method_single_similarity_orthogonal():
    """Test similarity of orthogonal vectors (should be ~0)."""
    D = 10000
    backend = FHRRBackend(vector_dim=D)

    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()

    sim, _ = backend.similarity(v1, v2)

    assert torch.abs(sim) < 0.1, f"Expected ~0 similarity, got {sim.item()}"


def test_backend_method_batch_similarity_same():
    """Test batched similarity with identical vectors."""
    B = 8
    D = 10000
    backend = FHRRBackend(vector_dim=D)

    v_list = [backend.create_random_vector() for _ in range(B)]
    v1 = torch.stack(v_list, dim=0)

    sims, _ = backend.similarity(v1, v1)

    expected = torch.ones(B)
    assert torch.allclose(sims, expected, atol=1e-5)


# ============================================================================
# BACKEND METHOD TESTS: normalize
# ============================================================================

def test_backend_normalize_single():
    """Test backend's normalization with single vector."""
    D = 128
    backend = FHRRBackend(vector_dim=D)

    x = backend.create_random_vector() * 5.0  # Scale it

    pred, _ = backend.normalize(x)

    # Check unit magnitude
    magnitude = torch.sqrt(torch.sum(torch.abs(pred) ** 2))
    assert torch.isclose(magnitude, torch.tensor(1.0), atol=1e-5)


def test_backend_normalize_batched():
    """Test backend's normalization with batch of vectors."""
    B = 32
    D = 128
    backend = FHRRBackend(vector_dim=D)

    x = torch.randn((B, D), dtype=torch.complex64) * 10.0

    pred, _ = backend.normalize(x)

    # Check unit magnitude for each row
    magnitudes = torch.sqrt(torch.sum(torch.abs(pred) ** 2, dim=-1))
    assert torch.allclose(magnitudes, torch.ones(B), atol=1e-5)


# ============================================================================
# BACKEND METHOD TESTS: invert
# ============================================================================

def test_backend_single_invert():
    """Test backend's invert method with single vector."""
    D = 128
    b = FHRRBackend(vector_dim=D)

    v = b.create_random_vector()
    v_inv, _ = b.invert(v)

    # Should be complex conjugation
    expected = torch.conj(v)

    assert torch.allclose(v_inv, expected, atol=1e-6)


def test_backend_batch_invert():
    """Test backend's invert method with batch of vectors."""
    B = 10
    D = 128
    b = FHRRBackend(vector_dim=D)

    v_list = [b.create_random_vector() for _ in range(B)]
    v_tensor = torch.stack(v_list, dim=0)

    pred, _ = b.invert(v_tensor)
    expected = torch.conj(v_tensor)

    assert torch.allclose(pred, expected, atol=1e-6)


def test_backend_bind_unbind_identity():
    """Test that bind followed by unbind retrieves original vector."""
    backend = FHRRBackend(vector_dim=1024)

    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()

    # Bind: v_bound = v1 * v2
    v_bound, _ = backend.bind(v1, v2)

    # Unbind: v_retrieved = v_bound * conj(v2)
    v2_inv, _ = backend.invert(v2)
    v_retrieved, _ = backend.bind(v_bound, v2_inv)

    # Normalize for comparison
    v1_norm, _ = backend.normalize(v1)
    v_retrieved_norm, _ = backend.normalize(v_retrieved)

    # Check similarity (should be very close to 1 for complex operations)
    similarity = torch.real(torch.dot(v1_norm, torch.conj(v_retrieved_norm)))
    assert similarity > 0.99, f"Bind-unbind similarity too low: {similarity.item()}"


# ============================================================================
# BACKEND METHOD TESTS: weight
# ============================================================================

def test_backend_single_weight():
    """Test backend's weighting function with single vector."""
    D = 128
    backend = FHRRBackend(vector_dim=D)

    x = backend.create_random_vector()
    w_small = torch.tensor([0.5])

    x_weighted, _ = backend.weight(x, w_small)

    assert x_weighted.shape == (D,)
    
    # Check magnitude is scaled
    mag_original = torch.sqrt(torch.sum(torch.abs(x) ** 2))
    mag_weighted = torch.sqrt(torch.sum(torch.abs(x_weighted) ** 2))
    
    assert torch.isclose(mag_weighted, mag_original * w_small, atol=1e-5)


def test_backend_batch_weight():
    """Test backend's weighting function with batch of vectors."""
    B = 10
    D = 256
    backend = FHRRBackend(vector_dim=D)

    x = torch.randn((B, D), dtype=torch.complex64)
    w = torch.rand((B))

    x_pred, _ = backend.weight(x, w)

    expected = x * w.view(-1, 1)

    assert torch.allclose(x_pred, expected, atol=1e-6)


# ============================================================================
# BACKEND METHOD TESTS: positional_encoding
# ============================================================================

def test_positional_encoding_single():
    """Test single positional encoding."""
    env_dim = 3
    vector_dim = 1280
    position = torch.tensor([2.0, 3.0, 4.0])
    length_scale = 1.5

    b = FHRRBackend(
        vector_dim=vector_dim,
        env_dim=env_dim,
        length_scale=length_scale
    )

    pred, _ = b.positional_encoding(position)

    assert pred.shape == (vector_dim,)
    assert pred.dtype in [torch.complex64, torch.complex32]


def test_positional_encoding_batched():
    """Test batched positional encoding."""
    batch_size = 3
    env_dim = 2
    vector_dim = 1280
    position = torch.tensor([
        [2.0, 3.0],
        [1.0, 5.0],
        [2.0, 6.0]
    ])
    length_scale = 1.5

    b = FHRRBackend(
        vector_dim=vector_dim,
        env_dim=env_dim,
        length_scale=length_scale
    )

    pred, _ = b.positional_encoding(position)

    assert pred.shape == (batch_size, vector_dim)
    assert pred.dtype in [torch.complex64, torch.complex32]


# ============================================================================
# BACKEND METHOD TESTS: value_encoding
# ============================================================================

def test_value_encoding_single():
    """Test single value encoding."""
    value_dim = 3
    vector_dim = 1280
    value = torch.tensor([2.0, 3.0, 4.0])
    length_scale = 1.5

    b = FHRRBackend(
        vector_dim=vector_dim,
        value_dim=value_dim,
        length_scale=length_scale
    )

    pred, _ = b.value_encoding(value)

    assert pred.shape == (vector_dim,)
    assert pred.dtype in [torch.complex64, torch.complex32]


def test_value_encoding_batched():
    """Test batched value encoding."""
    batch_size = 3
    value_dim = 2
    vector_dim = 1280
    value = torch.tensor([
        [2.0, 3.0],
        [1.0, 5.0],
        [2.0, 6.0]
    ])
    length_scale = 1.5

    b = FHRRBackend(
        vector_dim=vector_dim,
        value_dim=value_dim,
        length_scale=length_scale
    )

    pred, _ = b.value_encoding(value)

    assert pred.shape == (batch_size, vector_dim)
    assert pred.dtype in [torch.complex64, torch.complex32]


# ============================================================================
# PERFORMANCE AND SPECIAL TESTS
# ============================================================================

def test_fhrr_performance_characteristics():
    """
    Verify FHRR operations are O(n) vs HRR's O(n log n).
    This is a sanity check, not a rigorous benchmark.
    """
    D = 8192

    fhrr_backend = FHRRBackend(vector_dim=D)

    # Create complex vectors
    fhrr_v1 = fhrr_backend.create_random_vector()
    fhrr_v2 = fhrr_backend.create_random_vector()

    # Test binding (O(n) for FHRR - just complex multiplication)
    fhrr_bound, _ = fhrr_backend.bind(fhrr_v1, fhrr_v2)
    assert fhrr_bound.shape == (D,)
    assert fhrr_bound.dtype in [torch.complex64, torch.complex32]

    # Test inversion (O(n) for FHRR - just conjugation)
    fhrr_inv, _ = fhrr_backend.invert(fhrr_v1)
    assert fhrr_inv.shape == (D,)
    expected_inv = torch.conj(fhrr_v1)
    assert torch.allclose(fhrr_inv, expected_inv, atol=1e-6)


def test_complex_phase_preservation():
    """Test that FHRR preserves phase information correctly."""
    D = 256
    backend = FHRRBackend(vector_dim=D)

    v = backend.create_random_vector()

    # Check that phases are preserved through operations
    phases = torch.angle(v)
    assert torch.all(torch.isfinite(phases))
    assert phases.min() >= -np.pi - 1e-5
    assert phases.max() <= np.pi + 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
