"""
Basic tests for FHRR backend to verify core functionality.
"""
import pytest
import torch
import numpy as np

from hyperspace.backends.fhrr import FHRRBackend


def test_fhrr_backend_initialization():
    """Test that FHRRBackend can be instantiated."""
    backend = FHRRBackend(vector_dim=128)
    assert backend.vector_dim == 128
    assert backend.vector_dtype == torch.complex32
    assert backend.name == "FHRR"


def test_fhrr_create_random_vector():
    """Test that FHRR can create complex random vectors with unit magnitude."""
    backend = FHRRBackend(vector_dim=256)
    
    v = backend.create_random_vector()
    
    # Check it's a complex tensor
    assert v.dtype in [torch.complex64, torch.complex32]
    assert v.shape == (256,)
    
    # Check unit magnitude
    magnitudes = torch.abs(v)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)


def test_fhrr_create_multiple_orthogonal_vectors():
    """Test that multiple FHRR vectors are approximately orthogonal."""
    D = 10000
    backend = FHRRBackend(vector_dim=D)
    
    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()
    
    # Normalize first (each should already be unit magnitude)
    v1_norm = v1 / torch.sqrt(torch.sum(torch.abs(v1) ** 2))
    v2_norm = v2 / torch.sqrt(torch.sum(torch.abs(v2) ** 2))
    
    # Complex inner product similarity: Re(<v1, conj(v2)>)
    similarity = torch.real(torch.dot(v1_norm, torch.conj(v2_norm)))
    
    # For random unit vectors, similarity should be close to 0
    # Allow some tolerance proportional to 1/sqrt(D)
    tolerance = 5.0 / np.sqrt(D)  # ~0.05 for D=10000
    assert abs(similarity) < tolerance, f"Vectors not orthogonal; similarity={similarity.item()}"


def test_fhrr_bind_single():
    """Test single vector binding (complex multiplication)."""
    backend = FHRRBackend(vector_dim=128)
    
    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()
    
    v_bound, _ = backend.bind(v1, v2)
    
    # Check output is complex
    assert v_bound.dtype in [torch.complex64, torch.complex32]
    assert v_bound.shape == (128,)
    
    # Manual check: should be element-wise multiplication
    expected = v1 * v2
    assert torch.allclose(v_bound, expected, atol=1e-6)


def test_fhrr_bundle_single():
    """Test single vector bundling (complex addition)."""
    backend = FHRRBackend(vector_dim=128)
    
    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()
    
    v_bundled, _ = backend.bundle(v1, v2)
    
    # Check output is complex
    assert v_bundled.dtype in [torch.complex64, torch.complex32]
    assert v_bundled.shape == (128,)
    
    # Manual check: should be element-wise addition
    expected = v1 + v2
    assert torch.allclose(v_bundled, expected, atol=1e-6)


def test_fhrr_invert_single():
    """Test single vector inversion (complex conjugation)."""
    backend = FHRRBackend(vector_dim=128)
    
    v = backend.create_random_vector()
    v_inv, _ = backend.invert(v)
    
    # Check output is complex
    assert v_inv.dtype in [torch.complex64, torch.complex32]
    assert v_inv.shape == (128,)
    
    # Manual check: should be complex conjugate
    expected = torch.conj(v)
    assert torch.allclose(v_inv, expected, atol=1e-6)


def test_fhrr_normalize_single():
    """Test single vector normalization."""
    backend = FHRRBackend(vector_dim=128)
    
    # Create a non-normalized vector
    v = backend.create_random_vector() * 5.0  # Scale it
    
    v_norm, _ = backend.normalize(v)
    
    # Check unit magnitude
    magnitude = torch.sqrt(torch.sum(torch.abs(v_norm) ** 2))
    assert torch.isclose(magnitude, torch.tensor(1.0), atol=1e-5)


def test_fhrr_batch_bind():
    """Test batch vector binding."""
    backend = FHRRBackend(vector_dim=128)
    
    B = 16
    v1_list = [backend.create_random_vector() for _ in range(B)]
    v2_list = [backend.create_random_vector() for _ in range(B)]
    
    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)
    
    v_bound, _ = backend.bind(v1, v2)
    
    # Check output shape and type
    assert v_bound.shape == (B, 128)
    assert v_bound.dtype in [torch.complex64, torch.complex32]
    
    # Manual check
    expected = v1 * v2
    assert torch.allclose(v_bound, expected, atol=1e-6)


def test_fhrr_positional_encoding_single():
    """Test single positional encoding."""
    backend = FHRRBackend(vector_dim=256, env_dim=2, length_scale=1.0)
    
    position = torch.tensor([1.5, 2.3])
    
    encoded, _ = backend.positional_encoding(position)
    
    # Check output is complex
    assert encoded.dtype in [torch.complex64, torch.complex32]
    assert encoded.shape == (256,)


def test_fhrr_positional_encoding_batch():
    """Test batch positional encoding."""
    backend = FHRRBackend(vector_dim=256, env_dim=3, length_scale=1.0)
    
    B = 8
    positions = torch.randn(B, 3)
    
    encoded, _ = backend.positional_encoding(positions)
    
    # Check output is complex and correct shape
    assert encoded.dtype in [torch.complex64, torch.complex32]
    assert encoded.shape == (B, 256)


def test_fhrr_value_encoding_single():
    """Test single value encoding."""
    backend = FHRRBackend(vector_dim=256, value_dim=2, length_scale=1.0)
    
    value = torch.tensor([3.0, 4.0])
    
    encoded, _ = backend.value_encoding(value)
    
    # Check output is complex
    assert encoded.dtype in [torch.complex64, torch.complex32]
    assert encoded.shape == (256,)


def test_fhrr_bind_unbind_identity():
    """Test that bind followed by unbind (with inverse) retrieves original vector."""
    backend = FHRRBackend(vector_dim=1024)
    
    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()
    
    # Bind: v_bound = v1 ⊗ v2
    v_bound, _ = backend.bind(v1, v2)
    
    # Unbind: v_retrieved = v_bound ⊗ v2^{-1}
    v2_inv, _ = backend.invert(v2)
    v_retrieved, _ = backend.bind(v_bound, v2_inv)
    
    # Normalize for comparison
    v1_norm, _ = backend.normalize(v1)
    v_retrieved_norm, _ = backend.normalize(v_retrieved)
    
    # Check similarity (should be close to 1)
    similarity = torch.real(torch.dot(v1_norm, torch.conj(v_retrieved_norm)))
    assert similarity > 0.95, f"Bind-unbind similarity too low: {similarity.item()}"


def test_fhrr_performance_vs_hrr_binding():
    """
    Verify that FHRR binding is indeed O(n) vs HRR's O(n log n).
    This is more of a sanity check than a rigorous benchmark.
    """
    from hyperspace.backends.hrr import HRRBackend
    
    D = 8192
    
    fhrr_backend = FHRRBackend(vector_dim=D)
    hrr_backend = HRRBackend(vector_dim=D)
    
    # Create vectors
    fhrr_v1 = fhrr_backend.create_random_vector()
    fhrr_v2 = fhrr_backend.create_random_vector()
    
    hrr_v1 = hrr_backend.create_random_vector()
    hrr_v2 = hrr_backend.create_random_vector()
    
    # Just verify both work (actual timing would require more sophisticated setup)
    fhrr_bound, _ = fhrr_backend.bind(fhrr_v1, fhrr_v2)
    hrr_bound, _ = hrr_backend.bind(hrr_v1, hrr_v2)
    
    assert fhrr_bound.shape == (D,)
    assert hrr_bound.shape == (D,)
    
    # FHRR should produce complex output
    assert fhrr_bound.dtype in [torch.complex64, torch.complex32]
    # HRR should produce real output
    assert hrr_bound.dtype in [torch.float32, torch.float64]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
