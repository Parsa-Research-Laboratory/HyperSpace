import pytest

def test_ve_no_backend():
    """
    Test that the value encoder module doesn't assume a default backend.
    """
    from hyperspace.core.value_encoder import ValueEncoderModule

    with pytest.raises(TypeError):
        ValueEncoderModule()

def test_ve_true_backend():
    """
    Test that the ve module initializes with a value backend
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.value_encoder import ValueEncoderModule

    b = HRRBackend(vector_dim=128)
    _ = ValueEncoderModule(b)

def test_ve_invalid_backend():
    """
    Test that the pe module throws and error when an invalid
    backend is passed
    """
    from hyperspace.core.value_encoder import ValueEncoderModule

    with pytest.raises(TypeError):
        _ = ValueEncoderModule(5)

def test_value_encoding_module_invalid_x_type():
    """
    Test that the value_encoding_module throws an error
    when x isn't a Tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.value_encoder import ValueEncoderModule

    B: int = 16
    E: int = 3

    b = HRRBackend(vector_dim=128, env_dim=E)
    vem = ValueEncoderModule(b)

    x_single = np.zeros((E))
    x_batch = np.zeros((B, E))

    with pytest.raises(TypeError):
        vem(x_single)

    with pytest.raises(TypeError):
        vem(x_batch)

def test_value_encoding_module_invalid_x_dim():
    """
    Test that the value encoding module throws an error when
    x isn't single or batched
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.value_encoder import ValueEncoderModule

    b = HRRBackend(vector_dim=128)
    vem = ValueEncoderModule(b)

    x_invalid = torch.zeros((5, 5, 5))
    
    with pytest.raises(ValueError):
        vem(x_invalid)

def test_value_encoding_single_x():
    """
    test that the value encoding module works when given
    a single x value
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.value_encoder import ValueEncoderModule

    value_dim: int = 3
    vector_dim: int = 1280
    value = torch.from_numpy(np.array([2, 3, 4]))
    length_scale: float = 1.5

    b = HRRBackend(
        vector_dim=vector_dim,
        value_dim=value_dim,
        length_scale=length_scale
    )
    vem = ValueEncoderModule(
        backend=b
    )

    assert b.value_basis_vectors.shape == (value_dim, vector_dim)

    pred, _ = vem(value)
    gt, _ = b.value_encoding(value)

    assert pred.shape == (vector_dim,)
    assert gt.shape == (vector_dim,)

    # Should produce identical results
    assert torch.allclose(
        pred,
        gt,
        rtol=1e-5,
        atol=1e-7,
    )

@pytest.mark.skip(reason="Not Updated")
def test_value_encoding_batched_x():
    """
    test that the value encoding module works when given
    a batched x value
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    batch_size: int = 3
    value_dim: int = 2
    vector_dim: int = 1280
    value = torch.from_numpy(np.array([
        [2, 3],
        [1, 5],
        [2, 6]
    ]))
    length_scale: float = 1.5

    b = HRRBackend(
        vector_dim=vector_dim,
        value_dim=value_dim,
        length_scale=length_scale
    )

    assert b.value_basis_vectors.shape == (value_dim, vector_dim)

    # Calculate baseline results
    base = b.value_basis_vectors
    base = torch.fft.fft(base, dim=-1)

    gt = torch.zeros((batch_size, vector_dim))

    for bs in range(batch_size):
        components = torch.zeros((value_dim, vector_dim), dtype=torch.complex64)

        for ed in range(value_dim):
            v = base[ed]
            v = v ** (value[bs][ed] / length_scale)
            components[ed] = v

        components = torch.prod(components, axis=0)
        gt[bs] = torch.fft.ifft(components, dim=-1).real

    pred, _ = b.value_encoding(value)

    assert gt.shape == (batch_size, vector_dim)
    assert pred.shape == (batch_size, vector_dim)

    # Should produce identical results
    assert torch.allclose(
        gt,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )