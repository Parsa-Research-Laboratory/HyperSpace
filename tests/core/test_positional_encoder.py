import pytest

def test_pe_no_backend():
    """
    Test that the positional encoder module doesn't assume a default backend.
    """
    from hyperspace.core.positional_encoder import PositionalEncoderModule

    with pytest.raises(TypeError):
        PositionalEncoderModule()

def test_pe_true_backend():
    """
    Test that the pe module initializes with a value backend
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_encoder import PositionalEncoderModule

    b = HRRBackend(vector_dim=128)
    _ = PositionalEncoderModule(b)

def test_pe_invalid_backend():
    """
    Test that the pe module throws and error when an invalid
    backend is passed
    """
    from hyperspace.core.positional_encoder import PositionalEncoderModule

    with pytest.raises(TypeError):
        _ = PositionalEncoderModule(5)

def test_positional_encoding_module_invalid_x_type():
    """
    Test that the positional_encoding_module throws an error
    when x isn't a Tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_encoder import PositionalEncoderModule

    B: int = 16
    E: int = 3

    b = HRRBackend(vector_dim=128, env_dim=E)
    pem = PositionalEncoderModule(b)

    x_single = np.zeros((E))
    x_batch = np.zeros((B, E))

    with pytest.raises(TypeError):
        pem(x_single)

    with pytest.raises(TypeError):
        pem(x_batch)

def test_positional_encoding_module_invalid_x_dim():
    """
    Test that the positional encoding module throws an error when
    x isn't single or batched
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_encoder import PositionalEncoderModule

    b = HRRBackend(vector_dim=128)
    pem = PositionalEncoderModule(b)

    x_invalid = torch.zeros((5, 5, 5))
    
    with pytest.raises(ValueError):
        pem(x_invalid)