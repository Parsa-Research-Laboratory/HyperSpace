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