import pytest
from typing import List

def test_base_create_single_vector_base_arguments():
    """
    test the ability to generate random HRR vectors
    """
    import torch
    from torch import Generator, Tensor
    from hyperspace.backends.hrr import _base_create_single_vector

    vd: int = 128
    gen = Generator(device="cpu")
    v1 = _base_create_single_vector(
        vector_dim=vd,
        gen=gen
    )

    v2 = _base_create_single_vector(
        vector_dim=vd,
        gen=gen
    )

    dot = torch.dot(v1, v2)
    assert dot.abs() < 0.12, f"Vectors not orthogonal; dot={dot.item()}" 
    assert isinstance(v1, Tensor)
    assert len(v1.shape) == 1
    assert v1.shape[0] == vd

def test_base_create_single_vector_non_int_vector_dim():
    """
    test the ability to generate random HRR vectors
    """
    from torch import Generator
    from hyperspace.backends.hrr import _base_create_single_vector

    vd: float = 1.0

    with pytest.raises(TypeError):
        _ = _base_create_single_vector(
            vector_dim=vd,
            gen=Generator(device="cpu")
        )

def test_base_create_single_vector_negative_vector_dim():
    """
    test that the create random vector function catches
    negative dimensionalities
    """
    from torch import Generator
    from hyperspace.backends.hrr import _base_create_single_vector

    vd: int = -1

    with pytest.raises(ValueError):
        _ = _base_create_single_vector(
            vector_dim=vd,
            gen=Generator(device="cpu")
        )

def test_base_create_single_vector_invalid_generator():
    """
    test that the create random vector functon catches objects
    that are not true torch generators
    """
    from hyperspace.backends.hrr import _base_create_single_vector

    with pytest.raises(TypeError):
        _ = _base_create_single_vector(
            vector_dim=20,
            gen=int(5)
        )

def test_base_create_single_vector_invalid_eps_type():
    """
    test that the create random vector function catches
    when the eps argument is the wrong type
    """
    from torch import Generator
    from hyperspace.backends.hrr import _base_create_single_vector

    with pytest.raises(TypeError):
        _ = _base_create_single_vector(
            vector_dim=128,
            gen=Generator(),
            eps=int(5)
        )

def test_base_create_single_vector_negative_eps():
    """
    test that the create_single_vector function catches
    when eps is negative
    """
    from torch import Generator
    from hyperspace.backends.hrr import _base_create_single_vector

    with pytest.raises(ValueError):
        _ = _base_create_single_vector(
            vector_dim=128,
            gen=Generator(),
            eps=0.0
        )

def test_base_single_bind():
    """
    test the _base_single_bind function
    """
    import numpy as np
    from torch import Generator
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_single_bind
    )

    vd: int = 256
    gen = Generator()

    v1 = _base_create_single_vector(vd, gen)
    v2 = _base_create_single_vector(vd, gen)

    v_bind = _base_single_bind(v1, v2)

    v1_numpy = v1.numpy()
    v2_numpy = v2.numpy()

    v1_numpy_fft = np.fft.fft(v1_numpy)
    v2_numpy_fft = np.fft.fft(v2_numpy)
    v_out_numpy_fft = v1_numpy_fft * v2_numpy_fft
    v_out_numpy = np.fft.ifft(v_out_numpy_fft)

    assert np.allclose(v_out_numpy, v_bind.numpy(), rtol=1e-5, atol=1e-7)

def test_base_batch_bind_batch():
    """
    Test the batched bind function against NumPy FFT implementation.
    """
    import numpy as np
    import torch
    from torch import Generator

    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_batch_bind
    )

    B: int = 8     # batch size
    D: int = 256   # vector dimensionality

    gen = Generator().manual_seed(0)

    # Build batched tensors: (B, D)
    v1_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v2_list = [_base_create_single_vector(D, gen) for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)  # (B, D)
    v2 = torch.stack(v2_list, dim=0)  # (B, D)

    v_bind_batch = _base_batch_bind(v1, v2)  # (B, D)

    # NumPy reference
    v1_numpy = v1.numpy()  # (B, D)
    v2_numpy = v2.numpy()  # (B, D)

    v1_numpy_fft = np.fft.fft(v1_numpy, axis=-1)
    v2_numpy_fft = np.fft.fft(v2_numpy, axis=-1)
    v_out_numpy_fft = v1_numpy_fft * v2_numpy_fft
    v_out_numpy = np.fft.ifft(v_out_numpy_fft, axis=-1).real  # (B, D)

    assert np.allclose(
        v_out_numpy,
        v_bind_batch.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

# def test_base_single_fpe():
#     """
#     Test the functionality of the base fractional
#     binding implementation
#     """
#     raise NotImplementedError

# def test_base_single_bundle():
#     """
#     Test the functionality of the base fractional
#     binding implementation
#     """
#     raise NotImplementedError

# def test_base_single_bind():
#     """
#     Test the functionality of the base fractional
#     binding implementation
#     """
#     raise NotImplementedError
