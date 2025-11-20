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
    assert dot.abs() < 0.08, f"Vectors not orthogonal; dot={dot.item()}" 
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
