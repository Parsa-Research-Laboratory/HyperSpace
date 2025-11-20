import pytest
from typing import List

def test_base_create_single_vector_base_arguments():
    """
    test the ability to generate random HRR vectors
    """
    from torch import Generator, Tensor
    from hyperspace.backends.hrr import _base_create_single_vector

    vd: int = 128
    v = _base_create_single_vector(
        vector_dim=vd,
        gen=Generator(device="cpu")
    )

    assert isinstance(v, Tensor)
    assert len(v.shape) == 1
    assert v.shape[0] == vd

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
