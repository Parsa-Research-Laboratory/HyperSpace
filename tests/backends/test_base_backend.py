# -------------------------------------------------
# A series of unittests for the base_backend class
# -------------------------------------------------
import pytest
from typing import List

def test_no_arg_init():
    """
    test initialization with no argument; should
    raise an error
    """
    from hyperspace.backends.base import BaseBackend

    with pytest.raises(TypeError):
        _ = BaseBackend()

def test_vector_dim_init():
    """
    test initialization with multiple vector dimensionalities;
    should see internal variable according to argument
    """
    from hyperspace.backends.base import BaseBackend

    vector_dims: List[int] = [32, 64, 128]

    for d in vector_dims:
        b = BaseBackend(d)
        assert b.vector_dim == d

def test_device_init():
    """
    test initialization with different torch devices;
    the internal device should match the ground truth
    """
    from hyperspace.backends.base import BaseBackend
    import torch

    d: str = "cpu"
    gtd = torch.device(d)

    b = BaseBackend(vector_dim=256, device="cpu")

    assert b.device == gtd