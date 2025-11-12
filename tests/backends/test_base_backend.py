# -------------------------------------------------
# A series of unittests for the base_backend class
# -------------------------------------------------
import pytest
from typing import List

ENV_VECTOR_BUFFER_NAME: str = "env_basis_vectors"
VALUE_VECTOR_BUFFER_NAME: str = "value_basis_vectors"

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

def test_vector_type_init():
    """
    test initialization of different vector data types; should
    have an internal value that matches the desired type
    """
    from hyperspace.backends.base import BaseBackend
    import torch

    types: List[torch.dtype] = [torch.float32, torch.complex64]

    for t in types:
        b = BaseBackend(
            vector_dim=128,
            vector_dtype=t
        )
        assert b.vector_dtype == t

def test_exist_value_vector_buffer():
    """
    test if the register buffer for the value vectors exists
    """
    from hyperspace.backends.base import BaseBackend

    b = BaseBackend(256)
    
    assert VALUE_VECTOR_BUFFER_NAME in dict(b.named_buffers())

def test_exist_env_vector_buffer():
    """
    test if the register buffer for the env vectors exists
    """
    from hyperspace.backends.base import BaseBackend

    b = BaseBackend(256)
    
    assert ENV_VECTOR_BUFFER_NAME in dict(b.named_buffers())

def test_value_vector_dimensionality():
    """
    test the dimensionality of the value vector buffer
    """
    from hyperspace.backends.base import BaseBackend

    dims: List[int] = [32, 64, 128]

    for d in dims:
        b = BaseBackend(d)

        buffer = b.get_buffer(VALUE_VECTOR_BUFFER_NAME)

        assert len(buffer.shape) == 2
        assert buffer.shape[0] == 0
        assert buffer.shape[1] == d

def test_env_vector_dimensionality():
    """
    test the dimensionality of the env vector buffer
    """
    from hyperspace.backends.base import BaseBackend

    dims: List[int] = [32, 64, 128]

    for d in dims:
        b = BaseBackend(d)

        buffer = b.get_buffer(ENV_VECTOR_BUFFER_NAME)

        assert len(buffer.shape) == 2
        assert buffer.shape[0] == 0
        assert buffer.shape[1] == d

def test_value_vector_dtype():
    """
    test the data type of the value vector buffer
    """
    from hyperspace.backends.base import BaseBackend
    import torch

    types: List[torch.dtype] = [torch.float32, torch.complex64]

    for t in types:
        b = BaseBackend(
            vector_dim=128,
            vector_dtype=t
        )

        buffer = b.get_buffer(VALUE_VECTOR_BUFFER_NAME)

        assert buffer.dtype == t

def test_env_vector_dtype():
    """
    test the data type of the value vector buffer
    """
    from hyperspace.backends.base import BaseBackend
    import torch

    types: List[torch.dtype] = [torch.float32, torch.complex64]

    for t in types:
        b = BaseBackend(
            vector_dim=128,
            vector_dtype=t
        )

        buffer = b.get_buffer(ENV_VECTOR_BUFFER_NAME)

        assert buffer.dtype == t

def test_value_vector_device():
    """
    test the device of the value vector buffer
    """
    from hyperspace.backends.base import BaseBackend
    import torch

    d = "cpu"
    gtd = torch.device(d)
    b = BaseBackend(256, device=d)
    buffer = b.get_buffer(VALUE_VECTOR_BUFFER_NAME)
    assert buffer.device == gtd

def test_env_vector_device():
    """
    test the device of the env vector buffer
    """
    from hyperspace.backends.base import BaseBackend
    import torch

    d = "cpu"
    gtd = torch.device(d)
    b = BaseBackend(256, device=d)
    buffer = b.get_buffer(ENV_VECTOR_BUFFER_NAME)
    assert buffer.device == gtd

def test_not_implemented_create_random_vector():
    """
    ensure the `create_random_vector` method is not implememted
    """
    from hyperspace.backends.base import BaseBackend

    b = BaseBackend(128)

    with pytest.raises(NotImplementedError):
        b.create_random_vector()

def test_not_implemented_continuous_encoding():
    """
    ensure the `continuous_encoding` method is not implemented
    """
    from hyperspace.backends.base import BaseBackend
    import torch

    b = BaseBackend(128)
    dt1: torch.Tensor = torch.zeros(1)
    dt2: torch.Tensor = torch.zeros(1)

    with pytest.raises(NotImplementedError):
        b.continuous_encoding(dt1, dt2)
    
def test_not_implemented_bind():
    """
    ensure the `bind` method is not implemented
    """
    from hyperspace.backends.base import BaseBackend
    import torch

    b = BaseBackend(128)
    dt1: torch.Tensor = torch.zeros(1)
    dt2: torch.Tensor = torch.zeros(1)

    with pytest.raises(NotImplementedError):
        b.bind(dt1, dt2)