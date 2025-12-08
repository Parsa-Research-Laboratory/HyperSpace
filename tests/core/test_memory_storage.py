import pytest

def test_msm_no_backend():
    """
    Test that the memory storage module doesn't assume a default backend.
    """
    from hyperspace.core.memory_storage import MemoryStorageModule

    with pytest.raises(TypeError):
        MemoryStorageModule()

def test_msm_true_backend():
    """
    Test that the memory storage module initializes with a value backend
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    b = HRRBackend(vector_dim=128)
    _ = MemoryStorageModule(b)

def test_msm_invalid_backend():
    """
    Test that the memory storage module throws and error when an invalid
    backend is passed
    """
    from hyperspace.core.memory_storage import MemoryStorageModule

    with pytest.raises(TypeError):
        _ = MemoryStorageModule(5)

def test_msm_invalid_p_vector_type():
    """
    Test that the memory storage module throws an error when the type
    of the positional vectors is invalid
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule
    
    b = HRRBackend(vector_dim=128)
    msm = MemoryStorageModule(b)

    p_vector = np.zeros(10)
    v_vector = torch.zeros(10)
    prev_memory = torch.zeros(128)

    with pytest.raises(TypeError):
        msm(
            p_vectors=p_vector,
            v_vectors=v_vector,
            prev_memory=prev_memory
        )

def test_msm_invalid_v_vector_type():
    """
    Test that the memory storage module throws an error when the type
    of the value vectors is incorrect
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule
    
    b = HRRBackend(vector_dim=128)
    msm = MemoryStorageModule(b)

    p_vector = torch.zeros(10)
    v_vector = np.zeros(10)
    prev_memory = torch.zeros(128)

    with pytest.raises(TypeError):
        msm(
            p_vectors=p_vector,
            v_vectors=v_vector,
            prev_memory=prev_memory
        )

def test_msm_invalid_prev_memory_type():
    """
    Test that the memory storage module throws an error when the type
    of the previous memory vector is incorrect
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule
    
    b = HRRBackend(vector_dim=128)
    msm = MemoryStorageModule(b)

    p_vector = torch.zeros(10)
    v_vector = torch.zeros(10)
    prev_memory = np.zeros(128)

    with pytest.raises(TypeError):
        msm(
            p_vectors=p_vector,
            v_vectors=v_vector,
            prev_memory=prev_memory
        )

def test_msm_invalid_p_vector_shape():
    """
    Test that the memory storage modules throws an error when the
    shape of the positional vectors isn't single or batched
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    p_vec_invalid = torch.zeros((D, D, D))
    v_vec = torch.zeros((D))
    prev_memory = torch.zeros((D))

    with pytest.raises(ValueError):
        msm(p_vec_invalid, v_vec, prev_memory)

@pytest.mark.skip(reason="Not Implemented")
def test_msm_invalid_v_vector_shape():
    """
    Test that the memory storage module throws an error when the
    value vectors aren't single or batched
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_msm_invalid_prev_memory_shape():
    """
    Test that the memory storage module throws an error when the
    previous memory vector isn't the correct shape
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_msm_p_h_shape_mismatch():
    """
    Test that the memory storage module throws an error when the
    number of position and value don't match
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_msm_p_h_dim_mismatch():
    """
    test that the memory storage module throws an error when the
    dimensionality of the position and value vectors doesn't match
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_msm_p_prev_mismatch():
    """
    test that the memory storage module throws an error when the
    dimensionality of the position vectors and previous memory
    do not match
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_msm_single_storage():
    """
    test the memory storage module when storing a single point
    and value into the memory
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_msm_batched_storage():
    """
    test the memory storage module when storing a batch of points
    and values into the memory
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_msm_initialize_memory():
    """
    test the ability of the memory storage module to initialize an
    empty memory to store future information
    """
    pass
    