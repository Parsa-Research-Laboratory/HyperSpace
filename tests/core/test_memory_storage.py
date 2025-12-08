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

def test_msm_invalid_v_vector_shape():
    """
    Test that the memory storage module throws an error when the
    value vectors aren't single or batched
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    p_vec = torch.zeros((D, D))
    v_vec_invalid = torch.zeros((D, D, D))
    prev_memory = torch.zeros((D))

    with pytest.raises(ValueError):
        msm(p_vec, v_vec_invalid, prev_memory)

def test_msm_invalid_prev_memory_shape():
    """
    Test that the memory storage module throws an error when the
    previous memory vector isn't the correct shape
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    p_vec = torch.zeros((D, D))
    v_vec = torch.zeros((D, D))
    prev_memory_invalid = torch.zeros((D, D))

    with pytest.raises(ValueError):
        msm(p_vec, v_vec, prev_memory_invalid)

def test_msm_p_h_shape_mismatch():
    """
    Test that the memory storage module throws an error when the
    number of position and value don't match
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    p_vec = torch.zeros((D))
    v_vec = torch.zeros((D, D))
    prev_memory = torch.zeros((D))

    with pytest.raises(ValueError):
        msm(p_vec, v_vec, prev_memory)

def test_msm_p_prev_mismatch():
    """
    test that the memory storage module throws an error when the
    dimensionality of the position vectors and previous memory
    do not match
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    p_vec = torch.zeros((D, D))
    v_vec = torch.zeros((D, D))
    prev_memory_large = torch.zeros((D + 1))
    prev_memory_small = torch.zeros((D - 1))

    with pytest.raises(ValueError):
        msm(p_vec, v_vec, prev_memory_large)

    with pytest.raises(ValueError):
        msm(p_vec, v_vec, prev_memory_small)

def test_msm_initialize_memory():
    """
    test the ability of the memory storage module to initialize an
    empty memory to store future information
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    memory = msm.initialize_memory()

    assert isinstance(memory, torch.Tensor)
    assert memory.shape == (D,)

def test_msm_single_storage_no_prev():
    """
    test the memory storage module when storing a single point
    and value into the memory
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    v_vector = b.create_random_vector()
    p_vector = b.create_random_vector()
    prev_memory = msm.initialize_memory()

    pred, _ = msm(
        p_vectors=p_vector,
        v_vectors=v_vector,
        prev_memory=prev_memory
    )
    gt, _ = b.bind(v_vector, p_vector)

    assert torch.allclose(
        gt,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

def test_msm_single_storage_with_prev():
    """
    test the memory storage module when storing a single point
    and value into the previous memory
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    v_vector = b.create_random_vector()
    p_vector = b.create_random_vector()
    prev_memory = b.create_random_vector()

    pred, _ = msm(
        p_vectors=p_vector,
        v_vectors=v_vector,
        prev_memory=prev_memory
    )
    gt, _ = b.bind(v_vector, p_vector)
    gt, _ = b.bundle(gt, prev_memory)

    assert torch.allclose(
        gt,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

def test_msm_batched_storage_no_prev():
    """
    test the memory storage module when storing a batch of points
    and values into the memory
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule

    B: int = 2
    D: int = 128

    b = HRRBackend(vector_dim=D)
    msm = MemoryStorageModule(b)

    v_vectors = [b.create_random_vector() for _ in range(B)]
    p_vectors = [b.create_random_vector() for _ in range(B)]
    prev_memory = msm.initialize_memory()

    v_stack = torch.stack(v_vectors, dim=0)
    p_stack = torch.stack(p_vectors, dim=0)

    pred, _ = msm(
        p_vectors=p_stack,
        v_vectors=v_stack,
        prev_memory=prev_memory
    )
    gt, _ = b.bind(v_stack, p_stack)
    gt, _ = b.bundle(gt)

    assert torch.allclose(
        gt,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

@pytest.mark.skip(reason="Not Implemented")
def test_msm_batched_storage_with_prev():
    """
    test the memory storage module when storing a batch of points
    and values into the memory
    """
    pass