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