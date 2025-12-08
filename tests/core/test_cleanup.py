import pytest

def test_cm_no_backend():
    """
    Test that the cleanup module doesn't assume a default backend.
    """
    from hyperspace.core.cleanup import CleanupModule

    with pytest.raises(TypeError):
        CleanupModule()

def test_cm_true_backend():
    """
    Test that the cleanup module initializes with a value backend
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.cleanup import CleanupModule

    b = HRRBackend(vector_dim=128)
    _ = CleanupModule(b)

def test_cm_invalid_backend():
    """
    Test that the cm module throws and error when an invalid
    backend is passed
    """
    from hyperspace.core.cleanup import CleanupModule

    with pytest.raises(TypeError):
        _ = CleanupModule(5)