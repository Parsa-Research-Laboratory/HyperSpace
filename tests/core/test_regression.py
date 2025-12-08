import pytest

def test_rm_no_backend():
    """
    Test that the regression module doesn't assume a default backend.
    """
    from hyperspace.core.regression import RegressionModule

    with pytest.raises(TypeError):
        RegressionModule()

def test_rm_true_backend():
    """
    Test that the regression module initializes with a value backend
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    b = HRRBackend(vector_dim=128)
    _ = RegressionModule(b)

def test_rm_invalid_backend():
    """
    Test that the rm module throws and error when an invalid
    backend is passed
    """
    from hyperspace.core.regression import RegressionModule

    with pytest.raises(TypeError):
        _ = RegressionModule(5)