import pytest

def test_rm_no_backend():
    """
    Test that the regression module doesn't assume a default backend.
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16

    b = HRRBackend(vector_dim=D)
    c = torch.rand((B, D))
    v = torch.rand((B, 1))

    with pytest.raises(TypeError):
        RegressionModule(
            codebook=c,
            values=v
        )

def test_rm_true_backend():
    """
    Test that the regression module initializes with a value backend
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16

    b = HRRBackend(vector_dim=D)
    c = torch.rand((B, D))
    v = torch.rand((B, 1))

    _ = RegressionModule(b, c, v)

def test_rm_invalid_backend():
    """
    Test that the rm module throws and error when an invalid
    backend is passed
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16

    b = HRRBackend(vector_dim=D)
    c = torch.rand((B, D))
    v = torch.rand((B, 1))

    with pytest.raises(TypeError):
        _ = RegressionModule(5, c, v)