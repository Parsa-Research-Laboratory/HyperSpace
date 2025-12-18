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

def test_rm_codebook_invalid_type():
    """
    Test that the regression module throws an error when
    the codebook isn't a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16

    b = HRRBackend(vector_dim=D)
    c = np.random.random((B, D))
    v = torch.rand((B, 1))

    with pytest.raises(TypeError):
        _ = RegressionModule(b, c, v)

def test_rm_codebook_invalid_shape():
    """
    Test that the regression module throws an error when
    the codebook isn't the correct shape
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16

    b = HRRBackend(vector_dim=D)
    c_small = torch.rand((B))
    c_large = torch.rand((B, D, D))
    v = torch.rand((B, 1))

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c_small, v)

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c_large, v)

def test_rm_codebook_invalid_dim():
    """
    Test that the regression module throws an error when
    the codebook doesn't have the correct dimensionality
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16

    b = HRRBackend(vector_dim=D)
    c_small = torch.rand((B, D - 1))
    c_large = torch.rand((B, D + 1))
    v = torch.rand((B, 1))

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c_small, v)

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c_large, v)

def test_rm_values_invalid_type():
    """
    Test that the regression module throws an error when
    the values isn't a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16

    b = HRRBackend(vector_dim=D)
    c = torch.rand((B, D))
    v = np.random.random((B, 1))

    with pytest.raises(TypeError):
        _ = RegressionModule(b, c, v)

def test_rm_values_invalid_shape():
    """
    Test that the regression module throws an error when
    the values isn't the correct shape
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16

    b = HRRBackend(vector_dim=D)
    c = torch.rand((B, D))
    v_small = torch.rand((B,))
    v_large = torch.rand((B, D, D))

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c, v_small)

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c, v_large)

def test_rm_values_invalid_dim():
    """
    Test that the regression module throws an error when
    the values doesn't have the correct dimensionality
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16
    V: int = 3

    b = HRRBackend(vector_dim=D, value_dim=V)
    c = torch.rand((B, D))
    v_small = torch.rand((B, V - 1))
    v_large = torch.rand((B, V + 1))

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c, v_small)

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c, v_large)

def test_rm_method_invalid_type():
    """
    Test that the regression module throws an error when
    the method doesn't have the correct type
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16
    V: int = 3

    b = HRRBackend(vector_dim=D, value_dim=V)
    c = torch.rand((B, D))
    v = torch.rand((B, V))

    with pytest.raises(TypeError):
        rm = RegressionModule(b, c, v, method=int(5))

def test_rm_method_invalid_value():
    """
    Test that the regression module throws an error when
    the method doesn't match the list of valid methods
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.regression import RegressionModule

    D: int = 1024
    B: int = 16
    V: int = 3

    b = HRRBackend(vector_dim=D, value_dim=V)
    c = torch.rand((B, D))
    v = torch.rand((B, V))

    with pytest.raises(ValueError):
        rm = RegressionModule(b, c, v, method="SomeRandomMethod")

def test_rm_call_missing_v():
    """
    Test that calling the regression module without a vector
    throws an error
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import RegressionModule
    
    D: int = 1024
    B: int = 16
    V: int = 3

    b = HRRBackend(vector_dim=D, value_dim=V)
    c = torch.rand((B, D))
    v = torch.rand((B, V))

    rm = RegressionModule(b, c, v)

    inp = None

    with pytest.raises(TypeError):
        rm()

def test_rm_call_v_type():
    """
    test that calling the regression module with a vector that
    is Tensor throws an error
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import RegressionModule
    
    D: int = 1024
    B: int = 16
    V: int = 3

    b = HRRBackend(vector_dim=D, value_dim=V)
    c = torch.rand((B, D))
    v = torch.rand((B, V))

    rm = RegressionModule(b, c, v)

    inp = np.random.random((B, D))

    with pytest.raises(TypeError):
        rm(inp)

def test_rm_call_v_shape():
    """
    test that calling the regression module with a vector
    with an invalid shape throws an error
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import RegressionModule
    
    D: int = 1024
    B: int = 16
    V: int = 3

    b = HRRBackend(vector_dim=D, value_dim=V)
    c = torch.rand((B, D))
    v = torch.rand((B, V))

    rm = RegressionModule(b, c, v)

    inp = torch.rand((B, D, D))

    with pytest.raises(ValueError):
        rm(inp)

@pytest.mark.skip()
def test_rm_call_v_dim():
    """
    test that the call method checks the dimensionality
    of v
    """
    pass