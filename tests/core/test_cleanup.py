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
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.cleanup import CleanupModule

    B: int = 64
    D: int = 1024

    codebook = torch.rand((B, D))

    b = HRRBackend(vector_dim=D)
    _ = CleanupModule(b, codebook=codebook)

def test_cm_invalid_backend():
    """
    Test that the cm module throws and error when an invalid
    backend is passed
    """
    import torch
    from hyperspace.core.cleanup import CleanupModule

    B: int = 64
    D: int = 1024

    codebook = torch.rand((B, D))

    with pytest.raises(TypeError):
        _ = CleanupModule(5, codebook=codebook)

def test_cm_missing_values_and_codebook():
    """
    Test that the module throws an error when
    missing both values and codebook arguments
    """
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024

    b = HRRBackend(vector_dim=D)

    with pytest.raises(ValueError):
        CleanupModule(b)

def test_cm_both_values_and_codebook():
    """
    Test that the module throws an error when
    receiving both values and codebook arguments
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    values = torch.rand((B, V))
    codebook = torch.rand((B, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, values, codebook)

def test_cm_constructor_values_type():
    """
    Test that the module throws an error when
    values isn't a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    values = np.random.random((B, V))
    codebook = torch.rand((B, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(TypeError):
        CleanupModule(b, values, codebook)

@pytest.mark.skip(reason="NI")
def test_cm_constructor_values_shape():
    """
    Test that the module throws an error when
    values isn't a 2D Tensor
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_constructor_values_dim():
    """
    Test that the module throws an error when
    values doesn't match value_dim
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_constructor_values_generated_codebook():
    """
    Test the fidelity of the generated codebook
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_constructor_codebook_type():
    """
    Test that the module throws an error when
    values isn't a Tensor
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_constructor_codebook_shape():
    """
    Test that the module throws an error when
    codebook isn't a 2D Tensor
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_constructor_codebook_dim():
    """
    Test that the module throws an error when
    codebook doesn't match vector_dim
    """
    pass