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

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(TypeError):
        CleanupModule(b, values)

def test_cm_constructor_values_shape():
    """
    Test that the module throws an error when
    values isn't a 2D Tensor
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    values_small = torch.rand((B))
    values_large = torch.rand((B, V, V))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, values_small)

    with pytest.raises(ValueError):
        CleanupModule(b, values_large)

def test_cm_constructor_values_dim():
    """
    Test that the module throws an error when
    values doesn't match value_dim
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    values_small = torch.rand((B, V - 1))
    values_large = torch.rand((B, V + 1))
    codebook = torch.rand((B, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, values_small, codebook)

    with pytest.raises(ValueError):
        CleanupModule(b, values_large, codebook)

@pytest.mark.skip(reason="NI")
def test_cm_constructor_values_generated_codebook():
    """
    Test the fidelity of the generated codebook
    """
    pass

def test_cm_constructor_codebook_type():
    """
    Test that the module throws an error when
    values isn't a Tensor
    """
    import numpy as np
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    codebook = np.random.random((B, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(TypeError):
        CleanupModule(b, codebook)

def test_cm_constructor_codebook_shape():
    """
    Test that the module throws an error when
    codebook isn't a 2D Tensor
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    codebook_small = torch.rand((B))
    codebook_large = torch.rand((B, D, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, codebook_small)

    with pytest.raises(ValueError):
        CleanupModule(b, codebook_large)

def test_cm_constructor_codebook_dim():
    """
    Test that the module throws an error when
    codebook doesn't match vector_dim
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    codebook_small = torch.rand((B, D - 1))
    codebook_large = torch.rand((B, D + 1))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, codebook_small)

    with pytest.raises(ValueError):
        CleanupModule(b, codebook_large)

def test_cm_call_missing_v():
    """
    Test that calling the cleanup module without a vector
    throws an error
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    with pytest.raises(TypeError):
        cm()

def test_cm_call_v_type():
    """
    test that calling the cleanup module with a vector that
    is Tensor throws an error
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    v = np.random.random((B, D))

    with pytest.raises(TypeError):
        cm(v)

def test_cm_call_v_shape():
    """
    test that calling the cleanup module with a vector
    with an invalid shape throws an error
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    v = torch.rand((B, D, D))

    with pytest.raises(ValueError):
        cm(v)

def test_cm_call_v_dim():
    """
    test that the call method checks the dimensionality
    of v
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    v_single_small = torch.rand((D - 1,))
    v_single_large = torch.rand((D + 1,))
    v_batch_small = torch.rand((B, D - 1))
    v_batch_large = torch.rand((B, D + 1))

    with pytest.raises(ValueError):
        cm(v_single_small)

    with pytest.raises(ValueError):
        cm(v_single_large)

    with pytest.raises(ValueError):
        cm(v_batch_small)

    with pytest.raises(ValueError):
        cm(v_batch_large)

@pytest.mark.skip(reason="NI")
def test_cm_call_missing_method():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_non_string_method():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_invalid_method():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_single_value_predef_codebook_resonator():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_single_value_derived_codebook_resonator():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_multi_value_predef_codebook_hopfield():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_multi_value_derived_codebook_hopfield():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_single_value_predef_codebook():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_single_value_derived_codebook():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_multi_value_predef_codebook():
    """
    
    """
    pass

@pytest.mark.skip(reason="NI")
def test_cm_call_multi_value_derived_codebook():
    """
    
    """
    pass