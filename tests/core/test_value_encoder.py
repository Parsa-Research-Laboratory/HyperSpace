import pytest

def test_ve_no_backend():
    """
    Test that the value encoder module doesn't assume a default backend.
    """
    from hyperspace.core.value_encoder import ValueEncoderModule

    with pytest.raises(TypeError):
        ValueEncoderModule()

def test_ve_true_backend():
    """
    Test that the ve module initializes with a value backend
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.value_encoder import ValueEncoderModule

    b = HRRBackend(vector_dim=128)
    _ = ValueEncoderModule(b)

def test_ve_invalid_backend():
    """
    Test that the pe module throws and error when an invalid
    backend is passed
    """
    from hyperspace.core.value_encoder import ValueEncoderModule

    with pytest.raises(TypeError):
        _ = ValueEncoderModule(5)