import pytest

def test_pe_no_backend():
    """
    Test that the positional encoder module doesn't assume a default backend.
    """
    from hyperspace.core.positional_encoder import PositionalEncoderModule

    with pytest.raises(TypeError):
        PositionalEncoderModule()

def test_pe_true_backend():
    """
    Test that the pe module initializes with a value backend
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_encoder import PositionalEncoderModule

    b = HRRBackend(vector_dim=128)
    _ = PositionalEncoderModule(b)

def test_pe_invalid_backend():
    """
    Test that the pe module throws and error when an invalid
    backend is passed
    """
    from hyperspace.core.positional_encoder import PositionalEncoderModule

    with pytest.raises(TypeError):
        _ = PositionalEncoderModule(5)
