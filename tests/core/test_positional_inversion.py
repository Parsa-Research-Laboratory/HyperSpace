import pytest

def test_pi_no_backend():
    """
    Test that the positional inversion module doesn't assume a default backend.
    """
    from hyperspace.core.positional_inversion import PositionalInversionModule

    with pytest.raises(TypeError):
        PositionalInversionModule()

def test_pi_true_backend():
    """
    Test that the pi module initializes with a value backend
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_inversion import PositionalInversionModule

    b = HRRBackend(vector_dim=128)
    _ = PositionalInversionModule(b)

def test_pi_invalid_backend():
    """
    Test that the pe module throws and error when an invalid
    backend is passed
    """
    from hyperspace.core.positional_inversion import PositionalInversionModule

    with pytest.raises(TypeError):
        _ = PositionalInversionModule(5)