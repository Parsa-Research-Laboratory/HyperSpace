import pytest

def test_pi_no_backend():
    """
    Test that the positional inversion module doesn't assume a default backend.
    """
    import torch
    from hyperspace.core.positional_inversion import PositionalInversionModule

    B: int = 64
    E: int = 3

    p = torch.rand((B, E))

    with pytest.raises(TypeError):
        PositionalInversionModule(positions=p)

def test_pi_true_backend():
    """
    Test that the pi module initializes with a value backend
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_inversion import PositionalInversionModule

    B: int = 64
    D: int = 128
    E: int = 3

    b = HRRBackend(vector_dim=D, env_dim=E)
    p = torch.rand((B, E))

    _ = PositionalInversionModule(b, p)

def test_pi_invalid_backend():
    """
    Test that the pe module throws and error when an invalid
    backend is passed
    """
    import torch
    from hyperspace.core.positional_inversion import PositionalInversionModule

    B: int = 64
    E: int = 3

    p = torch.rand((B, E))

    with pytest.raises(TypeError):
        _ = PositionalInversionModule(5, p)