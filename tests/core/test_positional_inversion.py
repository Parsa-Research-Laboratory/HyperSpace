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

def test_pi_no_positions():
    """
    Test that the PI module throws an error when
    positions aren't provided
    """
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_inversion import PositionalInversionModule

    D: int = 128
    E: int = 3

    b = HRRBackend(vector_dim=D, env_dim=E)

    with pytest.raises(TypeError):
        _ = PositionalInversionModule(b)

@pytest.mark.skip(reason="Not Implemented")
def test_pi_invalid_positions_type():
    """
    Test that the PI module throws an error when
    positions aren't the correct type
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_invalid_positions_shape():
    """
    Test that the PI module throws an error when
    positions don't have the same env dimensionality
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_valid_position_generation():
    """
    Test that the PI module correctly generates the position vectors
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_position_vectors_type():
    """
    Test the type of the generated position vectors is correct
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_position_vectors_shape():
    """
    Test that the shape of the generated position vectors is correct
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_memory_type():
    """
    Test that the call method only accepts tensor memories
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_memory_shape():
    """
    That that memory shapes are correctly validated in
    the call method
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_single_pos_1D():
    """
    Test the PI module to return a single noisy value vector
    in a 1D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_single_pos_2D():
    """
    Test the PI module to return a single noisy value vector
    in a 2D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_single_pos_3D():
    """
    Test the PI module to return a single noisy value vector
    in a 3D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_multi_pos_1D():
    """
    Test the PI module to return a single noisy value vector
    in a 1D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_multi_pos_2D():
    """
    Test the PI module to return a single noisy value vector
    in a 2D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_multi_pos_3D():
    """
    Test the PI module to return a single noisy value vector
    in a 3D positional space
    """
    pass