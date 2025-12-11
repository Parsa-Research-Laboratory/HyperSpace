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

def test_pi_invalid_positions_type():
    """
    Test that the PI module throws an error when
    positions aren't the correct type
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_inversion import PositionalInversionModule

    B: int = 64
    D: int = 128
    E: int = 3

    b = HRRBackend(vector_dim=D, env_dim=E)
    p = np.random.random((B, E))

    with pytest.raises(TypeError):
        _ = PositionalInversionModule(b, p)

def test_pi_invalid_positions_shape():
    """
    Test that the PI module throws an error when
    positions don't have the same env dimensionality
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_inversion import PositionalInversionModule

    B: int = 64
    D: int = 128
    E: int = 3

    b = HRRBackend(vector_dim=D, env_dim=E)
    p_small_e = torch.rand((B, E - 1))
    p_big_e = torch.rand((B, E + 1))
    p_small_d = torch.rand((B))
    p_big_d = torch.rand((B, E, E))

    with pytest.raises(ValueError):
        PositionalInversionModule(b, p_small_e)

    with pytest.raises(ValueError):
        PositionalInversionModule(b, p_big_e)

    with pytest.raises(ValueError):
        PositionalInversionModule(b, p_small_d)

    with pytest.raises(ValueError):
        PositionalInversionModule(b, p_big_d)

def test_pi_valid_position_generation():
    """
    Test that the PI module correctly generates the position vectors
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_inversion import PositionalInversionModule

    B: int = 2
    D: int = 128
    E: int = 1

    b = HRRBackend(vector_dim=D, env_dim=E)
    p = torch.rand((B, E))

    pim = PositionalInversionModule(b, p)

    assert pim.inv_position_vectors.shape == (B, D)

    p_gt, _ = b.positional_encoding(p)
    p_gt, _ = b.invert(p_gt)

    assert torch.allclose(
        pim.inv_position_vectors,
        p_gt,
    )

def test_pi_call_memory_type():
    """
    Test that the call method only accepts tensor memories
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_inversion import PositionalInversionModule

    B: int = 2
    D: int = 128
    E: int = 1

    b = HRRBackend(vector_dim=D, env_dim=E)
    p = torch.rand((B, E))

    pim = PositionalInversionModule(b, p)

    m = np.random.random((B, D))

    with pytest.raises(TypeError):
        pim(m)

def test_pi_call_memory_shape():
    """
    That that memory shapes are correctly validated in
    the call method
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.positional_inversion import PositionalInversionModule

    B: int = 2
    D: int = 128
    E: int = 1

    b = HRRBackend(vector_dim=D, env_dim=E)
    p = torch.rand((B, E))

    pim = PositionalInversionModule(b, p)

    m_single_small_d = torch.rand((D - 1))
    m_single_big_d = torch.rand((D + 1))
    m_batch_small_d = torch.rand((B, D - 1))
    m_batch_big_d = torch.rand((B, D + 1))
    m_dim = torch.rand((B, D, D))

    with pytest.raises(ValueError):
        pim(m_single_small_d)

    with pytest.raises(ValueError):
        pim(m_single_big_d)

    with pytest.raises(ValueError):
        pim(m_batch_small_d)

    with pytest.raises(ValueError):
        pim(m_batch_big_d)

    with pytest.raises(ValueError):
        pim(m_dim)

def test_pi_call_single_pos_1D_orth():
    """
    Test the PI module to return a single noisy value vector
    in a 1D positional space
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core.memory_storage import MemoryStorageModule
    from hyperspace.core.positional_inversion import PositionalInversionModule

    D: int = 1024
    E: int = 1

    position = torch.tensor([[1.3]])
    axis_positions = torch.arange(0, 2, 0.1)
    axis_positions = axis_positions.unsqueeze(-1)

    assert position.shape == (1, 1)         # (batch_size, env_dim)
    assert axis_positions.shape == (20, 1)  # (batch_size, env_dim)

    b = HRRBackend(vector_dim=D, env_dim=E)
    msm = MemoryStorageModule(backend=b)
    pim = PositionalInversionModule(
        backend=b,
        positions=axis_positions
    )

    # construct the hypervector memory
    memory = msm.initialize_memory()
    
    pv, _ = b.positional_encoding(position) # (batch_size, vector_dim)
    vv = b.create_random_vector()           # (vector_dim,)
    vv = vv.unsqueeze(0)                    # (batch_size, vector_dim)

    assert pv.shape == (1, D)
    assert vv.shape == (1, D)

    memory, _ = msm(
        p_vectors=pv,
        v_vectors=vv,
        prev_memory=memory
    )

    vv_prime, _ = pim(memory)

    assert vv_prime.shape == (20, D)

    sims = torch.einsum("bd,ad->b", vv_prime, vv)

    assert sims.shape == (20,)
    assert torch.argmax(sims) == 13

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_single_pos_2D_orth():
    """
    Test the PI module to return a single noisy value vector
    in a 2D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_single_pos_3D_orth():
    """
    Test the PI module to return a single noisy value vector
    in a 3D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_multi_pos_1D_orth():
    """
    Test the PI module to return a single noisy value vector
    in a 1D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_multi_pos_2D_orth():
    """
    Test the PI module to return a single noisy value vector
    in a 2D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_multi_pos_3D_orth():
    """
    Test the PI module to return a single noisy value vector
    in a 3D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_single_pos_1D_value_2D():
    """
    Test the PI module to return a single noisy value vector
    in a 1D positional space
    """


    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_single_pos_2D_value_2D():
    """
    Test the PI module to return a single noisy value vector
    in a 2D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_multi_pos_1D_value_2D():
    """
    Test the PI module to return a single noisy value vector
    in a 1D positional space
    """
    pass

@pytest.mark.skip(reason="Not Implemented")
def test_pi_call_multi_pos_2D_value_2D():
    """
    Test the PI module to return a single noisy value vector
    in a 2D positional space
    """
    pass