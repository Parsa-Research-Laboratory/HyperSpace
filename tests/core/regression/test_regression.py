# tests/core/regression/test_regression_module.py

import pytest
import torch
import torch.nn as nn

from hyperspace.backends.hrr import HRRBackend
from hyperspace.core.regression.regression_module import (
    RegressionModule,
    _module_device,
    _create_dummy_input,
    _validate_model,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dims():
    return dict(D=1024, B=16, V=3)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture
def backend(dims, device):
    return HRRBackend(vector_dim=dims["D"], value_dim=dims["V"])


@pytest.fixture
def codebook(dims, device):
    return torch.rand((dims["B"], dims["D"]), device=device)


@pytest.fixture
def values(dims, device):
    return torch.rand((dims["B"], dims["V"]), device=device)


@pytest.fixture
def rm(backend, codebook, values):
    return RegressionModule(backend=backend, codebook=codebook, values=values)


# -----------------------------------------------------------------------------
# Utility function tests
# -----------------------------------------------------------------------------

def test_util_module_device_parameter_module(device):
    m = nn.Linear(8, 4).to(device)
    assert _module_device(m) == device


def test_util_module_device_buffer_only(device):
    class BufferOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buf", torch.ones(3))

    m = BufferOnly().to(device)
    assert _module_device(m) == device


def test_util_module_device_no_params_no_buffers_defaults_cpu():
    class Empty(nn.Module):
        def __init__(self):
            super().__init__()

    m = Empty()
    assert _module_device(m) == torch.device("cpu")


@pytest.mark.parametrize(
    "batched,expected_shape",
    [
        # NOTE: This reflects the CURRENT implementation in regression_module.py:
        # batched=True -> (16, feature_dim)
        # batched=False  -> (feature_dim,)
        (False, (1024,)),
        (True, (16, 1024)),
    ],
)
def test_util_create_dummy_input_shapes_and_device(device, batched, expected_shape):
    x = _create_dummy_input(feature_dim=1024, device=device, batched=batched)
    assert x.shape == expected_shape
    assert x.device == device
    assert x.dtype == torch.float32


def test_util_validate_model_accepts_linear(device):
    feature_dim = 32
    value_dim = 5
    m = nn.Linear(feature_dim, value_dim).to(device)

    # Should not raise
    _validate_model(model=m, feature_dim=feature_dim, value_dim=value_dim)


def test_util_validate_model_raises_on_wrong_output_dim(device):
    feature_dim = 32
    value_dim = 5

    # Wrong output dimension (value_dim + 1)
    m = nn.Linear(feature_dim, value_dim + 1).to(device)

    with pytest.raises(ValueError):
        _validate_model(model=m, feature_dim=feature_dim, value_dim=value_dim)


# -----------------------------------------------------------------------------
# RegressionModule constructor validation tests
# -----------------------------------------------------------------------------

def test_rm_no_backend(dims, device):
    """
    RegressionModule should not assume a default backend.
    """
    c = torch.rand((dims["B"], dims["D"]), device=device)
    v = torch.rand((dims["B"], 1), device=device)

    with pytest.raises(TypeError):
        RegressionModule(codebook=c, values=v)  # missing backend


def test_rm_true_backend(backend, codebook, values):
    """
    RegressionModule should initialize with a valid backend.
    """
    _ = RegressionModule(backend, codebook, values)


def test_rm_invalid_backend(codebook, values):
    """
    RegressionModule should raise if backend is not a BaseBackend.
    """
    with pytest.raises(TypeError):
        _ = RegressionModule(5, codebook, values)


def test_rm_codebook_invalid_type(backend, dims, device):
    """
    RegressionModule should raise if codebook is not a Tensor.
    """
    import numpy as np

    c = np.random.random((dims["B"], dims["D"]))
    v = torch.rand((dims["B"], dims["V"]), device=device)

    with pytest.raises(TypeError):
        _ = RegressionModule(backend, c, v)


def test_rm_codebook_invalid_shape(backend, dims, device):
    """
    RegressionModule should raise if codebook is not 2D.
    """
    c_small = torch.rand((dims["B"],), device=device)
    c_large = torch.rand((dims["B"], dims["D"], dims["D"]), device=device)
    v = torch.rand((dims["B"], dims["V"]), device=device)

    with pytest.raises(ValueError):
        _ = RegressionModule(backend, c_small, v)

    with pytest.raises(ValueError):
        _ = RegressionModule(backend, c_large, v)


def test_rm_codebook_invalid_dim(backend, dims, device):
    """
    RegressionModule should raise if codebook last-dim != backend.vector_dim.
    """
    c_small = torch.rand((dims["B"], dims["D"] - 1), device=device)
    c_large = torch.rand((dims["B"], dims["D"] + 1), device=device)
    v = torch.rand((dims["B"], dims["V"]), device=device)

    with pytest.raises(ValueError):
        _ = RegressionModule(backend, c_small, v)

    with pytest.raises(ValueError):
        _ = RegressionModule(backend, c_large, v)


def test_rm_values_invalid_type(backend, dims, device):
    """
    RegressionModule should raise if values is not a Tensor.
    """
    import numpy as np

    c = torch.rand((dims["B"], dims["D"]), device=device)
    v = np.random.random((dims["B"], dims["V"]))

    with pytest.raises(TypeError):
        _ = RegressionModule(backend, c, v)


def test_rm_values_invalid_shape(backend, dims, device):
    """
    RegressionModule should raise if values is not 2D.
    """
    c = torch.rand((dims["B"], dims["D"]), device=device)
    v_small = torch.rand((dims["B"],), device=device)
    v_large = torch.rand((dims["B"], dims["V"], dims["V"]), device=device)

    with pytest.raises(ValueError):
        _ = RegressionModule(backend, c, v_small)

    with pytest.raises(ValueError):
        _ = RegressionModule(backend, c, v_large)


def test_rm_values_invalid_dim(dims, device):
    """
    RegressionModule should raise if values last-dim != backend.value_dim.
    """
    D, B, V = dims["D"], dims["B"], dims["V"]
    b = HRRBackend(vector_dim=D, value_dim=V)

    c = torch.rand((B, D), device=device)
    v_small = torch.rand((B, V - 1), device=device)
    v_large = torch.rand((B, V + 1), device=device)

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c, v_small)

    with pytest.raises(ValueError):
        _ = RegressionModule(b, c, v_large)


def test_rm_method_invalid_type(backend, codebook, values):
    """
    RegressionModule should raise if method is not a string.
    """
    with pytest.raises(TypeError):
        _ = RegressionModule(backend, codebook, values, method=int(5))


def test_rm_method_invalid_value(backend, codebook, values):
    """
    RegressionModule should raise if method not in valid_methods.
    """
    with pytest.raises(ValueError):
        _ = RegressionModule(backend, codebook, values, method="SomeRandomMethod")


# -----------------------------------------------------------------------------
# RegressionModule call validation tests
# -----------------------------------------------------------------------------

def test_rm_call_missing_v(rm):
    """
    Calling RegressionModule without an argument should raise TypeError
    (python signature enforcement).
    """
    with pytest.raises(TypeError):
        rm()  # missing required positional argument


def test_rm_call_v_type(rm):
    """
    Calling RegressionModule with a non-Tensor should raise TypeError.
    """
    import numpy as np

    inp = np.random.random((16, 1024))
    with pytest.raises(TypeError):
        rm(inp)


def test_rm_call_v_shape_rejects_non_1d_or_2d(rm, dims, device):
    """
    Calling RegressionModule with ndim not in {1,2} should raise ValueError.
    """
    bad = torch.rand((dims["B"], dims["D"], 2), device=device)
    with pytest.raises(ValueError):
        rm(bad)


def test_rm_call_v_dim_rejects_mismatch(rm, dims, device):
    """
    Calling RegressionModule with last-dim != backend.vector_dim should raise ValueError.
    """
    D, B = dims["D"], dims["B"]

    inp_single_small = torch.rand((D - 1,), device=device)
    inp_single_large = torch.rand((D + 1,), device=device)
    inp_batch_small = torch.rand((B, D - 1), device=device)
    inp_batch_large = torch.rand((B, D + 1), device=device)

    with pytest.raises(ValueError):
        rm(inp_single_small)
    with pytest.raises(ValueError):
        rm(inp_single_large)
    with pytest.raises(ValueError):
        rm(inp_batch_small)
    with pytest.raises(ValueError):
        rm(inp_batch_large)


def test_rm_has_network_flags_default(rm):
    """
    RegressionModule should expose network_needed/network_ready flags in codebook mode.
    """
    assert hasattr(rm, "network_needed")
    assert hasattr(rm, "network_ready")
    assert rm.network_needed is False
    assert rm.network_ready is False


def test_rm_network_needed_with_neural(backend, codebook, values):
    """
    RegressionModule should set network_needed=True in neural mode.
    """
    rm_neural = RegressionModule(backend, codebook, values, method="neural")
    assert rm_neural.network_needed is True
    assert rm_neural.network_ready is False


def test_rm_call_with_no_loaded_network_raises(backend, codebook, values):
    """
    In neural mode, calling without a loaded network should raise AttributeError.
    """
    rm_neural = RegressionModule(backend, codebook, values, method="neural")
    assert rm_neural.network_needed is True
    assert rm_neural.network_ready is False

    with pytest.raises(AttributeError):
        rm_neural(codebook)  # any valid v triggers the "no network" guard


def test_rm_call_with_unneeded_loaded_network_raises(rm, codebook):
    """
    In non-neural mode, having network_ready=True should raise ValueError.
    """
    assert rm.network_needed is False
    assert rm.network_ready is False

    # Force the inconsistent state to hit the guard.
    rm.network_ready = True

    with pytest.raises(ValueError):
        rm(codebook)


# -----------------------------------------------------------------------------
# Neural network loading tests
# -----------------------------------------------------------------------------

def test_rm_load_neural_network_rejects_non_module(backend, codebook, values):
    rm_neural = RegressionModule(backend, codebook, values, method="neural")

    with pytest.raises(TypeError):
        rm_neural.load_neural_network(model=5)


def test_rm_load_neural_network_accepts_valid_model(device, backend, codebook, values, dims):
    rm_neural = RegressionModule(backend, codebook, values, method="neural")

    model = nn.Linear(dims["D"], dims["V"]).to(device)

    rm_neural.load_neural_network(model=model)

    assert rm_neural.network_ready is True
    assert hasattr(rm_neural, "model")
    assert rm_neural.model is model


def test_rm_load_neural_network_rejects_wrong_output_dim(device, backend, codebook, values, dims):
    rm_neural = RegressionModule(backend, codebook, values, method="neural")

    bad_model = nn.Linear(dims["D"], dims["V"] + 1).to(device)

    with pytest.raises(ValueError):
        rm_neural.load_neural_network(model=bad_model)


def test_rm_load_neural_network_sets_ready_even_if_method_not_neural(device, backend, codebook, values, dims):
    """
    NOTE: This reflects CURRENT behavior: load_neural_network does not check self.method.
    It will validate and set network_ready=True even in codebook mode.
    """
    rm_codebook = RegressionModule(backend, codebook, values, method="codebook")

    model = nn.Linear(dims["D"], dims["V"]).to(device)

    rm_codebook.load_neural_network(model=model)
    assert rm_codebook.network_ready is True
