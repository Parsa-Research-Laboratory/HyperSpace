import pytest
import torch
import torch.nn as nn

def test_base_import():
    """
    Verify that the DenseLinearModel can be imported successfully.

    This test ensures that the model module is discoverable
    and free of import-time errors caused by missing dependencies or
    circular imports.
    """
    from hyperspace.core.regression.models.dense_linear import DenseLinearModel

def test_passthrough_init_valid_feature_dim_value_positional():
    """
    Verify successful initialization of DenseLinearModel using positional arguments.

    This test ensures that valid feature_dim values provided via positional arguments
    are accepted and correctly stored on the model instance.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10

    m = DenseLinearModel(f, v)

    assert m.feature_dim == f

    f: int = 2048 * 3
    v: int = 10

    m = DenseLinearModel(f, v)

    assert m.feature_dim == f

def test_passthrough_init_valid_value_dim_value_positional():
    """
    Verify successful initialization of DenseLinearModel using positional arguments.

    This test ensures that valid value_dim values provided via positional arguments
    are accepted and correctly stored on the model instance.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10

    m = DenseLinearModel(f, v)

    assert m.value_dim == v

    f: int = 2048
    v: int = 10 * 100

    m = DenseLinearModel(f, v)

    assert m.value_dim == v

def test_passthrough_init_valid_feature_dim_value_keyword():
    """
    Verify successful initialization of DenseLinearModel using keyword arguments.

    This test ensures that valid feature_dim values provided via keyword arguments
    are accepted and correctly stored on the model instance.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10

    m = DenseLinearModel(
        feature_dim=f,
        value_dim=v
    )

    assert m.feature_dim == f

    f: int = 2048 * 3
    v: int = 10

    m = DenseLinearModel(
        feature_dim=f,
        value_dim=v
    )

    assert m.feature_dim == f

def test_passthrough_init_valid_value_dim_value_keyword():
    """
    Verify successful initialization of DenseLinearModel using keyword arguments.

    This test ensures that valid value_dim values provided via keyword arguments
    are accepted and correctly stored on the model instance.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10

    m = DenseLinearModel(
        feature_dim=f,
        value_dim=v
    )

    assert m.value_dim == v

    f: int = 2048
    v: int = 10 * 100

    m = DenseLinearModel(
        feature_dim=f,
        value_dim=v
    )

    assert m.value_dim == v

def test_init_extra_args():
    """
    Verify that DenseLinearModel supports positional initialization arguments.

    This test ensures that the model can be instantiated using positional
    arguments for feature and value dimensions without raising an error.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10

    m = DenseLinearModel(f, v)

    assert m.num_layers == 1

def test_init_num_layer_invalid_type():
    """
    Verify that DenseLinearModel enforces the type of num_layers.

    This test ensures that providing a non-integer value for the num_layers
    argument (e.g., a float) raises a TypeError, enforcing strict type
    requirements for layer configuration.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: float = 10.0

    with pytest.raises(TypeError):
        m = DenseLinearModel(f, v, num_layers=nl)

def test_init_num_layer_invalid_value():
    """
    Verify that DenseLinearModel enforces valid num_layers values.

    This test ensures that the num_layers argument must be a positive integer
    and that zero or negative values raise a ValueError, preventing invalid
    model configurations.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 0

    with pytest.raises(ValueError):
        m = DenseLinearModel(f, v, num_layers=nl)

    f: int = 2048
    v: int = 10
    nl: int = -1

    with pytest.raises(ValueError):
        m = DenseLinearModel(f, v, num_layers=nl)

def test_init_hidden_size_invalid_type():
    """
    Verify that DenseLinearModel enforces the type of hidden_size.

    This test ensures that providing a non-integer value for the hidden_size
    argument (e.g., a float) raises a TypeError, enforcing strict type
    requirements for hidden layer dimensionality.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 2
    hs: float = 100.0

    with pytest.raises(TypeError):
        m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs)

def test_init_hidden_size_invalid_int_value():
    """
    Verify that DenseLinearModel enforces valid hidden_size values.

    This test ensures that the hidden_size argument must be a positive integer
    and that zero or negative values raise a ValueError, preventing invalid
    hidden layer configurations.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 2
    hs: int = 0

    with pytest.raises(ValueError):
        m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs)

    f: int = 2048
    v: int = 10
    nl: int = 2
    hs: int = -1

    with pytest.raises(ValueError):
        m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs)

def test_init_hidden_size_invalid_list_value():
    """
    Verify that DenseLinearModel validates hidden_size when provided as a list.

    This test ensures that when hidden_size is specified as a sequence:
      - all elements must be integers (TypeError otherwise)
      - all elements must be strictly positive (ValueError otherwise)

    Invalid list-based hidden_size configurations should be rejected at
    initialization time.
    """
    from typing import List
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 2
    hs: List[float] = [10.0]

    with pytest.raises(TypeError):
        m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs)

    f: int = 2048
    v: int = 10
    nl: int = 3
    hs: List[int] = [0, 0]

    with pytest.raises(ValueError):
        m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs)

    f: int = 2048
    v: int = 10
    nl: int = 3
    hs: List[int] = [-1, -1]

    with pytest.raises(ValueError):
        m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs)

def test_init_hidden_act_invalid_type():
    """
    Verify DenseLinearModel requires hidden_act to be an nn.Module.

    This test ensures that providing a non-module activation specification
    (e.g., a Tensor) raises a TypeError.
    """
    import torch
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 2
    hs: int = 64
    ha = torch.zeros((10))

    with pytest.raises(TypeError):
        DenseLinearModel(f, v, nl, hs, ha)

def test_init_output_act_invalid_type():
    """
    Verify DenseLinearModel requires output_act to be an nn.Module.

    This test ensures that providing a non-module activation specification
    (e.g., a Tensor) raises a TypeError.
    """
    import torch
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 2
    hs: int = 64
    ha: torch.nn.Module = torch.nn.ReLU()
    oa = torch.zeros(10)

    with pytest.raises(TypeError):
        DenseLinearModel(f, v, nl, hs, ha, oa)

def test_init_num_layers_one_rejects_hidden_size_provided():
    """
    Verify DenseLinearModel rejects hidden_size when num_layers == 1.

    This test ensures that if the model is configured with a single layer
    (i.e., no hidden layers), the user cannot also provide a hidden_size.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 1
    hs: int = 64

    with pytest.raises(TypeError, match=r"hidden_size.*num_layers\s*==\s*1|num_layers\s*==\s*1.*hidden_size"):
        DenseLinearModel(f, v, num_layers=nl, hidden_size=hs)


def test_init_num_layers_one_rejects_hidden_act_provided():
    """
    Verify DenseLinearModel rejects hidden_act when num_layers == 1.

    This test ensures that if the model is configured with a single layer
    (i.e., no hidden layers), the user cannot also provide a hidden activation.
    """
    import torch
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 1
    ha: torch.nn.Module = torch.nn.ReLU()

    with pytest.raises(TypeError, match=r"hidden_act.*num_layers\s*==\s*1|num_layers\s*==\s*1.*hidden_act"):
        DenseLinearModel(f, v, num_layers=nl, hidden_act=ha)


def test_init_num_layers_one_allows_hidden_args_omitted():
    """
    Verify DenseLinearModel initializes when num_layers == 1 and hidden args omitted.

    This test ensures the "single-layer" configuration is supported and does not
    require hidden_size or hidden_act.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 1

    m = DenseLinearModel(f, v, num_layers=nl)

    assert m.num_layers == nl
    # Optional: only assert these exist if they are attributes on the class.
    # assert m.hidden_size is None
    # assert m.hidden_act is None


def test_init_num_layers_gt_one_requires_hidden_size():
    """
    Verify DenseLinearModel requires hidden_size when num_layers > 1.

    This test ensures that when hidden layers are present, a hidden_size must
    be provided.
    """
    import torch
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 2
    ha: torch.nn.Module = torch.nn.ReLU()

    with pytest.raises(TypeError, match=r"hidden_size.*required|required.*hidden_size"):
        DenseLinearModel(f, v, num_layers=nl, hidden_size=None, hidden_act=ha)


def test_init_num_layers_gt_one_requires_hidden_act():
    """
    Verify DenseLinearModel requires hidden_act when num_layers > 1.

    This test ensures that when hidden layers are present, a hidden activation
    must be provided.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 2
    hs: int = 64

    with pytest.raises(TypeError, match=r"hidden_act.*required|required.*hidden_act"):
        DenseLinearModel(f, v, num_layers=nl, hidden_size=hs, hidden_act=None)


def test_init_num_layers_gt_one_allows_hidden_args_provided():
    """
    Verify DenseLinearModel initializes when num_layers > 1 and hidden args provided.

    This test ensures that a valid multi-layer configuration is accepted when both
    hidden_size and hidden_act are provided.
    """
    import torch
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 2
    hs: int = 64
    ha: torch.nn.Module = torch.nn.ReLU()

    m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs, hidden_act=ha)

    assert m.num_layers == nl


def test_init_hidden_size_list_length_must_match_num_layers_minus_one():
    """
    Verify DenseLinearModel validates list-based hidden_size length.

    This test ensures that when hidden_size is provided as a list, it contains one
    entry per hidden layer, i.e., length == num_layers - 1.
    """
    import torch
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    ha: torch.nn.Module = torch.nn.ReLU()

    nl: int = 3
    hs_bad = [64]  # should be length 2

    with pytest.raises(ValueError, match=r"length.*num_layers\s*-\s*1|num_layers\s*-\s*1.*length"):
        DenseLinearModel(f, v, num_layers=nl, hidden_size=hs_bad, hidden_act=ha)

    nl = 4
    hs_bad = [64, 64]  # should be length 3

    with pytest.raises(ValueError, match=r"length.*num_layers\s*-\s*1|num_layers\s*-\s*1.*length"):
        DenseLinearModel(f, v, num_layers=nl, hidden_size=hs_bad, hidden_act=ha)


def test_init_hidden_size_list_valid_length_is_accepted():
    """
    Verify DenseLinearModel accepts list-based hidden_size with correct length.

    This test ensures that providing a hidden_size list with one entry per hidden
    layer is accepted.
    """
    import torch
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 3
    hs = [64, 64]  # num_layers - 1
    ha: torch.nn.Module = torch.nn.ReLU()

    m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs, hidden_act=ha)

    assert m.num_layers == nl

def test_init_num_layers_one_rejects_hidden_size_list():
    """
    Verify DenseLinearModel rejects list hidden_size when num_layers == 1.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f: int = 2048
    v: int = 10
    nl: int = 1
    hs = [64]  # still "provided"

    with pytest.raises(TypeError):
        DenseLinearModel(f, v, num_layers=nl, hidden_size=hs)

@pytest.mark.parametrize(
    "num_layers, hidden_size, hidden_act, output_act, expected_linear",
    [
        # num_layers == 1 => exactly one Linear (feature_dim -> value_dim)
        (1, None, None, None, 1),
        (1, None, None, nn.Identity(), 1),

        # num_layers > 1 => (num_layers - 1) hidden linears + 1 output linear = num_layers linears
        (2, 64, nn.ReLU(), None, 2),
        (3, 64, nn.ReLU(), nn.Identity(), 3),
        (4, [32, 64, 128], nn.GELU(), None, 4),
    ],
)
def test_dense_linear_model_builds_expected_number_of_linears(
    num_layers, hidden_size, hidden_act, output_act, expected_linear
):
    """
    Verify DenseLinearModel builds the expected number of Linear layers.

    For num_layers == 1: 1 Linear layer total.
    For num_layers > 1: num_layers Linear layers total (hidden + output).
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f, v = 128, 7
    m = DenseLinearModel(
        feature_dim=f,
        value_dim=v,
        num_layers=num_layers,
        hidden_size=hidden_size,
        hidden_act=hidden_act,
        output_act=output_act,
    )

    linear_count = sum(1 for mod in m.net.modules() if isinstance(mod, nn.Linear))
    assert linear_count == expected_linear

def test_dense_linear_model_hidden_act_count_matches_hidden_layers():
    """
    Verify hidden_act is applied once per hidden layer when num_layers > 1.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    f, v = 64, 3
    nl = 4  # 3 hidden layers
    hs = 16
    ha = nn.ReLU()

    m = DenseLinearModel(f, v, num_layers=nl, hidden_size=hs, hidden_act=ha)

    # Count how many modules are *the same object* as ha in the Sequential
    act_count = sum(1 for mod in m.net if mod is ha)
    assert act_count == nl - 1

@pytest.mark.parametrize("batch", [None, 1, 8])
@pytest.mark.parametrize(
    "num_layers, hidden_size, hidden_act, output_act",
    [
        (1, None, None, None),
        (2, 32, torch.nn.ReLU(), None),
        (3, [16, 32], torch.nn.GELU(), torch.nn.Identity()),
    ],
)
def test_dense_linear_model_forward_shapes(batch, num_layers, hidden_size, hidden_act, output_act):
    """
    Verify forward returns the correct output shape for 1D and batched inputs.
    """
    from hyperspace.core.regression.models import DenseLinearModel

    torch.manual_seed(0)

    f, v = 50, 6
    m = DenseLinearModel(
        feature_dim=f,
        value_dim=v,
        num_layers=num_layers,
        hidden_size=hidden_size,
        hidden_act=hidden_act,
        output_act=output_act,
    )

    if batch is None:
        x = torch.randn(f)
        y = m(x)
        assert y.shape == (v,)
    else:
        x = torch.randn(batch, f)
        y = m(x)
        assert y.shape == (batch, v)

    assert y.dtype == x.dtype
    assert y.device == x.device

def test_dense_linear_model_forward_is_deterministic_in_eval():
    """
    Verify forward is deterministic in eval mode for the same input.
    """
    from hyperspace.core.regression.models import DenseLinearModel
    import torch
    import torch.nn as nn

    torch.manual_seed(0)

    f, v = 32, 4
    m = DenseLinearModel(f, v, num_layers=3, hidden_size=16, hidden_act=nn.ReLU())
    m.eval()

    x = torch.randn(10, f)

    y1 = m(x)
    y2 = m(x)

    assert torch.allclose(y1, y2)

def test_dense_linear_model_forward_rejects_non_tensor():
    """
    Verify forward rejects non-tensor inputs with TypeError.
    """
    from hyperspace.core.regression.models import DenseLinearModel
    import torch.nn as nn

    m = DenseLinearModel(16, 3, num_layers=2, hidden_size=8, hidden_act=nn.ReLU())

    with pytest.raises(TypeError, match=r"x.*torch\.Tensor|Expected x"):
        m([1, 2, 3])  # type: ignore[arg-type]


def test_dense_linear_model_forward_rejects_invalid_ndim():
    """
    Verify forward rejects tensors that are not 1D or 2D.
    """
    from hyperspace.core.regression.models import DenseLinearModel
    import torch.nn as nn

    f, v = 16, 3
    m = DenseLinearModel(f, v, num_layers=2, hidden_size=8, hidden_act=nn.ReLU())

    x = torch.randn(2, 3, f)
    with pytest.raises(ValueError, match=r"1D or 2D|shape"):
        m(x)


def test_dense_linear_model_forward_rejects_wrong_feature_dim():
    """
    Verify forward rejects inputs whose last dimension != feature_dim.
    """
    from hyperspace.core.regression.models import DenseLinearModel
    import torch.nn as nn
    import torch

    f, v = 16, 3
    m = DenseLinearModel(f, v, num_layers=2, hidden_size=8, hidden_act=nn.ReLU())

    x = torch.randn(5, f + 1)
    with pytest.raises(ValueError, match=r"feature_dim|last dimension|match"):
        m(x)

def test_dense_linear_model_backward_runs():
    """
    Verify gradients flow through the model.
    """
    from hyperspace.core.regression.models import DenseLinearModel
    import torch
    import torch.nn as nn

    f, v = 20, 5
    m = DenseLinearModel(f, v, num_layers=3, hidden_size=16, hidden_act=nn.ReLU())

    x = torch.randn(7, f, requires_grad=True)
    y = m(x).sum()
    y.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()