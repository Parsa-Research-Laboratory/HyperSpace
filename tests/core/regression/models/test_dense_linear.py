import pytest

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