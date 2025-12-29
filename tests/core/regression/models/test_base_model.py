import pytest

def test_base_import():
    """
    Verify that the BaseRegressionModel can be imported successfully.

    This test ensures that the core regression model module is discoverable
    and free of import-time errors caused by missing dependencies or
    circular imports.
    """
    from hyperspace.core.regression.models.base_model import BaseRegressionModel

def test_base_init_no_arg():
    """
    Verify that BaseRegressionModel cannot be instantiated without arguments.

    This test ensures that the base regression model enforces its constructor
    contract and raises a TypeError when required initialization parameters
    are omitted.
    """
    from hyperspace.core.regression.models.base_model import BaseRegressionModel

    with pytest.raises(TypeError):
        BaseRegressionModel()

def test_base_init_missing_feature_dim():
    """
    Verify that BaseRegressionModel requires a feature dimension at initialization.

    This test ensures that providing only a value dimension is insufficient
    and that omitting the required feature_dim argument results in a TypeError,
    enforcing the constructor contract of the base model.
    """
    from hyperspace.core.regression.models.base_model import BaseRegressionModel

    with pytest.raises(TypeError):
        BaseRegressionModel(value_dim=10)

def test_base_init_missing_value_dim():
    """
    Verify that BaseRegressionModel requires a value dimension at initialization.

    This test ensures that providing only a feature dimension is insufficient
    and that omitting the required value_dim argument results in a TypeError,
    enforcing the constructor contract of the base model.
    """
    from hyperspace.core.regression.models.base_model import BaseRegressionModel

    with pytest.raises(TypeError):
        BaseRegressionModel(feature_dim=10)

       