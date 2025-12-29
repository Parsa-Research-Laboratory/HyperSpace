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