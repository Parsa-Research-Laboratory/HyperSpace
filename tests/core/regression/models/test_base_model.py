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
    TODO Finish Documentation
    """
    from hyperspace.core.regression.models.base_model import BaseRegressionModel

    with pytest.raises(TypeError):
        BaseRegressionModel()