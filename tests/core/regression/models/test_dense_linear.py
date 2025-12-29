import pytest

def test_base_import():
    """
    Verify that the DenseLinearModel can be imported successfully.

    This test ensures that the core regression model module is discoverable
    and free of import-time errors caused by missing dependencies or
    circular imports.
    """
    from hyperspace.core.regression.models.dense_linear import DenseLinearModel