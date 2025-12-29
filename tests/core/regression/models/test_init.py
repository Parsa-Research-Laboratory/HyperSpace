import pytest

def test_relative_import_dense_linear_model():
    """
    Verify that DenseLinearModel is exposed via the regression.models package.

    This test ensures that the package-level __init__.py correctly re-exports
    DenseLinearModel, enabling clean and stable public imports without requiring
    users to reference internal module paths.
    """
    from hyperspace.core.regression.models import DenseLinearModel