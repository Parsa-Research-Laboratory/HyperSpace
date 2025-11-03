from torch import Tensor

from .base_module import BaseModule

class RegressionModule(BaseModule):
    """
    Regression module for HyperSpace.

    This module provides regression operations for vectors using various methods
    such as pseudo-inverse and ridge regression.
    """
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def __call__(self, basis: Tensor, values: Tensor, method: str = "pseudo_inverse") -> Tensor:
        """
        Apply the specified regression method to the input tensor.
        """
        return self.backend.regression(basis, values, method)