from torch import Tensor

from ..backends.base import BaseBackend
from .base_module import BaseModule

class RegressionModule(BaseModule):
    """
    Regression module for HyperSpace.

    This module provides regression operations for vectors using various methods
    such as pseudo-inverse and ridge regression.
    """
    def __init__(self, backend):
        """
        Initialize the RegressionModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
        """
        super().__init__()
        self.backend: BaseBackend = backend

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(self.backend)}")

    def __call__(self, basis: Tensor, values: Tensor, method: str = "pseudo_inverse") -> Tensor:
        """
        Apply the specified regression method to the input tensor.
        """
        return self.backend.regression(basis, values, method)