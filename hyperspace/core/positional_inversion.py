from torch import Tensor

from ..backends.base import BaseBackend
from .base_module import BaseModule

class PositionalInversionModule(BaseModule):
    """
    Positional Inversion module for HyperSpace.

    This module provides positional inversion operations for vectors using various methods
    such as sinusoidal and learned positional inversions.
    """
    def __init__(self, backend: BaseBackend):
        """
        Initialize the PositionalInversionModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
        """
        super().__init__()
        self.backend: BaseBackend = backend

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(self.backend)}")

    def __call__(self, basis: Tensor, x: Tensor, method: str = "sinusoidal") -> Tensor:
        """
        Apply the specified positional inversion method to the input tensor.
        """
        return self.backend.positional_inversion(basis, x, method)