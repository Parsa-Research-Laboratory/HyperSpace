from torch import Tensor

from .base_module import BaseModule

class PositionalInversionModule(BaseModule):
    """
    Positional Inversion module for HyperSpace.

    This module provides positional inversion operations for vectors using various methods
    such as sinusoidal and learned positional inversions.
    """
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def __call__(self, basis: Tensor, x: Tensor, method: str = "sinusoidal") -> Tensor:
        """
        Apply the specified positional inversion method to the input tensor.
        """
        return self.backend.positional_inversion(basis, x, method)