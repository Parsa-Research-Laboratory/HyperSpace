from torch import Tensor
from .base_module import BaseModule

class PositionalEncoderModule(BaseModule):
    """
    Positional Encoder module for HyperSpace.

    This module provides positional encoding operations for vectors using various methods
    such as sinusoidal and learned positional encodings.
    """
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def __call__(self, basis: Tensor, x: Tensor, method: str = "sinusoidal") -> Tensor:
        """
        Apply the specified positional encoding method to the input tensor.
        """
        return self.backend.positional_encoding(basis, x, method)