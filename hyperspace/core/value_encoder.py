from torch import Tensor

from .base_module import BaseModule

class ValueEncoderModule(BaseModule):
    """
    Value Encoder module for HyperSpace.

    This module provides value encoding operations for vectors using various methods
    such as scalar and one-hot encodings.
    """
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def __call__(self, basis: Tensor, values: Tensor, method: str = "scalar") -> Tensor:
        """
        Apply the specified value encoding method to the input tensor.
        """
        return self.backend.value_encoding(basis, values, method)