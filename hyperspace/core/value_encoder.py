from torch import Tensor

from ..backends.base import BaseBackend
from .base_module import BaseModule

class ValueEncoderModule(BaseModule):
    """
    Value Encoder module for HyperSpace.

    This module provides value encoding operations for vectors using various methods
    such as scalar and one-hot encodings.
    """
    def __init__(self, backend: BaseBackend, value_dim: int = 1):
        super().__init__()
        
        self.backend: BaseBackend = backend
        self.value_dim: int = value_dim

        if self.value_dim < 1:
            raise ValueError("value_dim must be at least 1.")
        
        if self.value_dim > 1:
            raise ValueError("Currently only scalar (1D) value encoding is supported.")
        
        self.backend.initialize_value_basis_vectors(self.value_dim)

    def __call__(self, values: Tensor) -> Tensor:
        """
        Apply the specified value encoding method to the input tensor.
        """
        raise NotImplementedError("ValueEncoderModule is not yet implemented.")