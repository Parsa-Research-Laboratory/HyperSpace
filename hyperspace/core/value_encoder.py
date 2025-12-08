import torch
from torch import Tensor

from ..backends.base import BaseBackend
from .base_module import BaseModule

class ValueEncoderModule(BaseModule):
    """
    Value Encoder module for HyperSpace.

    This module provides value encoding operations for vectors using various methods
    such as scalar and one-hot encodings.
    """
    def __init__(self, backend: BaseBackend):
        """
        Initialize the ValueModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
        """
        super().__init__()
        self.backend: BaseBackend = backend

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(self.backend)}")

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the specified positional encoding method to the input tensor.

        Arguments:
            x : Tensor
                Input tensor of shape (num_samples, value_dim).

        Returns:
            Tuple[Tensor, dict]
                Encoded tensor of shape (num_samples, vectorD) and info dictionary.
        """

        # --------------------------------
        # validate input shapes
        # --------------------------------
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x should be a Tensor, got {type(x)}")
        if x.dim() != 2 and x.dim() != 1:
            raise ValueError(f"Input x must be a 1D or 2D tensor of shape (env_dim) or (batch_size, env_dim); got {x.dim()}")
        if x.shape[-1] != self.env_dim:
            raise ValueError(f"Input x's last dimension should have shape {self.env_dim}; got {x.shape[-1]}")

        # TODO: Add batched processing for embedded / smaller devices
        
        out, info = backend.value_encoding(x)

        return out, info