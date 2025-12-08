import torch
from torch import Tensor
from typing import Tuple

from ..backends.base import BaseBackend
from .base_module import BaseModule

class PositionalEncoderModule(BaseModule):
    """
    Positional Encoder module for HyperSpace.

    This module provides positional encoding operations for vectors using various methods
    such as sinusoidal and learned positional encodings.
    """
    def __init__(self, backend: BaseBackend):
        """
        Initialize the PositionalEncoderModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
        """
        super().__init__()
        self.backend: BaseBackend = backend

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(self.backend)}")

    def __call__(self, x: Tensor) -> Tuple[Tensor, dict]:
        """
        Apply the specified positional encoding method to the input tensor.

        Arguments:
            x : Tensor
                Input tensor of shape (num_samples, env_dim).

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
        
        out, info = backend.positional_encoding(x)

        return out, info