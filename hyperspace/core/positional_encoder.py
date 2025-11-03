from torch import Tensor

from ..backends.base import BaseBackend
from .base_module import BaseModule

class PositionalEncoderModule(BaseModule):
    """
    Positional Encoder module for HyperSpace.

    This module provides positional encoding operations for vectors using various methods
    such as sinusoidal and learned positional encodings.
    """
    def __init__(self, backend: BaseBackend, env_dim: int):
        """
        Initialize the PositionalEncoderModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
            env_dim : int
                The dimensionality of the environment to encode.
        """
        super().__init__()
        self.backend: BaseBackend = backend
        self.env_dim: int = env_dim

        self.backend.initialize_env_basis_vectors(self.env_dim)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the specified positional encoding method to the input tensor.
        """
        raise NotImplementedError("PositionalEncoderModule is not yet implemented.")