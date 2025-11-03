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

        if not isinstance(x, Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        
        if x.dim() != 2:
            raise ValueError("Input x must be a 2D tensor of shape (num_samples, env_dim).")
        
        if x.shape[1] != self.env_dim:
            raise ValueError(f"Input x must have shape (num_samples, {self.env_dim}).")
        
        # ----------------------------------------------------------
        # Start Shape: (num_samples, env_dim)
        # End Shape: (num_samples * env_dim,)
        # 
        # Also create a basis vector index tensor to map each value
        # to its corresponding basis vector. Then, encode each
        # value using the corresponding basis vector.
        # ----------------------------------------------------------
        x_flat = x.view(-1) # Shape: (num_samples * env_dim,)
        x_indexes = torch.arange(self.env_dim, device=x.device).repeat(x.shape[0]) # Shape: (num_samples * env_dim,)

        # ----------------------------------------------------------
        # Encode the flattened input using the backend's
        # continuous encoding
        # ----------------------------------------------------------
        # Shape: (num_samples * env_dim, vectorD)
        phi_x_flat, bind_info_dict = self.backend.continuous_encoding(x_flat, x_indexes)

        # ----------------------------------------------------------
        # Combine the individual axis encodings into a single
        # positional encoding for each sample by binding the
        # encodings together
        # ----------------------------------------------------------
        phi_x_flat = phi_x_flat.view(x.shape[0], self.env_dim, -1) # Shape: (num_samples, env_dim, vectorD)
        phi_x, bundle_info_dict = self.backend.bundle(phi_x_flat, dim=1) # Shape: (num_samples, vectorD)

        total_dict = {**bind_info_dict, **bundle_info_dict}
        return phi_x, total_dict