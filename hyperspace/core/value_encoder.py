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
        if not isinstance(values, Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        
        if values.dim() != 2:
            raise ValueError("Input values must be a 2D tensor of shape (num_samples, value_dim).")
        
        if values.shape[1] != self.value_dim:
            raise ValueError(f"Input x must have shape (num_samples, {self.value_dim}).")
        
        # ----------------------------------------------------------
        # Start Shape: (num_samples, env_dim)
        # End Shape: (num_samples * env_dim,)
        # 
        # Also create a basis vector index tensor to map each value
        # to its corresponding basis vector. Then, encode each
        # value using the corresponding basis vector.
        # ----------------------------------------------------------
        x_flat = values.view(-1) # Shape: (num_samples * env_dim,)
        x_indexes = torch.arange(self.value_dim, device=values.device).repeat(values.shape[0]) # Shape: (num_samples * env_dim,)

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
        phi_x_flat = phi_x_flat.view(values.shape[0], self.value_dim, -1) # Shape: (num_samples, env_dim, vectorD)
        phi_x, bundle_info_dict = self.backend.bundle(phi_x_flat, dim=1) # Shape: (num_samples, vectorD)

        total_dict = {**bind_info_dict, **bundle_info_dict}
        return phi_x, total_dict