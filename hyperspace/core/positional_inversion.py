from torch import Tensor

from ..backends.base import BaseBackend
from .base_module import BaseModule

class PositionalInversionModule(BaseModule):
    """
    Positional Inversion module for HyperSpace.

    This module provides positional inversion operations for vectors using various methods
    such as sinusoidal and learned positional inversions.
    """
    def __init__(self, backend: BaseBackend, positions: Tensor):
        """
        Initialize the PositionalInversionModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
            positions : Tensor
                The positions to invert and decode from
        """

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected backend to be a subclass of BaseBackend; got {type(backend)}")
        
        if not isinstance(positions, Tensor):
            raise TypeError(f"Expected positions to be a Tensor; got {type(positions)}")

        if positions.dim() != 2:
            raise ValueError(f"Expected positions to be a 2D Tensor with shape (batch_size, env_dim); got {positions.shape}")
        
        if positions.shape[-1] != backend.env_dim:
            raise ValueError(f"Expected the env_dim of positions to match the backend; got {positions.shape[-1]} and {backend.env_dim}")

        super().__init__()
        self.backend: BaseBackend = backend
        self.positions: Tensor = positions # (batch_size, env_dim)
        self.inv_position_vectors: Tensor = self._create_inv_position_vectors() # (batch_size, vector_dim)

        if self.inv_position_vectors.dim() != 2:
            raise ValueError(f"Expected position vectors to be a 2D Tensor with shape (batch_size, vector_dim); got {self.inv_position_vectors.shape}")
        
        if self.inv_position_vectors.shape[0] != self.positions.shape[0]:
            raise ValueError(f"Expected the number of position vectors to match the number of positions; got {self.inv_position_vectors.shape[0]} and {self.positions.shape[0]}")

        if self.inv_position_vectors.shape[-1] != backend.vector_dim:
            raise ValueError(f"Expected the vector_dim of position vectors to match the backend; got {self.inv_position_vectors.shape[-1]} and {backend.vector_dim}")

    def __call__(self, basis: Tensor, x: Tensor, method: str = "sinusoidal") -> Tensor:
        """
        Apply the specified positional inversion method to the input tensor.
        """
        return self.backend.positional_inversion(basis, x, method)
    
    def _create_inv_position_vectors(self) -> Tensor:
        """
        Create inverted positional encoding vectors for the stored positions.

        This method applies the backend positional encoding function to the
        internally stored position values, inverts then, and returns the
        resulting hypervector representations.

        Returns:
            Tensor: A tensor containing the positional encoding vectors
            corresponding to ``self.positions``.
        """
        pvs, _ = self.backend.positional_encoding(self.positions)
        ipvs, _ = self.backend.invert(pvs)
        return ipvs