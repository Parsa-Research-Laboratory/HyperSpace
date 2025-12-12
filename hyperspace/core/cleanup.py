import torch
from torch import Tensor
from typing import List, Optional, Union

from ..backends.base import BaseBackend
from .base_module import BaseModule

class CleanupModule(BaseModule):
    """
    Cleanup module for HyperSpace.

    This module provides cleanup operations for vectors using various methods
    such as resonator, identity, and hopfield network cleanup.
    """

    valid_methods: List[str] = [
        "resonator",
        "modern_hopfield"
    ]

    def __init__(self, backend: BaseBackend, values: Optional[Tensor] = None,
                 codebook: Optional[Tensor] = None):
        """
        Initialize the CleanupModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
            positions : Optional(Tensor)
                The specific values to clean over; these values will be encoded to
                hypervectors with the backend; shape = (batch_size, value_dim)
            codebook : Optional(Tensor)
                The specific codes to clean over; these values will not be
                manipulated with the backend; shape = (batch_size, vector_dim)
        """
        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(backend)}")
        
        if not isinstance(values, Tensor) and not isinstance(codebook, Tensor):
            raise ValueError(f"Must receive values or a codebook.")
        
        if isinstance(values, Tensor) and isinstance(codebook, Tensor):
            raise ValueError(f"Must receive values or codebook; not both.")
        
        super().__init__()
        self.backend: BaseBackend = backend
        self.values: Optional[Tensor] = values
        self.codebook: Union[Tensor, None] = codebook

        if self.values is not None:
            if not isinstance(self.values, Tensor):
                raise TypeError(f"Values must be a Tensor; got {type(self.values)}")
            
            if self.values.ndim != 2:
                raise ValueError(f"Values must be 2D with shape (batch_size, value_dim); got {self.values.shape}")
            
            if self.values.shape[-1] != self.backend.value_dim:
                raise ValueError(f"Values[-1] must match value dim; got {self.values.shape[-1]} and {self.backend.value_dim}")
            
            self.codebook, _ = self.backend.value_encoding(self.values)
            
        if self.codebook is not None:
            if not isinstance(self.codebook, Tensor):
                raise TypeError(f"Codebook must be a Tensor; got {type(self.codebook)}")
            
            if self.codebook.ndim != 2:
                raise ValueError(f"Codebook must be 2D with shape (batch_size, vector_dim); got {self.codebook.shape}")
            
            if self.codebook.shape[-1] != self.backend.vector_dim:
                raise ValueError(f"Codebook[-1] must match vector dim; got {self.codebook.shape[-1]} and {self.backend.vector_dim}")

        # final sanity check
        assert self.codebook is not None

    def __call__(self, tensor, method: str = "resonator"):
        """
        Apply the specified cleanup method to the input tensor.
        """
        return self.backend.cleanup(tensor, method)