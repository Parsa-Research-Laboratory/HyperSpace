import torch
from torch import Tensor
from typing import List, Optional, Tuple, Union

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
        
        if values is None and codebook is None:
            raise ValueError(f"Must receive values or a codebook.")
        
        if values is not None and codebook is not None:
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

    def __call__(self, v: Tensor, method: str = "resonator") -> Tuple[Tensor, dict]:
        """
        Apply the specified cleanup method to the input tensor.
        """

        if not isinstance(v, Tensor):
            raise TypeError(f"Expected v to be a Tensor; got {type(v)}")
        
        if v.ndim not in [1, 2]:
            raise ValueError(f"Expected v to be a 1D or 2D Tensor; got shape {v.shape}")
        
        if v.shape[-1] != self.backend.vector_dim:
            raise ValueError(f"Expected dimensionality of v to match the backend; got {v.shape[-1]} and {self.backend.vector_dim}")
        
        if method not in self.valid_methods:
            raise ValueError(f"Expected method to be on of [{self.valid_methods}]; got {method}")
        
        if method == "resonator":
            out, info_dict = self.backend._resonator_cleanup(v, self.codebook)
        elif method == "modern_hopfield":
            out, info_dict = self.backend._hopfield_cleanup(v, self.codebook)
        else:
            raise ValueError(f"received invalid cleanup method: {method}")

        return out, info_dict