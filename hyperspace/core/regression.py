from torch import Tensor
from typing import List

from ..backends.base import BaseBackend
from .base_module import BaseModule

class RegressionModule(BaseModule):
    """
    Regression module for HyperSpace.

    This module provides regression operations for vectors using various methods
    such as pseudo-inverse and ridge regression.
    """

    valid_methods: List[str] = ["codebook"]
    network_needed: bool = False
    network_ready: bool = False

    def __init__(self, backend: BaseBackend, codebook: Tensor, values: Tensor,
                 method: str = "codebook"):
        """
        Initialize the RegressionModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
            codebook : Tensor
                The set of vectors representing discrete values
            values : Tensor
                The discrete values represented by the codebook
            method: str
                The type of decoding method to leverage for the
                regression process
        """
        super().__init__()
        self.backend: BaseBackend = backend
        self.codebook: Tensor = codebook
        self.values: Tensor = values
        self.method: str = method

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(self.backend)}")
        
        if not isinstance(codebook, Tensor):
            raise TypeError(f"Expected the argued codebook to be a Tensor; got {type(codebook)}")
        
        if not isinstance(values, Tensor):
            raise TypeError(f"Expected the argued values to be Tensor; got {type(values)}")
        
        if not isinstance(method, str):
            raise TypeError(f"Expected the argued method to be string; got {type(method)}")
        
        if self.codebook.ndim != 2:
            raise ValueError(f"Codebook must be 2D with shape (batch_size, vector_dim); got {self.codebook.shape}")
        
        if self.codebook.shape[-1] != self.backend.vector_dim:
            raise ValueError(f"Codebook[-1] must match vector dim; got {self.codebook.shape[-1]} and {self.backend.vector_dim}")

        if self.values.ndim != 2:
            raise ValueError(f"Values must be 2D with shape (batch_size, value_dim); got {self.values.shape}")
        
        if self.values.shape[-1] != self.backend.value_dim:
            raise ValueError(f"Values[-1] must match value dim; got {self.values.shape[-1]} and {self.backend.value_dim}")
        
        if method not in self.valid_methods:
            raise ValueError(f"Expected method to be on of [{self.valid_methods}]; got {method}")
        
        if method in ["neural"]:
            self.network_needed = True

    def __call__(self, v: Tensor) -> Tensor:
        """
        Apply the specified regression method to the input tensor.
        """

        if not isinstance(v, Tensor):
            raise TypeError(f"Expected v to be a Tensor; got {type(v)}")
        
        if v.ndim not in [1, 2]:
            raise ValueError(f"Expected v to be a 1D or 2D Tensor; got shape {v.shape}")
        
        if v.shape[-1] != self.backend.vector_dim:
            raise ValueError(f"Expected dimensionality of v to match the backend; got {v.shape[-1]} and {self.backend.vector_dim}")

        return self.backend.regression(v, v, v)