from torch import Tensor
from typing import Tuple

from ..backends.base import BaseBackend
from .base_module import BaseModule

class MemoryStorageModule(BaseModule):
    """
    Memory Storage module for HyperSpace.

    This module provides memory storage operations for vectors using various methods
    such as key-value storage and associative memory.
    """
    def __init__(self, backend: BaseBackend):
        """
        Initialize the MemoryStorageModule.

        Arguments:
            backend : BaseBackend
                The backend to use for encoding operations.
        """
        super().__init__()
        self.backend: BaseBackend = backend

        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Expected the argued backend to extend the BaseBackend class; got {type(self.backend)}")

    def __call__(self, p_vectors: Tensor, v_vectors: Tensor, prev_memory: Tensor = None) -> Tuple[Tensor, dict]:
        """
        Apply the specified memory storage method to the input tensor.
        """

        if not isinstance(p_vectors, Tensor):
            raise TypeError(f"Expected p_vectors to be a torch.Tensor; got {type(p_vectors)}")

        if not isinstance(v_vectors, Tensor):
            raise TypeError(f"Expected v_vectors to be a torch.Tensor; got {type(p_vectors)}")

        if not isinstance(prev_memory, Tensor):
            raise TypeError(f"Expected prev_memory to be a torch.Tensor; got {type(prev_memory)}")

        # check that the p_vectors is single or batched
        if p_vectors.dim() != 1 and p_vectors.dim() != 2:
            raise ValueError(f"Expected p_vectors to be single (vector_dim) or batched (num_points, vector_dim); got {p_vectors.shape}")

        # check that the v_vectors is single or batched
        if v_vectors.dim() != 1 and v_vectors.dim() != 2:
            raise ValueError(f"Expected v_vectors to be single (vector_dim) or batched (num_points, vector_dim); got {v_vectors.shape}")

        # check that the prev_memory is single
        if prev_memory.dim() != 1:
            raise ValueError(f"Expected prev_memory to be single (vector_dim); got {prev_memory.shape}")
        
        # check that p_vectors and v_vectors have the same shape
        if p_vectors.shape != v_vectors.shape:
            raise ValueError(f"Expected p_vectors and v_vectors to have the same shape; got {p_vectors.shape} and {v_vectors.shape}")

        # check that the dimensionalities of the vectors match
        if p_vectors.shape[-1] != prev_memory.shape[-1]:
            raise ValueError(f"Expected all vectors to have the same dimensionality; got {p_vectors.shape[-1]} and {prev_memory.shape[-1]}")

        new_memory, _ = self.backend.bind(p_vectors, v_vectors)

        # combine multiple points into a single memory
        if new_memory.ndim > 1:
            new_memory, _ = self.backend.bundle(new_memory)

        new_memory, _ = self.backend.bundle(new_memory, prev_memory)

        return new_memory, {}

    def initialize_memory(self) -> Tensor:
        """
        Initialize and return the initial memory vector
        """
        return self.backend.create_empty_vector()
        