import torch
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
        
        if p_vectors.dim() != 3 or v_vectors.dim() != 3:
            raise ValueError("Input p_vectors and v_vectors must be 3D tensors of shape (num_samples, 1, vectorD).")
        
        if p_vectors.shape[1] != 1 or v_vectors.shape[1] != 1:
            raise ValueError("Input p_vectors and v_vectors must have shape (num_samples, 1, vectorD).")
        
        if p_vectors.shape[0] != v_vectors.shape[0]:
            raise ValueError("Input p_vectors and v_vectors must have the same number of samples.")
        
        if p_vectors.shape[2] != self.backend.vector_dim or v_vectors.shape[2] != self.backend.vector_dim:
            raise ValueError(f"Input vectors must have shape (num_samples, {self.backend.vector_dim}).")
        
        if prev_memory is not None:
            if not isinstance(prev_memory, Tensor):
                raise TypeError("Input prev_memory must be a torch.Tensor.")
            if prev_memory.dim() != 1 or prev_memory.shape[0] != self.backend.vector_dim:
                raise ValueError(f"Input prev_memory must be a 1D tensor of shape ({self.backend.vector_dim},).")
            
        # ----------------------------------------------------------------
        # Each position vector is bound to its corresponding value vector
        # ----------------------------------------------------------------
        pv_batch = torch.concatenate([p_vectors,v_vectors], dim=1)  # Shape: (num_samples, 2, vectorD)
        print(f"Binding {pv_batch.shape} position-value vector pairs into bound vectors...")
        bound_vectors, bind_info = self.backend.bind(pv_batch)

        if bound_vectors.shape != (p_vectors.shape[0], 1, self.backend.vector_dim):
            raise ValueError(f"Bound vectors have incorrect shape: {bound_vectors.shape}")

        # ----------------------------------------------------------------
        # Bundle all bound vectors into a single memory vector
        # ----------------------------------------------------------------
        if prev_memory is not None:
            bound_vectors = torch.vstack([bound_vectors, prev_memory.unsqueeze(0)])

        print(f"Bundling {bound_vectors.shape} bound vectors into memory...")

        memory, bundle_info = self.backend.bundle(bound_vectors)
        info_dict = {
            "memory_storage": {
                **bind_info,
                **bundle_info
            }
        }
        return memory, info_dict

        