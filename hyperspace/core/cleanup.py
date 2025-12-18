from torch import Tensor
from typing import List, Optional, Tuple, Union

from ..backends.base import BaseBackend
from .base_module import BaseModule

class CleanupModule(BaseModule):
    """
    CleanupModule performs associative cleanup of hypervectors using a
    backend-defined set of cleanup rules.

    The module constructs a cleanup codebook from either raw input values (which
    are encoded using the backend) or a precomputed set of hypervectors. At
    runtime, the module applies a chosen cleanup method—such as resonator dynamics
    or a modern Hopfield update—to move an input vector toward the nearest item in
    the codebook under the backend’s similarity metric.

    Supported cleanup methods depend on the backend but typically include:
        - "resonator": Iterative resonator-based attractor dynamics.
        - "modern_hopfield": A modern Hopfield-style update rule.
        - "identity": Returns the input unchanged (useful for debugging).

    This module provides both single-vector and batched cleanup, allowing it to be
    used in encoding pipelines, memory-retrieval systems, or iterative inference
    procedures within HyperSpace.
    """

    valid_methods: List[str] = [
        "resonator",
        "modern_hopfield"
    ]

    def __init__(self, backend: BaseBackend, values: Optional[Tensor] = None,
                 codebook: Optional[Tensor] = None):
        """
        Initialize a CleanupModule for performing hypervector cleanup operations.

        The module constructs a cleanup codebook either by:
        (1) encoding a set of input values into hypervectors using the backend, or
        (2) accepting a precomputed codebook of hypervectors directly.

        Exactly one of `values` or `codebook` must be provided. If `values` is
        supplied, each row is encoded into a hypervector via the backend’s
        value-encoding method. If `codebook` is supplied, it is used verbatim
        without modification.

        Parameters
        ----------
        backend : BaseBackend
            Backend instance that defines the vector dimensionality and provides
            encoding and cleanup operations.

        values : Optional[Tensor], default=None
            Tensor of raw values to encode into hypervectors. Must have shape
            (batch_size, value_dim). When provided, the module encodes these values
            into a cleanup codebook using `backend.value_encoding`.

        codebook : Optional[Tensor], default=None
            Precomputed cleanup codebook. Must be a tensor of shape
            (batch_size, vector_dim). These vectors are assumed to already reside in
            the backend’s hypervector space and will not be re-encoded.

        Raises
        ------
        TypeError
            If `backend` does not extend BaseBackend, or if `values`/`codebook` 
            are not tensors when provided.

        ValueError
            If neither or both of `values` and `codebook` are provided;
            if either input has incorrect dimensionality or mismatched backend 
            value/vector dimension requirements.

        Notes
        -----
        After initialization, `self.codebook` will always be a valid 2D tensor
        of encoded hypervectors suitable for cleanup procedures.
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

    def __call__(self, v: Tensor, method: str = "resonator", num_iters: int = 3, **kwargs) -> Tuple[Tensor, dict]:
        """
        Perform cleanup of an input vector using the specified cleanup method.

        This method selects and applies a cleanup procedure—such as resonator
        dynamics or a modern Hopfield update—to move the input vector toward the
        closest item in the codebook according to the backend's similarity
        structure. Cleanup can be performed on a single vector of shape (D,) or a
        batch of vectors of shape (B, D).

        Parameters
        ----------
        v : Tensor
            Input tensor to clean up. Must be either a 1D tensor of shape (D,) or a 
            2D tensor of shape (B, D), where D matches the backend's vector 
            dimensionality.
        
        method : str, optional
            The cleanup rule to apply. Supported options include:
            - `"resonator"`: Uses iterative resonator dynamics for cleanup.
            - `"modern_hopfield"`: Uses a modern Hopfield-style update rule.
            Defaults to `"resonator"`.

        num_iters : int, optional
            The number of times to repeat the cleanup operation

        **kwargs : Optional
            Any other method specific keyword arguments

        Returns
        -------
        Tensor
            The cleaned vector(s), matching the shape of the input.
        
        dict
            An information dictionary containing method-specific diagnostics.

        Raises
        ------
        TypeError
            If `v` is not a Tensor.
        ValueError
            If `v` does not have 1 or 2 dimensions;
            if its last dimension does not match the backend's vector dimensionality;
            or if an unsupported cleanup method is provided.
        """

        if not isinstance(v, Tensor):
            raise TypeError(f"Expected v to be a Tensor; got {type(v)}")
        
        if v.ndim not in [1, 2]:
            raise ValueError(f"Expected v to be a 1D or 2D Tensor; got shape {v.shape}")
        
        if v.shape[-1] != self.backend.vector_dim:
            raise ValueError(f"Expected dimensionality of v to match the backend; got {v.shape[-1]} and {self.backend.vector_dim}")
        
        if method not in self.valid_methods:
            raise ValueError(f"Expected method to be on of [{self.valid_methods}]; got {method}")
        
        if not isinstance(num_iters, int):
            raise TypeError(f"Expected num_iters to be an int; got {type(num_iters)}")
        
        if num_iters < 1:
            raise ValueError(f"Expected num_iters to be >= 1; got {num_iters}")
        
        if method == "resonator":
            out, info_dict = self.backend._resonator_cleanup(v, self.codebook, num_iters, **kwargs)
        elif method == "modern_hopfield":
            out, info_dict = self.backend._hopfield_cleanup(v, self.codebook, num_iters, **kwargs)
        else:
            raise ValueError(f"received invalid cleanup method: {method}")

        return out, info_dict