from typing import Tuple, Optional

from torch import Tensor

from ..backends.base import BaseBackend
from .base_module import BaseModule


class MemoryStorageModule(BaseModule):
    """
    MemoryStorageModule maintains and updates a single associative memory
    hypervector using a backend-defined binding and bundling scheme.

    At each call, the module binds a set of key hypervectors (`p_vectors`)
    with corresponding value hypervectors (`v_vectors`) and bundles the result
    into a persistent memory vector. This supports key–value style storage
    and retrieval in HyperSpace-style vector symbolic architectures.
    """

    def __init__(self, backend: BaseBackend):
        """
        Initialize a MemoryStorageModule.

        Parameters
        ----------
        backend : BaseBackend
            Backend instance that defines the hypervector dimensionality and
            provides `bind`, `bundle`, and `create_empty_vector` operations.

        Raises
        ------
        TypeError
            If `backend` does not extend `BaseBackend`.
        """
        super().__init__()
        self.backend: BaseBackend = backend

        if not isinstance(backend, BaseBackend):
            raise TypeError(
                f"Expected the argued backend to extend the BaseBackend class; "
                f"got {type(self.backend)}"
            )

    def __call__(
        self,
        p_vectors: Tensor,
        v_vectors: Tensor,
        prev_memory: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Update the stored memory with new key–value bindings.

        This method binds each position/key hypervector in `p_vectors` with the
        corresponding value hypervector in `v_vectors` using the backend's
        `bind` operation. If multiple key–value pairs are provided, their
        bindings are bundled into a single hypervector. The resulting update
        is then bundled with `prev_memory` to produce the new memory state.

        If `prev_memory` is ``None``, an empty memory vector is created via
        :meth:`initialize_memory` and used as the initial state.

        Parameters
        ----------
        p_vectors : Tensor
            Position or key hypervectors. Must be either:
            - shape ``(D,)`` for a single key, or
            - shape ``(N, D)`` for a batch of N keys.
        v_vectors : Tensor
            Value hypervectors corresponding to `p_vectors`. Must have the same
            shape as `p_vectors`.
        prev_memory : Tensor, optional
            Previous memory hypervector of shape ``(D,)``. If ``None``,
            a new empty memory vector is created and used as the starting state.

        Returns
        -------
        Tensor
            Updated memory hypervector of shape ``(D,)``.
        dict
            An information dictionary (currently empty, reserved for future
            diagnostics or metadata).

        Raises
        ------
        TypeError
            If `p_vectors` or `v_vectors` is not a `torch.Tensor`,
            or if `prev_memory` is not a `torch.Tensor` when provided.
        ValueError
            If `p_vectors` or `v_vectors` do not have 1 or 2 dimensions;
            if `p_vectors` and `v_vectors` have mismatched shapes;
            if `prev_memory` is not 1D; or if the vector dimensionalities are
            inconsistent.
        """

        if not isinstance(p_vectors, Tensor):
            raise TypeError(
                f"Expected p_vectors to be a torch.Tensor; got {type(p_vectors)}"
            )

        if not isinstance(v_vectors, Tensor):
            raise TypeError(
                f"Expected v_vectors to be a torch.Tensor; got {type(v_vectors)}"
            )

        # check that p_vectors is single or batched
        if p_vectors.dim() not in (1, 2):
            raise ValueError(
                "Expected p_vectors to be single (vector_dim) or batched "
                f"(num_points, vector_dim); got {p_vectors.shape}"
            )

        # check that v_vectors is single or batched
        if v_vectors.dim() not in (1, 2):
            raise ValueError(
                "Expected v_vectors to be single (vector_dim) or batched "
                f"(num_points, vector_dim); got {v_vectors.shape}"
            )

        # check that p_vectors and v_vectors have the same shape
        if p_vectors.shape != v_vectors.shape:
            raise ValueError(
                "Expected p_vectors and v_vectors to have the same shape; "
                f"got {p_vectors.shape} and {v_vectors.shape}"
            )

        # Initialize or validate prev_memory
        if prev_memory is None:
            prev_memory = self.initialize_memory()
        elif not isinstance(prev_memory, Tensor):
            raise TypeError(
                f"Expected prev_memory to be a torch.Tensor; got {type(prev_memory)}"
            )

        # check that prev_memory is single
        if prev_memory.dim() != 1:
            raise ValueError(
                f"Expected prev_memory to be single (vector_dim); got {prev_memory.shape}"
            )

        # check that the dimensionalities of the vectors match
        if p_vectors.shape[-1] != prev_memory.shape[-1]:
            raise ValueError(
                "Expected all vectors to have the same dimensionality; "
                f"got {p_vectors.shape[-1]} and {prev_memory.shape[-1]}"
            )

        # bind keys and values
        new_memory, _ = self.backend.bind(p_vectors, v_vectors)

        # combine multiple points into a single memory if batched
        if new_memory.ndim > 1:
            new_memory, _ = self.backend.bundle(new_memory)

        # update memory with previous state
        new_memory, _ = self.backend.bundle(new_memory, prev_memory)

        return new_memory, {}

    def initialize_memory(self) -> Tensor:
        """
        Create and return an empty memory hypervector.

        The returned vector has the backend's hypervector dimensionality and is
        suitable to be used as the initial `prev_memory` state.

        Returns
        -------
        Tensor
            An empty memory vector of shape ``(D,)`` on the backend's device.
        """
        return self.backend.create_empty_vector()