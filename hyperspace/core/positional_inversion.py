from torch import Tensor
from typing import Tuple

from ..backends.base import BaseBackend
from .base_module import BaseModule


class PositionalInversionModule(BaseModule):
    """
    PositionalInversionModule constructs inverse positional hypervectors for a
    fixed set of Euclidean positions and uses them to invert position-bound
    memories.

    Given a memory vector `m` that encodes information bound with positional
    hypervectors, this module binds `m` with precomputed inverse positional
    codes. The result is a set of hypervectors aligned with each stored
    position, which can then be decoded or cleaned up by downstream
    components.
    """

    def __init__(self, backend: BaseBackend, positions: Tensor):
        """
        Initialize a PositionalInversionModule.

        Parameters
        ----------
        backend : BaseBackend
            Backend instance that defines the environment and vector
            dimensionalities and provides positional encoding and inversion
            operations.
        positions : Tensor
            Tensor of Euclidean positions to support during inversion.
            Must have shape ``(N, env_dim)``, where ``env_dim`` matches
            ``backend.env_dim`` and ``N`` is the number of positions.

        Raises
        ------
        TypeError
            If `backend` is not a subclass of :class:`BaseBackend`,
            or if `positions` is not a :class:`torch.Tensor`.
        ValueError
            If `positions` is not 2D, if its trailing dimension does not match
            ``backend.env_dim``, or if the internally created inverse
            positional vectors do not have consistent shapes.
        """
        if not isinstance(backend, BaseBackend):
            raise TypeError(
                f"Expected backend to be a subclass of BaseBackend; got {type(backend)}"
            )

        if not isinstance(positions, Tensor):
            raise TypeError(f"Expected positions to be a Tensor; got {type(positions)}")

        if positions.dim() != 2:
            raise ValueError(
                "Expected positions to be a 2D Tensor with shape "
                f"(batch_size, env_dim); got {positions.shape}"
            )

        if positions.shape[-1] != backend.env_dim:
            raise ValueError(
                "Expected the env_dim of positions to match the backend; "
                f"got {positions.shape[-1]} and {backend.env_dim}"
            )

        super().__init__()
        self.backend: BaseBackend = backend
        self.positions: Tensor = positions  # (batch_size, env_dim)
        # (batch_size, vector_dim)
        self.inv_position_vectors: Tensor = self._create_inv_position_vectors()

        if self.inv_position_vectors.dim() != 2:
            raise ValueError(
                "Expected position vectors to be a 2D Tensor with shape "
                f"(batch_size, vector_dim); got {self.inv_position_vectors.shape}"
            )

        if self.inv_position_vectors.shape[0] != self.positions.shape[0]:
            raise ValueError(
                "Expected the number of position vectors to match the number of "
                f"positions; got {self.inv_position_vectors.shape[0]} and "
                f"{self.positions.shape[0]}"
            )

        if self.inv_position_vectors.shape[-1] != backend.vector_dim:
            raise ValueError(
                "Expected the vector_dim of position vectors to match the backend; "
                f"got {self.inv_position_vectors.shape[-1]} and {backend.vector_dim}"
            )

    def __call__(self, m: Tensor) -> Tuple[Tensor, dict]:
        """
        Invert a position-bound memory vector with respect to stored positions.

        This method binds the input memory hypervector `m` with the precomputed
        inverse positional hypervectors corresponding to :attr:`positions`.
        Intuitively, if `m` contains content bound with forward positional codes,
        this operation produces a set of content-like hypervectors aligned with
        each stored position.

        Parameters
        ----------
        m : Tensor
            Memory hypervector to invert. Must be a 1D tensor of shape
            ``(vector_dim,)``, where ``vector_dim`` matches
            ``backend.vector_dim``.

        Returns
        -------
        Tensor
            A tensor of shape ``(N, vector_dim)``, where ``N`` is the number of
            stored positions. Each row corresponds to the bound result
            ``m * inv_position_vectors[i]`` under the backend's binding
            operation.
        dict
            An information dictionary returned by the backend's `bind`
            implementation (may be empty).

        Raises
        ------
        TypeError
            If `m` is not a :class:`torch.Tensor`.
        ValueError
            If `m` is not 1D or if its dimensionality does not match
            ``backend.vector_dim``.
        """
        if not isinstance(m, Tensor):
            raise TypeError(f"Expected m to be a Tensor; got {type(m)}")

        if m.dim() != 1:
            raise ValueError(
                f"Expected m to be a 1D tensor with shape (vector_dim,); got {m.shape}"
            )

        if m.shape[-1] != self.backend.vector_dim:
            raise ValueError(
                "Expected m to have the same vector_dim as the backend; "
                f"got {m.shape[-1]} and {self.backend.vector_dim}"
            )

        return self.backend.bind(
            a=m,                        # (vector_dim,)
            b=self.inv_position_vectors # (batch_size, vector_dim)
        )

    def _create_inv_position_vectors(self) -> Tensor:
        """
        Create inverse positional encoding hypervectors for the stored positions.

        This method first applies the backend's positional encoding to
        :attr:`positions` to obtain positional hypervectors, then inverts
        those hypervectors using the backend's :meth:`invert` operation.

        Returns
        -------
        Tensor
            A tensor of shape ``(N, vector_dim)`` containing the inverse
            positional hypervectors corresponding to :attr:`positions`.
        """
        pvs, _ = self.backend.positional_encoding(self.positions)
        ipvs, _ = self.backend.invert(pvs)
        return ipvs
