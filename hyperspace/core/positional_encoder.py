import torch
from torch import Tensor
from typing import Tuple

from ..backends.base import BaseBackend
from .base_module import BaseModule


class PositionalEncoderModule(BaseModule):
    """
    PositionalEncoderModule maps Euclidean input coordinates into
    high-dimensional hypervectors using the backend's positional encoding.

    This module is a thin wrapper around the backend's
    :meth:`positional_encoding` implementation, providing a consistent
    interface for turning positions (e.g., spatial coordinates) into
    hypervectors that can be used in downstream HyperSpace components.
    """

    def __init__(self, backend: BaseBackend):
        """
        Initialize a PositionalEncoderModule.

        Parameters
        ----------
        backend : BaseBackend
            Backend instance that defines the hypervector dimensionality and
            provides the :meth:`positional_encoding` operation.

        Raises
        ------
        TypeError
            If `backend` does not extend :class:`BaseBackend`.
        """
        super().__init__()
        self.backend: BaseBackend = backend

        if not isinstance(backend, BaseBackend):
            raise TypeError(
                f"Expected the argued backend to extend the BaseBackend class; "
                f"got {type(self.backend)}"
            )

    def __call__(self, x: Tensor) -> Tuple[Tensor, dict]:
        """
        Encode input positions into high-dimensional hypervectors.

        This method forwards the input tensor `x` to the backend's
        positional encoding implementation. Inputs may be either a 
        single position vector or a batch of positions.

        Parameters
        ----------
        x : Tensor
            Input tensor of positions. Must be:
            - shape ``(D,)`` for a single position, or
            - shape ``(N, D)`` for a batch of N positions,
            where ``D`` *must equal* ``backend.value_dim``.

        Returns
        -------
        Tensor
            Encoded hypervectors corresponding to the input positions.
            Shape mirrors the input:
            - ``(vector_dim,)`` for a single input
            - ``(N, vector_dim)`` for a batch
        dict
            Additional information returned by the backend (may be empty).

        Raises
        ------
        TypeError
            If `x` is not a :class:`torch.Tensor`.

        ValueError
            If `x` is not 1D or 2D,
            or if the trailing dimension does not match ``backend.env_dim``.
        """
        # --------------------------------
        # validate input types
        # --------------------------------
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x should be a Tensor; got {type(x)}")

        if x.dim() not in (1, 2):
            raise ValueError(
                "Input x must be a 1D or 2D tensor of shape (env_dim) or "
                f"(batch_size, env_dim); got {x.dim()}"
            )

        # --------------------------------
        # validate spatial / Euclidean dimensionality
        # --------------------------------
        if x.shape[-1] != self.backend.env_dim:
            raise ValueError(
                f"Expected input dimensionality {self.backend.env_dim}; "
                f"got {x.shape[-1]}"
            )

        # TODO: Add batched processing for embedded / smaller devices

        out, info = self.backend.positional_encoding(x)

        return out, info
