from torch import Tensor
from typing import Tuple

from ...backends.base import BaseBackend
from ..base.base_module import BaseModule


class ValueEncoderModule(BaseModule):
    """
    ValueEncoderModule maps raw values into high-dimensional hypervectors
    using the backend's value-encoding mechanism.

    This module is a thin wrapper around the backend's
    :meth:`value_encoding` implementation, providing a consistent interface for
    turning scalar, categorical, or vector-valued inputs into hypervectors
    that can be used in downstream HyperSpace components.
    """

    def __init__(self, backend: BaseBackend):
        """
        Initialize a ValueEncoderModule.

        Parameters
        ----------
        backend : BaseBackend
            Backend instance that defines the hypervector dimensionality and
            provides the :meth:`value_encoding` operation.

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
        Encode input values into high-dimensional hypervectors.

        This method forwards the input tensor `x` to the backend's
        :meth:`value_encoding` implementation. Inputs may be provided either
        as a single value vector or as a batch of value vectors.

        Parameters
        ----------
        x : Tensor
            Input tensor of values. Must be either:
            - shape ``(D,)`` for a single value vector, or
            - shape ``(N, D)`` for a batch of N value vectors,
            where ``D`` *must equal* ``backend.value_dim``.

        Returns
        -------
        Tensor
            Encoded hypervectors corresponding to the input values.
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
            If `x` is not 1D or 2D, or if the trailing dimension does not
            match ``backend.value_dim``.
        """
        # --------------------------------
        # validate input types and shapes
        # --------------------------------
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x should be a Tensor; got {type(x)}")

        if x.dim() not in (1, 2):
            raise ValueError(
                "Input x must be a 1D or 2D tensor of shape (value_dim) or "
                f"(batch_size, value_dim); got {x.dim()}"
            )

        if x.shape[-1] != self.backend.value_dim:
            raise ValueError(
                f"Expected input value_dim {self.backend.value_dim}; "
                f"got {x.shape[-1]}"
            )

        # TODO: Add additional batched processing optimizations for small devices

        out, info = self.backend.value_encoding(x)

        return out, info
