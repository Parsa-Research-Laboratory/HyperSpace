from torch import Tensor
import torch
import torch.nn as nn
from typing import List, Union

from .base_model import BaseRegressionModel


class DenseLinearModel(BaseRegressionModel):
    """
    Fully-connected regression model with an optional stack of hidden layers.

    The model supports two configurations:
      - num_layers == 1: a single linear projection from feature_dim to value_dim
      - num_layers > 1: (num_layers - 1) hidden layers followed by an output layer

    Hidden layer widths may be specified as:
      - hidden_size: int (same width for all hidden layers)
      - hidden_size: List[int] (one width per hidden layer; length == num_layers - 1)

    Activations:
      - hidden_act is applied after each hidden layer (when num_layers > 1)
      - output_act is applied after the final output layer if provided
    """

    def __init__(
        self,
        feature_dim: int,
        value_dim: int,
        num_layers: int = 1,
        hidden_size: Union[None, int, List[int]] = None,
        hidden_act: Union[None, nn.Module] = None,
        output_act: Union[None, nn.Module] = None,
    ):
        super().__init__(feature_dim=feature_dim, value_dim=value_dim)

        # ----------------------
        # Individual Validation
        # ----------------------
        if not isinstance(num_layers, int):
            raise TypeError(f"Expected num_layers to be an integer; got {type(num_layers)}")

        if num_layers < 1:
            raise ValueError(f"Expected num_layers to be greater than zero; got {num_layers}")

        if hidden_size is not None and not isinstance(hidden_size, (int, list)):
            raise TypeError(f"Expected hidden_size to be one of [int, or List[int]]; got {type(hidden_size)}")

        if isinstance(hidden_size, int) and hidden_size < 1:
            raise ValueError(f"Expected hidden_size to be greater than zero; got {hidden_size}")

        if isinstance(hidden_size, list):
            if not all(isinstance(x, int) for x in hidden_size):
                types = [type(x) for x in hidden_size if not isinstance(x, int)]
                raise TypeError(f"Expected all elements to be integers; got invalid types: {types}")
            if not all(x > 0 for x in hidden_size):
                raise ValueError(f"Expected all elements to be greater than zero; got {hidden_size}")

        if hidden_act is not None and not isinstance(hidden_act, nn.Module):
            raise TypeError(f"Expected hidden_act to be nn.Module; got {type(hidden_act)}")

        if output_act is not None and not isinstance(output_act, nn.Module):
            raise TypeError(f"Expected output_act to be a nn.Module; got {type(output_act)}")

        # -------------------------
        # Combinatorial Validation
        # -------------------------
        if num_layers == 1 and hidden_size is not None:
            raise TypeError(
                "hidden_size must be None when num_layers == 1 (no hidden layers). "
                "Either set hidden_size=None or set num_layers > 1."
            )

        if num_layers == 1 and hidden_act is not None:
            raise TypeError(
                "hidden_act must be None when num_layers == 1 (no hidden layers). "
                "Either set hidden_act=None or set num_layers > 1."
            )

        if num_layers > 1 and hidden_size is None:
            raise TypeError(
                "hidden_size is required when num_layers > 1 (hidden layers present). "
                "Provide an int hidden_size, or a list of int sizes of length num_layers - 1."
            )

        if num_layers > 1 and hidden_act is None:
            raise TypeError(
                "hidden_act is required when num_layers > 1 (hidden layers present). "
                "Provide a hidden activation (e.g., a callable/nn.Module) or a list of "
                "activations of length num_layers - 1."
            )

        if isinstance(hidden_size, list) and len(hidden_size) != (num_layers - 1):
            raise ValueError(
                f"When hidden_size is a list, it must have length num_layers - 1 "
                f"(one entry per hidden layer). Got len(hidden_size)={len(hidden_size)} "
                f"but expected {num_layers - 1} (num_layers={num_layers})."
            )

        # ----------------------
        # Store config
        # ----------------------
        self.num_layers: int = num_layers
        self.hidden_size: Union[None, int, List[int]] = hidden_size
        self.hidden_act: Union[None, nn.Module] = hidden_act
        self.output_act: Union[None, nn.Module] = output_act

        # ----------------------
        # Build network
        # ----------------------
        self.net: nn.Module = self._build_network()

    def _build_network(self) -> nn.Module:
        """
        Construct the underlying torch module graph based on validated arguments.
        """
        layers: List[nn.Module] = []

        # num_layers == 1: single linear projection
        if self.num_layers == 1:
            layers.append(nn.Linear(self.feature_dim, self.value_dim))
            if self.output_act is not None:
                layers.append(self.output_act)
            return nn.Sequential(*layers)

        # num_layers > 1: hidden stack + output layer
        # Determine hidden widths per hidden layer (count == num_layers - 1)
        if isinstance(self.hidden_size, int):
            hidden_widths = [self.hidden_size] * (self.num_layers - 1)
        else:
            # mypy: validated earlier when num_layers > 1, hidden_size is not None
            hidden_widths = list(self.hidden_size)  # type: ignore[arg-type]

        in_dim = self.feature_dim

        # Hidden layers
        for hdim in hidden_widths:
            layers.append(nn.Linear(in_dim, hdim))
            # Note: you can optionally add Dropout/LayerNorm here later.
            layers.append(self.hidden_act)  # type: ignore[arg-type]
            in_dim = hdim

        # Output layer
        layers.append(nn.Linear(in_dim, self.value_dim))
        if self.output_act is not None:
            layers.append(self.output_act)

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the model to an input tensor.

        Arguments:
            x : Tensor
                Input tensor of shape (feature_dim,) or (batch_size, feature_dim).

        Returns:
            Tensor
                Output tensor of shape (value_dim,) or (batch_size, value_dim).
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor; got {type(x)}")

        if x.ndim not in (1, 2):
            raise ValueError(
                f"Expected x to be a 1D or 2D tensor with shape (feature_dim,) "
                f"or (batch_size, feature_dim); got shape {tuple(x.shape)}"
            )

        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected last dimension of x to match feature_dim={self.feature_dim}; "
                f"got {x.shape[-1]}"
            )

        return self.net(x)
