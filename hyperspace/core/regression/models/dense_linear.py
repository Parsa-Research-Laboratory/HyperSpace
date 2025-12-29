from torch import Tensor
import torch.nn as nn
from typing import List, Union

from .base_model import BaseRegressionModel

class DenseLinearModel(BaseRegressionModel):
    """
    TODO Finish Documentation
    """
    def __init__(self, feature_dim: int, value_dim: int,
                 num_layers: int = 1,
                 hidden_size: Union[None, int, List[int]] = None,
                 hidden_act: Union[None, nn.Module] = None,
                 output_act: Union[None, nn.Module] = None):
        """
        TODO Finish Documentation
        """
        super().__init__(
            feature_dim=feature_dim,
            value_dim=value_dim
        )

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
            # check the types of all elements
            if not all([isinstance(x, int) for x in hidden_size]):
                types = [type(x) for x in hidden_size]
                raise TypeError(f"Expected all elements to be integers; got {types}")

            # check the values of all elements
            if not all([x > 0 for x in hidden_size]):
                raise ValueError(f"Expected all elements to be greater than zero; got {hidden_size}")

        if hidden_act is not None and not isinstance(hidden_act, nn.Module):
            raise TypeError(f"Expected hidden_act to be nn.Module; got {type(hidden_act)}")
        
        if output_act is not None and not isinstance(output_act, nn.Module):
            raise TypeError(f"Expected output_act to be a nn.Module; got {type(output_act)}")

        # num_layers == 1 => no hidden layers, so these must be omitted
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

        # num_layers > 1 => hidden layers exist, so these are required
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

        # Per-hidden-layer sizes: one size per hidden layer => length == num_layers - 1
        if isinstance(hidden_size, list) and len(hidden_size) != (num_layers - 1):
            raise ValueError(
                f"When hidden_size is a list, it must have length num_layers - 1 "
                f"(one entry per hidden layer). Got len(hidden_size)={len(hidden_size)} "
                f"but expected {num_layers - 1} (num_layers={num_layers})."
            )

        self.num_layers: int = num_layers
        self.hidden_size: Union[None, int, List[int]] = hidden_size
        self.hidden_act: Union[None, nn.Module] = hidden_act
        self.output_act: Union[None, nn.Module] = output_act

    def forward(self, x: Tensor) -> Tensor:
        """
        TODO Finish Documentation
        """
        raise NotImplementedError