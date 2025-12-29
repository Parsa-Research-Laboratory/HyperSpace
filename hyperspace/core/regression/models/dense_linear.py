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
        
        if hidden_act is not None and not isinstance(hidden_act, nn.Module):
            raise TypeError(f"Expected hidden_act to be nn.Module; got {type(hidden_act)}")
        
        if output_act is not None and not isinstance(output_act, nn.Module):
            raise TypeError(f"Expected output_act to be a nn.Module; got {type(output_act)}")

        # -------------------------
        # Combinatorial Validation
        # -------------------------

        # TODO

        self.num_layers: int = num_layers
        self.hidden_size: Union[None, int, List[int]] = hidden_size
        self.hidden_act: Union[None, nn.Module] = hidden_act
        self.output_act: Union[None, nn.Module] = output_act

    def forward(self, x: Tensor) -> Tensor:
        """
        TODO Finish Documentation
        """
        raise NotImplementedError