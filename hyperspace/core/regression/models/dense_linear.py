from torch import Tensor

from .base_model import BaseRegressionModel

class DenseLinearModel(BaseRegressionModel):
    """
    TODO Finish Documentation
    """
    def __init__(self, feature_dim: int, value_dim: int):
        """
        TODO Finish Documentation
        """
        super().__init__(
            feature_dim=feature_dim,
            value_dim=value_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        TODO Finish Documentation
        """
        raise NotImplementedError