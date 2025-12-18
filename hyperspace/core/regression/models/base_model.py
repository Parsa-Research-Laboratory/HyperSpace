from torch import Tensor
import torch.nn as nn

class BaseRegressionModel(nn.Module):
    """
    Abstract base class (template) for regression models in HyperSpace.

    This class serves as a template for implementing custom regression models
    that can be used with the RegressionModule. All custom regression models
    should inherit from this class and implement the required abstract methods.

    The base class provides:
        - Input validation for feature dimensions
        - Standard initialization structure
        - Interface contract via abstract forward method

    Attributes:
        feature_dim : int
            The dimensionality of input features (vector_dim from the backend).

    Abstract Methods:
        forward(x: Tensor) -> Tensor
            Must be implemented by subclasses to define the forward pass.

    Example:
        Creating a custom regression model from this template::

            class CustomRegressionModel(BaseRegressionModel):
                def __init__(self, feature_dim: int, output_dim: int = 1):
                    super().__init__(feature_dim)
                    
                    # Define your custom layers
                    self.fc1 = nn.Linear(feature_dim, 128)
                    self.fc2 = nn.Linear(128, output_dim)
                    self.activation = nn.ReLU()
                
                def forward(self, x: Tensor) -> Tensor:
                    # Implement your custom forward pass
                    x = self.activation(self.fc1(x))
                    x = self.fc2(x)
                    return x

    Notes:
        - Subclasses must call super().__init__(feature_dim) in their constructor
        - The forward method must accept a Tensor and return a Tensor
        - Input tensors should have shape (..., feature_dim)
        - This class cannot be instantiated directly; use a concrete implementation
    """
    def __init__(self, feature_dim: int) -> None:
        """
        Initialize the base regression model with input validation.

        This constructor performs validation on the feature dimension and
        sets up the base nn.Module. Subclasses should call this via super()
        before initializing their own layers.

        Arguments:
            feature_dim : int
                The dimensionality of input features. Must be a positive integer
                greater than or equal to 1. This typically corresponds to the
                vector_dim of the HyperSpace backend.

        Raises:
            TypeError
                If feature_dim is not an integer.
            ValueError
                If feature_dim is less than 1.

        Example:
            In a subclass::

                def __init__(self, feature_dim: int, custom_param: int):
                    super().__init__(feature_dim)
                    # Initialize your custom layers here
                    self.custom_layer = nn.Linear(feature_dim, custom_param)
        """
        super().__init__(BaseRegressionModel)

        # -------------------
        # Validate Arguments
        # -------------------
        if not isinstance(feature_dim, int):
            raise TypeError(f"Expected feature_dim to be an integer; got {type(feature_dim)}")
        
        if feature_dim < 1:
            raise ValueError(f"Expected feature_dim to be >= 1; got {feature_dim}")
        
        # ---------------------
        # Set class attributes
        # ---------------------
        self.feature_dim: int = feature_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Abstract forward pass method - must be implemented by subclasses.

        This method defines the computation performed at every call and must
        be overridden by all subclasses to implement the specific regression
        model's forward pass logic.

        Arguments:
            x : Tensor
                Input tensor of shape (..., feature_dim) where the last
                dimension must match self.feature_dim.

        Returns:
            Tensor
                Output tensor from the regression model. The shape depends
                on the specific implementation but typically is (..., output_dim).

        Raises:
            NotImplementedError
                Always raised if called on the base class directly. Subclasses
                must provide their own implementation.

        Example:
            Implementation in a subclass::

                def forward(self, x: Tensor) -> Tensor:
                    # Validate input shape if needed
                    if x.shape[-1] != self.feature_dim:
                        raise ValueError(f"Expected input dim {self.feature_dim}, got {x.shape[-1]}")
                    
                    # Perform forward computation
                    x = self.layer1(x)
                    x = self.activation(x)
                    x = self.layer2(x)
                    return x
        """
        raise NotImplementedError
