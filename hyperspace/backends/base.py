import torch
from torch import Generator, Tensor
import torch.nn as nn
from typing import List, Tuple

class BaseBackend(nn.Module):
    """
    Abstract base class for HyperSpace backends.
    """
    def __init__(self,
                 name: str = "Base",
                 vector_dim: int = 128,
                 vector_dtype: torch.dtype = torch.float32,
                 device: str = "cpu",
                 seed: int = 42
        ):
        """
        Initialize the backend with the specified vector dimension and device.

        Arguments:
        ----------
        name: str
            A unique identifier for the backend
        vector_dim : int
            Dimension of the vectors to be used in the backend.
        vector_dtype: torch.dtype
            The type of values stored within the hypervectors.
        device : str
            Device to run computations on (e.g., 'cpu' or 'cuda').
        seed: int 
        """
        super().__init__()
        self.name: str = name
        self.vector_dim: int = vector_dim
        self.vector_dtype: torch.dtype = vector_dtype
        self.device = torch.device(device)
        self.generator = torch.Generator(
            device=self.device,
        ).manual_seed(seed)

        self.register_buffer(
            "env_basis_vectors",
            torch.empty(0, self.vector_dim, dtype=self.vector_dtype, device=self.device),
            persistent=False
        )

        self.register_buffer(
            "value_basis_vectors",
            torch.empty(0, self.vector_dim, dtype=self.vector_dtype, device=self.device),
            persistent=False
        )

    def create_random_vector(self) -> torch.Tensor:
        """
        Create a random vector of dimension self.vectorD.

        Returns:
        -------
        torch.Tensor
            A random vector of shape (self.vectorD, ).
        """
        raise NotImplementedError("create_random_vector method must be implemented by subclasses.")

    def continuous_encoding(self, x: Tensor, indexes: Tensor) -> Tuple[Tensor, dict]:
        """
        Abstract definition of the continuous encoding method (\\mathcal{E})
        from the HyperSpace paper.

        Arguments:
        ----------
        x : torch.Tensor
            Continuous value to be encoded. Shape should be (batch_size, ).
    
        indexes : torch.Tensor
            Indexes of the basis vectors to use for encoding. Shape should be (batch_size, ).

        Returns:
        -------
        torch.Tensor
            Encoded representation of the input value. Shape should be (batch_size, vectorD).
        dict
            Information dictionary containing any relevant metadata.
        """
        raise NotImplementedError("continuous_encoding method must be implemented by subclasses.")
    
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Abstract definition of the binding operation (\\otimes)
        from the HyperSpace paper.

        Arguments:
        ----------
        a : torch.Tensor
            First tensor to bind. Shape should be (batch_size, self.vectorD).
        b : torch.Tensor
            Second tensor to bind. Shape should be (batch_size, self.vectorD).
    
        Returns:
        -------
        torch.Tensor
            Result of the binding operation. Shape should be (batch_size, vectorD).
        """
        raise NotImplementedError("binding method must be implemented by subclasses.")
    
    def bundle(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Abstract definition of the bundling operation (\\oplus)
        from the HyperSpace paper.

        Arguments:
        ----------
        a : torch.Tensor
            First tensor to bundle. Shape should be (batch_size, self.vectorD).
        b : torch.Tensor
            Second tensor to bundle. Shape should be (batch_size, self.vectorD).
    
        Returns:
        -------
        torch.Tensor
            Result of the bundling operation. Shape should be (batch_size, vectorD).
        """
        raise NotImplementedError("bundling method must be implemented by subclasses.")
    
    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Abstract definition of the similarity measure
        from the HyperSpace paper.

        Arguments:
        ----------
        a : torch.Tensor
            First tensor for similarity computation. Shape should be (batch_size, self.vectorD).
        b : torch.Tensor
            Second tensor for similarity computation. Shape should be (batch_size, self.vectorD).
    
        Returns:
        -------
        torch.Tensor
            Similarity scores between the two tensors. Shape should be (batch_size, ).
        """
        raise NotImplementedError("similarity method must be implemented by subclasses.")
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Abstract definition of the normalization method
        from the HyperSpace paper.

        Arguments:
        ----------
        tensor : torch.Tensor
            Tensor to be normalized. Shape should be (batch_size, self.vectorD).
    
        Returns:
        -------
        torch.Tensor
            Normalized tensor. Shape should be (batch_size, vectorD).
        """
        raise NotImplementedError("normalize method must be implemented by subclasses.")
    
    def invert(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Abstract definition of the inversion method
        from the HyperSpace paper.

        Arguments:
        ----------
        tensor : torch.Tensor
            Tensor to be inverted. Shape should be (batch_size, self.vectorD).
    
        Returns:
        -------
        torch.Tensor
            Inverted tensor. Shape should be (batch_size, vectorD).
        """
        raise NotImplementedError("invert method must be implemented by subclasses.")
    
    def weight(self, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        """
        Abstract definition of the weighting method
        from the HyperSpace paper.

        Arguments:
        ----------
        tensor : torch.Tensor
            Tensor to be weighted. Shape should be (batch_size, self.vectorD).
        weight : float
            Weighting factor.
    
        Returns:
        -------
        torch.Tensor
            Weighted tensor. Shape should be (batch_size, vectorD).
        """
        raise NotImplementedError("weight method must be implemented by subclasses.")
    
    def regression(self, vectors: torch.Tensor, method: str) -> Tuple[torch.Tensor, float]:
        """
        Abstract definition of the regression method
        from the HyperSpace paper.

        Arguments:
        ----------
        vectors : torch.Tensor
            Input vectors for regression. Shape should be (num_samples, self.vectorD).
        method : str
            Regression method to be used. Could be "nearest_neighbor", "neural_network", etc.
    
        Returns:
        -------
        Tuple[torch.Tensor, float]
            A tuple containing the regression coefficients and the loss value.
        """

        if method == "nearest_neighbor":
            # Apply nearest neighbor regression
            values: torch.Tensor = self._nearest_neighbor_regression(vectors)
        elif method == "neural_network":
            # Apply neural network regression
            values: torch.Tensor = self._neural_network_regression(vectors)
        else:
            raise ValueError(f"Unknown regression method: {method}")

        return values
    
    def cleanup(self, tensor: torch.Tensor, method: str = "resonator") -> torch.Tensor:
        """
        Abstract definition of the cleanup method
        from the HyperSpace paper.

        Arguments:
        ----------
        tensor : torch.Tensor
            Tensor to be cleaned up. Shape should be (batch_size, self.vectorD).
        method : str
            Cleanup method to be used. Default is "resonator". Other options could be
            "identity" or "hopfield".
    
        Returns:
        -------
        torch.Tensor
            Cleaned up tensor. Shape should be (batch_size, vectorD).
        """
        if method == "resonator":
            # Apply resonator cleanup
            cleaned_vectors = self._resonator_cleanup(tensor)
        elif method == "identity":
            # pass through operation
            cleaned_vectors = tensor
        elif method == "hopfield":
            # Apply hopfield network cleanup
            cleaned_vectors = self._hopfield_cleanup(tensor)

        else:
            raise ValueError(f"Unknown cleanup method: {method}")

        return cleaned_vectors
    
    def initialize_env_basis_vectors(self, env_dim: int) -> None:
        """
        Initialize the environment basis vectors for positional encoding.

        Arguments:
        ----------
        env_dim : int
            Dimensionality of the environment.
        """
        raise NotImplementedError("initialize_env_basis_vectors method must be implemented by subclasses.")
    
    def initialize_value_basis_vectors(self, value_dim: int) -> None:
        """
        Initialize the value basis vectors for value encoding.

        Arguments:
        ----------
        value_dim : int
            Dimensionality of the values.
        """
        raise NotImplementedError("initialize_value_basis_vectors method must be implemented by subclasses.")
    
    def _nearest_neighbor_regression(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Private method to perform nearest neighbor regression.

        Arguments:
        ----------
        vectors : torch.Tensor
            Input vectors for regression. Shape should be (num_samples, self.vectorD).

        Returns:
        -------
        torch.Tensor
            Regression coefficients. Shape should be (num_samples, ).
        """
        raise NotImplementedError("_nearest_neighbor_regression method must be implemented by subclasses.")
    
    def _neural_network_regression(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Private method to perform neural network regression.

        Arguments:
        ----------
        vectors : torch.Tensor
            Input vectors for regression. Shape should be (num_samples, self.vectorD).

        Returns:
        -------
        torch.Tensor
            Regression coefficients. Shape should be (num_samples, ).
        """
        raise NotImplementedError("_neural_network_regression method must be implemented by subclasses.")
    
    def _resonator_cleanup(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Private method to perform resonator cleanup.

        Arguments:
        ----------
        tensor : torch.Tensor
            Tensor to be cleaned up. Shape should be (batch_size, self.vectorD).

        Returns:
        -------
        torch.Tensor
            Cleaned up tensor. Shape should be (batch_size, vectorD).
        """
        raise NotImplementedError("_resonator_cleanup method must be implemented by subclasses.")
    
    def _hopfield_cleanup(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Private method to perform hopfield network cleanup.

        Arguments:
        ----------
        tensor : torch.Tensor
            Tensor to be cleaned up. Shape should be (batch_size, self.vectorD).

        Returns:
        -------
        torch.Tensor
            Cleaned up tensor. Shape should be (batch_size, vectorD).
        """
        raise NotImplementedError("_hopfield_cleanup method must be implemented by subclasses.")