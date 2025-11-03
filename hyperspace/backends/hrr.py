
import numpy as np
import torch
from torch import Tensor

from .base import BaseBackend

class HRRBackend(BaseBackend):
    """
    Holographic Reduced Representations (HRR) backend implementation.

    Implements the continuous encoding, binding, and bundling operations
    as defined in the HyperSpace paper using HRR principles.
    """
    def __init__(self, vector_dim: int, device: str = "cpu"):
        super().__init__(vector_dim, device)
        self.name = "HRR"

    def create_random_vector(self, eps: float = 1e-3) -> Tensor:
        """
        Create a random vector of dimension self.vectorD.

        Arguments:
            eps : float
                Small value to avoid extreme angles in frequency domain.

        Returns:
            Tensor
                A random vector of dimension self.vectorD.
        """

        a = torch.rand((self.vector_dim - 1) // 2)
        sign = np.random.choice((-1, +1), len(a))
        
        sign = torch.from_numpy(sign).to(self.device)
        a = a.to(self.device)

        phi = sign * torch.pi * (eps + a * (1 - 2 * eps))

        if not torch.all(torch.abs(phi) >= torch.pi * eps):
            raise ValueError("Generated phi values are out of bounds (lower).")
        if not torch.all(torch.abs(phi) <= torch.pi * (1 - eps)):
            raise ValueError("Generated phi values are out of bounds (upper).")

        fv = torch.zeros(self.vector_dim, dtype=torch.complex64, device=self.device)
        fv[0] = 1
        fv[1:(self.vector_dim + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
        fv[(self.vector_dim // 2) + 1:] = torch.flip(torch.conj(fv[1:(self.vector_dim + 1) // 2]), dims=[0])

        if self.vector_dim % 2 == 0:
            fv[self.vector_dim // 2] = 1

        if not torch.allclose(torch.abs(fv), torch.ones(fv.shape, device=self.device)):
            raise ValueError("Generated frequency vector is not unit magnitude.")

        v = torch.fft.ifft(fv)
        v = v.real
        v = v.to(self.device)

        if not torch.allclose(torch.fft.fft(v), fv):
            raise ValueError("Inverse FFT did not produce the expected frequency vector.")

        if not torch.allclose(torch.linalg.norm(v), torch.ones(v.shape, device=self.device)):
            raise ValueError("Inverse FFT did not produce the expected norm.")

        return v
    
    def continuous_encoding(self, basis: Tensor, x: Tensor) -> Tensor:
        """
        Continuous encoding method for HRR backend.
        """
        return super().continuous_encoding(basis, x)
    
    def bind(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Binding operation for HRR backend using circular convolution.
        """
        return super().bind(a, b)

    def bundle(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Bundling operation for HRR backend using vector addition.
        """
        return super().bundle(a, b)
    
    def similarity(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Similarity operation for HRR backend using cosine similarity.
        """
        return super().similarity(a, b)
    
    def normalize(self, tensor: Tensor) -> Tensor:
        """
        Normalize the input tensor.
        """
        return super().normalize(tensor)
    
    def invert(self, tensor: Tensor) -> Tensor:
        """
        Invert the input tensor.
        """
        return super().invert(tensor)
    
    def weight(self, tensor: Tensor, weight: float) -> Tensor:
        """
        Apply weighting to the input tensor using the specified method.
        """
        return super().weight(tensor, weight)
    
    def initialize_env_basis_vectors(self, env_dim: int) -> None:
        """
        Initialize environment basis vectors for HRR backend.

        Arguments:
            env_dim : int
                The dimensionality of the environment to encode.
        """
        if env_dim < 1:
            raise ValueError("env_dim must be at least 1.")
        
        if not isinstance(env_dim, int):
            env_dim_new = int(env_dim)
            print(f"Warning: env_dim {env_dim} is not an integer. Converting to {env_dim_new}.")
            env_dim = env_dim_new

        for _ in range(env_dim):
            self.env_basis_vectors.append(self.create_random_vector())

    def initialize_value_basis_vectors(self, value_dim: int) -> None:
        """
        Initialize value basis vectors for HRR backend.

        Arguments:
            value_dim : int
                The dimensionality of the values to encode.
        """
        if value_dim < 1:
            raise ValueError("value_dim must be at least 1.")
        
        if not isinstance(value_dim, int):
            value_dim_new = int(value_dim)
            print(f"Warning: value_dim {value_dim} is not an integer. Converting to {value_dim_new}.")
            value_dim = value_dim_new

        for _ in range(value_dim):
            self.value_basis_vectors.append(self.create_random_vector())
    
    def _nearest_neighbor_regression(self, vectors: Tensor) -> Tensor:
        """
        Nearest neighbor regression for HRR backend.
        """
        return super()._nearest_neighbor_regression(vectors)
    
    def _neural_network_regression(self, vectors: Tensor) -> Tensor:
        """
        Neural network regression for HRR backend.
        """
        return super()._neural_network_regression(vectors)
    
    def _resonator_cleanup(self, tensor: Tensor) -> Tensor:
        """
        Resonator cleanup for HRR backend.
        """
        return super()._resonator_cleanup(tensor)
    
    def _hopfield_cleanup(self, tensor: Tensor) -> Tensor:
        """
        Hopfield cleanup for HRR backend.
        """
        return super()._hopfield_cleanup(tensor)
