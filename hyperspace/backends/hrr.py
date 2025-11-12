
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple

from .base import BaseBackend


def _base_batch_encode_impl(basis_log: Tensor, x_norm: Tensor) -> Tensor:
    """
    Fractional power encoding implementation with all bases
    already in complex log form.

    Arguments:
        basis_log : Tensor
            Logarithm of basis vectors. Shape should be (B,D).
        x_norm : Tensor
            Normalized input values. Shape should be (B,).

    Returns:
        Tensor
            Encoded representations in real domain. Shape
            should be (B,D).
    """
    y = torch.exp(x_norm.unsqueeze(1) * basis_log)
    y = torch.fft.ifft(y, dim=-1)
    return y.real

def _base_batch_bundle_impl(x: Tensor, dim: int) -> Tensor:
    """
    Batch bundling implementation for HRR backend.

    Arguments:
        x : Tensor
            Input tensor to bundle. Shape should be (N, M, D).
        dim : int
            Dimension along which to bundle.
    """
    # simple unweighted superposition along `dim`
    return x.sum(dim=dim)

def _base_batch_bind_impl(x: Tensor, fft_dim: int = 2, bind_dim: int = 1) -> Tensor:
    """
    Batch binding implementation for HRR backend using circular
    convolution.

    Arguments:
        x : Tensor
            Input tensor to bind. Shape should be (N, M, D).
        dim : int
            Dimension along which to bind.

    Returns:
        Tensor
            Bound tensor. Shape should be (N, 1, D).
    """
    x = torch.fft.fft(x, dim=fft_dim)
    x = x.prod(dim=bind_dim, keepdim=True)
    return torch.fft.ifft(x, dim=fft_dim).real

class HRRBackend(BaseBackend):
    """
    Holographic Reduced Representations (HRR) backend implementation.

    Implements the continuous encoding, binding, and bundling operations
    as defined in the HyperSpace paper using HRR principles.
    """
    def __init__(self, vector_dim: int, length_scale: float = 1.0, device: str = "cpu"):
        super().__init__(vector_dim, device)
        self.name = "HRR"
        self.length_scale: float = length_scale

        # Cache inverse length scale (replace division with mul)
        self.register_buffer(
            "_inv_length_scale",
            torch.tensor(1.0 / float(length_scale), dtype=torch.float32),
            persistent=False)
        
        # Cache other buffers
        self.register_buffer(
            "env_log_basis",
            torch.empty(0, self.vector_dim, dtype=torch.complex64, device=self.device),
            persistent=False
        )
        self.register_buffer(
            "value_basis_vectors",
            torch.empty(0, self.vector_dim, dtype=torch.complex64, device=self.device),
            persistent=False
        )
        self.register_buffer(
            "value_log_basis",
            torch.empty(0, self.vector_dim, dtype=torch.complex64, device=self.device),
            persistent=False
        )

        self._base_batch_encode_impl = torch.compile(
            _base_batch_encode_impl,
            mode="reduce-overhead",
            fullgraph=False
        )
        self._base_batch_bundle_impl = torch.compile(
            _base_batch_bundle_impl,
            mode="reduce-overhead",
            fullgraph=False
        )
        self._base_batch_bind_impl = torch.compile(
            _base_batch_bind_impl,
            mode="reduce-overhead",
            fullgraph=False
        )

    @torch.inference_mode()
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
    
    @torch.inference_mode()
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

        # --------------------------------
        # validate input shapes
        # --------------------------------
        if x.dim() != 1:
            raise ValueError("Input x must be a 1D tensor of shape (batch_size, ).")
        if indexes.dim() != 1:
            raise ValueError("Input indexes must be a 1D tensor of shape (batch_size, ).")
        if x.shape[0] != indexes.shape[0]:
            raise ValueError("Input x and indexes must have the same batch size.")
        
        # ------------------------------------------------
        # perform fractional power encoding of each value
        # ------------------------------------------------
        bases_log = self.env_log_basis.index_select(0, indexes)

        assert not torch.isnan(bases_log).any(), "NaN detected"

        x_norm = (x * self._inv_length_scale)

        assert not torch.isnan(x_norm).any(), "NaN detected"
        encoded = self._base_batch_encode_impl(bases_log, x_norm)
        assert not torch.isnan(encoded).any(), "NaN detected"

        if encoded.shape != (x.shape[0], self.vector_dim):
            raise ValueError(f"Encoded output has incorrect shape: {encoded.shape}")

        info_dict = {
            "original_values": x,
            "indexes": indexes,
        }

        return encoded, info_dict

    @torch.inference_mode()
    def bind(self, a: Tensor) -> Tuple[Tensor, dict]:
        """
        Binding operation for HRR backend using circular convolution. The function
        expects 3-dimensional tensors for batch processing. The function also expects
        the input tensors to have the same shape. The binding is performed along the
        dimension and we are assuming the vectors are in the complex domain.

        Arguments:
            a : Tensor
                Input tensor to bind. Shape should be (N, M, D).

        Returns:
            Tensor
                Bound tensor. Shape should be (N, D).
            dict
                Information dictionary.
        """

        if a.ndim != 3 :
            raise ValueError("Input tensor must be 3-dimensional for binding.")
        
        a_hat = self._base_batch_bind_impl(a)

        info_dict = {}

        return a_hat, info_dict

    def bundle(self, a: Tensor, dim: int = 1) -> Tensor:
        """
        Bundling operation for HRR backend using vector addition.
        """

        if a.ndim != 3:
            raise ValueError("Input tensor a must be 3-dimensional for bundling.")
        
        if a.shape[dim] < 1:
            raise ValueError(f"Cannot bundle along dimension dim={dim} with size less than 1.")
        
        if dim != 1:
            raise ValueError("Currently, only bundling along dimension 1 is supported.")

        if a.dim() < 3:
            raise ValueError("Input tensor a must have at least 2 dimensions for bundling.")

        bundle = self._base_batch_bundle_impl(a, dim)

        print(f"Bundled tensor shape: {bundle.shape}")
    
        if bundle.shape != (a.shape[0], a.shape[2]):
            raise ValueError(f"Bundled output has incorrect shape: {bundle.shape}")

        info_dict = {}

        return bundle, info_dict

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

        # Build a fresh basis locally (avoid touching buffers until ready)
        rows = []
        log_rows = []
        for _ in range(env_dim):
            v = self.create_random_vector()        # (D,)
            v_f = torch.fft.fft(v)                 # HRR often stores basis in freq; if you want time-domain, remove this
            # v_f = torch.clamp(v_f, min=1e-7)  # avoid log(0)
            rows.append(v)                       # ensure real (HRR base vectors are real in time; freq mag=1)
            log_rows.append(torch.log(v_f))

        env = torch.stack(rows).to(self.device)    # (env_dim, D)
        env = env.contiguous()

        log_env = torch.stack(log_rows).to(self.device)    # env_log = torch.clamp(log_env, min=-20.0)  # avoid extreme logs
        log_env = log_env.contiguous()

        assert not torch.isnan(env).any(), "NaN detected"

        # Update buffers IN-PLACE
        self.env_basis_vectors.resize_(env.shape).copy_(env)

        assert not torch.isnan(self.env_basis_vectors).any(), "NaN detected"

        self.env_log_basis.resize_(env.shape).copy_(log_env)

        assert not torch.isnan(self.env_log_basis).any(), "NaN detected"

    @torch.no_grad()
    def initialize_value_basis_vectors(self, value_dim: int) -> None:
        """
        Idempotent initializer for value basis vectors.
        Populates buffers in-place: value_basis_vectors (value_dim, D)
        and value_log_basis_fp16 (value_dim, D in fp16).
        """
        if not isinstance(value_dim, int):
            value_dim_new = int(value_dim)
            print(f"Warning: value_dim {value_dim} is not an integer. Converting to {value_dim_new}.")
            value_dim = value_dim_new
        if value_dim < 1:
            raise ValueError("value_dim must be at least 1.")

        # Build locally first
        rows = [self.create_random_vector() for _ in range(value_dim)]  # each (D,)
        rows = [torch.fft.fft(v) for v in rows] 
        vals = torch.stack(rows).to(self.device).contiguous()           # (value_dim, D)

        # Update buffers in-place (no re-register)
        self.value_basis_vectors.resize_(vals.shape).copy_(vals)

        # Precompute logs (clamp to keep log finite), store compact
        v_log = torch.log(self.value_basis_vectors)      # fp32 compute
        self.value_log_basis.resize_(v_log.shape).copy_(v_log)

    
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
