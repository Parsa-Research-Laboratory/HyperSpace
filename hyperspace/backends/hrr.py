
import numpy as np
import torch
from torch import device, Generator, Tensor
from typing import Tuple

from .base import BaseBackend

def _base_create_single_vector(
        vector_dim: int,
        gen: Generator,
        eps: float = 1e-3,
        dev: device = torch.device("cpu")
    ) -> Tensor:
    """
    Generate a randomly initialized HRR

    Arguments:
    ----------
        vector_dim: int
            The size of the vector to generate
        gen: torch.Generator
            A PyTorch generator to control the stochasticity of
            the random process
        eps: float
            A threshold for the upper and lower bounds of the
            fourier coefficients
        dev: torch.Device
            The device to generate the vector on

    Returns:
    --------
        v: torch.Tensor
            The randomly generated vector with shape (vector_dim). It
            should be noted this vector is returned the time domain.
    """
    if not isinstance(vector_dim, int):
        raise TypeError(f"vector_dim should be an integer; got {type(vector_dim)}")

    if vector_dim <= 0:
        raise ValueError(f"vector_dim should be > 0; got {vector_dim}")
    
    if not isinstance(gen, Generator):
        raise TypeError(f"gen should be a torch.Generator; got {type(gen)}")
    
    if not isinstance(eps, float):
        raise TypeError(f"eps should be a float; got {type(eps)}")

    if eps <= 0:
        raise ValueError(f"eps should be > 0; got {eps}")
    
    if not isinstance(dev, device):
        raise TypeError(f"dev should be a valid torch.device; got {type(dev)}")
    
    if dev != gen.device:
        raise AttributeError(f"the generator should be on the same device; got {gen.device} and {dev}")
    
    a = torch.rand((vector_dim - 1) // 2, generator=gen)
    sign = np.random.choice((-1, +1), len(a))
    
    sign = torch.from_numpy(sign).to(dev)
    a = a.to(dev)

    phi = sign * torch.pi * (eps + a * (1 - 2 * eps))

    if not torch.all(torch.abs(phi) >= torch.pi * eps):
        raise ValueError("Generated phi values are out of bounds (lower).")
    if not torch.all(torch.abs(phi) <= torch.pi * (1 - eps)):
        raise ValueError("Generated phi values are out of bounds (upper).")

    fv = torch.zeros(vector_dim, dtype=torch.complex64, device=dev)
    fv[0] = 1
    fv[1:(vector_dim + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
    fv[(vector_dim // 2) + 1:] = torch.flip(
        torch.conj(fv[1:(vector_dim + 1) // 2]), dims=[0]
    )

    if vector_dim % 2 == 0:
        fv[vector_dim // 2] = 1

    if not torch.allclose(torch.abs(fv), torch.ones(fv.shape, device=dev)):
        raise ValueError("Generated frequency vector is not unit magnitude.")

    v = torch.fft.ifft(fv)
    v = v.real
    v = v.to(dev)

    if not torch.allclose(torch.fft.fft(v), fv):
        raise ValueError("Inverse FFT did not produce the expected frequency vector.")

    if not torch.allclose(torch.linalg.norm(v), torch.ones(v.shape, device=dev)):
        raise ValueError("Inverse FFT did not produce the expected norm.")
    
    return v

def _base_single_bind(v1: Tensor, v2: Tensor) -> Tensor:
    """
    Bind two HRR vectors together

    Arguments:
    ----------
    1) v1: Tensor
        The first vector to bind together
    2) v2: Tensor
        The second vector to bind together

    Returns:
    --------
    1) v_out: Tensor
        The binded vector
    """

    if not isinstance(v1, Tensor):
        raise TypeError(f"expected v1 to be a Tensor; got {type(v1)}")
    
    if not isinstance(v2, Tensor):
        raise TypeError(f"expected v2 to be a Tensor; got {type(v2)}")
    
    if v1.shape != v2.shape:
        raise ValueError(f"expected v1 and v2 to have the same shape; got {v1.shape} and {v2.shape}")
    
    if len(v1.shape) != 1:
        raise ValueError(f"expected v1 to be a 1d vector; got {v1.shape}")
    
    v1_fft = torch.fft.fft(v1)
    v2_fft = torch.fft.fft(v2)
    v_out_fft = v1_fft * v2_fft
    v_out = torch.fft.ifft(v_out_fft).real

    return v_out

def _base_batch_bind(v1: Tensor, v2: Tensor) -> Tensor:
    """
    Batched HRR binding via FFT.

    v1: (B, D)
    v2: (B, D)
    returns: (B, D)
    """
    if not isinstance(v1, Tensor):
        raise TypeError(f"expected v1 to be a Tensor; got {type(v1)}")
    if not isinstance(v2, Tensor):
        raise TypeError(f"expected v2 to be a Tensor; got {type(v2)}")

    if v1.shape != v2.shape:
        raise ValueError(f"expected v1 and v2 to have the same shape; got {v1.shape} and {v2.shape}")
    if v1.ndim != 2:
        raise ValueError(f"expected v1 to be (B, D); got {v1.shape}")

    # FFT along the last dimension (D), broadcast across batch (B)
    v1_fft = torch.fft.fft(v1, dim=-1)
    v2_fft = torch.fft.fft(v2, dim=-1)

    v_out_fft = v1_fft * v2_fft
    v_out = torch.fft.ifft(v_out_fft, dim=-1).real

    return v_out

def _base_single_bundle(v1: Tensor, v2: Tensor) -> Tensor:
    """
    Bundle (superpose) two HRR vectors together via elementwise addition.

    Arguments:
    ----------
    1) v1: Tensor
        The first vector to bundle
    2) v2: Tensor
        The second vector to bundle

    Returns:
    --------
    1) v_out: Tensor
        The bundled vector
    """
    if not isinstance(v1, Tensor):
        raise TypeError(f"expected v1 to be a Tensor; got {type(v1)}")
    
    if not isinstance(v2, Tensor):
        raise TypeError(f"expected v2 to be a Tensor; got {type(v2)}")
    
    if v1.shape != v2.shape:
        raise ValueError(f"expected v1 and v2 to have the same shape; got {v1.shape} and {v2.shape}")
    
    if v1.ndim != 1:
        raise ValueError(f"expected v1 to be a 1d vector; got {v1.shape}")

    v_out = v1 + v2
    return v_out

def _base_batch_bundle(v1: Tensor, v2: Tensor) -> Tensor:
    """
    Batched HRR bundling via elementwise addition.

    v1: (B, D)
    v2: (B, D)
    returns: (B, D)
    """
    if not isinstance(v1, Tensor):
        raise TypeError(f"expected v1 to be a Tensor; got {type(v1)}")
    if not isinstance(v2, Tensor):
        raise TypeError(f"expected v2 to be a Tensor; got {type(v2)}")

    if v1.shape != v2.shape:
        raise ValueError(f"expected v1 and v2 to have the same shape; got {v1.shape} and {v2.shape}")
    
    if v1.ndim != 2:
        raise ValueError(f"expected v1 to be (B, D); got {v1.shape}")

    v_out = v1 + v2
    return v_out

def _base_single_fpe(basis: Tensor, power: float, length_scale: float) -> Tensor:
    """
    Single HRR fractional power encoding

    Arguments:
        basis: Tensor
            the random vector representing the basis of encoding; shape = (D)
        power: float
            The value to exponentiate the basis
        length_scale: float
            Adjust the kernel with between locations
    
    Returns:
        v_out: Tensor
            The fractional power encoded vector
    """

    if not isinstance(basis, Tensor):
        raise TypeError(f"expected basis to ba a Tensor; got {type(basis)}")
    
    power = float(power)
    length_scale = float(length_scale)
    
    if not isinstance(power, float):
        raise TypeError(f"expected power to be a float; got {type(power)}")
    
    if not isinstance(length_scale, float):
        raise TypeError(f"expected length_scale to be a float; got {type(length_scale)}")
    
    if basis.ndim != 1:
        raise ValueError(f"expected basis to be (D); got {basis.shape}")
    
    v_out: Tensor = torch.fft.fft(basis)
    v_out = v_out ** (power / length_scale)
    v_out = torch.fft.ifft(v_out).real

    return v_out

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

        v = _base_create_single_vector(
            vector_dim=self.vector_dim,
            gen=self.generator,
            eps=eps,
            dev=self.device
        )

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
