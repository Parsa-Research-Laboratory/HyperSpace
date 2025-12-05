
import numpy as np
import torch
from torch import device, Generator, Tensor
import torch.nn.functional as F
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

def _base_batch_fpe(basis: Tensor, powers: Tensor, length_scale: float) -> Tensor:
    """
    Single HRR fractional power encoding

    Arguments:
        basis: Tensor
            the random vector representing the basis of encoding; shape = (B, D)
        powers: Tensor
            The values to exponentiate the basis; shape = (B)
        length_scale: float
            Adjust the kernel with between locations
    
    Returns:
        v_out: Tensor
            The fractional power encoded vectors; shape = (B, D)
    """

    if not isinstance(basis, Tensor):
        raise TypeError(f"expected basis to ba a Tensor; got {type(basis)}")
    
    length_scale = float(length_scale)
    
    if not isinstance(powers, Tensor):
        raise TypeError(f"expected powers to be a Tensor; got {type(powers)}")
    
    if not isinstance(length_scale, float):
        raise TypeError(f"expected length_scale to be a float; got {type(length_scale)}")
    
    if basis.ndim != 2:
        raise ValueError(f"expected basis to be (B, D); got {basis.shape}")
    
    if powers.ndim != 1:
        raise ValueError(f"expected powers to be (B); got {powers.shape}")
    
    if basis.shape[0] != powers.shape[0]:
        raise ValueError(f"expected batch size of bases and power to match; got {basis.shape[0]} and {powers.shape[0]}")

    v_out: Tensor = torch.fft.fft(basis, dim=-1)
    v_out = v_out ** (powers / length_scale).unsqueeze(-1)
    v_out = torch.fft.ifft(v_out, dim=-1).real

    return v_out

def _base_single_value_encoding(x: Tensor, basis: Tensor, length_scale: float) -> Tensor:
    """
    Single HRR value encoding with multiple dimensions

    Arguments:
        x: Tensor
            The n-dimensional position to be encoded in a single hypervector;
            Shape = (value_dim)
        basis: Tensor
            The axis vectors for each dimension of x;
            Shape = (value_dim, vector_dim)
        length_scale: float
            The width of the kernel induced upon similarity

    Returns:
        v_out: Tensor
            The n-dimensional positon encoded as a vector
    """

    if not isinstance(x, Tensor):
        raise TypeError(f"expected x to be a Tensor; got {type(x)}")

    if not isinstance(basis, Tensor):
        raise TypeError(f"expected basis to ba a Tensor; got {type(basis)}")
    
    length_scale = float(length_scale)
    
    if not isinstance(length_scale, float):
        raise TypeError(f"expected length_scale to be a float; got {type(length_scale)}")
    
    if x.ndim != 1:
        raise ValueError(f"expected x to be (value_dim); got {x.shape}")

    if basis.ndim != 2:
        raise ValueError(f"expected basis to be (value_dim, vector_dim); got {basis.shape}")
    
    if x.shape[0] != basis.shape[0]:
        raise ValueError(f"Expected the ")
    
    basis_fft = torch.fft.fft(basis, dim=-1)
    v_out = basis_fft ** (x / length_scale).unsqueeze(-1)
    v_out = torch.prod(v_out, dim=0)
    v_out = torch.fft.ifft(v_out).real
    
    return v_out

def _base_batch_value_encoding(x: Tensor, basis: Tensor, length_scale: float) -> Tensor:
    """
    Batched HRR value encoding with multiple dimensions using bundling.
    
    Efficiently encodes multiple n-dimensional values into hypervectors by
    applying fractional power encoding (FPE) to each dimension's basis vector,
    then bundling them together via vector addition. All batches are processed
    in parallel.

    Arguments:
        x: Tensor
            Batched n-dimensional values to encode.
            Shape: (batch_size, value_dim)
        basis: Tensor
            The basis vectors for each dimension.
            Shape: (value_dim, vector_dim)
        length_scale: float
            Kernel width parameter controlling similarity decay.

    Returns:
        v_out: Tensor
            The encoded hypervectors.
            Shape: (batch_size, vector_dim)
    
    Example:
        >>> x_batch = torch.randn(32, 3)  # 32 samples of 3D values
        >>> basis = torch.randn(3, 256)   # 3 basis vectors of dim 256
        >>> encoded = _base_batch_value_encoding(x_batch, basis, length_scale=1.0)
        >>> encoded.shape
        torch.Size([32, 256])
    """
    # Type checking
    if not isinstance(x, Tensor):
        raise TypeError(f"expected x to be a Tensor; got {type(x)}")
    if not isinstance(basis, Tensor):
        raise TypeError(f"expected basis to be a Tensor; got {type(basis)}")
    
    length_scale = float(length_scale)
    if not isinstance(length_scale, float):
        raise TypeError(f"expected length_scale to be a float; got {type(length_scale)}")
    
    # Shape validation - ONLY accept 2D input
    if x.ndim != 2:
        raise ValueError(f"expected x to be (batch_size, value_dim); got {x.shape}")
    if basis.ndim != 2:
        raise ValueError(f"expected basis to be (value_dim, vector_dim); got {basis.shape}")
    
    # Validate value_dim matches
    if x.shape[1] != basis.shape[0]:
        raise ValueError(
            f"Expected x.shape[1] ({x.shape[1]}) to match basis.shape[0] ({basis.shape[0]})"
        )
    
    # Efficient batched computation in frequency domain
    # Step 1: Transform basis to frequency domain once for all batches
    # Shape: (value_dim, vector_dim)
    basis_fft = torch.fft.fft(basis, dim=-1)
    
    # Step 2: Apply FPE for all dimensions and batches
    # Broadcasting: (1, value_dim, vector_dim) ** (batch_size, value_dim, 1)
    # Result: (batch_size, value_dim, vector_dim)
    encoded_fft = basis_fft.unsqueeze(0) ** (x / length_scale).unsqueeze(-1)

    # Step 3: Bundle dimensions via summation for each batch
    # Sum along dim=1 (value_dim): (batch_size, value_dim, vector_dim) -> (batch_size, vector_dim)
    encoded_fft = torch.prod(encoded_fft, dim=1)
    
    # Step 4: Transform back to time domain for all batches
    # Shape: (batch_size, value_dim, vector_dim)
    encoded = torch.fft.ifft(encoded_fft, dim=-1).real
    
    return encoded


class HRRBackend(BaseBackend):
    """
    Holographic Reduced Representations (HRR) backend implementation.

    Implements the continuous encoding, binding, and bundling operations
    as defined in the HyperSpace paper using HRR principles.
    """
    def __init__(self, vector_dim: int, length_scale: float = 1.0, device: str = "cpu",
                 env_dim: int = 1, value_dim: int = 1):
        super().__init__(
            name="HRR",
            vector_dim=vector_dim,
            vector_dtype=torch.float32,
            device=device
        )

        self.length_scale: float = float(length_scale)
        if self.length_scale == 0.0:
            raise ValueError(f"length scale should be non-zero; received {self.length_scale}")
        
        if self.length_scale < 0:
            print(f"WARNING: received negative length scale value of {self.length_scale}")

        self.env_dim: int = int(env_dim)
        if self.env_dim <= 0:
            raise ValueError(f"env dim should be > 1; received {self.env_dim}")

        self.value_dim: int = int(value_dim)
        if self.value_dim <= 0:
            raise ValueError(f"value dim should be > 0; received {self.value_dim}")

        # -----------------------------
        # Compile HRR Specific Methods
        # -----------------------------
        self._comp_create_single_vector = torch.compile(_base_create_single_vector)
        self._comp_single_bind = torch.compile(_base_single_bind)
        self._comp_single_bundle = torch.compile(_base_single_bundle)
        self._comp_single_fpe = torch.compile(_base_single_fpe)
        self._comp_batch_bind = torch.compile(_base_batch_bind)
        self._comp_batch_bundle = torch.compile(_base_batch_bundle)
        self._comp_batch_fpe = torch.compile(_base_batch_fpe)

        # ----------------------------------------
        # Initialize all internal data structures
        # ----------------------------------------
        self.initialize_env_basis_vectors(self.env_dim)
        self.initialize_value_basis_vectors(self.value_dim)

    @torch.inference_mode()
    def create_random_vector(self, eps: float = 1e-3) -> Tensor:
        """
        Create a random vector of dimension D.

        Arguments:
            eps : float 
                Small value to avoid extreme angles in frequency domain.

        Returns:
            Tensor
                A random vector of dimension D.
        """
        return _base_create_single_vector(
            vector_dim=self.vector_dim,
            gen=self.generator,
            eps=eps,
            dev=self.device
        )

    @torch.inference_mode()
    def positional_encoding(self, x: Tensor) -> Tuple[Tensor, dict]:
        """
        Abstract definition of the continuous encoding method (\\mathcal{E})
        from the HyperSpace paper.

        Arguments:
        ----------
        x : torch.Tensor
            Continuous position to be encoded. Shape should be (batch_size, env_dim).

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
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x should be a Tensor, got {type(x)}")
        if x.dim() != 2 or x.dim() != 1:
            raise ValueError("Input x must be a 1D or 2D tensor of shape (env_dim) or (batch_size, env_dim).")
        if x.shape[-1] != self.env_dim:
            raise ValueError(f"Input x's last dimension should have shape {self.env_dim}; got {x.shape[-1]}")
        
        raise NotImplementedError()

        return encoded, info_dict
    
    @torch.inference_mode()
    def value_encoding(self, x: Tensor) -> Tuple[Tensor, dict]:
        """
        Abstract definition of the value encoding method (\\mathcal{V})
        from the HyperSpace paper.

        Arguments:
        ----------
        x : torch.Tensor
            Continuous value to be encoded. Shape should be (batch_size, value_dim).

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
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x should be a Tensor, got {type(x)}")
        if x.dim() != 2 or x.dim() != 1:
            raise ValueError("Input x must be a 1D or 2D tensor of shape (value_dim) or (batch_size, value_dim).")
        if x.shape[-1] != self.value_dim:
            raise ValueError(f"Input x's last dimensional should have shape {self.value_dim}; got {x.shape[-1]}")
        
        raise NotImplementedError()

        return encoded, info_dict

    @torch.inference_mode()
    def bind(self, a: Tensor, b: Tensor) -> Tuple[Tensor, dict]:
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

        if a.shape != b.shape:
            raise ValueError(f"Expected a and b to have the same shape; got {a.shape} and {b.shape}")
        
        if a.ndim == 1: # Single Bind
            out = self._comp_single_bind(a, b)
        elif a.ndim == 2: # Batch Bind
            out = self._comp_batch_bind(a, b)
        else:
            raise ValueError(f"Expected tensors to be single or two dimensional; got {a.ndim}")

        info_dict = {}

        return out, info_dict

    @torch.inference_mode()
    def bundle(self, a: Tensor, b: Tensor) -> Tuple[Tensor, dict]:
        """
        Bundling operation for HRR backend using vector addition (superposition).

        Bundling creates a superposition of two vectors through elementwise addition,
        allowing multiple vectors to be combined into a single representation. This
        operation supports both single vector and batched operations.

        Arguments:
            a : Tensor
                First input tensor to bundle. Shape should be either (D,) for single
                vectors or (B, D) for batch operations, where B is batch size and D
                is the vector dimension.
            b : Tensor
                Second input tensor to bundle. Must have the same shape as `a`.

        Returns:
            Tensor
                Bundled tensor with the same shape as inputs.
            dict
                Information dictionary (currently empty).
        """

        if a.shape != b.shape:
            raise ValueError(f"Expected a and b to have the same shape; got {a.shape} and {b.shape}")
        
        if a.ndim == 1: # Single Bind
            out = self._comp_single_bundle(a, b)
        elif a.ndim == 2: # Batch Bind
            out = self._comp_batch_bundle(a, b)
        else:
            raise ValueError(f"Expected tensors to be single or two dimensional; got {a.ndim}")

        info_dict = {}

        return out, info_dict

    def similarity(self, a: Tensor, b: Tensor) -> Tuple[Tensor, dict]:
        """
        Compute cosine similarity between two HRR vectors.

        Calculates the cosine similarity metric between two tensors, measuring
        the cosine of the angle between them. This operation supports both single
        vector and batched operations. The similarity values range from -1 (completely
        dissimilar) to 1 (identical).

        Arguments:
            a : Tensor
                First input tensor. Shape should be either (D,) for single vectors
                or (B, D) for batch operations, where B is batch size and D is the
                vector dimension.
            b : Tensor
                Second input tensor. Must have the same shape as `a`.

        Returns:
            Tensor
                Cosine similarity scores. Shape is scalar for single vectors or
                (B,) for batch operations.
            dict
                Information dictionary (currently empty).
        """

        if a.shape != b.shape:
            raise ValueError(f"Expected a and b to have the same shape; got {a.shape} and {b.shape}")
        
        if a.ndim == 1: # Single Bind
            out = F.cosine_similarity(a, b, dim=0)
        elif a.ndim == 2: # Batch Bind
            out = F.cosine_similarity(a, b, dim=1)
        else:
            raise ValueError(f"Expected tensors to be single or two dimensional; got {a.ndim}")

        info_dict = {}

        return out, info_dict
    
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
        rows = [self.create_random_vector() for _ in range(env_dim)]
        env = torch.stack(rows).to(self.device)    # (env_dim, D)
        env = env.contiguous()

        # Update buffers IN-PLACE
        self.env_basis_vectors.resize_(env.shape).copy_(env)

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
        vals = torch.stack(rows).to(self.device).contiguous()           # (value_dim, D)

        # Update buffers in-place (no re-register)
        self.value_basis_vectors.resize_(vals.shape).copy_(vals)
    
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
