
import numpy as np
import torch
from torch import device, Generator, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

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

def _base_list_bundle(v: Tensor) -> Tensor:
    """
    List HRR bundling where all vectors in the list are bundled together

    v: (B, D)
    returns: (D)
    """
    if not isinstance(v, Tensor):
        raise TypeError(f"expected v to be a Tensor; got {type(v)}")
    
    if v.ndim != 2:
        raise ValueError(f"expected v to be (B, D); got {v.shape}")

    v_out = torch.sum(v, dim=0)

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

def _base_single_normalize(x: Tensor) -> Tensor:
    """
    Normalize a single vector to unit L2 norm.

    This function rescales a one-dimensional input tensor so that its
    Euclidean (L2) norm is equal to 1. Normalization is performed using
    ``torch.nn.functional.normalize``, which is numerically stable and
    safely handles zero vectors.

    Parameters
    ----------
    x : Tensor
        A one-dimensional tensor of shape ``(vector_dim,)`` representing
        the input vector to be normalized.

    Returns
    -------
    Tensor
        A one-dimensional tensor of the same shape as ``x`` with unit
        L2 norm.

    Raises
    ------
    TypeError
        If ``x`` is not a ``torch.Tensor``.
    ValueError
        If ``x`` is not a one-dimensional tensor.
    """

    if not isinstance(x, Tensor):
        raise TypeError(f"Input x should be a Tensor, got {type(x)}")
    if x.dim() != 1:
        raise ValueError("Input x must be a 1D tensor of shape (vector_dim).")
    
    x = F.normalize(x, p=2, dim=0)

    return x

def _base_batch_normalize(x: Tensor) -> Tensor:
    """
    Normalize a batch of vector to unit L2 norm.

    This function rescales a one-dimensional input tensor so that its
    Euclidean (L2) norm is equal to 1. Normalization is performed using
    ``torch.nn.functional.normalize``, which is numerically stable and
    safely handles zero vectors.

    Parameters
    ----------
    x : Tensor
        A one-dimensional tensor of shape ``(batch_size, vector_dim)`` representing
        the input vector to be normalized.

    Returns
    -------
    Tensor
        A two-dimensional tensor of the same shape as ``x`` with unit
        L2 norm.

    Raises
    ------
    TypeError
        If ``x`` is not a ``torch.Tensor``.
    ValueError
        If ``x`` is not a two-dimensional tensor.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Input x should be a Tensor, got {type(x)}")
    if x.dim() != 2:
        raise ValueError("Input x must be a 2D tensor of shape (batch_size, vector_dim).")
    
    x = F.normalize(x, p=2, dim=-1)

    return x

def _base_single_invert(x: Tensor) -> Tensor:
    """
    Compute the inverse of a single HRR vector using frequency-domain conjugation.

    This function computes the approximate inverse of a single
    Holographic Reduced Representation (HRR) vector by transforming it
    into the frequency domain, applying complex conjugation, and
    transforming it back via the inverse FFT. The result corresponds to
    the circular correlation inverse used in HRR unbinding.

    Parameters
    ----------
    x : Tensor
        A one-dimensional real-valued tensor of shape (vector_dim,)
        representing an HRR vector.

    Returns
    -------
    Tensor
        A one-dimensional real-valued tensor of shape (vector_dim,)
        representing the inverse HRR vector.

    Raises
    ------
    TypeError
        If `x` is not a torch.Tensor.
    ValueError
        If `x` is not one-dimensional.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Input x should be a Tensor, got {type(x)}")
    if x.dim() != 1:
        raise ValueError("Input x must be a 1D tensor of shape (vector_dim).")
    
    out = torch.fft.fft(x)
    out = torch.conj(out)
    out = torch.fft.ifft(out).real

    return out

def _base_batch_invert(x: Tensor) -> Tensor:
    """
    Compute the inverse of a batch of HRR vectors via Fourier-domain conjugation.

    This function performs the HRR inverse operation by applying a Fast Fourier
    Transform (FFT) along the feature dimension, taking the complex conjugate
    in the frequency domain, and transforming back with the inverse FFT (IFFT).
    The operation is applied independently to each vector in the batch.

    Parameters
    ----------
    x : Tensor
        A real-valued tensor of shape (batch_size, vector_dim) containing a batch
        of HRR vectors to be inverted.

    Returns
    -------
    Tensor
        A real-valued tensor of shape (batch_size, vector_dim) containing the
        inverted HRR vectors.

    Raises
    ------
    TypeError
        If ``x`` is not a torch Tensor.
    ValueError
        If ``x`` is not a 2D tensor of shape (batch_size, vector_dim).
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Input x should be a Tensor, got {type(x)}")
    if x.dim() != 2:
        raise ValueError("Input x must be a 2D tensor of shape (batch_size, vector_dim).")
    
    out = torch.fft.fft(x, dim=-1)
    out = torch.conj(out)
    out = torch.fft.ifft(out, dim=-1).real

    return out

def _base_single_weight(x: Tensor, w: Tensor) -> Tensor:
    """
    Apply a scalar weight to a single HRR vector.

    This function performs element-wise scaling of a 1D input vector `x`
    by a single scalar value stored in a 1D tensor `w` of shape (1).

    Parameters
    ----------
    x : Tensor
        A 1D tensor of shape (vector_dim,) representing the input vector.
    w : Tensor
        A 1D tensor of shape (1,) containing the scalar weight.

    Returns
    -------
    Tensor
        A 1D tensor of shape (vector_dim,) representing the weighted vector.

    Raises
    ------
    TypeError
        If `x` or `w` is not a Tensor.
    ValueError
        If `x` is not 1D.
        If `w` is not 1D or does not have shape (1,).
    """

    if not isinstance(x, Tensor):
        raise TypeError(f"Input x should be a Tensor, got {type(x)}")
    
    if not isinstance(w, Tensor):
        raise TypeError(f"Input w should be a Tensor; got {type(w)}")

    if x.dim() != 1:
        raise ValueError("Input x must be a 1D tensor of shape (vector_dim).")
    
    if w.dim() != 1:
        raise ValueError("Input w must be a 1D tensor of shape (1).")
    
    if w.shape[0] != 1:
        raise ValueError("Input w must be a 1D tensor of shape (1).")
    
    out = x * w

    return out

def _base_batch_weight(x: Tensor, w: Tensor) -> Tensor:
    """
    Apply a scalar weight to each vector in a batch.

    This function scales each vector in the input batch `x` by the
    corresponding scalar weight in `w` using PyTorch broadcasting.

    Parameters
    ----------
    x : Tensor
        A 2D tensor of shape (batch_size, vector_dim) representing a batch
        of input vectors.
    w : Tensor
        A 1D tensor of shape (batch_size,) containing the scalar weights
        for each vector in the batch.

    Returns
    -------
    Tensor
        A 2D tensor of shape (batch_size, vector_dim) representing the
        batch of weighted vectors.

    Raises
    ------
    TypeError
        If `x` or `w` is not a Tensor.
    ValueError
        If `x` is not 2D.
        If `w` is not 1D.
        If the batch dimension of `w` does not match that of `x`.
    """

    if not isinstance(x, Tensor):
        raise TypeError(f"Input x should be a Tensor, got {type(x)}")
    
    if not isinstance(w, Tensor):
        raise TypeError(f"Input w should be a Tensor; got {type(w)}")

    if x.dim() != 2:
        raise ValueError("Input x must be a 2D tensor of shape (batch_size, vector_dim).")
    
    if w.dim() != 1:
        raise ValueError("Input w must be a 1D tensor of shape (batch_size).")
    
    if w.shape[0] != x.shape[0]:
        raise ValueError(f"Input w must have the same batch size of x; got {w.shape[0]} and {x.shape[0]}")
    
    out = x * w.view(-1, 1)

    return out

def _base_single_to_batch_bind(v: Tensor, batch: Tensor) -> Tensor:
    """
    Bind a single hypervector with every hypervector in a batch.

    Args:
        v: Single hypervector. Shape (D,).
        batch: Batch of hypervectors. Shape (B, D).

    Returns:
        out: Bound batch. Shape (B, D).
    """
    if v.ndim != 1:
        raise ValueError(f"Expected v to be 1D; got v.shape={v.shape}")
    if batch.ndim != 2:
        raise ValueError(f"Expected batch to be 2D; got batch.shape={batch.shape}")
    if v.shape[-1] != batch.shape[-1]:
        raise ValueError(
            f"Last dimension mismatch: v.shape={v.shape}, batch.shape={batch.shape}"
        )

    # Version 1 (simple): expand and reuse your batch kernel
    # You can later replace this with a custom fused/compiled implementation.
    v_expanded = v.unsqueeze(0).expand(batch.shape[0], -1)
    return _base_batch_bind(v_expanded, batch)

def _base_single_to_batch_bundle(v: Tensor, batch: Tensor) -> Tensor:
    """
    Bundle a single hypervector with every hypervector in a batch.

    Args:
        v: Single hypervector. Shape (D,).
        batch: Batch of hypervectors. Shape (B, D).

    Returns:
        out: Bundled batch. Shape (B, D).
    """
    if v.ndim != 1:
        raise ValueError(f"Expected v to be 1D; got v.shape={v.shape}")
    if batch.ndim != 2:
        raise ValueError(f"Expected batch to be 2D; got batch.shape={batch.shape}")
    if v.shape[-1] != batch.shape[-1]:
        raise ValueError(
            f"Last dimension mismatch: v.shape={v.shape}, batch.shape={batch.shape}"
        )

    # Simple version: expand and reuse batch bundle kernel.
    # You can replace this with a fused/compiled kernel later.
    v_expanded = v.unsqueeze(0).expand(batch.shape[0], -1)
    return _base_batch_bundle(v_expanded, batch)

def _base_list_bind(batch: Tensor) -> Tensor:
    """
    Sequentially bind all hypervectors in a batch into a single hypervector.

    Args:
        batch: Batch of hypervectors to bind together.
            Shape: (B, D), where B is batch size and D is vector dimension.

    Returns:
        out: Single hypervector representing the binding of all batch elements.
            Shape: (D,).
    """
    if batch.ndim != 2:
        raise ValueError(f"Expected batch to be 2D; got batch.shape={batch.shape}")

    # Simple reference implementation using your single-bind primitive.
    # You can replace this with a fused/compiled implementation later.
    out = batch[0]
    for i in range(1, batch.shape[0]):
        out = _base_single_bind(out, batch[i])
    return out

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
        self._comp_single_ve = torch.compile(_base_single_value_encoding)
        self._comp_single_normalize = torch.compile(_base_single_normalize)
        self._comp_single_invert = torch.compile(_base_single_invert)
        self._comp_single_weight = torch.compile(_base_single_weight)
        self._comp_batch_bind = torch.compile(_base_batch_bind)
        self._comp_batch_bundle = torch.compile(_base_batch_bundle)
        self._comp_batch_fpe = torch.compile(_base_batch_fpe)
        self._comp_batch_ve = torch.compile(_base_batch_value_encoding)
        self._comp_batch_normalize = torch.compile(_base_batch_normalize)
        self._comp_batch_invert = torch.compile(_base_batch_invert)
        self._comp_batch_weight = torch.compile(_base_batch_weight)
        self._comp_list_bundle = torch.compile(_base_list_bundle)
        self._comp_list_bind = torch.compile(_base_list_bind)
        self._comp_single_to_batch_bind = torch.compile(_base_single_to_batch_bind)
        self._comp_single_to_batch_bundle = torch.compile(_base_single_to_batch_bundle)
        

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
        if x.dim() != 2 and x.dim() != 1:
            raise ValueError(f"Input x must be a 1D or 2D tensor of shape (env_dim) or (batch_size, env_dim); got {x.dim()}")
        if x.shape[-1] != self.env_dim:
            raise ValueError(f"Input x's last dimension should have shape {self.env_dim}; got {x.shape[-1]}")

        if x.dim() == 1:
            out = self._comp_single_ve(
                x=x,
                basis=self.env_basis_vectors,
                length_scale=self.length_scale
            )
        elif x.dim() == 2:
            out = self._comp_batch_ve(
                x=x,
                basis=self.env_basis_vectors,
                length_scale=self.length_scale
            )
        else:
            raise ValueError(f"Expected tensors to be single or two dimensional; got {a.ndim}")

        info_dict = {}

        return out, info_dict
    
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
        if x.dim() != 2 and x.dim() != 1:
            raise ValueError("Input x must be a 1D or 2D tensor of shape (value_dim) or (batch_size, value_dim).")
        if x.shape[-1] != self.value_dim:
            raise ValueError(f"Input x's last dimensional should have shape {self.value_dim}; got {x.shape[-1]}")
        
        if x.dim() == 1:
            out = self._comp_single_ve(
                x=x,
                basis=self.value_basis_vectors,
                length_scale=self.length_scale
            )
        elif x.dim() == 2:
            out = self._comp_batch_ve(
                x=x,
                basis=self.value_basis_vectors,
                length_scale=self.length_scale
            )
        else:
            raise ValueError(f"Expected tensors to be single or two dimensional; got {a.ndim}")

        info_dict = {}

        return out, info_dict

    @torch.inference_mode()
    def bind(self, a: Tensor, b: Optional[Tensor] = None) -> Tuple[Tensor, dict]:
        """
        Binding operation for HRR backend using circular convolution.

        This method supports:
            Pairwise binding:
                * a: (D,),   b: (D,)     -> out: (D,)
                * a: (B, D), b: (B, D)   -> out: (B, D)

            Single↔batch binding:
                * a: (D,),   b: (B, D)   -> out: (B, D)
                * a: (B, D), b: (D,)     -> out: (B, D)

            List binding (reduction over batch):
                * a: (B, D), b: None     -> out: (D,)

        Args:
            a:
                First input tensor. Shape (D,) or (B, D).
            b:
                Second input tensor. If provided, must have shape (D,) or (B, D)
                with matching last dimension. If None and `a` is (B, D), performs
                list binding over the batch.

        Returns:
            out:
                Bound hypervector(s). Shape depends on the mode:
                - (D,) for single–single or list-binding
                - (B, D) for batch-related modes
            info_dict:
                Information dictionary (currently empty).
        """
        if not isinstance(a, Tensor):
            raise TypeError(f"Expected a to be a Tensor; got {type(a)}")
        if b is not None and not isinstance(b, Tensor):
            raise TypeError(f"Expected b to be a Tensor or None; got {type(b)}")

        if a.ndim not in (1, 2):
            raise ValueError(f"Expected a to be 1D or 2D; got a.ndim={a.ndim}")
        if b is not None and b.ndim not in (1, 2):
            raise ValueError(f"Expected b to be 1D or 2D; got b.ndim={b.ndim}")

        # Vector dimension check when b is present
        if b is not None:
            if a.shape[-1] != self.vector_dim or b.shape[-1] != self.vector_dim:
                raise ValueError(
                    f"Last dimension must be {self.vector_dim}; "
                    f"got a.shape={a.shape}, b.shape={b.shape}"
                )
        else:
            # Only a is provided; still sanity-check its last dim
            if a.shape[-1] != self.vector_dim:
                raise ValueError(
                    f"Last dimension must be {self.vector_dim}; got a.shape={a.shape}"
                )

        # ---- List binding when b is None ----
        if b is None:
            if a.ndim == 2:
                # (B, D) -> (D,)
                out = self._comp_list_bind(a)
            elif a.ndim == 1:
                # Reasonable identity behavior: binding a single vector list is itself.
                # If you prefer stricter semantics, you could raise here instead.
                out = a
            else:
                # Should be unreachable with ndim guard
                raise ValueError(f"Unsupported shape for a: {a.shape}")

        # ---- Pairwise and single↔batch binding when b is not None ----
        else:
            if a.ndim == 1 and b.ndim == 1:
                # (D,) with (D,)
                out = self._comp_single_bind(a, b)

            elif a.ndim == 2 and b.ndim == 2:
                # (B, D) with (B, D) (pairwise)
                if a.shape[0] != b.shape[0]:
                    raise ValueError(
                        f"Batch sizes must match for batched bind; "
                        f"got a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}"
                    )
                out = self._comp_batch_bind(a, b)

            elif a.ndim == 1 and b.ndim == 2:
                # (D,) with (B, D) → (B, D)
                out = self._comp_single_to_batch_bind(a, b)

            elif a.ndim == 2 and b.ndim == 1:
                # (B, D) with (D,) → (B, D)
                out = self._comp_single_to_batch_bind(b, a)

            else:
                raise ValueError(
                    f"Unsupported combination of shapes: a.shape={a.shape}, b.shape={b.shape}"
                )

        info_dict: dict = {}
        return out, info_dict

    @torch.inference_mode()
    def bundle(self, a: Tensor, b: Optional[Tensor] = None) -> Tuple[Tensor, dict]:
        """
        Bundling operation for HRR backend using vector addition (superposition).

        Bundling creates a superposition of vectors through elementwise addition,
        allowing multiple hypervectors to be combined into a single representation.
        This method supports:

        Pairwise bundling:
            * a: (D,),   b: (D,)     -> out: (D,)
            * a: (B, D), b: (B, D)   -> out: (B, D)

        Single↔batch bundling:
            * a: (D,),   b: (B, D)   -> out: (B, D)   (a bundled with each row of b)
            * a: (B, D), b: (D,)     -> out: (B, D)   (b bundled with each row of a)

        List bundling (reduction):
            * a: (B, D), b: None     -> out: (D,)    (superpose all rows in `a`)

        Args:
            a:
                First input tensor to bundle. Shape (D,) or (B, D).
            b:
                Second input tensor to bundle. If provided, must have shape
                (D,) or (B, D) with matching last dimension. If None and
                `a` is (B, D), performs list bundling over the batch.

        Returns:
            out:
                Bundled tensor with shape determined by the inputs.
            info_dict:
                Information dictionary (currently empty).
        """
        if not isinstance(a, Tensor):
            raise TypeError(f"Expected a to be a Tensor; got {type(a)}")
        if b is not None and not isinstance(b, Tensor):
            raise TypeError(f"Expected b to be a Tensor or None; got {type(b)}")

        if a.ndim not in (1, 2):
            raise ValueError(f"Expected a to be 1D or 2D; got a.ndim={a.ndim}")
        if b is not None and b.ndim not in (1, 2):
            raise ValueError(f"Expected b to be 1D or 2D; got b.ndim={b.ndim}")

        # Vector-dim consistency check if b is present
        if b is not None:
            if a.shape[-1] != self.vector_dim or b.shape[-1] != self.vector_dim:
                raise ValueError(
                    f"Last dimension must be {self.vector_dim}; "
                    f"got a.shape={a.shape}, b.shape={b.shape}"
                )

        # Case A: list bundling (reduce batch) when b is None
        if b is None:
            if a.ndim == 2:
                # Superpose all rows in the batch: (B, D) -> (D,)
                out = self._comp_list_bundle(a)
            elif a.ndim == 1:
                # Reasonable behavior: bundling a single vector alone = identity.
                # You can also choose to raise if you prefer strictness.
                out = a
            else:
                # Shouldn't be reachable with the ndim guard
                raise ValueError(f"Unsupported shape for a: {a.shape}")

        else:
            # Case B: pairwise / single↔batch bundling
            if a.ndim == 1 and b.ndim == 1:
                # (D,) + (D,) -> (D,)
                out = self._comp_single_bundle(a, b)

            elif a.ndim == 2 and b.ndim == 2:
                # (B, D) + (B, D) -> (B, D), require same batch size
                if a.shape[0] != b.shape[0]:
                    raise ValueError(
                        f"Batch sizes must match for batched bundle; "
                        f"got a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}"
                    )
                out = self._comp_batch_bundle(a, b)

            elif a.ndim == 1 and b.ndim == 2:
                # (D,) + (B, D) -> (B, D)
                out = self._comp_single_to_batch_bundle(a, b)

            elif a.ndim == 2 and b.ndim == 1:
                # (B, D) + (D,) -> (B, D)
                out = self._comp_single_to_batch_bundle(b, a)

            else:
                # Shouldn't happen with ndim guards
                raise ValueError(
                    f"Unsupported combination of shapes: a.shape={a.shape}, b.shape={b.shape}"
                )

        info_dict: dict = {}
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
    
    def normalize(self, x: Tensor) -> Tuple[Tensor, dict]:
        """
        Normalize a single or batched hypervector.

        This method applies the backend-specific normalization operator to either
        a single vector of shape ``(vector_dim,)`` or a batch of vectors of shape
        ``(batch_size, vector_dim)``. The normalization is dispatched to the
        appropriate single or batched compute kernel based on the dimensionality
        of the input.

        Parameters
        ----------
        x : Tensor
            Input tensor to normalize. Must be either a 1D tensor of shape
            ``(vector_dim,)`` or a 2D tensor of shape
            ``(batch_size, vector_dim)``.

        Returns
        -------
        out : Tensor
            The normalized output tensor with the same shape as the input.
        info : dict
            Dictionary containing auxiliary normalization metadata. Currently
            returned as an empty dictionary for API consistency.

        Raises
        ------
        TypeError
            If ``x`` is not a ``Tensor``.
        ValueError
            If ``x`` is not 1D or 2D, or if the last dimension does not match
            ``self.vector_dim``.
        """

        if not isinstance(x, Tensor):
            raise TypeError(f"Input x should be a Tensor, got {type(x)}")
        if x.dim() != 2 and x.dim() != 1:
            raise ValueError("Input x must be a 1D or 2D tensor of shape (vector_dim) or (batch_size, vector_dim).")
        if x.shape[-1] != self.vector_dim:
            raise ValueError(f"Input x's last dimensional should have shape {self.vector_dim}; got {x.shape[-1]}")

        if x.ndim == 1: # Single Bind
            out = self._comp_single_normalize(x)
        elif x.ndim == 2: # Batch Bind
            out = self._comp_batch_normalize(x)
        else:
            raise ValueError(f"Expected tensors to be single or two dimensional; got {x.ndim}")

        info = {}

        return out, info
    
    def invert(self, x: Tensor) -> Tuple[Tensor, dict]:
        """
        Invert one or more HRR vectors.

        This method computes the HRR inverse of the input using Fourier-domain
        conjugation. It supports both single-vector inversion and batched
        inversion, automatically dispatching to the appropriate backend
        implementation based on the dimensionality of the input.

        Parameters
        ----------
        x : Tensor
            A real-valued tensor representing one or more HRR vectors.
            - If 1D: shape (vector_dim,)
            - If 2D: shape (batch_size, vector_dim)

        Returns
        -------
        Tuple[Tensor, dict]
            A tuple ``(out, info)`` where:
            - ``out`` is a tensor of the same shape as ``x`` containing the inverted
            HRR vector(s).
            - ``info`` is a dictionary reserved for auxiliary information (currently empty).

        Raises
        ------
        TypeError
            If ``x`` is not a torch Tensor.
        ValueError
            If ``x`` is not 1D or 2D.
            If the last dimension of ``x`` does not match ``self.vector_dim``.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x should be a Tensor, got {type(x)}")
        if x.dim() != 2 and x.dim() != 1:
            raise ValueError("Input x must be a 1D or 2D tensor of shape (vector_dim) or (batch_size, vector_dim).")
        if x.shape[-1] != self.vector_dim:
            raise ValueError(f"Input x's last dimensional should have shape {self.vector_dim}; got {x.shape[-1]}")

        if x.ndim == 1: # Single Invert
            out = self._comp_single_invert(x)
        elif x.ndim == 2: # Batch Invert
            out = self._comp_batch_invert(x)
        else:
            raise ValueError(f"Expected tensors to be single or two dimensional; got {x.ndim}")

        info = {}

        return out, info
    
    def weight(self, x: Tensor, w: Tensor) -> Tuple[Tensor, dict]:
        """
        Apply scalar weighting to a single vector or a batch of vectors.

        This method scales each input vector in `x` by the corresponding weight
        in `w`. When `x` is a 1D tensor, a single weight must be provided.
        When `x` is a 2D tensor representing a batch, `w` must contain one
        weight per vector in the batch. The function automatically dispatches
        to the appropriate single-vector or batch-vector implementation.

        Parameters
        ----------
        x : Tensor
            A 1D tensor of shape (vector_dim,) or a 2D tensor of shape
            (batch_size, vector_dim) representing the input vector(s).
        w : Tensor
            A 1D tensor of shape (1,) for single-vector weighting, or
            (batch_size,) for batch weighting.

        Returns
        -------
        Tuple[Tensor, dict]
            A tuple containing:
            - A tensor of the same shape as `x` with the weighted vectors.
            - An empty info dictionary (reserved for future metadata).

        Raises
        ------
        TypeError
            If `x` or `w` is not a Tensor.
        ValueError
            If `x` is not 1D or 2D.
            If `x`'s last dimension does not match `self.vector_dim`.
            If `w` is not 1D.
            If batch sizes do not match for 2D input.
            If a single-vector input does not receive exactly one weight.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"Input x should be a Tensor, got {type(x)}")
        if not isinstance(w, Tensor):
            raise TypeError(f"Input w should be a Tensor, got {type(w)}")
        if x.dim() != 2 and x.dim() != 1:
            raise ValueError("Input x must be a 1D or 2D tensor of shape (vector_dim) or (batch_size, vector_dim).")
        if x.shape[-1] != self.vector_dim:
            raise ValueError(f"Input x's last dimensional should have shape {self.vector_dim}; got {x.shape[-1]}")
        if w.dim() != 1:
            raise ValueError(f"Expected w to be a matrix with shape (batch_size,); got {w.shape}")
        if x.dim() == 2 and x.shape[0] != w.shape[0]:
            raise ValueError(f"Expected the number of weights to match the number of vectors; got {x.shape[0]} vectors and {w.shape[0]} weights")
        if x.dim() == 1 and w.shape[0] != 1:
            raise ValueError(f"Expected one weight for the one argued vector; got {w.shape[0]} weights")
        
        if x.dim() == 1:
            out = self._comp_single_weight(x, w)
        elif x.dim() == 2:
            out = self._comp_batch_weight(x, w)
        else:
            raise ValueError(f"Received unknown x dimensionality; {x.dim()}")

        info = {}

        return out, info
    
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

    def create_empty_vector(self) -> Tensor:
        """
        Create an empty HRR vector initialized to all zeros.

        This method allocates a real-valued tensor of shape ``(vector_dim,)`` on the
        backend's configured device. The returned vector represents the neutral
        (zero) element in HRR space prior to any binding, bundling, or encoding
        operations.

        Returns
        -------
        Tensor
            A real-valued tensor of shape ``(vector_dim,)`` initialized to zeros and
            placed on ``self.device``.
        """
        v = torch.zeros((self.vector_dim))
        v = v.to(self.device)
        return v
    
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
