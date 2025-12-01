import pytest
from typing import List

def test_base_create_single_vector_base_arguments():
    """
    test the ability to generate random HRR vectors
    """
    import torch
    from torch import Generator, Tensor
    from hyperspace.backends.hrr import _base_create_single_vector

    vd: int = 256
    gen = Generator(device="cpu")
    v1 = _base_create_single_vector(
        vector_dim=vd,
        gen=gen
    )

    v2 = _base_create_single_vector(
        vector_dim=vd,
        gen=gen
    )

    dot = torch.dot(v1, v2)
    assert dot.abs() < 0.12, f"Vectors not orthogonal; dot={dot.item()}" 
    assert isinstance(v1, Tensor)
    assert len(v1.shape) == 1
    assert v1.shape[0] == vd

def test_base_create_single_vector_non_int_vector_dim():
    """
    test the ability to generate random HRR vectors
    """
    from torch import Generator
    from hyperspace.backends.hrr import _base_create_single_vector

    vd: float = 1.0

    with pytest.raises(TypeError):
        _ = _base_create_single_vector(
            vector_dim=vd,
            gen=Generator(device="cpu")
        )

def test_base_create_single_vector_negative_vector_dim():
    """
    test that the create random vector function catches
    negative dimensionalities
    """
    from torch import Generator
    from hyperspace.backends.hrr import _base_create_single_vector

    vd: int = -1

    with pytest.raises(ValueError):
        _ = _base_create_single_vector(
            vector_dim=vd,
            gen=Generator(device="cpu")
        )

def test_base_create_single_vector_invalid_generator():
    """
    test that the create random vector functon catches objects
    that are not true torch generators
    """
    from hyperspace.backends.hrr import _base_create_single_vector

    with pytest.raises(TypeError):
        _ = _base_create_single_vector(
            vector_dim=20,
            gen=int(5)
        )

def test_base_create_single_vector_invalid_eps_type():
    """
    test that the create random vector function catches
    when the eps argument is the wrong type
    """
    from torch import Generator
    from hyperspace.backends.hrr import _base_create_single_vector

    with pytest.raises(TypeError):
        _ = _base_create_single_vector(
            vector_dim=128,
            gen=Generator(),
            eps=int(5)
        )

def test_base_create_single_vector_negative_eps():
    """
    test that the create_single_vector function catches
    when eps is negative
    """
    from torch import Generator
    from hyperspace.backends.hrr import _base_create_single_vector

    with pytest.raises(ValueError):
        _ = _base_create_single_vector(
            vector_dim=128,
            gen=Generator(),
            eps=0.0
        )

def test_base_single_bind():
    """
    test the _base_single_bind function
    """
    import numpy as np
    from torch import Generator
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_single_bind
    )

    vd: int = 256
    gen = Generator()

    v1 = _base_create_single_vector(vd, gen)
    v2 = _base_create_single_vector(vd, gen)

    v_bind = _base_single_bind(v1, v2)

    v1_numpy = v1.numpy()
    v2_numpy = v2.numpy()

    v1_numpy_fft = np.fft.fft(v1_numpy)
    v2_numpy_fft = np.fft.fft(v2_numpy)
    v_out_numpy_fft = v1_numpy_fft * v2_numpy_fft
    v_out_numpy = np.fft.ifft(v_out_numpy_fft)

    assert np.allclose(v_out_numpy, v_bind.numpy(), rtol=1e-5, atol=1e-7)

def test_base_batch_bind_batch():
    """
    Test the batched bind function against NumPy FFT implementation.
    """
    import numpy as np
    import torch
    from torch import Generator

    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_batch_bind
    )

    B: int = 8     # batch size
    D: int = 256   # vector dimensionality

    gen = Generator().manual_seed(0)

    # Build batched tensors: (B, D)
    v1_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v2_list = [_base_create_single_vector(D, gen) for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)  # (B, D)
    v2 = torch.stack(v2_list, dim=0)  # (B, D)

    v_bind_batch = _base_batch_bind(v1, v2)  # (B, D)

    # NumPy reference
    v1_numpy = v1.numpy()  # (B, D)
    v2_numpy = v2.numpy()  # (B, D)

    v1_numpy_fft = np.fft.fft(v1_numpy, axis=-1)
    v2_numpy_fft = np.fft.fft(v2_numpy, axis=-1)
    v_out_numpy_fft = v1_numpy_fft * v2_numpy_fft
    v_out_numpy = np.fft.ifft(v_out_numpy_fft, axis=-1).real  # (B, D)

    assert np.allclose(
        v_out_numpy,
        v_bind_batch.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_single_bundle():
    """
    test the _base_single_bundle function
    """
    import numpy as np
    from torch import Generator

    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_single_bundle,
    )

    vd: int = 256
    gen = Generator().manual_seed(0)

    v1 = _base_create_single_vector(vd, gen)
    v2 = _base_create_single_vector(vd, gen)

    v_bundle = _base_single_bundle(v1, v2)

    v1_numpy = v1.numpy()
    v2_numpy = v2.numpy()

    v_out_numpy = v1_numpy + v2_numpy

    assert np.allclose(
        v_out_numpy,
        v_bundle.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_batch_bundle():
    """
    test the _base_batch_bundle function
    """
    import numpy as np
    import torch
    from torch import Generator

    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_batch_bundle,
    )

    B: int = 8
    D: int = 256

    gen = Generator().manual_seed(0)

    # Build batched tensors (B, D)
    v1_list = [_base_create_single_vector(D, gen) for _ in range(B)]
    v2_list = [_base_create_single_vector(D, gen) for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)

    v_bundle_batch = _base_batch_bundle(v1, v2)  # (B, D)

    v1_numpy = v1.numpy()
    v2_numpy = v2.numpy()

    v_out_numpy = v1_numpy + v2_numpy  # (B, D)

    assert np.allclose(
        v_out_numpy,
        v_bundle_batch.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_single_fpe():
    """
    test the _base_single_fpe function
    """
    import numpy as np
    from torch import Generator

    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_single_fpe,
    )

    D: int = 256

    gen = Generator().manual_seed(0)

    basis = _base_create_single_vector(D, gen)
    basis_numpy = basis.numpy()
    pow = np.random.random()
    length_scale = np.random.random()

    fpe_vector_torch = _base_single_fpe(basis, pow, length_scale)

    basis_fft = np.fft.fft(basis_numpy)
    basis_fft = basis_fft ** (pow / length_scale)
    fpe_numpy = np.fft.ifft(basis_fft).real

    assert np.allclose(
        fpe_vector_torch.numpy(),
        fpe_numpy,
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_batch_fpe():
    """
    test the _base_batch_fpe function
    """
    import numpy as np
    import torch
    from torch import Generator

    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_batch_fpe,
    )

    B: int = 8
    D: int = 256

    gen = Generator().manual_seed(0)

    # Build batched tensors (B, D)
    bases = [_base_create_single_vector(D, gen) for _ in range(B)]
    powers = [np.random.random() for _ in range(B)]

    bases_stack = torch.stack(bases, dim=0)
    powers_stack = torch.tensor(powers)

    length_scale = np.random.random()

    fpes_torch = _base_batch_fpe(bases_stack, powers_stack, length_scale)

    # numpy baseline
    fpes_numpy = np.zeros((B, D))

    for b in range(B):
        _b = bases[b].numpy()
        _p = powers[b]

        out = np.fft.fft(_b)
        out = out ** (_p / length_scale)
        out = np.fft.ifft(out).real

        fpes_numpy[b] = out

    assert np.allclose(
        fpes_numpy,
        fpes_torch.numpy(),
        rtol=1e-2,
        atol=1e-7,
    )

def test_backend_base_init():
    """
    Test the initialize HrrBackend
    """
    from hyperspace.backends.hrr import HRRBackend

    vdim: int = 128

    b = HRRBackend(
        vector_dim=vdim
    )

def test_length_scale_init():
    """
    Test the length scale arguments
    """
    from hyperspace.backends.hrr import HRRBackend

    vd: int = 128
    ls_list = [1, 2, 3.0, 0.00001, -1.0]

    for ls in ls_list:
        b = HRRBackend(
            vector_dim=vd,
            length_scale=ls
        )
        assert b.length_scale == float(ls)

def test_invalid_length_scale_init():
    """
    Test a zero-length scale initialization
    """
    from hyperspace.backends.hrr import HRRBackend

    vd: int = 128
    ls: float = 0.0

    with pytest.raises(ValueError):
        _ = HRRBackend(
            vector_dim=vd,
            length_scale=ls
        )

def test_method_exists_compiled_create_single_vector():
    """
    test that the compiled create single vector method exists
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    assert hasattr(b, "_comp_create_single_vector")

def test_method_exists_compiled_single_bind():
    """
    test that the compiled single bind function exists
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    assert hasattr(b, "_comp_single_bind")

def test_method_exists_compiled_single_bundle():
    """
    test that the compiled single bundle function exists
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    assert hasattr(b, "_comp_single_bundle")

def test_method_exists_compiled_single_fpe():
    """
    test that the compiled single fpe function exists
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    assert hasattr(b, "_comp_single_fpe")

def test_method_exists_compiled_batch_bind():
    """
    test that the compiled batch bind function exists
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    assert hasattr(b, "_comp_batch_bind")

def test_method_exists_compiled_batch_bundle():
    """
    test that the compiled batch bundle function exists
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    assert hasattr(b, "_comp_batch_bundle")

def test_method_exists_compiled_batch_fpe():
    """
    test that the compiled batch fpe function exists
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    assert hasattr(b, "_comp_batch_fpe")

def test_backend_method_create_random_vector_base():
    """
    test that the create_random_vector function calls
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    _ = b.create_random_vector()

def test_backend_method_create_random_vector_object_type():
    """
    test that the create_random_vector function returns a Tensor
    """
    from hyperspace.backends.hrr import HRRBackend
    from torch import Tensor

    b = HRRBackend(vector_dim=128)

    v = b.create_random_vector()
    assert isinstance(v, Tensor)

def test_backend_method_create_random_vector_device():
    """
    test the device of the vector returned from create_random_vector
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)
    v = b.create_random_vector()

    assert b.device == v.device

def test_backend_method_create_random_vector_data_type():
    """
    test the type of the vector returned from create_random_vector
    """
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)
    v = b.create_random_vector()

    assert v.dtype == b.vector_dtype

def test_backend_method_create_random_vector_shape():
    """
    test the shape of the vectors returned from create random vector
    """
    from hyperspace.backends.hrr import HRRBackend
    from typing import List

    v_dims: List[int] = [128, 256, 512]

    for d in v_dims:
        b = HRRBackend(vector_dim=d)
        v = b.create_random_vector()

        assert v.ndim == 1
        assert v.shape[0] == d

def test_backend_method_create_random_vector_orthogonality():
    """
    test the orthogonality of the vectors returned from the
    create random vector function
    """
    import torch
    from torch import Tensor
    from hyperspace.backends.hrr import HRRBackend

    vd: int = 256

    b = HRRBackend(vector_dim=vd)

    v1 = b.create_random_vector()
    v2 = b.create_random_vector()

    dot = torch.dot(v1, v2)
    assert dot.abs() < 0.12, f"Vectors not orthogonal; dot={dot.item()}" 
    assert isinstance(v1, Tensor)
    assert len(v1.shape) == 1
    assert v1.shape[0] == vd

def test_backend_method_single_bind():
    """
    test if the backend's binding function supports single binding
    """
    import numpy as np
    from torch import Tensor
    from hyperspace.backends.hrr import HRRBackend

    vector_dim: int = 2048

    backend = HRRBackend(
        vector_dim=vector_dim
    )

    v1: Tensor = backend.create_random_vector()
    v2: Tensor = backend.create_random_vector()

    # --------------------------
    # HyperSpace Implementation
    # --------------------------
    v_out_torch, _ = backend.bind(v1, v2)

    # --------------------------
    # NumPy Implementation
    # --------------------------
    v1_numpy: np.ndarray = v1.numpy()
    v2_numpy: np.ndarray = v2.numpy()

    v1_numpy_fft = np.fft.fft(v1_numpy)
    v2_numpy_fft = np.fft.fft(v2_numpy)

    v_out_numpy_fft = v1_numpy_fft * v2_numpy_fft

    v_out_numpy = np.fft.ifft(v_out_numpy_fft).real

    # -----------------
    # Similarity Check
    # -----------------
    assert np.allclose(
        v_out_numpy,
        v_out_torch.numpy(),
        rtol=1e-5,
        atol=1e-7
    )

def test_backend_method_batch_bind():
    """
    test if the backend's binding function supports batch binding
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 8     # batch size
    D: int = 256   # vector dimensionality

    backend = HRRBackend(vector_dim=D)

    # Build batched tensors: (B, D)
    v1_list = [backend.create_random_vector() for _ in range(B)]
    v2_list = [backend.create_random_vector() for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)  # (B, D)
    v2 = torch.stack(v2_list, dim=0)  # (B, D)

    v_bind_batch, _ = backend.bind(v1, v2)  # (B, D)

    # NumPy reference
    v1_numpy = v1.numpy()  # (B, D)
    v2_numpy = v2.numpy()  # (B, D)

    v1_numpy_fft = np.fft.fft(v1_numpy, axis=-1)
    v2_numpy_fft = np.fft.fft(v2_numpy, axis=-1)
    v_out_numpy_fft = v1_numpy_fft * v2_numpy_fft
    v_out_numpy = np.fft.ifft(v_out_numpy_fft, axis=-1).real  # (B, D)

    assert np.allclose(
        v_out_numpy,
        v_bind_batch.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_method_single_bundle():
    """
    test if the backend's bundling function supports single bundling
    """
    import numpy as np
    from torch import Generator

    from hyperspace.backends.hrr import HRRBackend

    D: int = 256

    backend = HRRBackend(vector_dim=D)

    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()

    v_bundle, _ = backend.bundle(v1, v2)

    v1_numpy = v1.numpy()
    v2_numpy = v2.numpy()

    v_out_numpy = v1_numpy + v2_numpy

    assert np.allclose(
        v_out_numpy,
        v_bundle.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_method_batch_bundle():
    """
    test the implementation of the backend's batch bundle method
    """

    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 8
    D: int = 256

    backend = HRRBackend(vector_dim=D)

    # Build batched tensors (B, D)
    v1_list = [backend.create_random_vector() for _ in range(B)]
    v2_list = [backend.create_random_vector() for _ in range(B)]

    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)

    v_bundle_batch, _ = backend.bundle(v1, v2)  # (B, D)

    v1_numpy = v1.numpy()
    v2_numpy = v2.numpy()

    v_out_numpy = v1_numpy + v2_numpy  # (B, D)

    assert np.allclose(
        v_out_numpy,
        v_bundle_batch.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_method_single_similarity_pre_normalized_same():
    """
    test the implementation of the backend's single similarity function
    with normalized vectors
    """

    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    D: int = 256

    backend = HRRBackend(vector_dim=D)

    # build the tensors
    v1 = backend.create_random_vector()

    sim, _ = backend.similarity(v1, v1)

    assert np.isclose(sim.numpy(), 1.0)
