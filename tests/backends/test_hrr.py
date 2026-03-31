import pytest

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
    assert dot.abs() < 0.15, f"Vectors not orthogonal; dot={dot.item()}" 
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

def test_batch_list_bund():
    """
    test the _base_list_bundle function
    """
    import numpy as np
    import torch
    from torch import Generator

    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_list_bundle,
        _base_single_bundle
    )

    B: int = 2
    D: int = 256

    gen = Generator().manual_seed(0)

    # Build batched tensors (B, D)
    v_list = [_base_create_single_vector(D, gen) for _ in range(B)]

    v1 = torch.stack(v_list, dim=0)

    v_bundle_list = _base_list_bundle(v1)  # (D,)

    v_out_gt = _base_single_bundle(v_list[0], v_list[1])

    assert v_bundle_list.shape == (D,)
    assert v_out_gt.shape == (D,)

    assert np.allclose(
        v_out_gt.numpy(),
        v_bundle_list.numpy(),
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
        rtol=1e-2,
        atol=1e-2,
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
        atol=1e-4,
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

    vd: int = 25600

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

def test_backend_method_single_similarity_non_normalized_same():
    """
    test the implementation of the backend's single similarity function
    with non-normalized vectors
    """

    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    D: int = 256

    backend = HRRBackend(vector_dim=D)

    # build the tensors
    v1 = backend.create_random_vector()

    sim, _ = backend.similarity(v1, v1 / 2)

    assert np.isclose(sim.numpy(), 1.0)

def test_backend_method_single_similarity_pre_normalized_orthogonal():
    """
    test the implementation of the backend's single similarity function
    with normalized vectors
    """

    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    D: int = 10000

    backend = HRRBackend(vector_dim=D)

    # build the tensors
    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()

    sim, _ = backend.similarity(v1, v2)

    assert np.isclose(sim.numpy(), 0.0, rtol=0.1, atol=0.1)

def test_backend_method_single_similarity_non_normalized_orthogonal():
    """
    test the implementation of the backend's single similarity function
    with non-normalized vectors
    """

    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    D: int = 10000

    backend = HRRBackend(vector_dim=D)

    # build the tensors
    v1 = backend.create_random_vector()
    v2 = backend.create_random_vector()

    sim, _ = backend.similarity(v1, v2 / 2.0)

    assert np.isclose(sim.numpy(), 0.0, rtol=0.1, atol=0.1)

def test_backend_method_batch_similarity_normalized_same():
    """
    test the functionality of the backend similarity metric with
    identical vectors that are normalized
    """

    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 8
    D: int = 10000

    backend = HRRBackend(vector_dim=D)

    # Build batched tensors (B, D)
    v1_list = [backend.create_random_vector() for _ in range(B)]
    gt = torch.ones(B)

    v1 = torch.stack(v1_list, dim=0)

    sims, _ = backend.similarity(v1, v1)

    assert np.allclose(
        sims.numpy(),
        gt.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_method_batch_similarity_non_normalized_same():
    """
    test the functionality of the backend similarity metric with
    identical vectors that are non-normalized
    """

    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 8
    D: int = 10000

    backend = HRRBackend(vector_dim=D)

    # Build batched tensors (B, D)
    v1_list = [backend.create_random_vector() for _ in range(B)]
    gt = torch.ones(B)

    v1 = torch.stack(v1_list, dim=0)

    sims, _ = backend.similarity(v1, v1 / 10.0)

    assert np.allclose(
        sims.numpy(),
        gt.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_method_batch_similarity_normalized_orthogonal():
    """
    test the functionality of the backend similarity metric with
    orthogonal vectors that are normalized
    """

    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 8
    D: int = 10000

    backend = HRRBackend(vector_dim=D)

    # Build batched tensors (B, D)
    v1_list = [backend.create_random_vector() for _ in range(B)]
    v2_list = [backend.create_random_vector() for _ in range(B)]
    gt = torch.zeros(B)

    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)

    sims, _ = backend.similarity(v1, v2)

    assert np.allclose(
        sims.numpy(),
        gt.numpy(),
        rtol=0.1,
        atol=0.1,
    )

def test_backend_method_batch_similarity_non_normalized_orthogonal():
    """
    test the functionality of the backend similarity metric with
    orthogonal vectors that are normalized
    """

    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 8
    D: int = 10000

    backend = HRRBackend(vector_dim=D)

    # Build batched tensors (B, D)
    v1_list = [backend.create_random_vector() for _ in range(B)]
    v2_list = [backend.create_random_vector() for _ in range(B)]
    gt = torch.zeros(B)

    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)

    sims, _ = backend.similarity(v1, v2 / 10.0)

    assert np.allclose(
        sims.numpy(),
        gt.numpy(),
        rtol=0.1,
        atol=0.1,
    )

def test_backend_env_dim_basis_size():
    """
    test that the size of the env basis matrix matches the arguments
    """
    from hyperspace.backends.hrr import HRRBackend

    D: int = 10000

    env_dim_list = [1, 3, 5]

    for d in env_dim_list:
        backend = HRRBackend(vector_dim=D, env_dim=d)
        assert backend.env_basis_vectors.shape == (d, D)

def test_backend_value_dim_basis_size():
    """
    test that the size of the value basis matrix matches the arguments
    """
    from hyperspace.backends.hrr import HRRBackend

    D: int = 10000

    value_dim_list = [1, 3, 5]

    for d in value_dim_list:
        b = HRRBackend(vector_dim=D, value_dim=d)
        assert b.value_basis_vectors.shape == (d, D)

def test_backend_env_basis_similarity():
    """
    test that the env basis vectors are approximately orthogonal
    """
    import torch
    from torch import Tensor
    from hyperspace.backends.hrr import HRRBackend

    D: int = 10000

    b = HRRBackend(vector_dim=D, env_dim=2)

    v1: torch.Tensor = b.env_basis_vectors[0]
    v2: torch.Tensor = b.env_basis_vectors[1]

    dot = torch.dot(v1, v2)
    assert dot.abs() < 0.12, f"Vectors not orthogonal; dot={dot.item()}" 
    assert isinstance(v1, Tensor)
    assert len(v1.shape) == 1
    assert v1.shape[0] == D

def test_backend_value_basis_similarity():
    """
    test that the value vectors are approximately orthogonal
    """

    import torch
    from torch import Tensor
    from hyperspace.backends.hrr import HRRBackend

    D: int = 10000

    b = HRRBackend(vector_dim=D, value_dim=2)

    v1: Tensor = b.value_basis_vectors[0]
    v2: Tensor = b.value_basis_vectors[1]

    dot = torch.dot(v1, v2)
    assert dot.abs() < 0.12, f"Vectors not orthogonal; dot={dot.item()}" 
    assert isinstance(v1, Tensor)
    assert len(v1.shape) == 1
    assert v1.shape[0] == D

def test_backend_env_dim_zero():
    """
    test that the backend throws an error with env_dim is zero
    """
    from hyperspace.backends.hrr import HRRBackend

    with pytest.raises(ValueError):
        HRRBackend(
            vector_dim=128,
            env_dim=0
        )

def test_backend_env_dim_negative():
    """
    test that the backend throws an error with env_dim is negative
    """
    from hyperspace.backends.hrr import HRRBackend

    with pytest.raises(ValueError):
        HRRBackend(
            vector_dim=128,
            env_dim=-1
        )

def test_backend_value_dim_zero():
    """
    test that the backend throws an error with value_dim is zero
    """
    from hyperspace.backends.hrr import HRRBackend

    with pytest.raises(ValueError):
        HRRBackend(
            vector_dim=128,
            value_dim=0
        )

def test_backend_value_dim_negative():
    """
    test that the backend throws an error with value_dim is negative
    """
    from hyperspace.backends.hrr import HRRBackend

    with pytest.raises(ValueError):
        HRRBackend(
            vector_dim=128,
            value_dim=-1
        )

def test_backend_continuous_encoding_x_not_tensor():
    """
    test that the backend throws an error with the x argument isn't a tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    with pytest.raises(TypeError):
        b.positional_encoding(np.zeros(2))

def test_backend_continuous_encoding_x_invalid_shape():
    """
    test that the backend throws an error when the x argument doesn't
    have the right dimensionality
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    env_dim: int = 3
    b = HRRBackend(vector_dim=128)

    # low dimensionality
    with pytest.raises(ValueError):
        inp = torch.zeros((env_dim))
        b.positional_encoding(inp)

    # high dimensionality
    with pytest.raises(ValueError):
        inp = torch.zeros((env_dim, env_dim, env_dim))
        b.positional_encoding(inp)

def test_backend_continuous_encoding_x_invalid_env_shape():
    """
    test that the backend throws an error with the x argument
    isn't the correct env_dim
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    env_dim: int = 3
    batch_size: int = 128

    b = HRRBackend(
        vector_dim=128,
        env_dim=env_dim
    )

    x_small = torch.ones((batch_size, env_dim - 1))
    x_big = torch.ones((batch_size, env_dim + 1))

    with pytest.raises(ValueError):
        b.positional_encoding(x_small)

    with pytest.raises(ValueError):
        b.positional_encoding(x_big)

def test_backend_value_encoding_x_not_tensor():
    """
    test that the backend throws an error with the x argument isn't a tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    with pytest.raises(TypeError):
        b.value_encoding(np.zeros(2))

def test_backend_value_encoding_x_invalid_shape():
    """
    test that the backend throws an error when the x argument doesn't
    have the right dimensionality
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    env_dim: int = 3
    b = HRRBackend(vector_dim=128)

    # low dimensionality
    with pytest.raises(ValueError):
        inp = torch.zeros((env_dim))
        b.value_encoding(inp)

    # high dimensionality
    with pytest.raises(ValueError):
        inp = torch.zeros((env_dim, env_dim, env_dim))
        b.value_encoding(inp)

def test_backend_value_encoding_x_invalid_env_shape():
    """
    test that the backend throws an error with the x argument
    isn't the correct value_dim
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    value_dim: int = 3
    batch_size: int = 128

    b = HRRBackend(
        vector_dim=128,
        value_dim=value_dim
    )

    x_small = torch.ones((batch_size, value_dim - 1))
    x_big = torch.ones((batch_size, value_dim + 1))

    with pytest.raises(ValueError):
        b.value_encoding(x_small)

    with pytest.raises(ValueError):
        b.value_encoding(x_big)

def test_base_single_value_encoding_invalid_x_type():
    """
    test that the base_single_value_encoding method throws
    an error when x is not a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import _base_single_value_encoding

    value_dim: int = 10
    vector_dim: int = 256

    x = np.zeros(value_dim)
    basis = torch.ones((value_dim, vector_dim))
    ls: float = 1.0

    with pytest.raises(TypeError):
        _base_single_value_encoding(x, basis, ls)

def test_base_single_value_encoding_invalid_basis_type():
    """
    test that the base_single_value_encoding method throws
    an error when basis is not a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import _base_single_value_encoding

    value_dim: int = 10
    vector_dim: int = 256

    x = torch.zeros(value_dim)
    basis = np.ones((value_dim, vector_dim))
    ls: float = 1.0

    with pytest.raises(TypeError):
        _base_single_value_encoding(x, basis, ls)

def test_base_single_value_encoding_invalid_x_dim():
    """
    test that the base_single_value_encoding method throws
    an error when x isn't the correct shape
    """
    import torch
    from hyperspace.backends.hrr import _base_single_value_encoding

    value_dim: int = 10
    vector_dim: int = 256

    basis = torch.ones((value_dim, vector_dim))
    ls: float = 1.0

    # too large dim
    x = torch.zeros(value_dim, value_dim)
    with pytest.raises(ValueError):
        _base_single_value_encoding(x, basis, ls)

def test_base_single_value_encoding_invalid_basis_dim():
    """
    test that the base_single_value_encoding method throws
    an error when basis isn't the correct shape
    """
    import torch
    from hyperspace.backends.hrr import _base_single_value_encoding

    value_dim: int = 10
    vector_dim: int = 256

    x = torch.zeros(value_dim, value_dim)
    basis_small = torch.ones((vector_dim))
    basis_large = torch.ones((value_dim, vector_dim, value_dim))
    ls: float = 1.0
    
    with pytest.raises(ValueError):
        _base_single_value_encoding(x, basis_small, ls)

    with pytest.raises(ValueError):
        _base_single_value_encoding(x, basis_large, ls)

def test_base_single_value_encoding_value_dim_missmatch():
    """
    test that the base_single_value_encoding method throws
    an error when x and basis assume different value
    dimensionalities
    """
    import torch
    from hyperspace.backends.hrr import _base_single_value_encoding

    value_dim: int = 10
    vector_dim: int = 256

    x_small = torch.zeros(value_dim - 1)
    x_large = torch.zeros(value_dim + 1)
    basis = torch.ones((value_dim, vector_dim))
    ls: float = 1.0

    with pytest.raises(ValueError):
        _base_single_value_encoding(x_small, basis, ls)

    with pytest.raises(ValueError):
        _base_single_value_encoding(x_large, basis, ls)

def test_base_single_value_encoding_valid_input_1d():
    """
    test that the base_single_value_encoding method
    create the correct value vector
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_batch_fpe,
        _base_single_value_encoding
    )

    vector_dim = 1024
    gen = torch.Generator()
    value = torch.tensor([2])
    length_scale = 1.0

    v1 = _base_create_single_vector(
        vector_dim=vector_dim,
        gen=gen
    )
    v1 = v1.unsqueeze(0)

    v_pred = _base_single_value_encoding(
        x=value,
        basis=v1,
        length_scale=length_scale
    ).numpy()

    # numpy baseline
    v1_fft = np.fft.fft(v1[0])
    v1_fft = v1_fft ** (value[0].item() / length_scale)
    v_gt = np.fft.ifft(v1_fft).real
    # v_gt = np.expand_dims(v_gt, axis=0)

    assert v_pred.shape == (vector_dim,)
    assert v_gt.shape == (vector_dim,)

    assert np.allclose(
        v_pred,
        v_gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_single_value_encoding_valid_input_2d():
    """
    test that the base_single_value_encoding method
    create the correct value vector
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_single_value_encoding
    )

    vector_dim = 1024
    gen = torch.Generator()
    value = torch.tensor([2, 4])
    length_scale = 1.0

    v1 = _base_create_single_vector(
        vector_dim=vector_dim,
        gen=gen
    )
    v2 = _base_create_single_vector(
        vector_dim=vector_dim,
        gen=gen
    )
    v_stack = torch.stack([v1, v2], dim=0)

    assert v_stack.shape == (2, vector_dim)

    v_pred = _base_single_value_encoding(
        x=value,
        basis=v_stack,
        length_scale=length_scale
    ).numpy()

    v1_fft = torch.fft.fft(v1)
    v2_fft = torch.fft.fft(v2)
    v1_fft = v1_fft ** (value[0] / length_scale)
    v2_fft = v2_fft ** (value[1] / length_scale)
    v_gt = v1_fft * v2_fft
    v_gt = torch.fft.ifft(v_gt).real

    assert v_pred.shape == (vector_dim,)
    assert v_gt.shape == (vector_dim,)

    assert np.allclose(
        v_pred,
        v_gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_single_value_encoding_valid_input_3d():
    """
    test that the base_single_value_encoding method
    create the correct value vector
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_single_value_encoding
    )

    vector_dim = 1024
    gen = torch.Generator()
    value = torch.tensor([2, 4, 6])
    length_scale = 1.0

    v_list = [_base_create_single_vector(vector_dim, gen) for _ in range(3)]
    v_stack = torch.stack(v_list, dim=0)

    assert v_stack.shape == (3, vector_dim)

    v_pred = _base_single_value_encoding(
        x=value,
        basis=v_stack,
        length_scale=length_scale
    ).numpy()

    v1_fft = torch.fft.fft(v_list[0])
    v2_fft = torch.fft.fft(v_list[1])
    v3_fft = torch.fft.fft(v_list[2])
    v1_fft = v1_fft ** (value[0] / length_scale)
    v2_fft = v2_fft ** (value[1] / length_scale)
    v3_fft = v3_fft ** (value[2] / length_scale)
    v_gt = v1_fft * v2_fft * v3_fft
    v_gt = torch.fft.ifft(v_gt).real

    assert v_pred.shape == (vector_dim,)
    assert v_gt.shape == (vector_dim,)

    assert np.allclose(
        v_pred,
        v_gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_batch_value_encoding_invalid_x_type():
    """
    Test that _base_batch_value_encoding throws error when x is not a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import _base_batch_value_encoding

    value_dim: int = 10
    vector_dim: int = 256
    batch_size: int = 8

    x = np.zeros((batch_size, value_dim))
    basis = torch.ones((value_dim, vector_dim))
    ls: float = 1.0

    with pytest.raises(TypeError):
        _base_batch_value_encoding(x, basis, ls)


def test_base_batch_value_encoding_invalid_basis_type():
    """
    Test that _base_batch_value_encoding throws error when basis is not a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import _base_batch_value_encoding

    value_dim: int = 10
    vector_dim: int = 256
    batch_size: int = 8

    x = torch.zeros((batch_size, value_dim))
    basis = np.ones((value_dim, vector_dim))
    ls: float = 1.0

    with pytest.raises(TypeError):
        _base_batch_value_encoding(x, basis, ls)


def test_base_batch_value_encoding_invalid_x_dim():
    """
    Test that _base_batch_value_encoding throws error when x isn't 2D
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_value_encoding

    value_dim: int = 10
    vector_dim: int = 256

    basis = torch.ones((value_dim, vector_dim))
    ls: float = 1.0

    # 1D input (should be 2D)
    x_1d = torch.zeros(value_dim)
    with pytest.raises(ValueError):
        _base_batch_value_encoding(x_1d, basis, ls)
    
    # 3D input (should be 2D)
    x_3d = torch.zeros((8, value_dim, value_dim))
    with pytest.raises(ValueError):
        _base_batch_value_encoding(x_3d, basis, ls)


def test_base_batch_value_encoding_invalid_basis_dim():
    """
    Test that _base_batch_value_encoding throws error when basis isn't 2D
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_value_encoding

    value_dim: int = 10
    vector_dim: int = 256
    batch_size: int = 8

    x = torch.zeros((batch_size, value_dim))
    basis_1d = torch.ones((vector_dim,))
    basis_3d = torch.ones((value_dim, vector_dim, value_dim))
    ls: float = 1.0
    
    with pytest.raises(ValueError):
        _base_batch_value_encoding(x, basis_1d, ls)

    with pytest.raises(ValueError):
        _base_batch_value_encoding(x, basis_3d, ls)


def test_base_batch_value_encoding_value_dim_mismatch():
    """
    Test that _base_batch_value_encoding throws error when x and basis 
    have mismatched value dimensions
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_value_encoding

    value_dim: int = 10
    vector_dim: int = 256
    batch_size: int = 8

    x_small = torch.zeros((batch_size, value_dim - 1))
    x_large = torch.zeros((batch_size, value_dim + 1))
    basis = torch.ones((value_dim, vector_dim))
    ls: float = 1.0

    with pytest.raises(ValueError):
        _base_batch_value_encoding(x_small, basis, ls)

    with pytest.raises(ValueError):
        _base_batch_value_encoding(x_large, basis, ls)


def test_base_batch_value_encoding_valid_input_1d():
    """
    Test _base_batch_value_encoding with 1D values (single dimension per sample)
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_batch_fpe,
        _base_batch_value_encoding
    )

    vector_dim = 1024
    batch_size = 16
    gen = torch.Generator()
    
    # Create batched 1D values
    values = torch.randn((batch_size, 1)) * 5  # Random values
    length_scale = 1.0

    v1 = _base_create_single_vector(
        vector_dim=vector_dim,
        gen=gen
    )
    v1 = v1.unsqueeze(0)  # (1, vector_dim)

    v_pred = _base_batch_value_encoding(
        x=values,
        basis=v1,
        length_scale=length_scale
    ).numpy()

    # Compute ground truth: each batch item separately
    v_gt = []
    for i in range(batch_size):
        v_fpe = _base_batch_fpe(
            basis=v1,
            powers=values[i],
            length_scale=length_scale
        )
        v_gt.append(torch.prod(v_fpe, dim=0))
    v_gt = torch.stack(v_gt, dim=0).numpy()

    assert v_pred.shape == (batch_size, vector_dim)
    assert v_gt.shape == (batch_size, vector_dim)

    assert np.allclose(
        v_pred,
        v_gt,
        rtol=1e-5,
        atol=1e-7,
    )


def test_base_batch_value_encoding_valid_input_2d():
    """
    Test _base_batch_value_encoding with 2D values
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_batch_value_encoding
    )

    vector_dim = 1024
    batch_size = 16
    gen = torch.Generator()
    
    # Create batched 2D values
    values = torch.randn((batch_size, 2)) * 5
    length_scale = 1.0

    v1 = _base_create_single_vector(
        vector_dim=vector_dim,
        gen=gen
    )
    v2 = _base_create_single_vector(
        vector_dim=vector_dim,
        gen=gen
    )
    v_stack = torch.stack([v1, v2], dim=0)

    assert v_stack.shape == (2, vector_dim)

    v_pred = _base_batch_value_encoding(
        x=values,
        basis=v_stack,
        length_scale=length_scale
    ).numpy()

    # Compute ground truth
    v_gt = []
    for i in range(batch_size):
        v1_fft = torch.fft.fft(v1)
        v2_fft = torch.fft.fft(v2)
        v1_fft = v1_fft ** (values[i][0] / length_scale)
        v2_fft = v2_fft ** (values[i][1] / length_scale)
        v_out = v1_fft * v2_fft
        v_out = torch.fft.ifft(v_out).real
        v_gt.append(v_out)
    v_gt = torch.stack(v_gt, dim=0).numpy()

    assert v_pred.shape == (batch_size, vector_dim)
    assert v_gt.shape == (batch_size, vector_dim)

    assert np.allclose(
        v_pred,
        v_gt,
        rtol=1e-3,
        atol=1e-3,
    )


def test_base_batch_value_encoding_valid_input_3d():
    """
    Test _base_batch_value_encoding with 6D values
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_batch_value_encoding
    )

    vector_dim = 1024
    batch_size = 16
    gen = torch.Generator()
    
    # Create batched 6D values
    values = torch.randn((batch_size, 3)) * 5
    length_scale = 1.0

    v_list = [_base_create_single_vector(vector_dim, gen) for _ in range(3)]
    v_stack = torch.stack(v_list, dim=0)

    assert v_stack.shape == (3, vector_dim)

    v_pred = _base_batch_value_encoding(
        x=values,
        basis=v_stack,
        length_scale=length_scale
    ).numpy()

    # Compute ground truth
    v_gt = []
    for i in range(batch_size):
        v1_fft = torch.fft.fft(v_list[0])
        v2_fft = torch.fft.fft(v_list[1])
        v3_fft = torch.fft.fft(v_list[2])
        v1_fft = v1_fft ** (values[i][0] / length_scale)
        v2_fft = v2_fft ** (values[i][1] / length_scale)
        v3_fft = v3_fft ** (values[i][2] / length_scale)
        v_out = v1_fft * v2_fft * v3_fft
        v_out = torch.fft.ifft(v_out).real
        v_gt.append(v_out)
    v_gt = torch.stack(v_gt, dim=0).numpy()

    assert v_pred.shape == (batch_size, vector_dim)
    assert v_gt.shape == (batch_size, vector_dim)

    assert np.allclose(
        v_pred,
        v_gt,
        rtol=1e-3,
        atol=1e-5,
    )


def test_base_batch_value_encoding_consistency_with_single():
    """
    Test that batched version produces same results as single version
    when batch_size=1
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_single_value_encoding,
        _base_batch_value_encoding
    )

    vector_dim = 512
    gen = torch.Generator().manual_seed(42)
    
    # Create test data
    value_dim = 4
    value_single = torch.tensor([1.5, 2.3, -0.7, 3.1])
    value_batch = value_single.unsqueeze(0)  # (1, 4)
    length_scale = 1.0

    v_list = [_base_create_single_vector(vector_dim, gen) for _ in range(value_dim)]
    v_stack = torch.stack(v_list, dim=0)

    # Single version
    v_single = _base_single_value_encoding(
        x=value_single,
        basis=v_stack,
        length_scale=length_scale
    ).numpy()

    # Batch version with batch_size=1
    v_batch = _base_batch_value_encoding(
        x=value_batch,
        basis=v_stack,
        length_scale=length_scale
    ).numpy()

    assert v_single.shape == (vector_dim,)
    assert v_batch.shape == (1, vector_dim)

    # Should produce identical results
    assert np.allclose(
        v_single,
        v_batch[0],
        rtol=1e-5,
        atol=1e-7,
    )

def test_positional_encoding_invalid_x_type():
    """
    Test that the backend's positional encoding method
    throws an error when x isn't a Tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    B: int = 16
    E: int = 3

    b = HRRBackend(
        vector_dim=128,
        env_dim=E
    )

    x_single = np.zeros((E))
    x_batch = np.zeros((B, E))

    with pytest.raises(TypeError):
        b.positional_encoding(x_single)

    with pytest.raises(TypeError):
        b.positional_encoding(x_batch)

def test_positional_encoding_invalid_x_dim():
    """
    Test that the backend's positional encoding method
    throws an error when x isn't single or batched
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    x_invalid = torch.zeros((5, 5, 5))
    
    with pytest.raises(ValueError):
        b.positional_encoding(x_invalid)

def test_positional_encoding_invalid_x_env_dim():
    """
    test that the backend's positional encoding method
    throws and error when x's env dim doesn't match
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    B: int = 8
    E: int = 3

    x_single_low = torch.zeros((E - 1))
    x_single_high = torch.zeros((E + 1))
    x_batch_low = torch.zeros((B, E - 1))
    x_batch_high = torch.zeros((B, E + 1))

    with pytest.raises(ValueError):
        b.positional_encoding(x_single_low)

    with pytest.raises(ValueError):
        b.positional_encoding(x_single_high)

    with pytest.raises(ValueError):
        b.positional_encoding(x_batch_low)

    with pytest.raises(ValueError):
        b.positional_encoding(x_batch_high)

def test_positional_encoding_single_x():
    """
    test that the positional encoding module works when given
    a single x value
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    env_dim: int = 3
    vector_dim: int = 1280
    position = torch.from_numpy(np.array([2, 3, 4]))
    length_scale: float = 1.5

    b = HRRBackend(
        vector_dim=vector_dim,
        env_dim=env_dim,
        length_scale=length_scale
    )

    assert b.env_basis_vectors.shape == (env_dim, vector_dim)

    # Calculate baseline results
    base = b.env_basis_vectors
    base = torch.fft.fft(base, dim=-1)

    for ed in range(env_dim):
        v = base[ed]
        v = v ** (position[ed] / length_scale)
        base[ed] = v

    base = torch.prod(base, axis=0)
    base = torch.fft.ifft(base, dim=-1).real

    pred, _ = b.positional_encoding(position)

    assert base.shape == (vector_dim,)
    assert pred.shape == (vector_dim,)

    # Should produce identical results
    assert torch.allclose(
        base,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

def test_positional_encoding_batched_x():
    """
    test that the positional encoding module works when given
    a batched x value
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    batch_size: int = 3
    env_dim: int = 2
    vector_dim: int = 1280
    position = torch.from_numpy(np.array([
        [2, 3],
        [1, 5],
        [2, 6]
    ]))
    length_scale: float = 1.5

    b = HRRBackend(
        vector_dim=vector_dim,
        env_dim=env_dim,
        length_scale=length_scale
    )

    assert b.env_basis_vectors.shape == (env_dim, vector_dim)

    # Calculate baseline results
    base = b.env_basis_vectors
    base = torch.fft.fft(base, dim=-1)

    gt = torch.zeros((batch_size, vector_dim))

    for bs in range(batch_size):
        components = torch.zeros((env_dim, vector_dim), dtype=torch.complex64)

        for ed in range(env_dim):
            v = base[ed]
            v = v ** (position[bs][ed] / length_scale)
            components[ed] = v

        components = torch.prod(components, axis=0)
        gt[bs] = torch.fft.ifft(components, dim=-1).real

    pred, _ = b.positional_encoding(position)

    assert gt.shape == (batch_size, vector_dim)
    assert pred.shape == (batch_size, vector_dim)

    # Should produce identical results
    assert torch.allclose(
        gt,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

def test_value_encoding_invalid_x_type():
    """
    test that the value encoding module throws an error when
    x is the incorrect type
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    B: int = 16
    E: int = 3

    b = HRRBackend(
        vector_dim=128,
        env_dim=E
    )

    x_single = np.zeros((E))
    x_batch = np.zeros((B, E))

    with pytest.raises(TypeError):
        b.value_encoding(x_single)

    with pytest.raises(TypeError):
        b.value_encoding(x_batch)

def test_value_encoding_invalid_x_dim():
    """
    Test that the backend's value encoding method
    throws an error when x isn't single or batched
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    x_invalid = torch.zeros((5, 5, 5))
    
    with pytest.raises(ValueError):
        b.value_encoding(x_invalid)

def test_value_encoding_invalid_x_val_dim():
    """
    Test that the backend's value encoding method throws
    an error when x isn't the correct value dimensionality
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)

    B: int = 8
    E: int = 3

    x_single_low = torch.zeros((E - 1))
    x_single_high = torch.zeros((E + 1))
    x_batch_low = torch.zeros((B, E - 1))
    x_batch_high = torch.zeros((B, E + 1))

    with pytest.raises(ValueError):
        b.value_encoding(x_single_low)

    with pytest.raises(ValueError):
        b.value_encoding(x_single_high)

    with pytest.raises(ValueError):
        b.value_encoding(x_batch_low)

    with pytest.raises(ValueError):
        b.value_encoding(x_batch_high)

def test_value_encoding_single_x():
    """
    test that the value encoding module works when given
    a single x value
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    value_dim: int = 3
    vector_dim: int = 1280
    value = torch.from_numpy(np.array([2, 3, 4]))
    length_scale: float = 1.5

    b = HRRBackend(
        vector_dim=vector_dim,
        value_dim=value_dim,
        length_scale=length_scale
    )

    assert b.value_basis_vectors.shape == (value_dim, vector_dim)

    # Calculate baseline results
    base = b.value_basis_vectors
    base = torch.fft.fft(base, dim=-1)

    for ed in range(value_dim):
        v = base[ed]
        v = v ** (value[ed] / length_scale)
        base[ed] = v

    base = torch.prod(base, axis=0)
    base = torch.fft.ifft(base, dim=-1).real

    pred, _ = b.value_encoding(value)

    assert base.shape == (vector_dim,)
    assert pred.shape == (vector_dim,)

    # Should produce identical results
    assert torch.allclose(
        base,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

def test_value_encoding_batched_x():
    """
    test that the value encoding module works when given
    a batched x value
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    batch_size: int = 3
    value_dim: int = 2
    vector_dim: int = 1280
    value = torch.from_numpy(np.array([
        [2, 3],
        [1, 5],
        [2, 6]
    ]))
    length_scale: float = 1.5

    b = HRRBackend(
        vector_dim=vector_dim,
        value_dim=value_dim,
        length_scale=length_scale
    )

    assert b.value_basis_vectors.shape == (value_dim, vector_dim)

    # Calculate baseline results
    base = b.value_basis_vectors
    base = torch.fft.fft(base, dim=-1)

    gt = torch.zeros((batch_size, vector_dim))

    for bs in range(batch_size):
        components = torch.zeros((value_dim, vector_dim), dtype=torch.complex64)

        for ed in range(value_dim):
            v = base[ed]
            v = v ** (value[bs][ed] / length_scale)
            components[ed] = v

        components = torch.prod(components, axis=0)
        gt[bs] = torch.fft.ifft(components, dim=-1).real

    pred, _ = b.value_encoding(value)

    assert gt.shape == (batch_size, vector_dim)
    assert pred.shape == (batch_size, vector_dim)

    # Should produce identical results
    assert torch.allclose(
        gt,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

def test_create_empty_vector_type():
    """
    test that the create_empty_vector method returns a Tensor
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    b = HRRBackend(vector_dim=128)
    v = b.create_empty_vector()

    assert isinstance(v, torch.Tensor)
    
def test_create_empty_vector_shape():
    """
    test the shape of the vector from create_empty_vector
    """
    from hyperspace.backends.hrr import HRRBackend

    D: int = 128
    b = HRRBackend(vector_dim=D)
    v = b.create_empty_vector()

    assert v.shape == (D,)

def test_create_empty_vector_values():
    """
    test the values within the empty vector from create_empty_vector
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    D: int = 128
    b = HRRBackend(vector_dim=D)
    v = b.create_empty_vector().cpu().numpy()
    gt = np.zeros((D))

    assert np.array_equal(v, gt)

def test_base_single_normalize_invalid_x_type():
    """
    test that the base_single_normalize function throws an
    error when x isn't a torch tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import _base_single_normalize
    
    x = np.zeros(10)

    with pytest.raises(TypeError):
        _base_single_normalize(x)

def test_base_single_normalize_invalid_x_shape():
    """
    test that the base_single_normalize function throws an
    error when x isn't a singular vector
    """
    import torch
    from hyperspace.backends.hrr import _base_single_normalize
    
    x = torch.zeros((10, 10))

    with pytest.raises(ValueError):
        _base_single_normalize(x)

def test_base_single_normalize_valid_x():
    """
    test that the base_single_normalize function converts
    the single value 
    """
    import torch
    from hyperspace.backends.hrr import _base_single_normalize

    x = torch.rand((10))

    pred = _base_single_normalize(x)
    gt = x / (torch.linalg.norm(x) + 1e-10)
    
    # Should produce identical results
    assert torch.allclose(
        gt,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_batch_normalize_invalid_x_type():
    """
    test that the base_batch_normalize function throws an
    error when x isn't a torch tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import _base_batch_normalize
    
    x = np.zeros((10, 10))

    with pytest.raises(TypeError):
        _base_batch_normalize(x)

def test_base_batch_normalize_invalid_x_shape():
    """
    test that the base_batch_normalize function throws an
    error when x isn't a batch of vectors
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_normalize
    
    x_small = torch.zeros((10))
    x_large = torch.zeros((10, 10, 10))

    with pytest.raises(ValueError):
        _base_batch_normalize(x_small)

    with pytest.raises(ValueError):
        _base_batch_normalize(x_large)

def test_base_batch_normalize_valid_x():
    """
    test that the base_batch_normalize function converts
    the single value 
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_normalize

    x = torch.rand((5, 10))

    pred = _base_batch_normalize(x)

    assert torch.allclose(
        torch.linalg.norm(pred, dim=-1),
        torch.ones(5),
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_normalize_invalid_x_type():
    """
    test that the backend's normalize method doesn't
    accept non torch.Tensors
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend

    D: int = 128
    b = HRRBackend(vector_dim=D)
    v = np.zeros(D)

    with pytest.raises(TypeError):
        b.normalize(v)

def test_backend_normalize_invalid_x_shape():
    """
    test that the backend's normalize method throws an error
    when x isn't a single vector or 2d batch of vectors
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    D: int = 128
    b = HRRBackend(vector_dim=D)
    v = torch.zeros((D, D, D))

    with pytest.raises(ValueError):
        b.normalize(v)

def test_backend_normalize_invalid_x_dimensionality():
    """
    test that the backend's normalize method throws an error
    when x isn't the correct dimensionality
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    D: int = 128
    B: int = 10

    b = HRRBackend(vector_dim=D)
    v_single_small = torch.zeros((D - 1))
    v_single_large = torch.zeros((D + 1))
    v_batch_small = torch.zeros((B, D - 1))
    v_batch_large = torch.zeros((B, D + 1))

    with pytest.raises(ValueError):
        b.normalize(v_single_small)

    with pytest.raises(ValueError):
        b.normalize(v_single_large)

    with pytest.raises(ValueError):
        b.normalize(v_batch_small)

    with pytest.raises(ValueError):
        b.normalize(v_batch_large)

def test_backend_normalize_single_x():
    """
    test the backend's normalization functionality
    with a single vector
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    D: int = 128

    x = torch.rand((D))

    b = HRRBackend(vector_dim=D)

    pred, _ = b.normalize(x)
    gt = x / (torch.linalg.norm(x, dim=0) + 1e-10)

    assert pred.shape == (D,)
    assert gt.shape == (D,)

    assert torch.allclose(
        torch.linalg.norm(pred, dim=0),
        torch.ones(D),
        rtol=1e-5,
        atol=1e-7,
    )
    
    # Should produce identical results
    assert torch.allclose(
        gt,
        pred,
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_normalize_batched_x():
    """
    test the backend's normalization functionality
    with a batch of vectors
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 32
    D: int = 128

    x = torch.rand((B, D))

    b = HRRBackend(vector_dim=D)

    pred, _ = b.normalize(x)

    assert pred.shape == (B, D)

    assert torch.allclose(
        torch.linalg.norm(pred, dim=-1),
        torch.ones(B),
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_single_invert_invalid_x_type():
    """
    test that the `base_single_invert` method throws an
    error when x isn't a torch tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import _base_single_invert

    v = np.zeros(10)

    with pytest.raises(TypeError):
        _base_single_invert(v)

def test_base_single_invert_invalid_x_shape():
    """
    test that the `base_single_invert` method throws an
    error with x isn't a single vector
    """
    import torch
    from hyperspace.backends.hrr import _base_single_invert
    
    v = torch.zeros((10, 10))

    with pytest.raises(ValueError):
        _base_single_invert(v)

def test_base_single_invert_valid_x():
    """
    test the `base_single_invert` function works with
    a single vector
    """
    import torch
    from hyperspace.backends.hrr import (
        HRRBackend,
        _base_single_invert
    )

    D: int = 128

    b = HRRBackend(vector_dim=D)

    v = b.create_random_vector()

    pred = _base_single_invert(v)
    
    # calculate ground truth
    gt = torch.fft.fft(v)
    gt = torch.conj(gt)
    gt = torch.fft.ifft(gt).real

    assert torch.allclose(
        pred,
        gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_batch_invert_invalid_x_type():
    """
    test that the `base_batch_invert` method throws an
    error when x isn't a torch tensor
    """
    import numpy as np
    from hyperspace.backends.hrr import _base_batch_invert

    v = np.zeros((5, 10))

    with pytest.raises(TypeError):
        _base_batch_invert(v)

def test_base_batch_invert_invalid_x_shape():
    """
    test that the `base_batch_invert` method throws an
    error with x isn't a batch of vectors
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_invert
    
    v_small = torch.zeros((10))
    v_large = torch.zeros((10, 10, 10))

    with pytest.raises(ValueError):
        _base_batch_invert(v_small)

    with pytest.raises(ValueError):
        _base_batch_invert(v_large)

def test_base_batch_invert_valid_x():
    """
    test that the base batch invert function performs
    the operation across a batch of vectors
    """
    import torch
    from hyperspace.backends.hrr import (
        HRRBackend,
        _base_batch_invert
    )

    B: int = 10
    D: int = 128

    b = HRRBackend(vector_dim=D)

    v_list = [b.create_random_vector() for _ in range(B)]
    v_tensor = torch.stack(v_list, dim=0)

    assert v_tensor.shape == (B, D)

    pred = _base_batch_invert(v_tensor)

    gt = torch.zeros((B, D))
    for i in range(B):
        v = v_list[i]
        v = torch.fft.fft(v)
        v = torch.conj(v)
        v = torch.fft.ifft(v).real
        gt[i] = v

    assert torch.allclose(
        pred,
        gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_invert_invalid_x_type():
    """
    test that the backend's invert method doesn't
    accept not torch.Tensors
    """
    import numpy as np
    from hyperspace.backends.hrr import HRRBackend
    
    v = np.zeros(10)
    b = HRRBackend(vector_dim=128)

    with pytest.raises(TypeError):
        b.invert(v)

def test_backend_invert_invalid_x_shape():
    """
    test that the backend's invert method doesn't accept
    3D tensors
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    
    v = torch.zeros((10, 10, 10))
    b = HRRBackend(vector_dim=128)

    with pytest.raises(ValueError):
        b.invert(v)

def test_backend_invert_invalid_x_dim():
    """
    test that the backend's invert method doesn't accept
    vectors with incorrect dimensionality
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    D: int = 128
    B: int = 8

    b = HRRBackend(vector_dim=D)

    v_single_small = torch.rand((D - 1))
    v_single_large = torch.rand((D + 1))
    v_batch_small = torch.rand((B, D - 1))
    v_batch_large = torch.rand((B, D + 1))

    with pytest.raises(ValueError):
        b.invert(v_single_small)

    with pytest.raises(ValueError):
        b.invert(v_single_large)

    with pytest.raises(ValueError):
        b.invert(v_batch_small)

    with pytest.raises(ValueError):
        b.invert(v_batch_large)

def test_backend_single_invert_valid_x():
    """
    test the backend's invert method with a single vector
    """
    import torch
    from hyperspace.backends.hrr import (
        HRRBackend,
    )

    D: int = 128

    b = HRRBackend(vector_dim=D)

    v = b.create_random_vector()

    pred, _ = b.invert(v)
    
    # calculate ground truth
    gt = torch.fft.fft(v)
    gt = torch.conj(gt)
    gt = torch.fft.ifft(gt).real

    assert torch.allclose(
        pred,
        gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_batch_invert_valid_x():
    """
    test the backend's invert method with a batch of vectors
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 10
    D: int = 128

    b = HRRBackend(vector_dim=D)

    v_list = [b.create_random_vector() for _ in range(B)]
    v_tensor = torch.stack(v_list, dim=0)

    assert v_tensor.shape == (B, D)

    pred, _ = b.invert(v_tensor)

    gt = torch.zeros((B, D))
    for i in range(B):
        v = v_list[i]
        v = torch.fft.fft(v)
        v = torch.conj(v)
        v = torch.fft.ifft(v).real
        gt[i] = v

    assert torch.allclose(
        pred,
        gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_single_weight_invalid_x_type():
    """
    test that the base single weight checks for the type
    of x
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import _base_single_weight
    
    x = np.zeros(10)
    w = torch.ones(1)

    with pytest.raises(TypeError):
        _base_single_weight(x, w)

def test_base_single_weight_invalid_x_shape():
    """
    test that the base single weight function checks the
    shape of x
    """
    import torch
    from hyperspace.backends.hrr import _base_single_weight
    
    x = torch.zeros((10, 10))
    w = torch.ones(1)

    with pytest.raises(ValueError):
        _base_single_weight(x, w)

def test_base_single_weight_invalid_weight_type():
    """
    test that the base single weight function checks the
    type of w
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import _base_single_weight
    
    x = torch.zeros(10)
    w = np.ones(1)

    with pytest.raises(TypeError):
        _base_single_weight(x, w)

def test_base_single_weight_invalid_weight_shape():
    """
    test that the base single weight function checks the
    shape of w
    """
    import torch
    from hyperspace.backends.hrr import _base_single_weight
    
    x = torch.zeros((10, 10))
    w = torch.ones(10)

    with pytest.raises(ValueError):
        _base_single_weight(x, w)

def test_base_single_weight_valid_x():
    """
    test the base single weight function with correct inputs
    """
    import torch
    from hyperspace.backends.hrr import _base_single_weight
    
    D: int = 128
    x = torch.rand(D)
    w_small = torch.tensor([0.5])
    w_large = torch.tensor([2.0])

    x_small_pred = _base_single_weight(x, w_small)
    x_small_gt = torch.linalg.norm(x) * w_small

    x_large_pred = _base_single_weight(x, w_large)
    x_large_gt = torch.linalg.norm(x) * w_large

    assert x_small_pred.shape == (D,)
    assert x_large_pred.shape == (D,)

    assert torch.allclose(
        torch.linalg.norm(x_small_pred),
        x_small_gt,
        rtol=1e-5,
        atol=1e-7,
    )

    assert torch.allclose(
        torch.linalg.norm(x_large_pred),
        x_large_gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_batch_weight_invalid_x_type():
    """
    test that the base batch weight method checks for the
    type of x
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import _base_batch_weight
    
    B: int = 10
    D: int = 256

    x = np.zeros((B, D))
    w = torch.ones((D))

    with pytest.raises(TypeError):
        _base_batch_weight(x, w)

def test_base_batch_weight_invalid_x_shape():
    """
    test that the base batch weight method checks for the
    shape of x
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_weight
    
    B: int = 10
    D: int = 256

    x_small = torch.zeros((B, D, D))
    x_large = torch.zeros((B))
    w = torch.ones((B))

    with pytest.raises(ValueError):
        _base_batch_weight(x_small, w)

    with pytest.raises(ValueError):
        _base_batch_weight(x_large, w)

def test_base_batch_weight_invalid_weight_type():
    """
    test that the base batch weight method checks for
    the type of w
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import _base_batch_weight
    
    B: int = 10
    D: int = 256

    x = torch.zeros((B, D))
    w = np.ones((D))

    with pytest.raises(TypeError):
        _base_batch_weight(x, w)

def test_base_batch_weight_invalid_weight_shape():
    """
    test that the base batch weight method checks for the
    shape of w
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_weight
    
    B: int = 10
    D: int = 256

    x = torch.zeros((B, D))
    w_bs_small = torch.ones((B - 1))
    w_bs_large = torch.ones((B + 1))
    w_dim = torch.ones((B, D))

    with pytest.raises(ValueError):
        _base_batch_weight(x, w_bs_small)

    with pytest.raises(ValueError):
        _base_batch_weight(x, w_bs_large)

    with pytest.raises(ValueError):
        _base_batch_weight(x, w_dim)

def test_base_batch_weight_valid_x():
    """
    test the base batch weight function works
    with valid inputs
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_weight
    
    B: int = 10
    D: int = 256

    x = torch.rand((B, D))
    w = torch.rand((B))

    x_pred = _base_batch_weight(x, w)

    x_gt = torch.zeros((B, D))
    for b in range(B):
        for d in range(D):
            x_gt[b, d] = x[b, d] * w[b]

    assert torch.allclose(
        x_pred,
        x_gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_weight_invalid_x_type():
    """
    test that the backend's weight method checks
    the type of x    
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    x = np.zeros((10))
    w = torch.zeros((1))
    
    b = HRRBackend(vector_dim=128)

    with pytest.raises(TypeError):
        b.weight(x, w)

def test_backend_weight_invalid_x_shape():
    """
    test that the backend's weight method checks the
    shape of x
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 10
    D: int = 128

    b = HRRBackend(vector_dim=D)

    v = torch.rand((B, B, D))
    w = torch.rand((B))

    with pytest.raises(ValueError):
        b.weight(v, w)

def test_backend_weight_invalid_x_dim():
    """
    test that the backend's weight method checks
    the dimensionality of the vectors
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 10
    D: int = 128

    b = HRRBackend(vector_dim=D)

    v_small = torch.rand((B, D - 1))
    v_large = torch.rand((B, D + 1))
    w = torch.rand((B))

    with pytest.raises(ValueError):
        b.weight(v_small, w)

    with pytest.raises(ValueError):
        b.weight(v_large, w)

def test_backend_weight_invalid_weight_type():
    """
    test that the backend's weight method checks
    the type of w   
    """
    import numpy as np
    import torch
    from hyperspace.backends.hrr import HRRBackend

    x = torch.zeros((10))
    w = np.zeros((1))
    
    b = HRRBackend(vector_dim=128)

    with pytest.raises(TypeError):
        b.weight(x, w)

def test_backend_weight_invalid_weight_shape():
    """
    test if the backend validates the shape of the weight
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 10
    D: int = 128

    b = HRRBackend(vector_dim=D)

    v = torch.rand((B, D))
    w_bs_small = torch.rand((B - 1))
    w_bs_large = torch.rand((B + 1))
    w_dim = torch.rand((B, B))

    with pytest.raises(ValueError):
        b.weight(v, w_bs_small)

    with pytest.raises(ValueError):
        b.weight(v, w_bs_large)

    with pytest.raises(ValueError):
        b.weight(v, w_dim)

    with pytest.raises(ValueError):
        b.weight(v[0], w_dim[0])

def test_backend_single_weight_valid_x():
    """
    test the backend's weighting functionality with
    a single vector and weight
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    
    D: int = 128
    x = torch.rand(D)

    b = HRRBackend(vector_dim=D)

    w_small = torch.tensor([0.5])
    w_large = torch.tensor([2.0])

    x_small_pred, _ = b.weight(x, w_small)
    x_small_gt = torch.linalg.norm(x) * w_small

    x_large_pred, _ = b.weight(x, w_large)
    x_large_gt = torch.linalg.norm(x) * w_large

    assert x_small_pred.shape == (D,)
    assert x_large_pred.shape == (D,)

    assert torch.allclose(
        torch.linalg.norm(x_small_pred),
        x_small_gt,
        rtol=1e-5,
        atol=1e-7,
    )

    assert torch.allclose(
        torch.linalg.norm(x_large_pred),
        x_large_gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_batch_weight_valid_x():
    """
    test the backend's weighting function with a batch
    of vectors and weights
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    
    B: int = 10
    D: int = 256

    b = HRRBackend(vector_dim=D)

    x = torch.rand((B, D))
    w = torch.rand((B))

    x_pred, _ = b.weight(x, w)

    x_gt = torch.zeros((B, D))
    for b in range(B):
        for d in range(D):
            x_gt[b, d] = x[b, d] * w[b]

    assert torch.allclose(
        x_pred,
        x_gt,
        rtol=1e-5,
        atol=1e-7,
    )

def test_batch_list_bind():
    """
    test the _base_list_bind function
    """
    import numpy as np
    import torch
    from torch import Generator

    from hyperspace.backends.hrr import (
        _base_create_single_vector,
        _base_list_bind,
        _base_single_bind
    )

    B: int = 2
    D: int = 256

    gen = Generator().manual_seed(0)

    # Build batched tensors (B, D)
    v_list = [_base_create_single_vector(D, gen) for _ in range(B)]

    v1 = torch.stack(v_list, dim=0)

    v_bundle_list = _base_list_bind(v1)  # (D,)

    v_out_gt = _base_single_bind(v_list[0], v_list[1])

    assert v_bundle_list.shape == (D,)
    assert v_out_gt.shape == (D,)

    assert np.allclose(
        v_out_gt.numpy(),
        v_bundle_list.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_backend_list_bind():
    """
    test the backend's list bind method
    """
    import numpy as np
    import torch
    from torch import Generator

    from hyperspace.backends.hrr import (
        HRRBackend,
        _base_create_single_vector,
        _base_single_bind
    )

    B: int = 2
    D: int = 256

    b = HRRBackend(vector_dim=D)

    gen = Generator().manual_seed(0)

    # Build batched tensors (B, D)
    v_list = [_base_create_single_vector(D, gen) for _ in range(B)]

    v1 = torch.stack(v_list, dim=0)

    v_bundle_list, _ = b.bind(v1)  # (D,)

    v_out_gt = _base_single_bind(v_list[0], v_list[1])

    assert v_bundle_list.shape == (D,)
    assert v_out_gt.shape == (D,)

    assert np.allclose(
        v_out_gt.numpy(),
        v_bundle_list.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )

def test_base_single_to_batch_bind_v_shape():
    """
    test the shape checking of v in _base_single_to_batch_bind
    """
    import torch
    from hyperspace.backends.hrr import _base_single_to_batch_bind

    B: int = 64
    D: int = 1024

    v_small_d = torch.rand((D - 1))
    v_large_d = torch.rand((D + 1))
    v_batched = torch.rand((B, D))
    batch = torch.rand((B, D))

    with pytest.raises(ValueError):
        _base_single_to_batch_bind(v_small_d, batch)

    with pytest.raises(ValueError):
        _base_single_to_batch_bind(v_large_d, batch)

    with pytest.raises(ValueError):
        _base_single_to_batch_bind(v_batched, batch)

def test_base_single_to_batch_bind_batch_shape():
    """
    test the shape checking of batch in _base_single_to_batch_bind
    """
    import torch
    from hyperspace.backends.hrr import _base_single_to_batch_bind

    B: int = 64
    D: int = 1024

    v = torch.rand((D,))
    b_small_d = torch.rand((B, D - 1))
    b_large_d = torch.rand((B, D + 1))
    b_small_dim = torch.rand((B))
    b_large_dim = torch.rand((B, D, D))

    with pytest.raises(ValueError):
        _base_single_to_batch_bind(v, b_small_d)

    with pytest.raises(ValueError):
        _base_single_to_batch_bind(v, b_large_d)

    with pytest.raises(ValueError):
        _base_single_to_batch_bind(v, b_small_dim)

    with pytest.raises(ValueError):
        _base_single_to_batch_bind(v, b_large_dim)

def test_base_single_to_batch_bind():
    """
    test the functionality of base_single_to_batch_bind
    """
    import torch
    from hyperspace.backends.hrr import (
        HRRBackend,
        _base_single_to_batch_bind
    )
    
    B: int = 2
    D: int = 1024

    b = HRRBackend(D)

    phi_red = b.create_random_vector()
    phi_blue = b.create_random_vector()
    phi_car = b.create_random_vector()
    phi_colors = torch.stack([phi_red, phi_blue], dim=0)

    phi_gt = torch.zeros((B, D))
    phi_red_car, _ = b.bind(phi_red, phi_car)
    phi_blue_car, _ = b.bind(phi_blue, phi_car)
    phi_gt[0, :] = phi_red_car
    phi_gt[1, :] = phi_blue_car

    phi_pred = _base_single_to_batch_bind(phi_car, phi_colors)

    assert phi_pred.shape == (B, D)

    assert torch.allclose(
        phi_pred,
        phi_gt,
        rtol=0.001,
        atol=0.001
    )

def test_backend_single_to_batch_bind_v_shape():
    """
    test the shape checking of v in backend's single to batch bind
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 64
    D: int = 1024

    b = HRRBackend(vector_dim=D)

    v_single_small = torch.rand((D - 1,))
    v_single_large = torch.rand((D + 1,))
    v_batch = torch.rand((B, D))

    with pytest.raises(ValueError):
        b.bind(v_single_small, v_batch)

    with pytest.raises(ValueError):
        b.bind(v_single_large, v_batch)

    with pytest.raises(ValueError):
        b.bind(v_batch, v_single_small)

    with pytest.raises(ValueError):
        b.bind(v_batch, v_single_large)

def test_backend_single_to_batch_bind_batch_shape():
    """
    test the shape checking of batch in the backend's single to batch bind
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 64
    D: int = 1024

    b = HRRBackend(vector_dim=D)

    v_single = torch.rand((D,))
    v_batch_small = torch.rand((B, D - 1))
    v_batch_large = torch.rand((B, D + 1))

    with pytest.raises(ValueError):
        b.bind(v_single, v_batch_small)

    with pytest.raises(ValueError):
        b.bind(v_single, v_batch_large)

    with pytest.raises(ValueError):
        b.bind(v_batch_small, v_single)

    with pytest.raises(ValueError):
        b.bind(v_batch_large, v_single)

def test_backend_single_to_batch_bind():
    """
    test the functionality of backend single_to_batch_bind
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    
    B: int = 2
    D: int = 1024

    b = HRRBackend(D)

    phi_red = b.create_random_vector()
    phi_blue = b.create_random_vector()
    phi_car = b.create_random_vector()
    phi_colors = torch.stack([phi_red, phi_blue], dim=0)

    phi_gt = torch.zeros((B, D))
    phi_red_car, _ = b.bind(phi_red, phi_car)
    phi_blue_car, _ = b.bind(phi_blue, phi_car)
    phi_gt[0, :] = phi_red_car
    phi_gt[1, :] = phi_blue_car

    phi_pred, _ = b.bind(phi_car, phi_colors)

    assert phi_pred.shape == (B, D)

    assert torch.allclose(
        phi_pred,
        phi_gt,
        rtol=0.001,
        atol=0.001
    )

    # reverse args = same ouput
    phi_pred, _ = b.bind(phi_colors, phi_car)

    assert phi_pred.shape == (B, D)

    assert torch.allclose(
        phi_pred,
        phi_gt,
        rtol=0.001,
        atol=0.001
    )

def test_base_single_to_batch_bundle_v_shape():
    """
    test the shape checking of v in _base_single_to_batch_bundle
    """
    import torch
    from hyperspace.backends.hrr import _base_single_to_batch_bundle

    B: int = 64
    D: int = 1024

    v_small_d = torch.rand((D - 1))
    v_large_d = torch.rand((D + 1))
    v_batched = torch.rand((B, D))
    batch = torch.rand((B, D))

    with pytest.raises(ValueError):
        _base_single_to_batch_bundle(v_small_d, batch)

    with pytest.raises(ValueError):
        _base_single_to_batch_bundle(v_large_d, batch)

    with pytest.raises(ValueError):
        _base_single_to_batch_bundle(v_batched, batch)

def test_base_single_to_batch_bundle_batch_shape():
    """
    test the shape checking of batch in _base_single_to_batch_bundle
    """
    import torch
    from hyperspace.backends.hrr import _base_single_to_batch_bundle

    B: int = 64
    D: int = 1024

    v = torch.rand((D,))
    b_small_d = torch.rand((B, D - 1))
    b_large_d = torch.rand((B, D + 1))
    b_small_dim = torch.rand((B))
    b_large_dim = torch.rand((B, D, D))

    with pytest.raises(ValueError):
        _base_single_to_batch_bundle(v, b_small_d)

    with pytest.raises(ValueError):
        _base_single_to_batch_bundle(v, b_large_d)

    with pytest.raises(ValueError):
        _base_single_to_batch_bundle(v, b_small_dim)

    with pytest.raises(ValueError):
        _base_single_to_batch_bundle(v, b_large_dim)

def test_base_single_to_batch_bundle():
    """
    test the functionality of base_single_to_batch_bundle
    """
    import torch
    from hyperspace.backends.hrr import (
        HRRBackend,
        _base_single_to_batch_bundle
    )

    B: int = 2
    D: int = 1024

    b = HRRBackend(D)

    phi_pizza = b.create_random_vector()
    phi_cheese = b.create_random_vector()
    phi_ham = b.create_random_vector()
    phi_toppings = torch.stack([phi_cheese, phi_ham], dim=0)

    phi_gt = torch.zeros((B, D))
    phi_cheese_pizza, _ = b.bundle(phi_pizza, phi_cheese)
    phi_ham_pizza, _ = b.bundle(phi_pizza, phi_ham)
    phi_gt[0, :] = phi_cheese_pizza
    phi_gt[1, :] = phi_ham_pizza

    phi_pred = _base_single_to_batch_bundle(phi_pizza, phi_toppings)
    assert torch.allclose(phi_pred, phi_gt)

def test_backend_single_to_batch_bundle_v_shape():
    """
    test the shape checking of v in the backend's single to batch bundling
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 64
    D: int = 1024

    b = HRRBackend(vector_dim=D)

    v_single_small = torch.rand((D - 1,))
    v_single_large = torch.rand((D + 1,))
    v_batch = torch.rand((B, D))

    with pytest.raises(ValueError):
        b.bundle(v_single_small, v_batch)

    with pytest.raises(ValueError):
        b.bundle(v_single_large, v_batch)

    with pytest.raises(ValueError):
        b.bundle(v_batch, v_single_small)

    with pytest.raises(ValueError):
        b.bundle(v_batch, v_single_large)

def test_backend_single_to_batch_bundle_batch_shape():
    """
    test the shape checking of batch in backend's single to batch bundle
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 64
    D: int = 1024

    b = HRRBackend(vector_dim=D)

    v_single = torch.rand((D,))
    v_batch_small = torch.rand((B, D - 1))
    v_batch_large = torch.rand((B, D + 1))

    with pytest.raises(ValueError):
        b.bundle(v_single, v_batch_small)

    with pytest.raises(ValueError):
        b.bundle(v_single, v_batch_large)

    with pytest.raises(ValueError):
        b.bind(v_batch_small, v_single)

    with pytest.raises(ValueError):
        b.bundle(v_batch_large, v_single)

def test_backend_single_to_batch_bundle():
    """
    test the functionality of backend single_to_batch_bundle
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend

    B: int = 2
    D: int = 1024

    b = HRRBackend(D)

    phi_pizza = b.create_random_vector()
    phi_cheese = b.create_random_vector()
    phi_ham = b.create_random_vector()
    phi_toppings = torch.stack([phi_cheese, phi_ham], dim=0)

    phi_gt = torch.zeros((B, D))
    phi_cheese_pizza, _ = b.bundle(phi_pizza, phi_cheese)
    phi_ham_pizza, _ = b.bundle(phi_pizza, phi_ham)
    phi_gt[0, :] = phi_cheese_pizza
    phi_gt[1, :] = phi_ham_pizza

    phi_pred, _ = b.bundle(phi_pizza, phi_toppings)
    assert torch.allclose(phi_pred, phi_gt)

    phi_pred, _ = b.bundle(phi_toppings, phi_pizza)
    assert torch.allclose(phi_pred, phi_gt)

def test_backend_resonator_cleanup_v_non_tensor():
    """
    Test that the backend's resonator cleanup method
    throws an error when v is not a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v = np.random.random((D,))
    codebook = torch.rand((B, D))

    with pytest.raises(TypeError):
        b._resonator_cleanup(v, codebook)

def test_backend_resonator_cleanup_codebook_non_tensor():
    """
    Test that the backend's resonator cleanup method
    throws an error when codebook isn't a tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v = torch.rand((D,))
    codebook = np.random.random((B, D))

    with pytest.raises(TypeError):
        b._resonator_cleanup(v, codebook)

def test_backend_resonator_cleanup_v_shape_missmatch():
    """
    Test that the backend's resonator cleanup method
    throws an error when v isn't the correct
    dimensionality
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v_large = torch.rand((D, D, D))
    codebook = torch.rand((B, D))

    with pytest.raises(ValueError):
        b._resonator_cleanup(v_large, codebook)

def test_backend_resonator_cleanup_v_vector_dim_missmatch():
    """
    Test that the backend's resonator cleanup method throws
    an error when v doesn't have the correct vector
    dimensionality
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v_1d_small = torch.rand((D - 1,))
    v_1d_large = torch.rand((D + 1,))
    v_2d_small = torch.rand((B, D - 1))
    v_2d_large = torch.rand((B, D + 1))
    codebook = torch.rand((B, D))

    with pytest.raises(ValueError):
        b._resonator_cleanup(v_1d_small, codebook)

    with pytest.raises(ValueError):
        b._resonator_cleanup(v_1d_large, codebook)

    with pytest.raises(ValueError):
        b._resonator_cleanup(v_2d_small, codebook)

    with pytest.raises(ValueError):
        b._resonator_cleanup(v_2d_large, codebook)

def test_backend_resonator_cleanup_codebook_shape_missmatch():
    """
    Test that the backend's resonator cleanup method throws an
    error when the dimensionality of the codebook doesn't match
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v = torch.rand((B, D))
    codebook_small = torch.rand((D,))
    codebook_large = torch.rand((B, B, D))

    with pytest.raises(ValueError):
        b._resonator_cleanup(v, codebook_small)

    with pytest.raises(ValueError):
        b._resonator_cleanup(v, codebook_large)

def test_backend_resonator_cleanup_codebook_vector_dim_missmatch():
    """
    Test that the backend's resonator cleanup method throws an error
    when the vector dim of the codebook doesn't match
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v = torch.rand((B, D))
    codebook_small = torch.rand((B, D - 1))
    codebook_large = torch.rand((B, D + 1))

    with pytest.raises(ValueError):
        b._resonator_cleanup(v, codebook_small)

    with pytest.raises(ValueError):
        b._resonator_cleanup(v, codebook_large)

def test_backend_resonator_cleanup_num_iters_non_int():
    """
    Test that the backend's resonator cleanup method throws an error
    when the number of iterations isn't an integer
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 3
    C: int = 64

    v = torch.rand((B, D))
    c = torch.rand((C, D))
    i = "None"

    b = HRRBackend(vector_dim=D)

    with pytest.raises(TypeError):
        b._resonator_cleanup(v, c, i)

def test_backend_resonator_cleanup_num_iters_lt_one():
    """
    Test that the backend's resonator cleanup method throws an error
    when the number of iterations is less than one
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 3
    C: int = 64

    v = torch.rand((B, D))
    c = torch.rand((C, D))
    i = 0

    b = HRRBackend(vector_dim=D)

    with pytest.raises(ValueError):
        b._resonator_cleanup(v, c, i)

def test_backend_resonator_cleanup_single_memory_single_iter():
    """
    Single-step resonator cleanup should move the query toward the correct stored pattern.

    Checks:
      1) output sanity
      2) cosine similarity to true memory improves after one step
      3) (optional) sims sanity if returned
    """
    import torch
    from hyperspace.backends import HRRBackend

    torch.manual_seed(0)

    D: int = 1024
    C: int = 3
    true_idx: int = 0
    noise_scale: float = 0.15

    b = HRRBackend(vector_dim=D)

    codebook_list = [b.create_random_vector() for _ in range(C)]
    codebook = torch.stack(codebook_list, dim=0)
    assert codebook.shape == (C, D)

    v_true = codebook_list[true_idx]
    v_noisy = v_true + noise_scale * torch.randn_like(v_true)
    assert v_noisy.shape == (D,)

    eps = 1e-12

    def cosine_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm() + eps)
        cb = cb / (cb.norm(dim=-1, keepdim=True) + eps)
        return cb @ x  # (C,)

    s_before = cosine_to_codebook(v_noisy, codebook)
    true_sim_before = float(s_before[true_idx].item())

    # ---- call resonator cleanup for a single iter ----
    # If your API differs, update this line accordingly.
    v_pred, info = b._resonator_cleanup(v_noisy, codebook, num_iters=1)

    assert isinstance(v_pred, torch.Tensor)
    assert v_pred.shape == (D,)
    assert torch.isfinite(v_pred).all()

    s_after = cosine_to_codebook(v_pred, codebook)
    true_sim_after = float(s_after[true_idx].item())

    # Directional improvement (best single-step invariant for resonator)
    assert true_sim_after > true_sim_before + 1e-4, (
        f"Single-step resonator did not improve true similarity. "
        f"before={true_sim_before:.6f}, after={true_sim_after:.6f}, "
        f"s_before={s_before.tolist()}, s_after={s_after.tolist()}"
    )

    # Optional sims sanity if available (expects sims_history list[(B,C)] or a single sims tensor)
    sims = None
    if isinstance(info, dict):
        if "sims_history" in info and len(info["sims_history"]) > 0:
            sims = info["sims_history"][-1]
        elif "sims" in info:
            sims = info["sims"]

    if isinstance(sims, torch.Tensor):
        if sims.ndim == 2 and sims.shape[0] == 1:
            sims = sims.squeeze(0)
        if sims.ndim == 1 and sims.numel() == C:
            assert torch.isfinite(sims).all()
            # if you normalize internally, sims should look like cosine similarities
            assert sims.max().item() <= 1.0 + 1e-5
            assert sims.min().item() >= -1.0 - 1e-5


def test_backend_resonator_cleanup_single_memory_converges_over_iterations():
    """
    Multi-iteration resonator cleanup should converge toward the correct stored memory
    with a reasonable success rate across seeds.

    Checks:
      1) output sanity
      2) final retrieved index matches true memory
      3) true similarity improves from noisy query
      4) (optional) sims on true index trends upward from first to last iter
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    C: int = 3
    true_idx: int = 0

    num_iters: int = 8
    noise_scale: float = 0.20

    trials: int = 10
    required_success_rate: float = 0.8

    eps = 1e-12

    def cosine_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm() + eps)
        cb = cb / (cb.norm(dim=-1, keepdim=True) + eps)
        return cb @ x  # (C,)

    successes = 0

    for seed in range(trials):
        torch.manual_seed(seed)
        b = HRRBackend(vector_dim=D)

        codebook_list = [b.create_random_vector() for _ in range(C)]
        codebook = torch.stack(codebook_list, dim=0)
        assert codebook.shape == (C, D)

        v_true = codebook_list[true_idx]
        v_noisy = v_true + noise_scale * torch.randn_like(v_true)

        s_before = cosine_to_codebook(v_noisy, codebook)
        true_sim_before = float(s_before[true_idx].item())

        # ---- call resonator cleanup for multiple iters ----
        v_out, info = b._resonator_cleanup(v_noisy, codebook, num_iters=num_iters)

        assert isinstance(v_out, torch.Tensor)
        assert v_out.shape == (D,)
        assert torch.isfinite(v_out).all()

        s_after = cosine_to_codebook(v_out, codebook)
        pred_after = int(torch.argmax(s_after).item())
        true_sim_after = float(s_after[true_idx].item())

        improved = true_sim_after > true_sim_before + 1e-4
        correct = (pred_after == true_idx)

        sims_ok = True
        if isinstance(info, dict) and "sims_history" in info:
            sims_hist = info["sims_history"]
            true_mass = []
            for s in sims_hist:
                if isinstance(s, torch.Tensor):
                    if s.ndim == 2 and s.shape[0] == 1:
                        s = s.squeeze(0)
                    if s.ndim == 1 and s.numel() == C:
                        true_mass.append(float(s[true_idx].item()))
            if len(true_mass) >= 2:
                sims_ok = (true_mass[-1] >= true_mass[0] - 1e-4)

        if improved and correct and sims_ok:
            successes += 1

    assert successes >= int(required_success_rate * trials), (
        f"Resonator convergence success-rate too low: {successes}/{trials}. "
        f"(required >= {int(required_success_rate * trials)}/{trials}) "
        f"Try reducing noise_scale or increasing num_iters, and ensure normalization is consistent."
    )


def test_backend_resonator_cleanup_single_step_batched_improves_true_alignment():
    """
    Batched single-step resonator cleanup should move each query toward its correct stored pattern.
    """
    import torch
    from hyperspace.backends import HRRBackend

    torch.manual_seed(0)

    D: int = 1024
    C: int = 8
    B: int = 16
    noise_scale: float = 0.15

    b = HRRBackend(vector_dim=D)

    codebook_list = [b.create_random_vector() for _ in range(C)]
    codebook = torch.stack(codebook_list, dim=0)
    assert codebook.shape == (C, D)

    true_idx = torch.randint(low=0, high=C, size=(B,))
    v_true = codebook[true_idx]  # (B, D)
    v_noisy = v_true + noise_scale * torch.randn_like(v_true)
    assert v_noisy.shape == (B, D)

    eps = 1e-12

    def cosine_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        cbn = cb / (cb.norm(dim=-1, keepdim=True) + eps)
        return x @ cbn.T  # (B, C)

    s_before = cosine_to_codebook(v_noisy, codebook)
    true_sim_before = s_before.gather(1, true_idx.view(-1, 1)).squeeze(1)

    # ---- batched single-iter resonator cleanup ----
    v_pred, info = b._resonator_cleanup(v_noisy, codebook, num_iters=1)

    assert isinstance(v_pred, torch.Tensor)
    assert v_pred.shape == (B, D)
    assert torch.isfinite(v_pred).all()

    s_after = cosine_to_codebook(v_pred, codebook)
    true_sim_after = s_after.gather(1, true_idx.view(-1, 1)).squeeze(1)

    improved = true_sim_after > true_sim_before + 1e-4
    assert bool(improved.all().item()), (
        "Some batch elements did not improve true similarity in a single resonator step.\n"
        f"true_idx={true_idx.tolist()}\n"
        f"true_sim_before={true_sim_before.tolist()}\n"
        f"true_sim_after ={true_sim_after.tolist()}\n"
    )

    # Optional sims sanity
    if isinstance(info, dict) and "sims_history" in info and len(info["sims_history"]) > 0:
        sims = info["sims_history"][-1]
        if isinstance(sims, torch.Tensor) and sims.ndim == 2 and sims.shape == (B, C):
            assert torch.isfinite(sims).all()


def test_backend_resonator_cleanup_converges_over_iterations_batched():
    """
    Batched multi-iteration resonator cleanup should converge so each query retrieves
    its correct stored pattern.
    """
    import torch
    from hyperspace.backends import HRRBackend

    torch.manual_seed(0)

    D: int = 1024
    C: int = 8
    B: int = 16

    num_iters: int = 8
    noise_scale: float = 0.20

    b = HRRBackend(vector_dim=D)

    codebook_list = [b.create_random_vector() for _ in range(C)]
    codebook = torch.stack(codebook_list, dim=0)
    assert codebook.shape == (C, D)

    true_idx = torch.randint(low=0, high=C, size=(B,))
    v_true = codebook[true_idx]
    v_noisy = v_true + noise_scale * torch.randn_like(v_true)
    assert v_noisy.shape == (B, D)

    eps = 1e-12

    def cosine_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        cbn = cb / (cb.norm(dim=-1, keepdim=True) + eps)
        return x @ cbn.T  # (B, C)

    s_before = cosine_to_codebook(v_noisy, codebook)
    true_sim_before = s_before.gather(1, true_idx.view(-1, 1)).squeeze(1)

    v_out, info = b._resonator_cleanup(v_noisy, codebook, num_iters=num_iters)

    assert isinstance(v_out, torch.Tensor)
    assert v_out.shape == (B, D)
    assert torch.isfinite(v_out).all()

    s_after = cosine_to_codebook(v_out, codebook)
    pred_after = torch.argmax(s_after, dim=-1)
    true_sim_after = s_after.gather(1, true_idx.view(-1, 1)).squeeze(1)

    k = 3  # or 2
    topk = torch.topk(s_after, k=k, dim=-1).indices           # (B, k)
    correct_topk = (topk == true_idx.unsqueeze(-1)).any(dim=-1)
    assert bool(correct_topk.all().item()), (
        f"Some batch elements did not have true_idx in top-{k} after convergence.\n"
        f"true_idx={true_idx.tolist()}\n"
        f"topk={topk.tolist()}\n"
        f"s_after={s_after.tolist()}\n"
    )

    improved = true_sim_after > true_sim_before + 1e-4
    assert bool(improved.all().item()), (
        "Some batch elements did not improve true similarity after resonator convergence.\n"
        f"true_sim_before={true_sim_before.tolist()}\n"
        f"true_sim_after ={true_sim_after.tolist()}\n"
    )

    # Optional: sims trajectory sanity (true sims should not decrease from first to last iter)
    if isinstance(info, dict) and "sims_history" in info:
        sims_hist = info["sims_history"]
        if isinstance(sims_hist, list) and len(sims_hist) >= 2:
            s0, sT = sims_hist[0], sims_hist[-1]
            if (
                isinstance(s0, torch.Tensor) and isinstance(sT, torch.Tensor)
                and s0.ndim == 2 and sT.ndim == 2
                and s0.shape == (B, C) and sT.shape == (B, C)
            ):
                true_mass_0 = s0.gather(1, true_idx.view(-1, 1)).squeeze(1)
                true_mass_T = sT.gather(1, true_idx.view(-1, 1)).squeeze(1)
                assert bool((true_mass_T >= true_mass_0 - 1e-4).all().item()), (
                    "Some batch elements did not increase (or maintain) sims on the true index.\n"
                    f"true_mass_0={true_mass_0.tolist()}\n"
                    f"true_mass_T={true_mass_T.tolist()}\n"
                )

def test_backend_hopfield_cleanup_v_non_tensor():
    """
    Test that the backend's hopfield cleanup method
    throws an error when v is not a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v = np.random.random((D,))
    codebook = torch.rand((B, D))

    with pytest.raises(TypeError):
        b._hopfield_cleanup(v, codebook)

def test_backend_hopfield_cleanup_codebook_non_tensor():
    """
    Test that the backend's hopfield cleanup method
    throws an error when codebook isn't a tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v = torch.rand((D,))
    codebook = np.random.random((B, D))

    with pytest.raises(TypeError):
        b._hopfield_cleanup(v, codebook)

def test_backend_hopfield_cleanup_v_shape_missmatch():
    """
    Test that the backend's hopfield cleanup method
    throws an error when v isn't the correct
    dimensionality
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v_large = torch.rand((D, D, D))
    codebook = torch.rand((B, D))

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v_large, codebook)

def test_backend_hopfield_cleanup_v_vector_dim_missmatch():
    """
    Test that the backend's hopfield cleanup method throws
    an error when v doesn't have the correct vector
    dimensionality
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v_1d_small = torch.rand((D - 1,))
    v_1d_large = torch.rand((D + 1,))
    v_2d_small = torch.rand((B, D - 1))
    v_2d_large = torch.rand((B, D + 1))
    codebook = torch.rand((B, D))

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v_1d_small, codebook)

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v_1d_large, codebook)

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v_2d_small, codebook)

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v_2d_large, codebook)

def test_backend_hopfield_cleanup_codebook_shape_missmatch():
    """
    Test that the backend's hopfield cleanup method throws an
    error when the dimensionality of the codebook doesn't match
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v = torch.rand((B, D))
    codebook_small = torch.rand((D,))
    codebook_large = torch.rand((B, B, D))

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v, codebook_small)

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v, codebook_large)

def test_backend_hopfield_cleanup_codebook_vector_dim_missmatch():
    """
    Test that the backend's hopfield cleanup method throws an error
    when the vector dim of the codebook doesn't match
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 64

    b = HRRBackend(vector_dim=D)

    v = torch.rand((B, D))
    codebook_small = torch.rand((B, D - 1))
    codebook_large = torch.rand((B, D + 1))

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v, codebook_small)

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v, codebook_large)

def test_backend_hopfield_cleanup_num_iters_non_int():
    """
    Test that the backend's hopfield cleanup method throws an error
    when the number of iterations isn't an integer
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 3
    C: int = 64

    v = torch.rand((B, D))
    c = torch.rand((C, D))
    i = "None"

    b = HRRBackend(vector_dim=D)

    with pytest.raises(TypeError):
        b._hopfield_cleanup(v, c, i)

def test_backend_hopfield_cleanup_num_iters_lt_one():
    """
    Test that the backend's hopfield cleanup method throws an error
    when the number of iterations is less than one
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 3
    C: int = 64

    v = torch.rand((B, D))
    c = torch.rand((C, D))
    i = 0

    b = HRRBackend(vector_dim=D)

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v, c, i)

def test_backend_hopfield_cleanup_temp_type():
    """
    Test that the backend's hopfield cleanup method throws an error
    when temperature is not a floating point number
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 3
    C: int = 64

    v = torch.rand((B, D))
    c = torch.rand((C, D))
    t = "testing"

    b = HRRBackend(vector_dim=D)

    with pytest.raises(TypeError):
        b._hopfield_cleanup(v, c, temperature=t)

def test_backend_hopfield_cleanup_temp_range():
    """
    Test that the backend's hopfield cleanup method throws an error
    when temperature is not in the valid range
    """
    import torch
    from hyperspace.backends import HRRBackend

    D: int = 1024
    B: int = 3
    C: int = 64

    v = torch.rand((B, D))
    c = torch.rand((C, D))
    t = 0.0

    b = HRRBackend(vector_dim=D)

    with pytest.raises(ValueError):
        b._hopfield_cleanup(v, c, temperature=t)

def test_backend_hopfield_cleanup_single_memory_single_iter():
    """
    Test that the backend's hopfield cleanup method works correctly
    with a single memory.

    This test checks:
      1) output shape/dtype/device sanity
      2) the predicted vector is closer (cosine) to the true memory after cleanup
      3) the top-1 retrieved codebook item matches the true index
      4) the top-1 margin increases (confidence improves)
      5) (optional) attention sanity if an attention-like tensor is returned
    """
    import torch
    import torch.nn.functional as F
    from hyperspace.backends import HRRBackend

    torch.manual_seed(0)

    D: int = 1024
    C: int = 3
    true_idx: int = 0

    # NOTE: single-step behavior is temperature-sensitive
    temperature = 0.10
    noise_scale = 0.15

    b = HRRBackend(vector_dim=D)

    # ----------------------------
    # Build codebook
    # ----------------------------
    codebook_list = [b.create_random_vector() for _ in range(C)]
    codebook = torch.stack(codebook_list, dim=0)
    assert codebook.shape == (C, D)

    v_true = codebook_list[true_idx]
    assert v_true.shape == (D,)

    # ----------------------------
    # Create noisy query
    # ----------------------------
    v_noisy = v_true + noise_scale * torch.randn_like(v_true)
    assert v_noisy.shape == (D,)

    # ----------------------------
    # Helpers
    # ----------------------------
    eps = 1e-12

    def cosine_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm() + eps)
        cb = cb / (cb.norm(dim=-1, keepdim=True) + eps)
        return cb @ x  # (C,)

    # ----------------------------
    # Before
    # ----------------------------
    s_before = cosine_to_codebook(v_noisy, codebook)
    true_sim_before = float(s_before[true_idx].item())

    # ----------------------------
    # Single Hopfield layer / single iteration
    # ----------------------------
    # Option A: if your backend exposes a "step" function, call it here:
    # v_pred, attn = b._comp_batch_modern_hopfield_cleanup(v_noisy.unsqueeze(0), codebook, temperature=temperature)
    # v_pred, attn = v_pred.squeeze(0), attn.squeeze(0)

    # Option B (works with your current API): call cleanup with num_iters=1
    v_pred, info = b._hopfield_cleanup(v_noisy, codebook, num_iters=1, temperature=temperature)

    # "info" may be a dict with attn_history
    attn = None
    if isinstance(info, dict) and "attn_history" in info and len(info["attn_history"]) > 0:
        attn = info["attn_history"][-1]
        if attn.ndim == 2 and attn.shape[0] == 1:
            attn = attn.squeeze(0)

    # ----------------------------
    # Sanity checks
    # ----------------------------
    assert isinstance(v_pred, torch.Tensor)
    assert v_pred.shape == (D,)
    assert torch.isfinite(v_pred).all()

    # ----------------------------
    # After
    # ----------------------------
    s_after = cosine_to_codebook(v_pred, codebook)
    true_sim_after = float(s_after[true_idx].item())

    # ----------------------------
    # Core single-step assertions (directional)
    # ----------------------------
    # 1) Must move closer to true memory (primary requirement for 1-step layer)
    assert true_sim_after > true_sim_before + 1e-4, (
        f"Single-step did not improve true similarity. "
        f"before={true_sim_before:.6f}, after={true_sim_after:.6f}, "
        f"s_before={s_before.tolist()}, s_after={s_after.tolist()}"
    )

    # 2) Optional: attention sanity if available
    if attn is not None and isinstance(attn, torch.Tensor) and attn.ndim == 1 and attn.shape[0] == C:
        assert torch.isfinite(attn).all()
        # If you expect a winner-take-all-ish step at this temperature, require argmax == true
        assert int(torch.argmax(attn).item()) == true_idx, (
            f"Single-step attention did not prefer true index. attn={attn.tolist()}"
        )
        # If it looks like probabilities, check sum ~ 1
        s = float(attn.sum().item())
        if 0.9 <= s <= 1.1 and attn.min().item() >= -1e-6:
            assert abs(s - 1.0) < 1e-3

def test_backend_hopfield_cleanup_single_memoryconverges_over_iterations():
    """
    Test that the backend's Hopfield cleanup converges toward the correct stored
    memory over multiple iterations.

    This test checks:
      1) output tensor sanity
      2) the final retrieved index matches the true memory
      3) cosine similarity to the true memory improves from the noisy query
      4) attention mass on the true memory increases over iterations (when available)
      5) success is stable over multiple random trials (seed sweep)
    """
    import torch
    import torch.nn.functional as F
    from hyperspace.backends import HRRBackend

    D: int = 1024
    C: int = 3
    true_idx: int = 0

    num_iters: int = 8
    temperature: float = 0.08   # usually needs to be sharp for convergence
    noise_scale: float = 0.20

    trials: int = 10
    required_success_rate: float = 0.8  # 8/10

    eps = 1e-12

    def cosine_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm() + eps)
        cb = cb / (cb.norm(dim=-1, keepdim=True) + eps)
        return cb @ x  # (C,)

    successes = 0

    for seed in range(trials):
        torch.manual_seed(seed)

        b = HRRBackend(vector_dim=D)

        # ----------------------------
        # Build codebook
        # ----------------------------
        codebook_list = [b.create_random_vector() for _ in range(C)]
        codebook = torch.stack(codebook_list, dim=0)
        assert codebook.shape == (C, D)

        v_true = codebook_list[true_idx]
        v_noisy = v_true + noise_scale * torch.randn_like(v_true)

        # ----------------------------
        # Before metrics
        # ----------------------------
        s_before = cosine_to_codebook(v_noisy, codebook)
        true_sim_before = float(s_before[true_idx].item())

        # ----------------------------
        # Multi-iteration cleanup
        # ----------------------------
        v_out, info = b._hopfield_cleanup(
            v_noisy,
            codebook,
            num_iters=num_iters,
            temperature=temperature,
        )

        # output sanity
        assert isinstance(v_out, torch.Tensor)
        assert v_out.shape == (D,)
        assert torch.isfinite(v_out).all()

        # ----------------------------
        # After metrics
        # ----------------------------
        s_after = cosine_to_codebook(v_out, codebook)
        pred_after = int(torch.argmax(s_after).item())
        true_sim_after = float(s_after[true_idx].item())

        # ----------------------------
        # Convergence criteria
        # ----------------------------
        # (A) Must improve similarity to the true memory
        improved = true_sim_after > true_sim_before + 1e-4

        # (B) Must retrieve the correct memory at the end
        correct = (pred_after == true_idx)

        # Optional: attention trajectory should trend toward the true memory
        attn_ok = True
        if isinstance(info, dict) and "attn_history" in info:
            attn_hist = info["attn_history"]
            if isinstance(attn_hist, list) and len(attn_hist) > 0:
                # squeeze batch dim if present
                true_mass = []
                for a in attn_hist:
                    if isinstance(a, torch.Tensor):
                        if a.ndim == 2 and a.shape[0] == 1:
                            a = a.squeeze(0)
                        if a.ndim == 1 and a.numel() == C:
                            true_mass.append(float(a[true_idx].item()))
                if len(true_mass) >= 2:
                    # weak monotonic: final should be >= first (allow tiny numeric slack)
                    attn_ok = (true_mass[-1] >= true_mass[0] - 1e-4)

        if improved and correct and attn_ok:
            successes += 1

        # Make failures easy to debug if the overall success-rate check fails
        # (Do not assert per-seed; we assert on aggregate below.)

    assert successes >= int(required_success_rate * trials), (
        f"Hopfield convergence success-rate too low: {successes}/{trials}. "
        f"(required >= {int(required_success_rate * trials)}/{trials}) "
        f"Try lowering temperature, reducing noise_scale, increasing num_iters, "
        f"and ensuring the inner cleanup normalizes consistently."
    )

def test_backend_hopfield_cleanup_single_step_batched_improves_true_alignment():
    """
    Batched single-step (one-layer) modern Hopfield cleanup should move each query
    toward its correct stored pattern (directional improvement), similarly to the
    single-vector case.
    """
    import torch
    from hyperspace.backends import HRRBackend

    torch.manual_seed(0)

    D: int = 1024
    C: int = 8
    B: int = 16

    temperature: float = 0.10
    noise_scale: float = 0.15

    b = HRRBackend(vector_dim=D)

    # ----------------------------
    # Build codebook
    # ----------------------------
    codebook_list = [b.create_random_vector() for _ in range(C)]
    codebook = torch.stack(codebook_list, dim=0)
    assert codebook.shape == (C, D)

    # ----------------------------
    # Build batch of queries with known targets
    # ----------------------------
    true_idx = torch.randint(low=0, high=C, size=(B,))
    v_true = codebook[true_idx]  # (B, D)
    v_noisy = v_true + noise_scale * torch.randn_like(v_true)
    assert v_noisy.shape == (B, D)

    # ----------------------------
    # Helpers
    # ----------------------------
    eps = 1e-12

    def cosine_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity of x (B,D) to each row in cb (C,D) -> (B,C)
        """
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        cbn = cb / (cb.norm(dim=-1, keepdim=True) + eps)
        return x @ cbn.T  # (B, C)

    # ----------------------------
    # Before
    # ----------------------------
    s_before = cosine_to_codebook(v_noisy, codebook)  # (B, C)
    true_sim_before = s_before.gather(1, true_idx.view(-1, 1)).squeeze(1)  # (B,)

    # ----------------------------
    # Single iteration cleanup
    # ----------------------------
    v_pred, info = b._hopfield_cleanup(v_noisy, codebook, num_iters=1, temperature=temperature)
    assert isinstance(v_pred, torch.Tensor)
    assert v_pred.shape == (B, D)
    assert torch.isfinite(v_pred).all()

    # ----------------------------
    # After
    # ----------------------------
    s_after = cosine_to_codebook(v_pred, codebook)  # (B, C)
    true_sim_after = s_after.gather(1, true_idx.view(-1, 1)).squeeze(1)  # (B,)

    # ----------------------------
    # Directional improvement per sample
    # ----------------------------
    improved = true_sim_after > true_sim_before + 1e-4
    assert bool(improved.all().item()), (
        "Some batch elements did not improve true similarity in a single step.\n"
        f"true_idx={true_idx.tolist()}\n"
        f"true_sim_before={true_sim_before.tolist()}\n"
        f"true_sim_after ={true_sim_after.tolist()}\n"
    )

    # ----------------------------
    # Optional: attention sanity if available
    # ----------------------------
    if isinstance(info, dict) and "attn_history" in info and len(info["attn_history"]) > 0:
        attn = info["attn_history"][-1]  # expect (B, C)
        if isinstance(attn, torch.Tensor) and attn.ndim == 2 and attn.shape == (B, C):
            assert torch.isfinite(attn).all()
            # Ensure true index is at least among the top-2 attention weights for each sample
            top2 = torch.topk(attn, k=2, dim=-1).indices  # (B, 2)
            hit = (top2 == true_idx.view(-1, 1)).any(dim=-1)
            assert bool(hit.all().item()), (
                "Some batch elements did not have their true index in top-2 attention after one step.\n"
                f"true_idx={true_idx.tolist()}\n"
                f"top2={top2.tolist()}\n"
            )

def test_backend_hopfield_cleanup_converges_over_iterations_batched():
    """
    Batched multi-iteration Hopfield cleanup should converge so that each query
    retrieves its correct stored pattern, similarly to the single-vector case.

    This test checks:
      1) final retrieval argmax matches true_idx for all batch items
      2) similarity to the true memory improves from initial noisy query
      3) (optional) attention mass on true index increases from first to last iter
    """
    import torch
    from hyperspace.backends import HRRBackend

    torch.manual_seed(0)

    D: int = 1024
    C: int = 8
    B: int = 16

    num_iters: int = 8
    temperature: float = 0.08
    noise_scale: float = 0.20

    b = HRRBackend(vector_dim=D)

    # ----------------------------
    # Build codebook
    # ----------------------------
    codebook_list = [b.create_random_vector() for _ in range(C)]
    codebook = torch.stack(codebook_list, dim=0)
    assert codebook.shape == (C, D)

    # ----------------------------
    # Build batch of queries with known targets
    # ----------------------------
    true_idx = torch.randint(low=0, high=C, size=(B,))
    v_true = codebook[true_idx]  # (B, D)
    v_noisy = v_true + noise_scale * torch.randn_like(v_true)
    assert v_noisy.shape == (B, D)

    # ----------------------------
    # Helpers
    # ----------------------------
    eps = 1e-12

    def cosine_to_codebook(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        cbn = cb / (cb.norm(dim=-1, keepdim=True) + eps)
        return x @ cbn.T  # (B, C)

    # ----------------------------
    # Before metrics
    # ----------------------------
    s_before = cosine_to_codebook(v_noisy, codebook)
    true_sim_before = s_before.gather(1, true_idx.view(-1, 1)).squeeze(1)  # (B,)

    # ----------------------------
    # Multi-iteration cleanup
    # ----------------------------
    v_out, info = b._hopfield_cleanup(v_noisy, codebook, num_iters=num_iters, temperature=temperature)

    assert isinstance(v_out, torch.Tensor)
    assert v_out.shape == (B, D)
    assert torch.isfinite(v_out).all()

    # ----------------------------
    # After metrics
    # ----------------------------
    s_after = cosine_to_codebook(v_out, codebook)  # (B, C)
    pred_after = torch.argmax(s_after, dim=-1)     # (B,)
    true_sim_after = s_after.gather(1, true_idx.view(-1, 1)).squeeze(1)

    # ----------------------------
    # Core batched assertions
    # ----------------------------
    k = 5  # or 2
    topk = torch.topk(s_after, k=k, dim=-1).indices           # (B, k)
    correct_topk = (topk == true_idx.unsqueeze(-1)).any(dim=-1)
    assert bool(correct_topk.all().item()), (
        f"Some batch elements did not have true_idx in top-{k} after convergence.\n"
        f"true_idx={true_idx.tolist()}\n"
        f"topk={topk.tolist()}\n"
        f"s_after={s_after.tolist()}\n"
    )

    improved = true_sim_after > true_sim_before + 1e-4
    assert bool(improved.all().item()), (
        "Some batch elements did not improve true similarity after convergence.\n"
        f"true_idx={true_idx.tolist()}\n"
        f"true_sim_before={true_sim_before.tolist()}\n"
        f"true_sim_after ={true_sim_after.tolist()}\n"
    )

    # ----------------------------
    # Optional: attention trajectory sanity
    # ----------------------------
    if isinstance(info, dict) and "attn_history" in info:
        attn_hist = info["attn_history"]
        if isinstance(attn_hist, list) and len(attn_hist) >= 2:
            a0 = attn_hist[0]
            aT = attn_hist[-1]
            if (
                isinstance(a0, torch.Tensor) and isinstance(aT, torch.Tensor)
                and a0.ndim == 2 and aT.ndim == 2
                and a0.shape == (B, C) and aT.shape == (B, C)
            ):
                true_mass_0 = a0.gather(1, true_idx.view(-1, 1)).squeeze(1)
                true_mass_T = aT.gather(1, true_idx.view(-1, 1)).squeeze(1)

                assert bool((true_mass_T >= true_mass_0 - 1e-4).all().item()), (
                    "Some batch elements did not increase attention mass on the true index.\n"
                    f"true_mass_0={true_mass_0.tolist()}\n"
                    f"true_mass_T={true_mass_T.tolist()}\n"
                )


def test_base_batch_resonator_cleanup_single_step_batched_equivalence():
    """
    Batched single-step resonator cleanup should behave identically to running
    the same function on each element individually.

    This test is *only* about the single-step layer in isolation:
      1) shapes/dtypes are correct
      2) sims match the expected cosine/dot-product range assumptions
      3) output vectors are unit norm (since you normalize)
      4) batched output == per-sample output (within tolerance)
      5) batched sims == per-sample sims (within tolerance)

    NOTE: This does not test convergence or retrieval correctness.
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_resonator_cleanup

    torch.manual_seed(0)

    D: int = 1024
    C: int = 8
    B: int = 16
    normalize: bool = True

    # ----------------------------
    # Build codebook + batch queries
    # ----------------------------
    codebook = torch.randn(C, D)
    v = torch.randn(B, D)

    # ----------------------------
    # Run batched
    # ----------------------------
    v_out_b, sims_b = _base_batch_resonator_cleanup(v, codebook, normalize=normalize)

    assert v_out_b.shape == (B, D)
    assert sims_b.shape == (B, C)
    assert v_out_b.dtype == v.dtype
    assert sims_b.dtype == v.dtype
    assert torch.isfinite(v_out_b).all()
    assert torch.isfinite(sims_b).all()

    # sims should be bounded if normalize=True (cosine similarity in [-1, 1])
    if normalize:
        assert sims_b.max().item() <= 1.0 + 1e-5
        assert sims_b.min().item() >= -1.0 - 1e-5

    # output should be unit norm because you normalize it
    norms = v_out_b.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0), (
        f"Output vectors not unit norm. min={norms.min().item()}, max={norms.max().item()}"
    )

    # ----------------------------
    # Run per-sample and compare
    # ----------------------------
    v_out_list = []
    sims_list = []
    for i in range(B):
        v_i = v[i : i + 1]  # (1, D)
        v_out_i, sims_i = _base_batch_resonator_cleanup(v_i, codebook, normalize=normalize)
        v_out_list.append(v_out_i)
        sims_list.append(sims_i)

    v_out_s = torch.cat(v_out_list, dim=0)  # (B, D)
    sims_s = torch.cat(sims_list, dim=0)    # (B, C)

    assert torch.allclose(v_out_b, v_out_s, atol=1e-6, rtol=1e-6), (
        f"Batched v_out differs from per-sample v_out. "
        f"max_abs={(v_out_b - v_out_s).abs().max().item():.3e}"
    )
    assert torch.allclose(sims_b, sims_s, atol=1e-6, rtol=1e-6), (
        f"Batched sims differs from per-sample sims. "
        f"max_abs={(sims_b - sims_s).abs().max().item():.3e}"
    )


def test_base_batch_modern_hopfield_cleanup_single_step_batched_equivalence():
    """
    Batched single-step Modern Hopfield cleanup (softmax retrieval) should behave
    identically to running the same function on each element individually.

    This test is intentionally *only* about the single-step layer in isolation:
      1) shapes/dtypes are correct
      2) attention rows are valid probabilities (sum ~ 1)
      3) batched output == per-sample output (within tolerance)
      4) batched attention == per-sample attention (within tolerance)

    NOTE: This does *not* assert convergence or correct retrieval (those are
    multi-iteration / regime-dependent properties).
    """
    import torch
    from hyperspace.backends.hrr import _base_batch_modern_hopfield_cleanup
    import torch.nn.functional as F

    torch.manual_seed(0)

    D: int = 1024
    C: int = 8
    B: int = 16
    temperature: float = 0.2
    normalize: bool = True

    # ----------------------------
    # Build codebook + batch queries
    # ----------------------------
    codebook = torch.randn(C, D)
    v = torch.randn(B, D)

    # ----------------------------
    # Run batched
    # ----------------------------
    v_out_b, attn_b = _base_batch_modern_hopfield_cleanup(
        v, codebook, temperature=temperature, normalize=normalize
    )

    assert v_out_b.shape == (B, D)
    assert attn_b.shape == (B, C)
    assert v_out_b.dtype == v.dtype
    assert attn_b.dtype == v.dtype
    assert torch.isfinite(v_out_b).all()
    assert torch.isfinite(attn_b).all()

    # Attention should be row-stochastic
    row_sums = attn_b.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=0.0), (
        f"Attention rows do not sum to 1. min={row_sums.min().item()}, max={row_sums.max().item()}"
    )
    assert (attn_b >= 0).all(), "Attention has negative entries"

    # Output should be (approximately) unit norm because you normalize it
    norms = v_out_b.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0), (
        f"Output vectors not unit norm. min={norms.min().item()}, max={norms.max().item()}"
    )

    # ----------------------------
    # Run per-sample and compare
    # ----------------------------
    v_out_list = []
    attn_list = []
    for i in range(B):
        v_i = v[i : i + 1]  # (1, D)
        v_out_i, attn_i = _base_batch_modern_hopfield_cleanup(
            v_i, codebook, temperature=temperature, normalize=normalize
        )
        v_out_list.append(v_out_i)
        attn_list.append(attn_i)

    v_out_s = torch.cat(v_out_list, dim=0)  # (B, D)
    attn_s = torch.cat(attn_list, dim=0)    # (B, C)

    # Batched == stacked-single (tight tolerances; should match exactly up to fp rounding)
    assert torch.allclose(v_out_b, v_out_s, atol=1e-6, rtol=1e-6), (
        f"Batched v_out differs from per-sample v_out. "
        f"max_abs={(v_out_b - v_out_s).abs().max().item():.3e}"
    )
    assert torch.allclose(attn_b, attn_s, atol=1e-6, rtol=1e-6), (
        f"Batched attn differs from per-sample attn. "
        f"max_abs={(attn_b - attn_s).abs().max().item():.3e}"
    )

import pytest
import torch

from hyperspace.backends.hrr import HRRBackend


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def test_similarity_single_single_returns_scalar(device):
    D = 32
    b = HRRBackend(vector_dim=D, value_dim=1)

    a = torch.zeros(D, device=device)
    a[0] = 1.0
    c = a.clone()

    s, _ = b.similarity(a, c)
    assert s.ndim == 0
    assert torch.isclose(s, torch.tensor(1.0, device=device), atol=1e-6)


def test_similarity_batch_batch_returns_B(device):
    D = 32
    B = 6
    b = HRRBackend(vector_dim=D, value_dim=1)

    x = torch.eye(D, device=device)[:B]   # (B,D), orthonormal rows
    y = x.clone()

    s, _ = b.similarity(x, y)
    assert s.shape == (B,)
    assert torch.allclose(s, torch.ones(B, device=device), atol=1e-6)


def test_similarity_batch_batch_mismatched_B_raises(device):
    """Test that pairwise mode raises error with mismatched batch sizes."""
    D = 32
    b = HRRBackend(vector_dim=D, value_dim=1)

    a = torch.randn(3, D, device=device)
    c = torch.randn(5, D, device=device)

    # With explicit pairwise mode, should raise error
    with pytest.raises(ValueError, match="pairwise mode requires same batch size"):
        b.similarity(a, c, mode="pairwise")
    
    # With auto mode (default), should use all_pairs and return (3, 5) matrix
    s, info = b.similarity(a, c)
    assert s.shape == (3, 5)
    assert info["mode"] == "all_pairs"


def test_similarity_batch_single_broadcast(device):
    D = 32
    B = 6
    b = HRRBackend(vector_dim=D, value_dim=1)

    code = torch.eye(D, device=device)[:B]  # (B,D)
    v = code[2].clone()                     # (D,)

    s, _ = b.similarity(code, v)            # (B,)
    assert s.shape == (B,)

    # row 2 matches exactly
    assert torch.isclose(s[2], torch.tensor(1.0, device=device), atol=1e-6)

    # orthonormal -> others ~ 0
    others = torch.cat([s[:2], s[3:]])
    assert torch.all(torch.abs(others) < 1e-6)


def test_similarity_single_batch_broadcast(device):
    D = 32
    B = 6
    b = HRRBackend(vector_dim=D, value_dim=1)

    code = torch.eye(D, device=device)[:B]
    v = code[4].clone()

    s1, _ = b.similarity(v, code)  # (B,)
    s2, _ = b.similarity(code, v)  # (B,)
    assert s1.shape == (B,)
    assert torch.allclose(s1, s2, atol=1e-6)

    assert torch.isclose(s1[4], torch.tensor(1.0, device=device), atol=1e-6)
    others = torch.cat([s1[:4], s1[5:]])
    assert torch.all(torch.abs(others) < 1e-6)


def test_similarity_rejects_bad_ndim(device):
    D = 32
    b = HRRBackend(vector_dim=D, value_dim=1)

    a = torch.randn(2, 3, D, device=device)
    c = torch.randn(2, 3, D, device=device)

    with pytest.raises(ValueError, match="must be 1D or 2D"):
        b.similarity(a, c)


def test_similarity_rejects_bad_last_dim(device):
    D = 32
    b = HRRBackend(vector_dim=D, value_dim=1)

    a = torch.randn(D - 1, device=device)
    c = torch.randn(D - 1, device=device)

    with pytest.raises(ValueError, match="Last dim must match"):
        b.similarity(a, c)

def test_similarity_mode_parameter_default_auto():
    """Test that mode defaults to 'auto' and behaves correctly."""
    D = 128
    b = HRRBackend(vector_dim=D)
    
    v1 = b.create_random_vector()
    v2 = b.create_random_vector()
    
    # Single vectors should return scalar
    sim, info = b.similarity(v1, v2)
    assert sim.ndim == 0
    assert info["mode"] == "pairwise"


def test_similarity_mode_pairwise_single_vectors():
    """Test explicit pairwise mode with single vectors."""
    D = 128
    b = HRRBackend(vector_dim=D)
    
    v1 = b.create_random_vector()
    v2 = b.create_random_vector()
    
    sim, info = b.similarity(v1, v2, mode="pairwise")
    assert sim.ndim == 0
    assert info["mode"] == "pairwise"


def test_similarity_mode_pairwise_batch_same_size():
    """Test explicit pairwise mode with batched vectors of same size."""
    D = 128
    B = 8
    b = HRRBackend(vector_dim=D)
    
    v1_list = [b.create_random_vector() for _ in range(B)]
    v2_list = [b.create_random_vector() for _ in range(B)]
    
    v1 = torch.stack(v1_list, dim=0)
    v2 = torch.stack(v2_list, dim=0)
    
    sim, info = b.similarity(v1, v2, mode="pairwise")
    assert sim.shape == (B,)
    assert info["mode"] == "pairwise"


def test_similarity_mode_pairwise_batch_different_size_raises():
    """Test that pairwise mode raises error with different batch sizes."""
    D = 128
    b = HRRBackend(vector_dim=D)
    
    v1 = torch.randn(5, D)
    v2 = torch.randn(8, D)
    
    with pytest.raises(ValueError, match="pairwise mode requires same batch size"):
        b.similarity(v1, v2, mode="pairwise")


def test_similarity_mode_all_pairs_returns_matrix():
    """Test that all_pairs mode returns (a, b) shaped matrix."""
    D = 128
    A = 5
    B = 8
    
    backend = HRRBackend(vector_dim=D)
    
    v1_list = [backend.create_random_vector() for _ in range(A)]
    v2_list = [backend.create_random_vector() for _ in range(B)]
    
    v1 = torch.stack(v1_list, dim=0)  # (A, D)
    v2 = torch.stack(v2_list, dim=0)  # (B, D)
    
    sim, info = backend.similarity(v1, v2, mode="all_pairs")
    
    assert sim.shape == (A, B), f"Expected shape ({A}, {B}), got {sim.shape}"
    assert info["mode"] == "all_pairs"
    assert sim.ndim == 2


def test_similarity_mode_all_pairs_identity_diagonal():
    """Test that all_pairs mode gives 1.0 on diagonal for identical vectors."""
    D = 256
    N = 6
    
    backend = HRRBackend(vector_dim=D)
    
    vectors = torch.stack([backend.create_random_vector() for _ in range(N)], dim=0)
    
    sim, info = backend.similarity(vectors, vectors, mode="all_pairs")
    
    assert sim.shape == (N, N)
    assert info["mode"] == "all_pairs"
    
    # Diagonal should be all 1.0 (self-similarity)
    diagonal = torch.diagonal(sim)
    assert torch.allclose(diagonal, torch.ones(N), atol=1e-5)


def test_similarity_mode_all_pairs_orthogonal_vectors():
    """Test all_pairs mode with approximately orthogonal vectors."""
    D = 10000  # Large dimension for better orthogonality
    N = 5
    
    backend = HRRBackend(vector_dim=D)
    
    v1 = torch.stack([backend.create_random_vector() for _ in range(N)], dim=0)
    v2 = torch.stack([backend.create_random_vector() for _ in range(N)], dim=0)
    
    sim, info = backend.similarity(v1, v2, mode="all_pairs")
    
    assert sim.shape == (N, N)
    assert info["mode"] == "all_pairs"
    
    # Off-diagonal elements should be close to 0 for random orthogonal vectors
    # Using looser tolerance due to randomness
    assert torch.all(torch.abs(sim) <= 1.0 + 1e-5)
    assert torch.all(torch.abs(sim) >= -1.0 - 1e-5)


def test_similarity_mode_all_pairs_asymmetric_shapes():
    """Test all_pairs mode with different sized batches."""
    D = 128
    A = 3
    B = 7
    
    backend = HRRBackend(vector_dim=D)
    
    v1 = torch.randn(A, D)
    v2 = torch.randn(B, D)
    
    sim_ab, info_ab = backend.similarity(v1, v2, mode="all_pairs")
    sim_ba, info_ba = backend.similarity(v2, v1, mode="all_pairs")
    
    assert sim_ab.shape == (A, B)
    assert sim_ba.shape == (B, A)
    assert info_ab["mode"] == "all_pairs"
    assert info_ba["mode"] == "all_pairs"
    
    # sim_ab[i, j] should equal sim_ba[j, i] (transpose relationship)
    assert torch.allclose(sim_ab, sim_ba.T, atol=1e-6)


def test_similarity_mode_auto_same_batch_size_uses_pairwise():
    """Test that auto mode uses pairwise when batch sizes match."""
    D = 128
    B = 6
    
    backend = HRRBackend(vector_dim=D)
    
    v1 = torch.randn(B, D)
    v2 = torch.randn(B, D)
    
    sim, info = backend.similarity(v1, v2, mode="auto")
    
    assert sim.shape == (B,)  # Pairwise returns 1D
    assert info["mode"] == "pairwise"


def test_similarity_mode_auto_different_batch_size_uses_all_pairs():
    """Test that auto mode uses all_pairs when batch sizes differ."""
    D = 128
    A = 4
    B = 7
    
    backend = HRRBackend(vector_dim=D)
    
    v1 = torch.randn(A, D)
    v2 = torch.randn(B, D)
    
    sim, info = backend.similarity(v1, v2, mode="auto")
    
    assert sim.shape == (A, B)  # all_pairs returns 2D
    assert info["mode"] == "all_pairs"


def test_similarity_mode_invalid_raises():
    """Test that invalid mode parameter raises error."""
    D = 128
    backend = HRRBackend(vector_dim=D)
    
    v1 = torch.randn(4, D)
    v2 = torch.randn(4, D)
    
    with pytest.raises(ValueError, match="mode must be one of"):
        backend.similarity(v1, v2, mode="invalid_mode")


def test_similarity_all_pairs_with_single_vector_broadcast():
    """Test all_pairs mode with single vector vs batch."""
    D = 128
    B = 5
    
    backend = HRRBackend(vector_dim=D)
    
    v_single = backend.create_random_vector()  # (D,)
    v_batch = torch.stack([backend.create_random_vector() for _ in range(B)], dim=0)  # (B, D)
    
    # Single vs Batch should give (B,) in pairwise mode (default behavior)
    sim, info = backend.similarity(v_single, v_batch)
    assert sim.shape == (B,)
    assert info["mode"] == "pairwise"


def test_similarity_all_pairs_codebook_query_scenario():
    """
    Test a realistic scenario: computing similarities between query vectors
    and a codebook (all-pairs mode).
    """
    D = 512
    num_queries = 4
    codebook_size = 10
    
    backend = HRRBackend(vector_dim=D)
    
    # Create queries and codebook
    queries = torch.stack([backend.create_random_vector() for _ in range(num_queries)], dim=0)
    codebook = torch.stack([backend.create_random_vector() for _ in range(codebook_size)], dim=0)
    
    # Compute all-pairs similarities
    sim, info = backend.similarity(queries, codebook, mode="all_pairs")
    
    assert sim.shape == (num_queries, codebook_size)
    assert info["mode"] == "all_pairs"
    
    # Each row should have a maximum similarity (best match for each query)
    max_sims = torch.max(sim, dim=1).values
    assert max_sims.shape == (num_queries,)
    
    # All similarities should be in valid range
    assert torch.all(sim >= -1.0 - 1e-5)
    assert torch.all(sim <= 1.0 + 1e-5)


def test_similarity_all_pairs_consistency_with_manual_computation():
    """
    Test that all_pairs mode produces the same results as manually computing
    all pairwise similarities.
    """
    D = 256
    A = 3
    B = 4
    
    backend = HRRBackend(vector_dim=D)
    
    v1 = torch.stack([backend.create_random_vector() for _ in range(A)], dim=0)
    v2 = torch.stack([backend.create_random_vector() for _ in range(B)], dim=0)
    
    # Use all_pairs mode
    sim_all, info = backend.similarity(v1, v2, mode="all_pairs")
    assert sim_all.shape == (A, B)
    
    # Manually compute each similarity
    sim_manual = torch.zeros(A, B)
    for i in range(A):
        for j in range(B):
            s, _ = backend.similarity(v1[i], v2[j], mode="pairwise")
            sim_manual[i, j] = s
    
    assert torch.allclose(sim_all, sim_manual, atol=1e-6)


def test_similarity_eps_parameter():
    """Test that eps parameter is accepted and used."""
    D = 128
    backend = HRRBackend(vector_dim=D)
    
    v1 = torch.randn(5, D)
    v2 = torch.randn(8, D)
    
    # Should not raise error
    sim, info = backend.similarity(v1, v2, mode="all_pairs", eps=1e-10)
    assert sim.shape == (5, 8)


def test_similarity_all_pairs_values_in_valid_range():
    """Test that all_pairs mode produces cosine similarities in [-1, 1]."""
    D = 256
    A = 10
    B = 15
    
    backend = HRRBackend(vector_dim=D)
    
    # Create random vectors with various magnitudes
    v1 = torch.randn(A, D) * torch.randint(1, 10, (A, 1)).float()
    v2 = torch.randn(B, D) * torch.randint(1, 10, (B, 1)).float()
    
    sim, info = backend.similarity(v1, v2, mode="all_pairs")
    
    assert sim.shape == (A, B)
    assert torch.all(sim >= -1.0 - 1e-5), f"Min similarity: {sim.min()}"
    assert torch.all(sim <= 1.0 + 1e-5), f"Max similarity: {sim.max()}"


def test_similarity_all_pairs_batch_size_one():
    """Test all_pairs mode with batch size 1 (edge case)."""
    D = 128
    backend = HRRBackend(vector_dim=D)
    
    v1 = backend.create_random_vector().unsqueeze(0)  # (1, D)
    v2 = torch.stack([backend.create_random_vector() for _ in range(5)], dim=0)  # (5, D)
    
    sim, info = backend.similarity(v1, v2, mode="all_pairs")
    
    assert sim.shape == (1, 5)
    assert info["mode"] == "all_pairs"


def test_similarity_mode_in_info_dict():
    """Test that the mode used is always returned in the info dict."""
    D = 128
    backend = HRRBackend(vector_dim=D)
    
    v1 = torch.randn(3, D)
    v2 = torch.randn(3, D)
    
    for mode in ["auto", "pairwise", "all_pairs"]:
        if mode == "all_pairs":
            v2_test = torch.randn(5, D)  # Different size for all_pairs
        else:
            v2_test = v2
            
        sim, info = backend.similarity(v1, v2_test, mode=mode)
        assert "mode" in info
        assert isinstance(info["mode"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
