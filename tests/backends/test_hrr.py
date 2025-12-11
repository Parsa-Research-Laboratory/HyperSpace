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
        rtol=1e-5,
        atol=1e-7,
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
        rtol=1e-5,
        atol=1e-7,
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
