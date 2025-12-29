import pytest

def test_cm_no_backend():
    """
    Test that the cleanup module doesn't assume a default backend.
    """
    from hyperspace.core import CleanupModule

    with pytest.raises(TypeError):
        CleanupModule()

def test_cm_true_backend():
    """
    Test that the cleanup module initializes with a value backend
    """
    import torch
    from hyperspace.backends.hrr import HRRBackend
    from hyperspace.core import CleanupModule

    B: int = 64
    D: int = 1024

    codebook = torch.rand((B, D))

    b = HRRBackend(vector_dim=D)
    _ = CleanupModule(b, codebook=codebook)

def test_cm_invalid_backend():
    """
    Test that the cm module throws and error when an invalid
    backend is passed
    """
    import torch
    from hyperspace.core import CleanupModule

    B: int = 64
    D: int = 1024

    codebook = torch.rand((B, D))

    with pytest.raises(TypeError):
        _ = CleanupModule(5, codebook=codebook)

def test_cm_missing_values_and_codebook():
    """
    Test that the module throws an error when
    missing both values and codebook arguments
    """
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024

    b = HRRBackend(vector_dim=D)

    with pytest.raises(ValueError):
        CleanupModule(b)

def test_cm_both_values_and_codebook():
    """
    Test that the module throws an error when
    receiving both values and codebook arguments
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    values = torch.rand((B, V))
    codebook = torch.rand((B, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, values, codebook)

def test_cm_constructor_values_type():
    """
    Test that the module throws an error when
    values isn't a Tensor
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    values = np.random.random((B, V))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(TypeError):
        CleanupModule(b, values)

def test_cm_constructor_values_shape():
    """
    Test that the module throws an error when
    values isn't a 2D Tensor
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    values_small = torch.rand((B))
    values_large = torch.rand((B, V, V))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, values_small)

    with pytest.raises(ValueError):
        CleanupModule(b, values_large)

def test_cm_constructor_values_dim():
    """
    Test that the module throws an error when
    values doesn't match value_dim
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    values_small = torch.rand((B, V - 1))
    values_large = torch.rand((B, V + 1))
    codebook = torch.rand((B, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, values_small, codebook)

    with pytest.raises(ValueError):
        CleanupModule(b, values_large, codebook)

def test_cm_constructor_values_generated_codebook():
    """
    Test the fidelity of the generated codebook
    """
    import torch
    from torch import Tensor
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    V: int = 3
    B: int = 64

    b = HRRBackend(
        vector_dim=D,
        value_dim=V
    )

    values = torch.rand((B, V))
    value_vectors_gt, _ = b.value_encoding(values)
    
    cm = CleanupModule(
        backend=b,
        values=values
    )

    assert isinstance(cm.codebook, Tensor)

    assert torch.allclose(
        value_vectors_gt,
        cm.codebook
    )

def test_cm_constructor_codebook_type():
    """
    Test that the module throws an error when
    values isn't a Tensor
    """
    import numpy as np
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    codebook = np.random.random((B, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(TypeError):
        CleanupModule(b, codebook)

def test_cm_constructor_codebook_shape():
    """
    Test that the module throws an error when
    codebook isn't a 2D Tensor
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    codebook_small = torch.rand((B))
    codebook_large = torch.rand((B, D, D))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, codebook_small)

    with pytest.raises(ValueError):
        CleanupModule(b, codebook_large)

def test_cm_constructor_codebook_dim():
    """
    Test that the module throws an error when
    codebook doesn't match vector_dim
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    V: int = 3
    D: int = 1024
    B: int = 64

    codebook_small = torch.rand((B, D - 1))
    codebook_large = torch.rand((B, D + 1))

    b = HRRBackend(vector_dim=D, value_dim=V)

    with pytest.raises(ValueError):
        CleanupModule(b, codebook_small)

    with pytest.raises(ValueError):
        CleanupModule(b, codebook_large)

def test_cm_call_missing_v():
    """
    Test that calling the cleanup module without a vector
    throws an error
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    with pytest.raises(TypeError):
        cm()

def test_cm_call_v_type():
    """
    test that calling the cleanup module with a vector that
    is Tensor throws an error
    """
    import numpy as np
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    v = np.random.random((B, D))

    with pytest.raises(TypeError):
        cm(v)

def test_cm_call_v_shape():
    """
    test that calling the cleanup module with a vector
    with an invalid shape throws an error
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    v = torch.rand((B, D, D))

    with pytest.raises(ValueError):
        cm(v)

def test_cm_call_v_dim():
    """
    test that the call method checks the dimensionality
    of v
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    v_single_small = torch.rand((D - 1,))
    v_single_large = torch.rand((D + 1,))
    v_batch_small = torch.rand((B, D - 1))
    v_batch_large = torch.rand((B, D + 1))

    with pytest.raises(ValueError):
        cm(v_single_small)

    with pytest.raises(ValueError):
        cm(v_single_large)

    with pytest.raises(ValueError):
        cm(v_batch_small)

    with pytest.raises(ValueError):
        cm(v_batch_large)

def test_cm_non_string_method():
    """
    test that the call method throws an error when the
    requested method isn't a string
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)

    with pytest.raises(ValueError):
        cm = CleanupModule(
            backend=b,
            codebook=codebook,
            method=int(20)
        )

def test_cm_call_num_iters_non_int():
    """
    test that the cleanup modules call method throws
    an error with num_iters isn't an int
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    v = torch.rand((D,))

    with pytest.raises(TypeError):
        cm(v, float(0.1))

def test_cm_call_num_iters_lt_one():
    """
    test that the cleanup modules call method throws
    an error with num_iters is less than 1
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    B: int = 64

    codebook = torch.rand((B, D))
    b = HRRBackend(vector_dim=D)
    cm = CleanupModule(
        backend=b,
        codebook=codebook
    )

    v = torch.rand((D,))

    with pytest.raises(ValueError):
        cm(v, 0)

def test_cm_call_single_value_predef_codebook_resonator():
    """
    test the cleanup ability of the cleanup module with a
    single vector, predefined codebook and a resonator
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

    D: int = 1024
    C: int = 3

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

    cm = CleanupModule(b, codebook=codebook)

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

    v_out, info = cm(v_noisy, num_iters=num_iters)

    assert isinstance(v_out, torch.Tensor)
    assert v_out.shape == (B, D)
    assert torch.isfinite(v_out).all()

    s_after = cosine_to_codebook(v_out, codebook)
    pred_after = torch.argmax(s_after, dim=-1)
    true_sim_after = s_after.gather(1, true_idx.view(-1, 1)).squeeze(1)

    MIN_PASS_RATE = 0.75  # 75%

    # ----------------------------
    # Top-1 correctness threshold
    # ----------------------------
    correct = (pred_after == true_idx)              # bool tensor, shape (B,)
    correct_rate = correct.float().mean().item()    # in [0, 1]

    assert correct_rate >= MIN_PASS_RATE, (
        f"Top-1 retrieval pass rate below threshold after resonator convergence.\n"
        f"required>={MIN_PASS_RATE:.0%}, got={correct_rate:.0%}\n"
        f"num_correct={int(correct.sum().item())}/{correct.numel()}\n"
        f"true_idx={true_idx.tolist()}\n"
        f"pred_after={pred_after.tolist()}\n"
    )

    # ----------------------------------------
    # Similarity improvement rate threshold
    # ----------------------------------------
    improved = (true_sim_after > true_sim_before + 1e-4)  # bool tensor, shape (B,)
    improved_rate = improved.float().mean().item()

    assert improved_rate >= MIN_PASS_RATE, (
        f"True-similarity improvement pass rate below threshold after resonator convergence.\n"
        f"required>={MIN_PASS_RATE:.0%}, got={improved_rate:.0%}\n"
        f"num_improved={int(improved.sum().item())}/{improved.numel()}\n"
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

def test_cm_call_multi_value_predef_codebook_hopfield():
    """
    test the cleanup ability of the cleanup module with a
    batch of vectors, a predefined codebook, and a hopfield
    """
    import torch
    from hyperspace.backends import HRRBackend
    from hyperspace.core import CleanupModule

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

    cm = CleanupModule(
        backend=b,
        codebook=codebook,
        method="modern_hopfield"
    )

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
    v_out, info = cm(v_noisy, num_iters=num_iters, temperature=temperature)

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