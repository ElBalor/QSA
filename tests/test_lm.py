import math
import torch

from qsa import QSAttention, QSDecoderBlock
from qsa.rope import apply_phase_rope
from qsa.model import QSAReferenceLM, QSAConfig

torch.manual_seed(0)


def test_phase_rope_relative_offset_property():
    """Rotating Q by pos_q and K by pos_k must depend only on pos_q - pos_k."""
    d_k = 8
    q_r = torch.randn(1, 1, 1, d_k)
    q_i = torch.randn(1, 1, 1, d_k)
    k_r = torch.randn(1, 1, 1, d_k)
    k_i = torch.randn(1, 1, 1, d_k)

    def hermitian(qr, qi, kr, ki):
        return (qr * kr + qi * ki).sum(), (qi * kr - qr * ki).sum()

    # Same relative offset (2), different absolute positions
    qr1, qi1 = apply_phase_rope(q_r, q_i, torch.tensor([3]))
    kr1, ki1 = apply_phase_rope(k_r, k_i, torch.tensor([1]))
    qr2, qi2 = apply_phase_rope(q_r, q_i, torch.tensor([7]))
    kr2, ki2 = apply_phase_rope(k_r, k_i, torch.tensor([5]))

    s1_r, s1_i = hermitian(qr1, qi1, kr1, ki1)
    s2_r, s2_i = hermitian(qr2, qi2, kr2, ki2)
    assert torch.allclose(s1_r, s2_r, atol=1e-5)
    assert torch.allclose(s1_i, s2_i, atol=1e-5)

    # Different relative offset must (generically) differ
    kr3, ki3 = apply_phase_rope(k_r, k_i, torch.tensor([2]))
    s3_r, _ = hermitian(qr1, qi1, kr3, ki3)
    assert not torch.allclose(s1_r, s3_r, atol=1e-4)


def test_rope_is_isometry():
    """Rotation must preserve the norm of each complex channel pair."""
    r = torch.randn(2, 4, 10, 8)
    i = torch.randn(2, 4, 10, 8)
    r_rot, i_rot = apply_phase_rope(r, i, torch.arange(10))
    norm_before = (r**2 + i**2).sum(-1)
    norm_after = (r_rot**2 + i_rot**2).sum(-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5)


def test_kv_cache_equals_full_forward():
    """Incremental cached attention must match recomputing the full prefix."""
    attn = QSAttention(d_model=16, n_heads=4, causal=True, use_rope=True)
    attn.eval()
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        full = attn(x)
        # Feed token by token with the cache
        out_last = None
        cache = None
        for t in range(6):
            step = x[:, t : t + 1]
            if cache is None:
                res = attn(step, use_cache=True)
            else:
                res = attn(step, past_kv=cache, use_cache=True)
            out_t, cache = res
            out_last = out_t if t == 0 else out_t
        cached_full = out_last  # only the last slot; compare all slots below

        # Rebuild full outputs slot by slot
        outs = []
        cache = None
        for t in range(6):
            step = x[:, t : t + 1]
            if cache is None:
                res = attn(step, use_cache=True)
            else:
                res = attn(step, past_kv=cache, use_cache=True)
            out_t, cache = res
            outs.append(out_t)
        incremental = torch.cat(outs, dim=1)
    assert torch.allclose(full, incremental, atol=1e-5)


def test_cache_length_and_positions():
    from qsa.cache import ComplexKVCache
    cache = ComplexKVCache(2, 4, 8)
    assert len(cache) == 0
    k = torch.randn(2, 4, 3, 8)
    cache.update(k, torch.zeros_like(k), k, torch.zeros_like(k))
    assert len(cache) == 3
    pos = cache.positions(2)
    assert pos.tolist() == [3, 4]
    cache.update(k, torch.zeros_like(k), k, torch.zeros_like(k))
    assert len(cache) == 6  # 3 + 3
    kr, ki, vr, vi = cache.get()
    assert kr.shape == (2, 4, 6, 8)


def test_lm_forward_loss_and_shape():
    cfg = QSAConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = QSAReferenceLM(cfg)
    idx = torch.randint(0, 64, (2, 12))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 12, 64)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()


def test_lm_generation_matches_full_forward_scores():
    """Single-step greedy continuation must equal full-forward greedy."""
    cfg = QSAConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = QSAReferenceLM(cfg)
    model.eval()
    prompt = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        gen = model.generate(prompt.clone(), max_new_tokens=3, temperature=0.0)
        logits_full, _ = model(gen[:, :5])
        greedy_first = logits_full[:, -1, :].argmax(dim=-1)
    assert gen.shape == (1, 8)
    assert gen[0, 5].item() == greedy_first[0].item()


def test_generation_deterministic_greedy_and_cache_effect():
    cfg = QSAConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = QSAReferenceLM(cfg)
    model.eval()
    prompt = torch.randint(0, 64, (1, 4))
    a = model.generate(prompt.clone(), max_new_tokens=6, temperature=0.0)
    b = model.generate(prompt.clone(), max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)
    # Sampled path also runs (shape check only)
    s = model.generate(prompt.clone(), max_new_tokens=3, temperature=0.9, top_k=10)
    assert s.shape == (1, 7)


def test_softmax_twin_runs():
    cfg = QSAConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, d_ff=64, kind="softmax")
    model = QSAReferenceLM(cfg)
    idx = torch.randint(0, 64, (2, 12))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 12, 64)
    assert torch.isfinite(loss)
    gen = model.generate(idx[:, :3].clone(), max_new_tokens=2, temperature=0.0)
    assert gen.shape == (2, 5)


def test_block_cache_interface():
    block = QSDecoderBlock(d_model=16, n_heads=4, d_ff=32)
    block.eval()
    x = torch.randn(1, 5, 16)
    z = torch.zeros_like(x)
    with torch.no_grad():
        r_full, _ = block(x, z)
        outs = []
        cache = None
        for t in range(5):
            res = block(x[:, t : t + 1], z[:, t : t + 1], use_cache=True) if cache is None else \
                block(x[:, t : t + 1], z[:, t : t + 1], past_kv=cache, use_cache=True)
            r_t, i_t, cache = res
            outs.append(r_t)
        r_inc = torch.cat(outs, dim=1)
    assert torch.allclose(r_full, r_inc, atol=1e-5)
