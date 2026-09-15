# QSA: Quantum Self-Attention
### Complex Projective Attention with Constructive and Destructive Interference

[![PyPI](https://img.shields.io/badge/pip%20install-qsa--attention-blue)](https://pypi.org/project/qsa-attention/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22250418.svg)](https://doi.org/10.5281/zenodo.22250418)

**QSA** is a drop-in replacement for softmax attention that lifts the real probability simplex into complex projective space **CP^{n-1}**. By preserving phase information, QSA unlocks **constructive and destructive interference**, **native subtractive logic**, and **antisymmetric relational encoding** — capabilities impossible for softmax.

> Softmax can only accumulate. QSA can cancel.

---

## The Five Steps

| Step | Softmax attention | QSA |
| :--- | :--- | :--- |
| 1. Embedding | Real Q, K, V | Complex Q̃ = Q_r + i·Q_i |
| 2. Scores | QKᵀ (real dot product) | Hermitian interference S = ⟨Q̃, K̃⟩_C — imaginary part is antisymmetric |
| 3. Normalize | softmax → real simplex, phase destroyed | α = S / (‖S‖₂ · √d_k) → CP^{n-1}, phase preserved |
| 4. Mix | Σ wᵢⱼVⱼ (accumulate only, w ≥ 0) | H = Σ αᵢⱼ·Ṽⱼ (coherent superposition, native cancellation) |
| 5. Readout | — | Born rule: output = Re(H) |

The Ω(N²) separation theorem proves Born-rule readouts are strictly more expressive than any affine-softmax model.

---

## 📦 Installation

```bash
pip install qsa-attention
```

Or from source:

```bash
git clone https://github.com/ElBalor/QSA
cd QSA
pip install -e .
```

Requires Python ≥ 3.8 and PyTorch ≥ 2.0. No custom autograd — gradients flow via Wirtinger calculus, native in PyTorch.

---

## 🛠️ Quick Start

### Attention module

```python
import torch
from qsa import QSAttention, spectral_clamp

attn = QSAttention(d_model=512, n_heads=8, dropout=0.05)
x = torch.randn(2, 128, 512)          # (batch, seq, d_model)
out = attn(x)                          # real stream, residual included
out_i = attn(x, return_imaginary=True) # also get the imaginary stream

# Stability: after every optimizer step
spectral_clamp(attn, gamma=1.0)
```

### Full decoder block + Log-Born LM head

```python
import torch
from qsa import QSDecoderBlock, LogBornHead

block = QSDecoderBlock(d_model=512, n_heads=8, d_ff=2048)
head = LogBornHead(d_model=512, vocab_size=32000)

r = torch.randn(2, 128, 512)   # real hidden state
i = torch.zeros_like(r)        # imaginary stream (trainable in full models)
r, i = block(r, i)
logits = head(r, i)            # clamped to [-50, 50]
```

### NecroGraft integration (graft QSA into a frozen model)

```python
from qsa import ComplexLinear, ExpandedComplexLinear, zero_frozen_gradients

base = ComplexLinear(512, 512)          # e.g. lifted from a frozen model
graft = ExpandedComplexLinear(base, expansion_dim=128)

r, i = graft(x_real, x_imag)   # y = W_f·x + λ·(W_g·x_g), λ = Yaka Gate
# ... train; W_f stays frozen (zero forgetting) ...
merged = graft.merge()          # zero-overhead single ComplexLinear
```

### Full language model with complex KV cache

```python
import torch
from qsa import QSAReferenceLM, QSAConfig

model = QSAReferenceLM(QSAConfig(vocab_size=256, d_model=256, n_heads=4, n_layers=6))
idx = torch.randint(0, 256, (1, 16))
logits, loss = model(idx, targets=idx)         # train
out = model.generate(idx, max_new_tokens=64, temperature=0.0)  # cache-backed

# Controlled baseline: identical body, softmax attention instead
softmax_model = QSAReferenceLM(QSAConfig(kind="softmax"))
```

### Train on causal arithmetic (the paper's stress-test)

```bash
python training/train_lm.py --steps 20000 --device cuda          # QSA + softmax
python training/train_lm.py --kind qsa --steps 50000 --device cuda  # longer run
```

Byte-level, no tokenizer needed. Logs train vs held-out longer-expression loss (OOD generalization) and samples greedy completions. Spectral clamping (γ=1.0) runs after every optimizer step, per the paper's stability regimen.

---

## 📊 Reproducing the paper's stress-tests

The harmonic interference benchmark (constructive vs destructive classification, d_model=16, 4 heads — QSA 99.83% vs softmax 97.80% at full scale in the paper) is included:

```bash
python benchmarks/harmonic_interference.py --epochs 30 --n 2000
```

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v
```

23 tests covering: Hermitian product equivalence to complex multiplication, causal masking (zero, not -inf — correct in CP space), padding masks, **the analytic destructive-interference proof** (phase-opposed scores → exact zero output; softmax on identical magnitudes → provably the value itself), graft zero-init and exact merge, spectral clamping bounds, NaN-free training of an 8-block stack, **phase-RoPE relative-offset property and isometry**, **complex KV cache ≡ full forward equivalence**, and greedy generation determinism for both QSA and softmax twins.

---

## 📐 The Math

Given real input X, project to complex Q̃, K̃, Ṽ. The interference matrix and normalization:

$$S_{ij} = \sum_d \tilde{Q}_{id}\,\overline{\tilde{K}_{jd}} \in \mathbb{C}, \qquad \alpha_{ij} = \frac{S_{ij}}{\|S_i\|_2 \cdot \sqrt{d_k}}$$

Coherent superposition and Born readout:

$$H_i = \sum_j \alpha_{ij}\,\tilde{V}_j \in \mathbb{C}^{d_k}, \qquad \text{Output}_i[k] = \operatorname{Re}(H_i[k])$$

When phases align (Δφ ≈ 0): constructive interference. When opposed (Δφ ≈ π): destructive interference. Standard attention occupies only the non-negative orthant of what QSA can express.

**Stability mechanisms** (all in the package): 0.9 residual scaling, Xavier(0.1) real / zero imaginary init, elastic spectral clamping (γ = 1.0) after every optimizer step, Log-Born clamping to [-50, 50].

---

## 📜 Citation

```bibtex
@misc{yaka2026qsa,
      title={Quantum Self-Attention (QSA): Complex Projective Attention with
             Constructive and Destructive Interference},
      author={Heylel Yaka},
      year={2026},
      publisher={Zenodo},
      doi={10.5281/zenodo.22250418},
      url={https://doi.org/10.5281/zenodo.22250418}
}
```

---

## 🧛‍♂️ Acknowledgements

**Built by Heylel Yaka (Eric Heylel Danjuma Yaka)**
*Independent Researcher, Abuja, Nigeria*

Forged in the **Grimoire of Elbàlor** by The Digital Necromancer. 💀🔥

*The softmax era is ending.*
