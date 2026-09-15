import torch
import torch.nn as nn

from .complex_linear import ComplexLinear


class ExpandedComplexLinear(nn.Module):
    """Pure W-Expansion for complex linear layers (NecroGraft integration).

    Keeps the base ComplexLinear frozen (zero forgetting) and adds trainable
    expansion columns fed by a learned projection, with a learnable gate
    lambda (the Yaka Gate). The expansion is zero-initialized so the graft
    starts from exactly zero contribution.

    Forward: y = W_f x + lambda * (W_g x_g)

    Two variants:

    * ``rms_scaling=True`` (paper default, used in the X graft): the
      projection reads the real stream only and is rescaled per-token to the
      input's RMS. The merge is then an *approximation* (the per-token
      rescale is input-dependent and cannot be folded into W).
    * ``rms_scaling=False``: a complex-proper expansion — both streams pass
      through a ComplexLinear projection and a ComplexLinear expansion, so
      the graft is itself a complex linear map and ``merge()`` is *exact*
      (complex matrix products compose: W_merged = W_f + lambda * W_g W_p).
    """

    def __init__(self, base_linear: ComplexLinear, expansion_dim: int, rms_scaling: bool = True):
        super().__init__()
        if not isinstance(base_linear, ComplexLinear):
            raise TypeError("ExpandedComplexLinear expects a ComplexLinear base")
        self.base = base_linear  # complex linear with frozen weights
        self.in_features = base_linear.fc_r.in_features
        self.out_features = base_linear.fc_r.out_features
        self.expansion_dim = expansion_dim
        self.rms_scaling = rms_scaling

        if rms_scaling:
            # Paper listing: real-only projection, separate real/imag expansions
            self.W_proj = nn.Linear(self.in_features, expansion_dim, bias=False)
            self.W_g_r = nn.Linear(expansion_dim, self.out_features, bias=False)
            self.W_g_i = nn.Linear(expansion_dim, self.out_features, bias=False)
            self.gate = nn.Parameter(torch.tensor(1.0))
            nn.init.zeros_(self.W_g_r.weight)
            nn.init.zeros_(self.W_g_i.weight)
            nn.init.xavier_uniform_(self.W_proj.weight, gain=0.1)
        else:
            # Complex-proper expansion: exactly mergeable
            self.W_proj = ComplexLinear(self.in_features, expansion_dim, bias=False)
            self.W_g = ComplexLinear(expansion_dim, self.out_features, bias=False)
            self.gate = nn.Parameter(torch.tensor(1.0))
            nn.init.zeros_(self.W_g.fc_r.weight)
            nn.init.zeros_(self.W_g.fc_i.weight)
            nn.init.xavier_uniform_(self.W_proj.fc_r.weight, gain=0.1)
            nn.init.zeros_(self.W_proj.fc_i.weight)

        # Freeze base
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x_real, x_imag):
        # Base output (frozen)
        base_r, base_i = self.base(x_real, x_imag)

        if self.rms_scaling:
            # Expansion path (real stream drives the projection, per the paper)
            x_proj = self.W_proj(x_real)
            # RMS scaling: preserve the input's scale into the expansion space
            rms_x = torch.sqrt(torch.mean(x_real.detach() ** 2, dim=-1, keepdim=True) + 1e-5)
            rms_p = torch.sqrt(torch.mean(x_proj.detach() ** 2, dim=-1, keepdim=True) + 1e-5)
            x_proj = x_proj * (rms_x / (rms_p + 1e-6))
            g_r = self.W_g_r(x_proj)
            g_i = self.W_g_i(x_proj)
        else:
            p_r, p_i = self.W_proj(x_real, x_imag)
            g_r, g_i = self.W_g(p_r, p_i)

        # Apply learned Yaka gate
        lambda_ = self.gate
        return base_r + lambda_ * g_r, base_i + lambda_ * g_i

    def merge(self) -> ComplexLinear:
        """Merge the expansion into the base for zero-overhead inference.

        Returns the merged ComplexLinear; the expansion path is discarded
        afterwards.

        * ``rms_scaling=False``: exact — the merged weights are
          W_f + lambda * (W_g @ W_proj) as complex matrix products.
        * ``rms_scaling=True``: approximate (paper's additive merge); the
          per-token RMS rescale cannot be folded into W.
        """
        with torch.no_grad():
            if self.rms_scaling:
                self.base.fc_r.weight.data += self.gate * (self.W_g_r.weight @ self.W_proj.weight)
                self.base.fc_i.weight.data += self.gate * (self.W_g_i.weight @ self.W_proj.weight)
            else:
                g_r, g_i = self.W_g.fc_r.weight, self.W_g.fc_i.weight
                p_r, p_i = self.W_proj.fc_r.weight, self.W_proj.fc_i.weight
                # Complex matrix product W_g @ W_proj
                dW_r = g_r @ p_r - g_i @ p_i
                dW_i = g_r @ p_i + g_i @ p_r
                self.base.fc_r.weight.data += self.gate * dW_r
                self.base.fc_i.weight.data += self.gate * dW_i
        return self.base


def zero_frozen_gradients(model: nn.Module) -> None:
    """Mask gradients on frozen base columns after backward().

    The paper's train loop zeroes the frozen columns' gradient every step:

        m.fc_r.weight.grad[:, :m.in_features].zero_()

    Note: within ExpandedComplexLinear the base ComplexLinear is fully frozen
    (requires_grad=False), so in this package the loop is a safety net for
    manually-unfrozen bases; call it between backward() and optimizer.step().
    """
    for m in model.modules():
        if isinstance(m, ExpandedComplexLinear):
            for fc in (m.base.fc_r, m.base.fc_i):
                if fc.weight.grad is not None:
                    fc.weight.grad.zero_()
