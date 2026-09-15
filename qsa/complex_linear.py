import math
import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """Linear layer for complex-valued tensors.

    Implements y = W x for x = (x_real, x_imag) using two real-valued linear
    layers, equivalent to multiplication by the complex weight matrix
    W = W_r + i * W_i:

        y_real = W_r x_real - W_i x_imag
        y_imag = W_r x_imag + W_i x_real

    Initialization follows the QSA paper: real weights use Xavier uniform with
    a small gain, imaginary weights start at zero so the layer initially
    behaves like its real-valued projection.
    """

    def __init__(self, in_features, out_features, bias=False, init_scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc_r = nn.Linear(in_features, out_features, bias=bias)
        self.fc_i = nn.Linear(in_features, out_features, bias=bias)
        # Initialization: real ~ Xavier, imaginary ~ zeros
        nn.init.xavier_uniform_(self.fc_r.weight, gain=init_scale)
        nn.init.zeros_(self.fc_i.weight)
        if bias:
            nn.init.zeros_(self.fc_r.bias)
            nn.init.zeros_(self.fc_i.bias)

    def forward(self, x_real, x_imag):
        return (
            self.fc_r(x_real) - self.fc_i(x_imag),
            self.fc_r(x_imag) + self.fc_i(x_real),
        )

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.fc_r.bias is not None}"


def spectral_clamp(module, gamma=1.0, max_iter=10):
    """Elastic spectral clamping (QSA stability mechanism).

    Clamp the spectral norm of all ComplexLinear weights to ``gamma`` using
    power iteration on each real/imaginary component matrix. Call after every
    optimizer step for a Lyapunov-style stability guarantee.
    """
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, ComplexLinear):
                for param_name in ["fc_r", "fc_i"]:
                    weight = getattr(child, param_name).weight.data
                    # Estimate spectral norm via power iteration
                    u = torch.randn(weight.size(0), device=weight.device)
                    v = torch.randn(weight.size(1), device=weight.device)
                    u_norm = torch.norm(u)
                    v_norm = torch.norm(v)
                    if u_norm == 0 or v_norm == 0:
                        continue
                    u = u / u_norm
                    v = v / v_norm
                    sigma = None
                    for _ in range(max_iter):
                        v = torch.mv(weight.t(), u)
                        v_norm = torch.norm(v)
                        if v_norm == 0:
                            break
                        v = v / v_norm
                        u = torch.mv(weight, v)
                        u_norm = torch.norm(u)
                        if u_norm == 0:
                            break
                        u = u / u_norm
                        sigma = u_norm
                    if sigma is not None and sigma > gamma:
                        weight.data *= gamma / sigma
