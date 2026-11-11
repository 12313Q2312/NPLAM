"""Equations (2)-(4), Zhu et al., DOI:10.1007/s11704-023-2662-3.

Implemented independently. Gates are signed real parameters, not probabilities.
The linear path does NOT use (1-g2), including in legacy mode.
"""
import torch
from torch import nn
from torch.nn import functional as F


class NPLAM(nn.Module):
    def __init__(self, features, hidden=50, mode="paper", ablation="none"):
        super().__init__()
        if features < 1 or hidden < 1:
            raise ValueError("features and hidden must be positive")
        if mode not in {"paper", "legacy"}:
            raise ValueError("mode must be paper or legacy")
        if ablation not in {"none", "no_fs", "no_sd", "no_reg", "no_l1", "no_l2", "train_hidden"}:
            raise ValueError("unknown ablation")
        self.features, self.hidden, self.mode, self.ablation = features, hidden, mode, ablation
        v, b = torch.randn(features, hidden), torch.randn(features, hidden)
        if ablation == "train_hidden":
            self.v, self.b = nn.Parameter(v), nn.Parameter(b)
        else:
            self.register_buffer("v", v)
            self.register_buffer("b", b)
        self.g1 = nn.Parameter(torch.ones(features), requires_grad=ablation != "no_fs")
        self.g2 = nn.Parameter(torch.empty(features).uniform_(0.69, 0.71), requires_grad=ablation != "no_sd")
        if ablation == "no_sd":
            self.g2.data.fill_(1)
        self.w = nn.Parameter(torch.ones(features))
        self.a = nn.Parameter(torch.zeros(features, hidden))
        self.beta = nn.Parameter(torch.zeros(()))

    def contributions(self, x):
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError("expected [samples, features]")
        z = (x * self.g1 * self.g2).unsqueeze(-1) * self.v + self.b
        h = F.relu(z) if self.mode == "paper" else F.leaky_relu(z, negative_slope=0.2)
        return (h * self.a).sum(-1) + x * self.g1 * self.w

    def forward(self, x):
        return self.beta + self.contributions(x).sum(-1)

    def penalty_terms(self):
        reduce = torch.sum if self.mode == "paper" else torch.mean
        return reduce(self.g1.abs()), reduce(self.g2.abs()), reduce(self.a.abs())

    def regularization(self, lambdas):
        l1, l2, l3 = lambdas
        if self.ablation in {"no_fs", "no_l1"}:
            l1 = 0
        if self.ablation in {"no_sd", "no_l2"}:
            l2 = 0
        if self.ablation == "no_reg":
            l1 = l2 = l3 = 0
        return sum(l * p for l, p in zip((l1, l2, l3), self.penalty_terms()))

    def structure(self, fs_threshold=0.05, sd_threshold=0.05):
        """0=irrelevant, 1=linear, 2=nonlinear; explicit fixed thresholds."""
        selected = self.g1.detach().abs() > fs_threshold
        nonlinear = (self.g1.detach() * self.g2.detach()).abs() > sd_threshold
        return torch.where(selected, 1 + nonlinear.long(), 0).cpu().numpy()

    def effective_linear_weights(self):
        return self.g1 * self.w


class NAM(nn.Module):
    """Single hidden layer NAM-style baseline, not original ensemble/ExU NAM."""
    def __init__(self, features, hidden=50):
        super().__init__()
        self.v = nn.Parameter(torch.randn(features, hidden))
        self.b = nn.Parameter(torch.randn(features, hidden))
        self.a = nn.Parameter(torch.randn(features, hidden) * 0.01)
        self.beta = nn.Parameter(torch.zeros(()))

    def contributions(self, x):
        return (F.relu(x.unsqueeze(-1) * self.v + self.b) * self.a).sum(-1)

    def forward(self, x):
        return self.beta + self.contributions(x).sum(-1)

    def regularization(self, lambdas):
        return self.a.new_zeros(())
