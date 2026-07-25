import torch
from torch.distributions import Weibull
from .base import AttentionScoreSampler
from .registry import register


@register("weibull")
class WeibullSampler(AttentionScoreSampler):
    def __init__(
        self,
        param: float,
    ):
        super().__init__()
        self.k = param

    def forward(
        self, 
        logexp: torch.Tensor,
    ) -> torch.Tensor:
        # (B,1,Nk) or (B,Nq,Nk)
        k = torch.full_like(input=logexp, fill_value=self.k)
        lamb = torch.exp(logexp) / torch.exp(torch.lgamma(1 + 1.0/k))
        return Weibull(scale=lamb, concentration=k)