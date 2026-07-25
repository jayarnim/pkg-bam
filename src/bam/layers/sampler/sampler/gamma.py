import torch
from torch.distributions import Gamma
from .base import AttentionScoreSampler
from .registry import register


@register("gamma")
class GammaSampler(AttentionScoreSampler):
    def __init__(
        self,
        param: float,
    ):
        super().__init__()
        self.beta = param

    def forward(
        self, 
        logexp: torch.Tensor,
    ) -> torch.Tensor:
        # (B,1,Nk) or (B,Nq,Nk)
        beta = torch.full_like(input=logexp, fill_value=self.beta)
        alpha = torch.exp(beta) * beta
        return Gamma(concentration=alpha, rate=beta)