import torch
from torch.distributions import LogNormal
from .base import AttentionScoreSampler
from .registry import register


@register("lognormal")
class LogNormalSampler(AttentionScoreSampler):
    def __init__(
        self,
        param: float,
    ):
        super().__init__()
        self.scale = param

    def forward(
        self, 
        logexp: torch.Tensor,
    ) -> torch.Tensor:
        # (B,1,Nk) or (B,Nq,Nk)
        scale = torch.full_like(input=logexp, fill_value=self.scale)
        loc = logexp - 0.5 * (scale ** 2)
        return LogNormal(loc=loc, scale=scale)