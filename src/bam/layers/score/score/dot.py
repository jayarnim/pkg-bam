import torch
from .base import AttentionScoreFunction
from .registry import register


@register("dot")
class DotProduct(AttentionScoreFunction):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        q: (B,Nq,D)
        k: (B,Nk,D)
        """
        # (B,Nq,D)
        # (B,Nk,D) -> (B,D,Nk)
        # (B,Nq,D) x (B,D,Nk) -> (B,Nq,Nk)
        return q @ k.transpose(-2, -1)