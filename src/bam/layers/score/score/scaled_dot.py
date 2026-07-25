import math
import torch
from .base import AttentionScoreFunction
from .registry import register


@register("scaled")
class ScaledDotProduct(AttentionScoreFunction):
    def __init__(
        self, 
        dim: int,
        **kwargs,
    ):
        super().__init__()
        self.scale = math.sqrt(dim)

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
        return q @ k.transpose(-2, -1) / self.scale