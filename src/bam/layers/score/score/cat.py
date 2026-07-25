import torch
import torch.nn as nn
from .base import AttentionScoreFunction
from .registry import register


@register("cat")
class Concatenation(AttentionScoreFunction):
    def __init__(
        self,
        dim: int,
        dropout: float,
        **kwargs,
    ):
        """
        Reference: He et al., "NAIS: Neural attentive item similarity model for recommendation", IEEE 2018.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(dim, 1),
        )

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        q: (B,Nq,D)
        k: (B,Nk,D)
        """
        # (B,Nq,D) -> (B,Nq,1,D)
        q = q.unsqueeze(2)
        # (B,Nk,D) -> (B,1,Nk,D)
        k = k.unsqueeze(1)
        # (B,Nq,Nk,2D)
        cat = torch.cat([q, k], dim=-1)
        # (B,Nq,Nk,2D) -> (B,Nq,Nk,1) -> (B,Nq,Nk)
        return self.mlp(cat).squeeze(-1)