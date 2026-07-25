import torch
import torch.nn as nn


class PriorScoreFunction(nn.Module):
    def __init__(
        self, 
        dim: int,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=dim, 
                out_features=dim,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=dim, 
                out_features=1, 
                bias=False,
            ),
            nn.Softmax(dim=1),
        )

    def forward(
        self, 
        k: torch.Tensor,
    ) -> torch.Tensor:
        # (B,Nk,D) -> (B,Nk,1) -> (B,1,Nk)
        return self.mlp(k).transpose(2,1)