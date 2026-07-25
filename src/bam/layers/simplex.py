import torch
import torch.nn.functional as F
import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(
        self,
        beta: float=0.5,
    ):
        """
        Reference: Fan et al., "Bayesian Attention Modules", NeurIPS 2020.
        """
        super().__init__()
        self.beta = beta

    def forward(
        self, 
        scores: torch.Tensor,
    ) -> torch.Tensor:
        # NUMERATOR ==========
        numerator = F.relu(scores)

        # DENOMINATOR ==========
        numerator_sum = numerator.sum(dim=-1, keepdim=True)
        denominator = (numerator_sum).pow(self.beta)

        # ATTENTION WEIGHTS ==========
        weights = numerator / (denominator + 1e-8)

        # STABILIZE ==========
        return torch.nan_to_num(
            input=weights,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )