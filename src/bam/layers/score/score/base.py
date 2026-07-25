import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AttentionScoreFunction(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError