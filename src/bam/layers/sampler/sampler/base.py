import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AttentionScoreSampler(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, 
        logexp: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError