from dataclasses import dataclass
import torch


@dataclass
class ModelOutput:
    context: torch.Tensor
    kld: torch.Tensor