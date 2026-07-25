from . import sampler
from .sampler.registry import SAMPLER_REGISTRY
from .sampler.base import AttentionScoreSampler


def build(
    name: str, 
    param: float,
) -> AttentionScoreSampler:
    cls = SAMPLER_REGISTRY[name]
    return cls(param)