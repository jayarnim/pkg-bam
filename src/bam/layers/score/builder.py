from . import score
from .score.registry import SCORE_REGISTRY
from .score.base import AttentionScoreFunction
from typing import Any


def build(
    name: str, 
    **params,
) -> AttentionScoreFunction:
    cls = SCORE_REGISTRY[name]
    return cls(**params)