import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from .layers.kld import *
from .layers.broadcast import BroadCast
from .layers.score import build as build_score
from .layers.prior import PriorScoreFunction
from .layers.sampler import build as build_sampler
from .layers.simplex import LinearProjection
from .layers.output import ModelOutput


class BayesianAttentionModules(nn.Module):
    def __init__(
        self, 
        score: str,
        sampler: str,
        dim: int, 
        param_q: float,
        param_p: float,
        beta: int,
        dropout: float=None,
    ):
        super().__init__()
        # BROADCAST
        self.broadcast = BroadCast()
        # VARIATIONAL DIST. EXPECTED VALUE FUNCTION
        self.score_q = build_score(
            name=score,
            dim=dim,
            dropout=dropout,
        )
        # PRIOR DIST. EXPECTED VALUE FUNCTION
        self.score_p = PriorScoreFunction(
            dim=dim,
        )
        # VARIATIONAL DIST.
        self.q = build_sampler(
            name=sampler,
            param=param_q,
        )
        # PRIOR DIST.
        self.p = build_sampler(
            name=(
                "lognormal"
                if sampler=="lognormal"
                else "gamma"
            ),
            param=param_p,
        )
        # SIMPLEX PROJECTION
        self.simplex = LinearProjection(
            beta=beta,
        )

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dimension:
        ---
        INPUT:
            - q: (B,D) or (B,Nq,D)
            - k: (Nk,D) or (B,Nk,D)
            - v: (Nk,D) or (B,Nk,D)
            - mask: (Nk,) or (B,Nk) or (B,Nq,Nk)
        OUTPUT:
            - context: (B,D) or (B,Nq,D)
            - kld: (B,)
        """
        
        # BROADCASTING ==========
        # q: -> (B,1,D) or (B,Nq,D)
        # k: -> (1,Nk,D) or (B,Nk,D)
        # v: -> (1,Nk,D) or (B,Nk,D)
        # mask: -> (1,1,Nk) or (B,1,Nk) or (B,Nq,Nk)
        q, k, v, mask = self.broadcast(q=q, k=k, v=v, mask=mask)

        # ATTENTION SCORES ==========
        # VARIATIONAL DIST.: (B,Nq,Nk)
        logexp = self.score_q(q=q, k=k)
        q_dist = self.q(logexp)
        # PRIOR DIST.: (B,1,Nk)
        logexp = self.score_p(k=k)
        p_dist = self.p(logexp)
        # SCORE: (B,Nq,Nk)
        scores = (
            q_dist.rsample()
            if self.training
            else q_dist.mean
        )

        # KLD ==========
        # (B,Nq,Nk)
        kld = kl_divergence(p=q_dist, q=p_dist)

        # MASKING ==========
        # (B,Nq,Nk)
        if mask is not None:
            scores = scores.masked_fill(mask=mask, value=0)
            kld = kld.masked_fill(mask=mask, value=0)
        
        # WEIGHTS ==========
        # (B,Nq,Nk)
        weights = self.simplex(scores)

        # WEIGHTED SUM ==========
        # (B,Nq,Nk) x (B,Nk,D) -> (B,Nq,D)
        # (B,1,Nk) x (B,Nk,D) -> (B,1,D) -> (B,D)
        context = (weights @ v).squeeze(1)

        # KLD MEAN ==========
        if mask is not None:
            # IF MASK SHAPE IS (1,1,Nk) or (B,1,Nk) -> (B,Nq,Nk)
            # INVALID MASK -> VALID MASK
            VALID = (~mask).expand_as(kld)
            # (B,Nq,Nk) -> (B,)
            kld_batch = (
                (kld * VALID).sum(dim=(1,2))
                / VALID.sum(dim=(1,2)).clamp_min(1)
            )
        else:
            # (B,Nq,Nk) -> (B,)
            kld_batch = kld.mean(dim=(1, 2))

        return ModelOutput(
            context=context, 
            kld=kld_batch,
        )
