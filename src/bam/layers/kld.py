import torch
from torch.distributions import Weibull, Gamma, register_kl


@register_kl(Weibull, Gamma)
def weibull_gamma(
    p: Weibull, 
    q: Gamma,
) -> torch.Tensor:
    """
    Analytic KL Divergence: KL[Weibull(k, lamb) || Gamma(alpha, beta)]\\
    Reference: Fan et al., "Bayesian Attention Modules", NeurIPS 2020.
    """
    k, lamb = p.concentration, p.scale
    alpha, beta = q.concentration, q.rate

    const = -torch.special.digamma(torch.ones_like(k))
    gamma_term = torch.exp(torch.lgamma(1.0 + 1.0 / k))
    
    eps = 1e-8
    
    kld = (
        (const / k) - torch.log(lamb * k + eps)
        + ((alpha - 1) / k) * (const + torch.log(lamb + eps))
        + beta * lamb * gamma_term
        - alpha
    )
    
    return torch.clamp(kld, min=0.0)