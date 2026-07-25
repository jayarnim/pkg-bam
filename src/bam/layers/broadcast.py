import torch
import torch.nn as nn


class BroadCast(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # INPUT DIMENSIONS VALIDATION ==========
        if q.ndim not in (2, 3):
            raise ValueError(f"Unsupported query shape: {q.shape}")

        if k.ndim not in (2, 3):
            raise ValueError(f"Unsupported key shape: {k.shape}")

        if v.ndim not in (2, 3):
            raise ValueError(f"Unsupported value shape: {v.shape}")

        if k.ndim != v.ndim:
            raise ValueError(
                "Key and Value must have the same number of dimensions."
            )

        if mask is not None:
            if mask.ndim not in (1, 2, 3):
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

        # BROADCASTING ==========
        # (B,D) -> (B,1,D) 
        if q.ndim==2:
            q = q.unsqueeze(1)

        # (Nk,D) -> (1,Nk,D)
        if k.ndim==2:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        if mask is not None:
            # SHARED MEMORY: (Nk,) -> (1,1,Nk)
            if mask.ndim==1:
                mask = mask.unsqueeze(0).unsqueeze(0)
            # BATCH-SPECIFIC MEMORY: (B,Nk) -> (B,1,Nk)
            elif mask.ndim==2:
                mask = mask.unsqueeze(1)
            # QUERY-SPECIFIC MEMORY: (B,Nq,Nk)
            else:
                mask = mask

        return q, k, v, mask