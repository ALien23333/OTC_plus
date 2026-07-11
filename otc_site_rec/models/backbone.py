"""MF-BPR 与 LightGCN 骨干网络（论文 Section 3.1）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, n_brands: int, n_regions: int, dim: int):
        super().__init__()
        self.brand_emb = nn.Embedding(n_brands, dim)
        self.region_emb = nn.Embedding(n_regions, dim)
        nn.init.xavier_normal_(self.brand_emb.weight)
        nn.init.xavier_normal_(self.region_emb.weight)

    def forward_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.brand_emb.weight, self.region_emb.weight

    def predict_all(self) -> torch.Tensor:
        u, v = self.forward_embeddings()
        return u @ v.t()

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        reg_lambda: float = 1e-4,
    ) -> torch.Tensor:
        u_e = self.brand_emb(users)
        p_e = self.region_emb(pos)
        n_e = self.region_emb(neg)
        loss = F.softplus(n_e.mul(u_e).sum(1) - p_e.mul(u_e).sum(1)).mean()
        reg = reg_lambda * (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / len(users)
        return loss + reg


class LightGCN(nn.Module):
    def __init__(self, n_brands: int, n_regions: int, dim: int, graph: torch.Tensor, n_layers: int = 2):
        super().__init__()
        self.n_brands = n_brands
        self.n_regions = n_regions
        self.n_layers = n_layers
        self.register_buffer("graph", graph)
        self.brand_emb = nn.Embedding(n_brands, dim)
        self.region_emb = nn.Embedding(n_regions, dim)
        nn.init.xavier_normal_(self.brand_emb.weight)
        nn.init.xavier_normal_(self.region_emb.weight)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.brand_emb.weight, self.region_emb.weight])
        embs = [all_emb]
        g = self.graph
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[: self.n_brands], out[self.n_brands :]

    def forward_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.propagate()

    def predict_all(self) -> torch.Tensor:
        u, v = self.forward_embeddings()
        return u @ v.t()

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        reg_lambda: float = 1e-4,
    ) -> torch.Tensor:
        u_all, v_all = self.propagate()
        u_e = u_all[users]
        p_e = v_all[pos]
        n_e = v_all[neg]
        loss = F.softplus(n_e.mul(u_e).sum(1) - p_e.mul(u_e).sum(1)).mean()
        reg = reg_lambda * (
            self.brand_emb(users).norm(2).pow(2)
            + self.region_emb(pos).norm(2).pow(2)
            + self.region_emb(neg).norm(2).pow(2)
        ) / len(users)
        return loss + reg
