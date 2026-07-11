"""Recall@K 与 nDCG@K，全排序（all-ranking）评估。"""
from __future__ import annotations

import numpy as np
from typing import Dict, List


def recall_ndcg_at_k(
    scores: np.ndarray,
    test_pos: Dict[int, List[int]],
    train_mask: np.ndarray | None,
    k: int = 20,
) -> tuple[float, float]:
    """
    scores: [n_brands, n_regions] 预测分数
    train_mask: 训练集正样本位置置 -inf，避免推荐已开店区域
    """
    if train_mask is not None:
        scores = scores.copy()
        scores[train_mask] = -np.inf

    recalls, ndcgs = [], []
    for u, items in test_pos.items():
        if not items:
            continue
        row = scores[u]
        topk_idx = np.argpartition(-row, min(k, len(row) - 1))[:k]
        topk_idx = topk_idx[np.argsort(-row[topk_idx])]

        hits = sum(1 for v in items if v in topk_idx)
        recalls.append(hits / len(items))

        dcg = 0.0
        for rank, idx in enumerate(topk_idx):
            if idx in items:
                dcg += 1.0 / np.log2(rank + 2)
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(items), k)))
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))
