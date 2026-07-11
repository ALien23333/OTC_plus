"""多源城融合推理（论文 Eq.9-10）。"""
from __future__ import annotations

import numpy as np
from typing import Dict

from .transport import gw_transport_plan, inference_scores, project_embeddings


def _rescale_inference(P_target: np.ndarray, P_infer: np.ndarray) -> np.ndarray:
    """将跨城推理分数调整到与骨干预测相近的尺度，避免 γ 过大破坏排序。"""
    std_t = float(np.std(P_target)) + 1e-8
    std_i = float(np.std(P_infer)) + 1e-8
    return P_infer * (std_t / std_i)


def otc_fuse(
    P_target: np.ndarray,
    brand_emb_target: np.ndarray,
    region_emb_target: np.ndarray,
    source_payloads: list[tuple[str, np.ndarray, np.ndarray]],
    gammas: Dict[str, float],
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    P_target: 目标城原始预测 [M_t, N_t]
    source_payloads: [(city_name, brand_emb_s, region_emb_s), ...]
    gammas: {source_city_name: gamma}
    """
    fused = P_target.copy()
    for name, u_s, v_s in source_payloads:
        gamma = gammas.get(name, 0.0)
        if gamma <= 0:
            continue
        T_u = gw_transport_plan(u_s, brand_emb_target, epsilon=epsilon)
        T_v = gw_transport_plan(v_s, region_emb_target, epsilon=epsilon)
        u_proj = project_embeddings(T_u, u_s)
        v_proj = project_embeddings(T_v, v_s)
        P_infer = inference_scores(u_proj, v_proj)
        P_infer = _rescale_inference(P_target, P_infer)
        fused += gamma * P_infer
    return fused
