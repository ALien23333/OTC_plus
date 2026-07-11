"""Gromov-Wasserstein 最优传输（论文 Eq.2,6,7,8）。"""
from __future__ import annotations

import numpy as np
import ot


def pairwise_sq_dist(E: np.ndarray) -> np.ndarray:
    """L2 平方距离矩阵 c(x,y)=||x-y||^2。"""
    sum_sq = np.sum(E ** 2, axis=1, keepdims=True)
    return sum_sq + sum_sq.T - 2 * (E @ E.T)


def gw_transport_plan(
    E_source: np.ndarray,
    E_target: np.ndarray,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    计算源域 M_s × 目标域 M_t 的 GW 传输计划 T (行和为均匀边际)。
    使用 POT: ot.gromov_wasserstein
    """
    M_s, M_t = E_source.shape[0], E_target.shape[0]
    C1 = pairwise_sq_dist(E_source)
    C2 = pairwise_sq_dist(E_target)
    p = ot.unif(M_s)
    q = ot.unif(M_t)
    T = ot.gromov.gromov_wasserstein(
        C1,
        C2,
        p,
        q,
        loss_fun="square_loss",
        epsilon=epsilon,
        verbose=False,
        max_iter=50000,
    )
    return T


def project_embeddings(T: np.ndarray, E_source: np.ndarray) -> np.ndarray:
    """
    Eq.(6)(7): 将源城嵌入投影到目标城。
    POT 的 T 列和为 q_j=1/M_t，乘以 M_t 得到凸组合系数（和为 1）。
    """
    M_t = T.shape[1]
    return (T.T @ E_source) * M_t


def inference_scores(
    U_proj: np.ndarray,
    V_proj: np.ndarray,
) -> np.ndarray:
    """Eq.(8): P_st = U_proj @ V_proj^T"""
    return U_proj @ V_proj.T
