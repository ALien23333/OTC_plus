"""
自己的改进方案：OTC-Attn（可学习跨城融合）

论文局限：
1. γ 需网格搜索，且过大时性能下降（Fig.4）
2. GW 完全基于嵌入几何，忽略跨城「同名品牌」的强先验
3. 各源城贡献相同结构，未区分城市相似度

改进思路：
1. 同名品牌先验 GW：跨城同名 Brand 传输矩阵行级对齐
2. 验证集自动学权重：对每个源城在 [0,1.5] 网格搜索（替代手工 γ 与不稳定梯度注意力）

运行: python solution_improved.py --target Chicago --model mf --skip-train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import OT_EPSILON, TOP_K
from data.loader import CityDataset, load_all_cities
from metrics import recall_ndcg_at_k
from otc.fusion import _rescale_inference
from otc.transport import gw_transport_plan, inference_scores, project_embeddings
from train import train_one_city


def gw_with_name_prior(
    E_source: np.ndarray,
    E_target: np.ndarray,
    names_source: list[str],
    names_target: list[str],
    prior_strength: float = 0.3,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    GW 传输计划 + 同名品牌行级融合。
    先求标准 GW，再将有同名匹配的行向 one-hot 对齐方向插值，并保持行边际 p。
    """
    M_s, M_t = len(names_source), len(names_target)
    p = np.ones(M_s) / M_s
    T = gw_transport_plan(E_source, E_target, epsilon=epsilon)
    tgt_index = {n: j for j, n in enumerate(names_target)}

    for i, ns in enumerate(names_source):
        if ns not in tgt_index:
            continue
        j = tgt_index[ns]
        one_hot = np.zeros(M_t, dtype=np.float64)
        one_hot[j] = 1.0
        T[i] = (1.0 - prior_strength) * T[i] + prior_strength * one_hot * p[i]
        row_sum = T[i].sum()
        if row_sum > 0:
            T[i] /= row_sum
            T[i] *= p[i]
    return T


def fuse_with_weights(
    P_base: np.ndarray,
    infer_list: list[np.ndarray],
    weights: dict[str, float],
    source_names: list[str],
) -> np.ndarray:
    fused = P_base.copy()
    for name, inf in zip(source_names, infer_list):
        fused += weights.get(name, 0.0) * inf
    return fused


def tune_source_weights(
    P_base: np.ndarray,
    infer_list: list[np.ndarray],
    source_names: list[str],
    val_pos: dict,
    train_mask: np.ndarray,
    candidates: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """在验证集上为每个源城学习权重（与 run_otc 相同策略，避免注意力塌缩到单城）。"""
    candidates = candidates or [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    rec0, ndcg0 = recall_ndcg_at_k(P_base, val_pos, train_mask, TOP_K)
    best_score = 0.5 * rec0 + 0.5 * ndcg0
    best_w = {name: 0.0 for name in source_names}

    for name in source_names:
        local_best_g, local_best = 0.0, best_score
        for g in candidates:
            trial = dict(best_w)
            trial[name] = g
            fused = fuse_with_weights(P_base, infer_list, trial, source_names)
            rec, ndcg = recall_ndcg_at_k(fused, val_pos, train_mask, TOP_K)
            score = 0.5 * rec + 0.5 * ndcg
            if score > local_best + 1e-6:
                local_best_g, local_best = g, score
        best_w[name] = local_best_g
        best_score = local_best

    fused = fuse_with_weights(P_base, infer_list, best_w, source_names)
    return fused, best_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mf")
    parser.add_argument("--target", default="Chicago")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--skip-train", action="store_true", help="使用 checkpoints 中已有嵌入")
    args = parser.parse_args()

    cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
    datasets = load_all_cities(cities)
    emb = {}
    for city, ds in datasets.items():
        ckpt = Path("checkpoints") / args.model / city
        if args.skip_train and (ckpt / "brand_emb.npy").exists():
            u = np.load(ckpt / "brand_emb.npy")
            v = np.load(ckpt / "region_emb.npy")
        else:
            print(f"Training {city} ...")
            _, u, v = train_one_city(ds, args.model, args.device, args.epochs)
        emb[city] = (u, v, ds.brand_names)

    target = args.target
    ds_t = datasets[target]
    u_t, v_t, brand_names_t = emb[target]
    P_t = u_t @ v_t.T
    train_mask = ds_t.train.toarray().astype(bool)

    infer_scores = []
    for src in [c for c in cities if c != target]:
        u_s, v_s, names_s = emb[src]
        u_t, v_t, _ = emb[target]
        T_u = gw_with_name_prior(u_s, u_t, names_s, brand_names_t, epsilon=OT_EPSILON)
        T_v = gw_transport_plan(emb[src][1], v_t, epsilon=OT_EPSILON)
        u_p = project_embeddings(T_u, u_s)
        v_p = project_embeddings(T_v, v_s)
        inf = inference_scores(u_p, v_p)
        infer_scores.append(_rescale_inference(P_t, inf))

    source_names = [c for c in cities if c != target]
    rec0, ndcg0 = recall_ndcg_at_k(P_t, ds_t.test_pos, train_mask, TOP_K)
    print("Tuning source weights on validation set ...")
    fused, weights = tune_source_weights(
        P_t, infer_scores, source_names, ds_t.val_pos, train_mask
    )
    rec1, ndcg1 = recall_ndcg_at_k(fused, ds_t.test_pos, train_mask, TOP_K)

    print(f"\n=== 改进方案 OTC-Name+AutoW ({args.model}, target={target}) ===")
    print(f"Backbone  Recall@{TOP_K}={rec0:.4f}  nDCG@{TOP_K}={ndcg0:.4f}")
    print(f"Improved  Recall@{TOP_K}={rec1:.4f}  nDCG@{TOP_K}={ndcg1:.4f}")
    imp_r = (rec1 - rec0) / max(rec0, 1e-8) * 100
    imp_n = (ndcg1 - ndcg0) / max(ndcg0, 1e-8) * 100
    print(f"Improvement: Recall {imp_r:+.2f}%  nDCG {imp_n:+.2f}%")
    print(f"Source weights: {weights}")


if __name__ == "__main__":
    main()
