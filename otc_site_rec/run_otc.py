"""OTC 完整流程：Algorithm 1 — 训练 → GW 投影 → 跨城推理 → 融合。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import OTC_GAMMA, OT_EPSILON, TOP_K
from data.loader import load_all_cities
from metrics import recall_ndcg_at_k
from otc.fusion import otc_fuse
from train import train_one_city


def tune_gamma(
    P_base: np.ndarray,
    brand_t: np.ndarray,
    region_t: np.ndarray,
    sources: list[tuple[str, np.ndarray, np.ndarray]],
    val_pos: dict,
    train_mask: np.ndarray,
    candidates: list[float] | None = None,
) -> dict[str, float]:
    """在验证集上网格搜索每个源城的 γ（步长 0.5，论文范围 (0,5]）。"""
    # 论文 (0,5]；实践中过大 γ 易负迁移（Fig.4），搜索范围收紧
    candidates = candidates or [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    rec0, ndcg0 = recall_ndcg_at_k(P_base, val_pos, train_mask, TOP_K)
    best_score = 0.5 * rec0 + 0.5 * ndcg0
    best_gammas = {name: 0.0 for name, _, _ in sources}

    for name, _, _ in sources:
        local_best_g, local_best = 0.0, best_score
        for g in candidates:
            trial = dict(best_gammas)
            trial[name] = g
            fused = otc_fuse(P_base, brand_t, region_t, sources, trial, OT_EPSILON)
            rec, ndcg = recall_ndcg_at_k(fused, val_pos, train_mask, TOP_K)
            score = 0.5 * rec + 0.5 * ndcg
            if score > local_best + 1e-6:
                local_best = score
                local_best_g = g
        best_gammas[name] = local_best_g
        best_score = local_best
    return best_gammas


def main():
    parser = argparse.ArgumentParser(description="OTC 跨城选址推荐复现")
    parser.add_argument("--model", choices=["mf", "lightgcn"], default="mf")
    parser.add_argument("--target", default="Chicago")
    parser.add_argument("--gamma", type=float, default=None, help="固定 γ；不设则在验证集搜索")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--skip-train", action="store_true", help="使用已有 checkpoint")
    args = parser.parse_args()

    all_cities = ["Chicago", "NYC", "Singapore", "Tokyo"]
    datasets = load_all_cities(all_cities)
    target = args.target
    sources_names = [c for c in all_cities if c != target]

    emb_store: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for city, ds in datasets.items():
        ckpt_dir = Path("checkpoints") / args.model / city
        if args.skip_train and (ckpt_dir / "brand_emb.npy").exists():
            u = np.load(ckpt_dir / "brand_emb.npy")
            v = np.load(ckpt_dir / "region_emb.npy")
            scores = np.load(ckpt_dir / "scores.npy")
        else:
            print(f"Training backbone on {city} ...")
            _, u, v = train_one_city(ds, args.model, args.device, args.epochs, checkpoint_dir=ckpt_dir)
            scores = u @ v.T
        emb_store[city] = (scores, u, v)

    ds_t = datasets[target]
    P_t, brand_t, region_t = emb_store[target]
    train_mask = ds_t.train.toarray().astype(bool)

    source_payloads = [(c, emb_store[c][1], emb_store[c][2]) for c in sources_names]

    # 基线
    rec0, ndcg0 = recall_ndcg_at_k(P_t, ds_t.test_pos, train_mask, TOP_K)
    print(f"\n[{target}] Backbone Recall@{TOP_K}={rec0:.4f}  nDCG@{TOP_K}={ndcg0:.4f}")

    if args.gamma is not None:
        gammas = {c: args.gamma for c in sources_names}
    else:
        print("Tuning gamma on validation set ...")
        gammas = tune_gamma(P_t, brand_t, region_t, source_payloads, ds_t.val_pos, train_mask)
    print("Gammas:", gammas)

    fused = otc_fuse(P_t, brand_t, region_t, source_payloads, gammas, OT_EPSILON)
    rec1, ndcg1 = recall_ndcg_at_k(fused, ds_t.test_pos, train_mask, TOP_K)
    print(f"[{target}] OTC Recall@{TOP_K}={rec1:.4f}  nDCG@{TOP_K}={ndcg1:.4f}")
    imp_r = (rec1 - rec0) / max(rec0, 1e-8) * 100
    imp_n = (ndcg1 - ndcg0) / max(ndcg0, 1e-8) * 100
    print(f"Improvement: Recall +{imp_r:.2f}%  nDCG +{imp_n:.2f}%")

    out = Path("results") / f"otc_{args.model}_{target}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": target,
                "model": args.model,
                "gammas": gammas,
                "backbone": {"recall": rec0, "ndcg": ndcg0},
                "otc": {"recall": rec1, "ndcg": ndcg1},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
