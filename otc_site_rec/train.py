"""单城市训练骨干模型并保存嵌入。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    EMBED_DIM,
    EPOCHS,
    LR,
    N_LAYERS,
    REG_LAMBDA,
    SEED,
    TOP_K,
    WEIGHT_DECAY,
)
from data.loader import CityDataset, load_all_cities
from metrics import recall_ndcg_at_k
from models.backbone import LightGCN, MatrixFactorization


def train_one_city(
    dataset: CityDataset,
    model_name: str = "mf",
    device: str = "cpu",
    epochs: int = EPOCHS,
    patience: int = 10,
    checkpoint_dir: Path | None = None,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n_b, n_r = dataset.n_brands, dataset.n_regions

    if model_name == "mf":
        model = MatrixFactorization(n_b, n_r, EMBED_DIM).to(device)
    elif model_name == "lightgcn":
        model = LightGCN(n_b, n_r, EMBED_DIM, dataset.graph.to(device), N_LAYERS).to(device)
    else:
        raise ValueError(model_name)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_ndcg, stale, best_state = -1.0, 0, None

    train_mask = dataset.train.toarray().astype(bool)

    for epoch in range(1, epochs + 1):
        model.train()
        u, pos, neg = dataset.sample_bpr_batch(BATCH_SIZE, rng)
        loss = model.bpr_loss(
            torch.tensor(u, device=device),
            torch.tensor(pos, device=device),
            torch.tensor(neg, device=device),
            REG_LAMBDA,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                scores = model.predict_all().cpu().numpy()
            _, ndcg = recall_ndcg_at_k(scores, dataset.val_pos, train_mask, TOP_K)
            if ndcg > best_val_ndcg:
                best_val_ndcg = ndcg
                stale = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        u_emb, v_emb = model.forward_embeddings()
        u_np, v_np = u_emb.cpu().numpy(), v_emb.cpu().numpy()
        scores = model.predict_all().cpu().numpy()

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        np.save(checkpoint_dir / "brand_emb.npy", u_np)
        np.save(checkpoint_dir / "region_emb.npy", v_np)
        np.save(checkpoint_dir / "scores.npy", scores)
        with open(checkpoint_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"city": dataset.city, "model": model_name, "n_brands": n_b, "n_regions": n_r},
                f,
                ensure_ascii=False,
                indent=2,
            )

    return model, u_np, v_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mf", "lightgcn"], default="mf")
    parser.add_argument("--city", default=None, help="单城市；默认训练全部四城")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    cities = [args.city] if args.city else ["Chicago", "NYC", "Singapore", "Tokyo"]
    datasets = load_all_cities(cities)
    out_root = Path(args.out) / args.model

    for city, ds in datasets.items():
        print(f"\n=== Training {args.model} on {city} ===")
        ckpt = out_root / city
        _, u, v = train_one_city(ds, args.model, args.device, args.epochs, checkpoint_dir=ckpt)
        train_mask = ds.train.toarray().astype(bool)
        scores = u @ v.T
        rec, ndcg = recall_ndcg_at_k(scores, ds.test_pos, train_mask, TOP_K)
        print(f"[{city}] Test Recall@{TOP_K}={rec:.4f}  nDCG@{TOP_K}={ndcg:.4f}")


if __name__ == "__main__":
    main()
