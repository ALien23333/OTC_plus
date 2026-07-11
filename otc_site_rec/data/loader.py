"""OpenSiteRec 数据加载：品牌-区域二部图，5-core + 按品牌划分 train/val/test。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

from config import DATA_ROOT, MIN_BRAND_SITES, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


def load_city_matrix(city: str, data_root: Path = DATA_ROOT) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    从 {city}_KG_plus.pkl 构建品牌×区域 0/1 矩阵 P。
    返回: P, brand2id, region2id
    """
    pkl = data_root / city / f"{city}_KG_plus.pkl"
    df = pd.read_pickle(pkl)
    df = df[["Brand", "Region"]].drop_duplicates()

    brand2id = {b: i for i, b in enumerate(sorted(df["Brand"].unique()))}
    region2id = {r: i for i, r in enumerate(sorted(df["Region"].unique()))}

    rows, cols = [], []
    for _, row in df.iterrows():
        rows.append(brand2id[row["Brand"]])
        cols.append(region2id[row["Region"]])
    P = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(brand2id), len(region2id)),
    )
    return P, brand2id, region2id


def filter_5core(P: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray]:
    """保留至少有 MIN_BRAND_SITES 个站点的品牌（论文 5-core）。返回子矩阵与保留行索引。"""
    brand_deg = np.array(P.sum(axis=1)).flatten()
    keep = np.where(brand_deg >= MIN_BRAND_SITES)[0]
    return P[keep, :], keep


def split_by_brand(
    P: sp.csr_matrix,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, Dict[int, List[int]]]:
    """
    每个品牌将其 POI 按 70/10/20 划分到 train/val/test（论文 4.1.2）。
    返回训练矩阵、验证/测试正样本字典 {brand_id: [region_ids]}。
    """
    rng = np.random.default_rng(SEED)
    n_brands, n_regions = P.shape
    train = sp.lil_matrix(P.shape)
    val_pos: Dict[int, List[int]] = {}
    test_pos: Dict[int, List[int]] = {}

    for u in range(n_brands):
        pos = P[u].nonzero()[1].tolist()
        if len(pos) < MIN_BRAND_SITES:
            continue
        rng.shuffle(pos)
        n = len(pos)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(0, int(n * VAL_RATIO))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = n - n_val - n_test

        train_pos = pos[:n_train]
        val_items = pos[n_train : n_train + n_val]
        test_items = pos[n_train + n_val :]

        for v in train_pos:
            train[u, v] = 1
        if val_items:
            val_pos[u] = val_items
        if test_items:
            test_pos[u] = test_items

    return train.tocsr(), val_pos, test_pos


def build_norm_adj(train: sp.csr_matrix) -> torch.Tensor:
    """LightGCN 归一化邻接矩阵（与 OpenSiteRec baseline 一致）。"""
    n_users, n_items = train.shape
    adj = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float64).tolil()
    R = train.tolil()
    adj[:n_users, n_users:] = R
    adj[n_users:, :n_users] = R.T
    adj = adj.todok()
    rowsum = np.maximum(np.array(adj.sum(axis=1)).flatten(), 1e-8)
    d_inv = np.power(rowsum, -0.5)
    d_mat = sp.diags(d_inv)
    norm_adj = d_mat.dot(adj).dot(d_mat).tocsr()
    coo = norm_adj.tocoo()
    indices = torch.from_numpy(
        np.vstack([coo.row, coo.col]).astype(np.int64)
    )
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices, values, torch.Size(coo.shape)
    ).coalesce()


class CityDataset:
    """单城市训练/评估所需数据结构。"""

    def __init__(self, city: str, data_root: Path = DATA_ROOT):
        self.city = city
        P_raw, brand2id_full, self.region2id = load_city_matrix(city, data_root)
        P, kept_rows = filter_5core(P_raw)
        id2brand_full = {i: b for b, i in brand2id_full.items()}
        self.brand_names = [id2brand_full[r] for r in kept_rows]
        self.brand2id = {b: i for i, b in enumerate(self.brand_names)}
        self.n_brands, self.n_regions = P.shape
        self.train, self.val_pos, self.test_pos = split_by_brand(P)
        self.train_pairs = list(zip(*self.train.nonzero()))
        self.graph = build_norm_adj(self.train)

    def sample_bpr_batch(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        users, pos_items, neg_items = [], [], []
        all_pos = self.train.tolil()
        for _ in range(batch_size):
            u = rng.integers(0, self.n_brands)
            pos_list = all_pos[u].nonzero()[1]
            if len(pos_list) == 0:
                continue
            pos = rng.choice(pos_list)
            neg = rng.integers(0, self.n_regions)
            while neg in pos_list:
                neg = rng.integers(0, self.n_regions)
            users.append(u)
            pos_items.append(pos)
            neg_items.append(neg)
        return (
            np.array(users, dtype=np.int64),
            np.array(pos_items, dtype=np.int64),
            np.array(neg_items, dtype=np.int64),
        )


def load_all_cities(cities: List[str] | None = None) -> Dict[str, CityDataset]:
    cities = cities or ["Chicago", "NYC", "Singapore", "Tokyo"]
    return {c: CityDataset(c) for c in cities}
