# OTC 跨城选址推荐 — 论文复现与改进方案

对应论文：**Optimal Transport Enhanced Cross-City Site Recommendation**（SIGIR 2024）  
DOI: [10.1145/3626772.3657757](https://doi.org/10.1145/3626772.3657757)

完整实验报告（含负效果→正效果过程与改进方案）：**[实验报告_OTC选址推荐复现与改进.md](./实验报告_OTC选址推荐复现与改进.md)**

## 工作三部分

| 阶段 | 内容 | 命令 |
|------|------|------|
| 1 | 初次复现曾出现 γ=0 或负迁移 | 见报告 Part 1 |
| 2 | **论文 OTC**：验证集逐源城调 γ | `run_otc.py`（不加 `--gamma`） |
| 3 | **改进方案**：同名品牌 GW + 自动权重 | `solution_improved.py` |

## 环境准备

```powershell
cd d:\project\otc_site_rec
pip install -r requirements.txt
```

数据目录：`d:\project\data_repo`（[OpenSiteRec](https://github.com/HestiaSky/OpenSiteRec)）

## 运行步骤

### Step 1：训练四城骨干（MF）

```powershell
python train.py --model mf --epochs 100 --device cpu
```

### Step 2：论文复现（验证集搜索每个源城的 γ）

```powershell
python run_otc.py --model mf --target Chicago --skip-train --device cpu
python run_otc.py --model mf --target NYC --skip-train --device cpu
python run_otc.py --model mf --target Singapore --skip-train --device cpu
python run_otc.py --model mf --target Tokyo --skip-train --device cpu
```

### Step 3：改进方案

```powershell
python solution_improved.py --model mf --target Chicago --skip-train --device cpu
```

## 项目结构

```
otc_site_rec/
├── train.py                 # 单城 MF 训练
├── run_otc.py               # 论文 OTC 复现
├── solution_improved.py     # 改进方案 OTC-Name+AutoW
├── otc/                     # GW 传输与融合
└── 实验报告_OTC选址推荐复现与改进.md
```

## 参考文献

- Li et al., SIGIR 2024 — OTC  
- Li et al., OpenSiteRec — [arXiv:2307.00856](https://arxiv.org/abs/2307.00856)
