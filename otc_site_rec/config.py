from pathlib import Path

# OpenSiteRec 数据目录（已 clone 到 data_repo）
DATA_ROOT = Path(__file__).resolve().parent.parent / "data_repo"
CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]

# 与论文/OpenSiteRec benchmark 对齐的默认超参
EMBED_DIM = 64
N_LAYERS = 2
LR = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 200
BATCH_SIZE = 2048
REG_LAMBDA = 1e-4
MIN_BRAND_SITES = 5  # 5-core
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
TOP_K = 20
SEED = 42

# OTC：源城权重 γ，论文在 (0,5] 网格搜索，默认 1.0
OTC_GAMMA = 1.0
OT_EPSILON = 1e-9
# GW 源城实体过多时子采样上限（加速且减轻 numItermax 警告）
GW_MAX_ENTITIES = 800
