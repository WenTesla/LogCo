import torch
import os

# ===================== SM 配置 =====================
BERT_PATH = os.getenv("BERT_PATH", "bert-base-uncased")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = os.getenv("DATASET", "Spirit") # BGL / HDFS / Thunderbird / Spirit
GROUPED_LOGS_PATH = os.path.join("outputs", DATASET, "grouped_logs.csv")
SAVE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    DATASET,
    "bert_edl_best.pth",
)

# ===================== 训练超参数 =====================
MAX_LEN = int(os.getenv("MAX_LEN", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
LR = float(os.getenv("LR", "2e-5"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "2"))
FEATURE_BATCH_SIZE = int(os.getenv("FEATURE_BATCH_SIZE", "512"))
ANNEALING = float(os.getenv("ANNEALING", "0.01"))

# ===================== 划分策略（与 LLM/RAG 对齐） =====================
SPLIT_MODE = os.getenv("SPLIT_MODE", os.getenv("RAG_SPLIT_MODE", "ordered")) # ordered / random
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", os.getenv("RAG_TRAIN_RATIO", "0.7")))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", os.getenv("RAG_RANDOM_SEED", "42")))

# ===================== 其他 =====================
TOP_RATIO = float(os.getenv("TOP_RATIO", "0.2"))
top_ratio = TOP_RATIO  # backward compatibility
