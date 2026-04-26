import torch
import os

class Config:
    # ================ 模型与路径 ================
    # BERT_PATH = "../PLM/bert-base-uncased"  # 你的BERT路径
    BERT_PATH = "bert-base-uncased"  # 直接用 huggingface 在线模型，100%不报错

    
    # SAVE_MODEL_PATH =  "bert_edl_best.pth"   # 保存模型
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET = "Spirit" # 数据集名称 Spirit,BGL,Thunderbird
    GROUPED_LOGS_PATH = os.path.join("outputs",DATASET,"grouped_logs.csv") # 分组后的CSV文件 regex_contents,Templates,Label 其中Templates是日志模板列表，Label是0/1，0表示正常，1表示异常
    SAVE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", DATASET, "bert_edl_best.pth")
    # print(f"Model will be saved to: {SAVE_MODEL_PATH}")
    # ================ 训练超参数 ================
    MAX_LEN = 128
    BATCH_SIZE = 192
    EPOCHS = 50
    LR = 2e-5
    NUM_CLASSES = 2
    # ================ 数据集划分（新加） ================
    SPLIT_MODE = "ordered"       # random / ordered
    TRAIN_RATIO = 0.7             # 训练集比例

    # ================ BERT特征提取（仅Templates） ================
    FEATURE_BATCH_SIZE = 256

    # ================ EDL 相关 ================
    ANNEALING = 0.01  # KL退火系数


    # ================ 其他 ================
    RANDOM_SEED = 42
    top_ratio   = 0.2  # 用于不确定性分析的分组比例


# 全局实例化
config = Config()