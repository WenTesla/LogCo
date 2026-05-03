import config
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
from Model import BertEDL
from LogDataset import LogDataset
import os

# ===================== 加载数据集 & 划分 =====================
full_dataset = LogDataset(
    config.GROUPED_LOGS_PATH,
    bert_path=config.BERT_PATH,
    max_len=config.MAX_LEN,
    batch_size=config.FEATURE_BATCH_SIZE,
    device=config.DEVICE,
)
train_dataset, test_dataset = full_dataset.split(
    mode=config.SPLIT_MODE,
    train_ratio=config.TRAIN_RATIO,
    random_seed=config.RANDOM_SEED,
)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# ===================== 加载模型 =====================
model = BertEDL(num_classes=config.NUM_CLASSES).to(config.DEVICE)
model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=config.DEVICE))
model.eval()

# ===================== 存储结果 =====================
all_preds = []
all_labels = []
all_probs = []
all_uncertainties = []
all_templates = []

# ===================== 推理（带进度条） =====================
print("\n🚀 日志异常检测推理中...")
with torch.no_grad():
    pbar = tqdm(test_loader, desc="Testing")
    for batch in pbar:
        features = batch["features"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)

        alpha = model(features)
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        pred = torch.argmax(prob, dim=1)
        uncertainty = 2.0 / S.squeeze()

        # 保存结果
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(prob[:, 1].cpu().numpy())
        all_uncertainties.extend(uncertainty.cpu().numpy())

# 取出测试集原始日志模板
test_indices = test_dataset.indices
df = pd.read_csv(config.GROUPED_LOGS_PATH)
all_templates = [df.loc[i, "Templates"] for i in test_indices]

# ===================== 异常检测专用指标 =====================
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
auc = roc_auc_score(all_labels, all_probs)

print("\n" + "=" * 65)
print("📊 日志异常检测专用评估指标")
print("=" * 65)
print(f"Precision (精确率)      : {prec:.4f}")
print(f"Recall    (召回率)      : {rec:.4f}")
print(f"F1        (F1分数)      : {f1:.4f}")
print(f"AUC       (AUC分数)     : {auc:.4f}")
print("\n详细分类报告：")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Anomaly"], digits=4, zero_division=0))

# ===================== 构建结果表 =====================
df_result = pd.DataFrame({
    "TestIndex": test_indices,
    "Templates": all_templates,
    "Label": all_labels,
    "Pred": all_preds,
    "AnomalyScore": all_probs,
    "Uncertainty": all_uncertainties
})

# 保存全测试集结果（用于级联全量评估）
save_full_pred_path = os.path.join("outputs", config.DATASET, "results", "sm_test_predictions.csv")
df_result.to_csv(save_full_pred_path, index=False, encoding="utf-8")
print(f"✅ 全测试集预测已保存至：{save_full_pred_path}")
