import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
import config
from Model import BertEDL
from LogDataset import LogDataset
import os
from datetime import datetime

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

# ===================== 保存 高不确定性样本 =====================
df_high_unc = df_result.sort_values(by="Uncertainty", ascending=False)
top_ratio = float(config.TOP_RATIO)
topk = int(len(df_high_unc) * top_ratio)
df_high_unc = df_high_unc.head(topk)
save_high_path = os.path.join("outputs", config.DATASET, "results", f"high_uncertain_samples.csv")
df_high_unc.to_csv(save_high_path, index=False, encoding="utf-8")

# ===================== 保存 低不确定性样本（新增） =====================
# 按不确定性 升序排列 → 取前20%
df_low_unc = df_result.sort_values(by="Uncertainty", ascending=True)
df_low_unc = df_low_unc.head(topk)  # 同样保存 20%
save_low_path = os.path.join("outputs", config.DATASET, "results", f"low_uncertain_samples.csv")
df_low_unc.to_csv(save_low_path, index=False, encoding="utf-8")

# ===================== 追加保存每次推理指标（JSON） =====================
tn = int(((np.array(all_labels) == 0) & (np.array(all_preds) == 0)).sum())
fp = int(((np.array(all_labels) == 0) & (np.array(all_preds) == 1)).sum())
fn = int(((np.array(all_labels) == 1) & (np.array(all_preds) == 0)).sum())
tp = int(((np.array(all_labels) == 1) & (np.array(all_preds) == 1)).sum())

history_path = os.path.join("outputs", config.DATASET, "results", "inference_metrics_history.json")
history_record = {
    "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": config.DATASET,
    "config": {
        "bert_path": config.BERT_PATH,
        "device": str(config.DEVICE),
        "split_mode": config.SPLIT_MODE,
        "train_ratio": float(config.TRAIN_RATIO),
        "max_len": int(config.MAX_LEN),
        "feature_batch_size": int(config.FEATURE_BATCH_SIZE),
        "batch_size": int(config.BATCH_SIZE),
        "num_classes": int(config.NUM_CLASSES),
        "lr": float(config.LR),
        "epochs": int(config.EPOCHS),
        "random_seed": int(config.RANDOM_SEED),
        "top_ratio": float(top_ratio),
    },
    "metrics": {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "test_size": int(len(all_labels)),
    },
    "artifacts": {
        "high_uncertain_file": save_high_path,
        "low_uncertain_file": save_low_path,
    },
}

history = []
if os.path.exists(history_path):
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                history = loaded
    except Exception:
        history = []

history.append(history_record)
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

# ===================== 输出提示 =====================
print(f"\n✅ 高不确定性样本已保存至：{save_high_path}")
print(f"✅ 共保存 {len(df_high_unc)} 条（前 {top_ratio*100:.0f}% 不确定性最高样本）")

print(f"\n✅ 低不确定性样本已保存至：{save_low_path}")
print(f"✅ 共保存 {len(df_low_unc)} 条（前 {top_ratio*100:.0f}% 不确定性最低样本）")
print(f"✅ 全测试集预测已保存至：{save_full_pred_path}")
print(f"✅ 本次推理指标已追加保存至：{history_path}")
