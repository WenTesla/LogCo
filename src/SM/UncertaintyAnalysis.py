import config
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from Model import BertEDL
from LogDataset import LogDataset
import os

# ===================== 加载 =====================
full_dataset = LogDataset(
    config.GROUPED_LOGS_PATH,
    bert_path=config.BERT_PATH,
    max_len=config.MAX_LEN,
    batch_size=config.FEATURE_BATCH_SIZE,
    device=config.DEVICE,
)
train_dataset, val_dataset, test_dataset = full_dataset.split_train_val_test(
    mode=config.SPLIT_MODE,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    random_seed=config.RANDOM_SEED,
)
split_map = {
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset,
}
if config.UNCERTAINTY_SPLIT not in split_map:
    raise ValueError("UNCERTAINTY_SPLIT must be train/val/test")
uncertainty_dataset = split_map[config.UNCERTAINTY_SPLIT]
test_loader = DataLoader(uncertainty_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = BertEDL(num_classes=config.NUM_CLASSES).to(config.DEVICE)
model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=config.DEVICE))
model.eval()

# ===================== 计算不确定性 =====================
uncertainties = []
correct = []
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    pbar = tqdm(test_loader, desc="不确定性评估")
    for batch in pbar:
        features = batch["features"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)

        alpha = model(features)
        S = alpha.sum(dim=1)
        prob = alpha / S.unsqueeze(-1)
        pred = prob.argmax(dim=1)
        unc = 2.0 / S

        uncertainties.extend(unc.cpu().numpy())
        correct.extend((pred == labels).cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(prob[:, 1].cpu().numpy())

# ===================== 分组 =====================
unc = np.array(uncertainties)
cor = np.array(correct)
thresholds = np.quantile(unc, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
groups = np.digitize(unc, thresholds)
num_groups = len(thresholds) + 1

error_rates = []
group_counts = []
group_correct_counts = []
group_error_counts = []
group_confusion = []
for g in range(num_groups):
    mask = groups == g
    cnt = int(mask.sum())
    group_counts.append(cnt)
    if cnt == 0:
        group_correct_counts.append(0)
        group_error_counts.append(0)
        error_rates.append(0.0)
        group_confusion.append({"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        continue

    correct_cnt = int(cor[mask].sum())
    error_cnt = cnt - correct_cnt
    group_correct_counts.append(correct_cnt)
    group_error_counts.append(error_cnt)
    acc = cor[mask].mean()
    error_rates.append(100 * (1 - acc))

    labels_g = np.array(all_labels)[mask]
    preds_g = np.array(all_preds)[mask]
    group_confusion.append(
        {
            "tn": int(((labels_g == 0) & (preds_g == 0)).sum()),
            "fp": int(((labels_g == 0) & (preds_g == 1)).sum()),
            "fn": int(((labels_g == 1) & (preds_g == 0)).sum()),
            "tp": int(((labels_g == 1) & (preds_g == 1)).sum()),
        }
    )

# ===================== 输出 =====================
print("\n==========================================")
print(" Group | Samples | Ratio(%) | Correct | Error | Error Rate (%)")
total_cnt = len(unc)
for i, er in enumerate(error_rates):
    ratio = (group_counts[i] / total_cnt * 100) if total_cnt > 0 else 0.0
    print(
        f" G{i+1:<1}    | {group_counts[i]:7d} | {ratio:7.2f} | "
        f"{group_correct_counts[i]:7d} | {group_error_counts[i]:5d} | {er:12.2f}%"
    )

print("\n==========================================")
print(" Group | TN(Label0 Pred0) | FP(Label0 Pred1) | FN(Label1 Pred0) | TP(Label1 Pred1)")
for i, cm in enumerate(group_confusion):
    print(
        f" G{i+1:<2}   | {cm['tn']:16d} | {cm['fp']:16d} | "
        f"{cm['fn']:16d} | {cm['tp']:16d}"
    )

# ===================== 导出高/低不确定样本 =====================
test_indices = uncertainty_dataset.indices
source_df = pd.read_csv(config.GROUPED_LOGS_PATH)
all_templates = [source_df.loc[i, "Templates"] for i in test_indices]

df_result = pd.DataFrame(
    {
        "TestIndex": test_indices,
        "Templates": all_templates,
        "Label": all_labels,
        "Pred": all_preds,
        "AnomalyScore": all_probs,
        "Uncertainty": uncertainties,
    }
)

top_ratio = float(config.TOP_RATIO)
topk = int(len(df_result) * top_ratio)
highest_group_id = int(groups.max()) if len(groups) > 0 else -1
df_result["UncertaintyGroup"] = groups + 1
high_df = df_result[df_result["UncertaintyGroup"] == highest_group_id + 1].copy()
low_df = df_result.sort_values(by="Uncertainty", ascending=True).head(topk)

save_pred_path = os.path.join(
    "outputs",
    config.DATASET,
    "results",
    f"{config.UNCERTAINTY_SPLIT}_sm_predictions.csv",
)
save_high_path = os.path.join(
    "outputs",
    config.DATASET,
    "results",
    f"{config.UNCERTAINTY_SPLIT}_high_uncertain_samples.csv",
)
save_low_path = os.path.join(
    "outputs",
    config.DATASET,
    "results",
    f"{config.UNCERTAINTY_SPLIT}_low_uncertain_samples.csv",
)
os.makedirs(os.path.dirname(save_high_path), exist_ok=True)
df_result.to_csv(save_pred_path, index=False, encoding="utf-8")
high_df.to_csv(save_high_path, index=False, encoding="utf-8")
low_df.to_csv(save_low_path, index=False, encoding="utf-8")

print(f"\n✅ {config.UNCERTAINTY_SPLIT} 集预测已保存至 {save_pred_path}")
print(f"\n✅ 高不确定性样本已保存至 {save_high_path}")
print(f"✅ 共保存 {len(high_df)} 条（最高不确定度分组 G{highest_group_id + 1}）")
print(f"✅ 低不确定性样本已保存至 {save_low_path}")
print(f"✅ 共保存 {len(low_df)} 条（前 {top_ratio * 100:.0f}% 不确定性最低样本）")

# ===================== 绘图 =====================
plt.figure(figsize=(7,4))
group_labels = [f"G{i+1}" for i in range(num_groups)]
bar_colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_groups))
plt.bar(group_labels, error_rates, color=bar_colors)
plt.title(f"{config.DATASET} {config.UNCERTAINTY_SPLIT} Uncertainty vs Error Rate")
plt.ylabel("Error Rate (%)")
plt.xlabel("Uncertainty Group")
plt.tight_layout()
plot_path = os.path.join(
    "outputs",
    config.DATASET,
    "results",
    f"{config.DATASET}_{config.UNCERTAINTY_SPLIT}_unc_error.png",
)
plt.savefig(plot_path)
print(f"✅ 不确定性分析图已保存至 {plot_path}")
