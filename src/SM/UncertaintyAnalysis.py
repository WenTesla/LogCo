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
_, test_dataset = full_dataset.split(
    mode=config.SPLIT_MODE,
    train_ratio=config.TRAIN_RATIO,
    random_seed=config.RANDOM_SEED,
)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

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
for g in range(num_groups):
    mask = groups == g
    cnt = int(mask.sum())
    group_counts.append(cnt)
    if cnt == 0:
        group_correct_counts.append(0)
        group_error_counts.append(0)
        error_rates.append(0.0)
        continue

    correct_cnt = int(cor[mask].sum())
    error_cnt = cnt - correct_cnt
    group_correct_counts.append(correct_cnt)
    group_error_counts.append(error_cnt)
    acc = cor[mask].mean()
    error_rates.append(100 * (1 - acc))

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

# ===================== 导出高/低不确定样本 =====================
test_indices = test_dataset.indices
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
high_df = df_result.sort_values(by="Uncertainty", ascending=False).head(topk)
low_df = df_result.sort_values(by="Uncertainty", ascending=True).head(topk)

save_high_path = os.path.join("outputs", config.DATASET, "results", "high_uncertain_samples.csv")
save_low_path = os.path.join("outputs", config.DATASET, "results", "low_uncertain_samples.csv")
high_df.to_csv(save_high_path, index=False, encoding="utf-8")
low_df.to_csv(save_low_path, index=False, encoding="utf-8")

print(f"\n✅ 高不确定性样本已保存至：{save_high_path}")
print(f"✅ 共保存 {len(high_df)} 条（前 {top_ratio * 100:.0f}% 不确定性最高样本）")
print(f"✅ 低不确定性样本已保存至：{save_low_path}")
print(f"✅ 共保存 {len(low_df)} 条（前 {top_ratio * 100:.0f}% 不确定性最低样本）")

# ===================== 绘图 =====================
plt.figure(figsize=(7,4))
group_labels = [f"G{i+1}" for i in range(num_groups)]
bar_colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_groups))
plt.bar(group_labels, error_rates, color=bar_colors)
plt.title(f"{config.DATASET} Uncertainty vs Error Rate")
plt.ylabel("Error Rate (%)")
plt.xlabel("Uncertainty Group")
plt.tight_layout()
plt.savefig(os.path.join("outputs", config.DATASET,"results", f"{config.DATASET}_unc_error.png"))
print(f"✅ 不确定性分析图已保存至 {os.path.join('outputs', config.DATASET,'results', f'{config.DATASET}_unc_error.png')}")
