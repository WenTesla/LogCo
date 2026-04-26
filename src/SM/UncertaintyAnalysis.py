import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
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

# ===================== 分组 =====================
unc = np.array(uncertainties)
cor = np.array(correct)
thresholds = np.quantile(unc, [0.2, 0.4, 0.6, 0.8])
groups = np.digitize(unc, thresholds)

error_rates = []
group_counts = []
group_correct_counts = []
group_error_counts = []
for g in range(5):
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

# ===================== 绘图 =====================
plt.figure(figsize=(7,4))
plt.bar([f"G{i+1}" for i in range(5)], error_rates, color=['g','c','y','orange','red'])
plt.title(f"{config.DATASET} Uncertainty vs Error Rate")
plt.ylabel("Error Rate (%)")
plt.xlabel("Uncertainty Group")
plt.tight_layout()
plt.savefig(os.path.join("outputs", config.DATASET,"results", f"{config.DATASET}_unc_error.png"))
print(f"✅ 不确定性分析图已保存至 {os.path.join('outputs', config.DATASET,'results', f'{config.DATASET}_unc_error.png')}")
