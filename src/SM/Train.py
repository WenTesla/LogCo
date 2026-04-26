import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from config import config
from Model import BertEDL, edl_loss
from LogDataset import LogDataset

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
    train_ratio=config.TRAIN_RATIO
)

# 加载器
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# ===================== 模型 =====================
model = BertEDL(num_classes=config.NUM_CLASSES).to(config.DEVICE)

# ===================== 优化器 =====================
optim = AdamW(model.parameters(), lr=config.LR)
total_steps = len(train_loader) * config.EPOCHS
scheduler = get_linear_schedule_with_warmup(optim, 0, total_steps)

# ===================== 用于保存每轮 Loss（新增） =====================
loss_history = []  # 存储所有epoch的loss

# ===================== 训练（带tqdm进度条） =====================
model.train()
print(f"\n🚀 开始训练 {config.DATASET} 日志异常检测")
for epoch in range(config.EPOCHS):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

    for batch in pbar:
        features = batch["features"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)

        alpha = model(features)
        loss = edl_loss(alpha, labels, annealing=config.ANNEALING)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    # 保存当前epoch的loss（新增）
    loss_history.append({
        "epoch": epoch + 1,
        "avg_loss": avg_loss
    })

# ===================== 保存 Loss 到 CSV（新增） =====================
loss_save_dir = os.path.join("outputs", config.DATASET, "results")
os.makedirs(loss_save_dir, exist_ok=True)  # 自动创建文件夹
loss_save_path = os.path.join(loss_save_dir, f"train_loss.csv")

df_loss = pd.DataFrame(loss_history)
df_loss.to_csv(loss_save_path, index=False, encoding="utf-8")

# 保存模型
torch.save(model.state_dict(), config.SAVE_MODEL_PATH)

print(f"\n💾 模型保存: {config.SAVE_MODEL_PATH}")
print(f"📉 训练 Loss 已保存: {loss_save_path}")
