import os

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import config
from LogDataset import LogDataset
from Model import BertEDL, edl_loss


def _get_subset_indices(subset):
    return list(subset.indices)


def _load_feedback_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 LLM 二次检测结果: {path}")

    df = pd.read_csv(path)
    required_cols = {"TestIndex", "llm_pred"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} 缺少列: {sorted(missing_cols)}")

    df = df.copy()
    df["llm_pred"] = pd.to_numeric(df["llm_pred"], errors="coerce")
    mask = df["llm_pred"].isin([0, 1])

    if "llm_status" in df.columns:
        status = df["llm_status"].astype(str).str.strip().str.lower()
        mask &= status.isin(["normal", "anomaly"])

    if config.INCREMENTAL_CONFLICT_ONLY:
        if "Pred" not in df.columns:
            raise ValueError("INCREMENTAL_CONFLICT_ONLY=true 时 LLM CSV 必须包含 Pred 列")
        sm_pred = pd.to_numeric(df["Pred"], errors="coerce")
        conflict = pd.Series(False, index=df.index)
        valid = df["llm_pred"].notna() & sm_pred.notna()
        conflict.loc[valid] = (
            df.loc[valid, "llm_pred"].astype(int).to_numpy()
            != sm_pred.loc[valid].astype(int).to_numpy()
        )
        mask &= conflict

    feedback_df = df.loc[mask].copy()
    feedback_df["TestIndex"] = pd.to_numeric(feedback_df["TestIndex"], errors="coerce")
    feedback_df = feedback_df.dropna(subset=["TestIndex"])
    feedback_df["TestIndex"] = feedback_df["TestIndex"].astype(int)
    feedback_df["llm_pred"] = feedback_df["llm_pred"].astype(int)
    feedback_df = feedback_df.drop_duplicates(subset=["TestIndex"], keep="last")

    if feedback_df.empty:
        raise ValueError("没有可用于增量微调的 LLM 反馈样本")

    return feedback_df


def _sample_replay_indices(train_indices, feedback_count):
    if not train_indices:
        raise ValueError("原始训练集为空，无法抽取 replay 样本")

    target = int(feedback_count * config.INCREMENTAL_REPLAY_RATIO)
    target = max(target, config.INCREMENTAL_MIN_REPLAY)
    target = min(target, len(train_indices))
    if config.INCREMENTAL_MAX_REPLAY > 0:
        target = min(target, config.INCREMENTAL_MAX_REPLAY)

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    perm = torch.randperm(len(train_indices), generator=generator)[:target].tolist()
    return [train_indices[i] for i in perm]


def _build_incremental_dataset(full_dataset, train_indices, feedback_df):
    dataset_size = len(full_dataset)
    valid_feedback = feedback_df[
        (feedback_df["TestIndex"] >= 0) & (feedback_df["TestIndex"] < dataset_size)
    ].copy()
    dropped = len(feedback_df) - len(valid_feedback)
    if dropped:
        print(f"⚠️ 丢弃 {dropped} 条 TestIndex 越界的反馈样本")
    if valid_feedback.empty:
        raise ValueError("过滤 TestIndex 越界后没有可用反馈样本")

    feedback_indices = valid_feedback["TestIndex"].tolist()
    feedback_features = full_dataset.features[feedback_indices]
    feedback_labels = torch.tensor(valid_feedback["llm_pred"].tolist(), dtype=torch.long)
    feedback_weights = torch.full(
        (len(valid_feedback),),
        float(config.INCREMENTAL_FEEDBACK_WEIGHT),
        dtype=torch.float32,
    )

    replay_indices = _sample_replay_indices(train_indices, len(valid_feedback))
    replay_features = full_dataset.features[replay_indices]
    all_labels = torch.tensor(full_dataset.labels, dtype=torch.long)
    replay_labels = all_labels[replay_indices]
    replay_weights = torch.full(
        (len(replay_indices),),
        float(config.INCREMENTAL_REPLAY_WEIGHT),
        dtype=torch.float32,
    )

    features = torch.cat([replay_features, feedback_features], dim=0)
    labels = torch.cat([replay_labels, feedback_labels], dim=0)
    weights = torch.cat([replay_weights, feedback_weights], dim=0)

    used_feedback = valid_feedback.copy()
    used_feedback["feedback_weight"] = float(config.INCREMENTAL_FEEDBACK_WEIGHT)

    print("\n========== Incremental Finetune Data ==========")
    print(f"Replay samples          : {len(replay_indices)}")
    print(f"LLM feedback samples    : {len(valid_feedback)}")
    print(f"Conflict only           : {config.INCREMENTAL_CONFLICT_ONLY}")
    print(f"Replay weight           : {config.INCREMENTAL_REPLAY_WEIGHT}")
    print(f"Feedback weight         : {config.INCREMENTAL_FEEDBACK_WEIGHT}")

    return TensorDataset(features, labels, weights), used_feedback


def main():
    full_dataset = LogDataset(
        config.GROUPED_LOGS_PATH,
        bert_path=config.PRETRAINED_MODEL_PATH,
        max_len=config.MAX_LEN,
        batch_size=config.FEATURE_BATCH_SIZE,
        device=config.DEVICE,
    )
    train_dataset, _, _ = full_dataset.split_train_val_test(
        mode=config.SPLIT_MODE,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED,
    )

    train_indices = _get_subset_indices(train_dataset)
    feedback_df = _load_feedback_df(config.INCREMENTAL_LLM_CSV)
    finetune_dataset, used_feedback = _build_incremental_dataset(
        full_dataset=full_dataset,
        train_indices=train_indices,
        feedback_df=feedback_df,
    )

    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=(config.DEVICE.type == "cuda"),
        num_workers=0,
    )

    model = BertEDL(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.INCREMENTAL_BASE_MODEL_PATH, map_location=config.DEVICE))
    model.train()

    optim = AdamW(model.parameters(), lr=config.INCREMENTAL_LR)
    total_steps = len(finetune_loader) * config.INCREMENTAL_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optim, 0, total_steps)

    loss_history = []
    print(f"\n🚀 开始增量微调 {config.DATASET} SM 分类头")
    for epoch in range(config.INCREMENTAL_EPOCHS):
        total_loss = 0.0
        pbar = tqdm(finetune_loader, desc=f"Incremental Epoch {epoch + 1}/{config.INCREMENTAL_EPOCHS}")

        for features, labels, weights in pbar:
            features = features.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            weights = weights.to(config.DEVICE, non_blocking=True)

            alpha = model(features)
            loss = edl_loss(
                alpha,
                labels,
                num_classes=config.NUM_CLASSES,
                annealing=config.ANNEALING,
                sample_weight=weights,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(finetune_loader)
        loss_history.append({"epoch": epoch + 1, "avg_loss": avg_loss})
        # print(f"✅ Incremental Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(config.INCREMENTAL_SAVE_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.INCREMENTAL_SAVE_MODEL_PATH)

    result_dir = os.path.join("outputs", config.DATASET, "results")
    os.makedirs(result_dir, exist_ok=True)
    loss_path = os.path.join(result_dir, "incremental_finetune_loss.csv")
    used_feedback_path = os.path.join(result_dir, "incremental_feedback_used.csv")
    pd.DataFrame(loss_history).to_csv(loss_path, index=False, encoding="utf-8")
    used_feedback.to_csv(used_feedback_path, index=False, encoding="utf-8")

    print(f"\n💾 增量微调模型保存: {config.INCREMENTAL_SAVE_MODEL_PATH}")
    print(f"📉 增量微调 Loss 已保存: {loss_path}")
    print(f"🧾 已使用反馈样本已保存: {used_feedback_path}")


if __name__ == "__main__":
    main()
