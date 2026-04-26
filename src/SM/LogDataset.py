import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import ast
import numpy as np
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer
from transformers.utils import logging as hf_logging
import hashlib
from pathlib import Path

hf_logging.set_verbosity_error()

class LogDataset(Dataset):
    def __init__(self, csv_path, bert_path, max_len=128, batch_size=128, device=None, cache_dir=None):
        self.df = pd.read_csv(csv_path)
        csv_file = Path(csv_path).resolve()
        dataset_name = csv_file.parent.name
        if cache_dir is None:
            cache_root = csv_file.parent / "bert_cache"
        else:
            cache_root = Path(cache_dir) / dataset_name
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache_root)

        csv_stat = csv_file.stat()
        cache_key_raw = f"{csv_file}|{csv_stat.st_size}|{csv_stat.st_mtime_ns}|{bert_path}|{max_len}"
        cache_key = hashlib.md5(cache_key_raw.encode("utf-8")).hexdigest()[:16]
        cache_prefix = f"{dataset_name}_{csv_file.stem}_{cache_key}"
        self.feat_cache = cache_root / f"{cache_prefix}_features.npy"
        self.label_cache = cache_root / f"{cache_prefix}_labels.npy"

        # 检查列
        if "Templates" not in self.df.columns or "Label" not in self.df.columns:
            raise ValueError(f"必须包含 Templates 和 Label 列！当前列：{self.df.columns.tolist()}")

        self.templates = self.df["Templates"].tolist()
        self.labels = self.df["Label"].tolist()

        # 设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==========================================
        # 关键优化：有缓存就直接加载，没有才提取
        # ==========================================
        if self.feat_cache.exists() and self.label_cache.exists():
            print("✅ 从缓存加载 BERT 特征，秒加载！")
            self.features = torch.tensor(np.load(self.feat_cache), dtype=torch.float32)
            self.labels = np.load(self.label_cache).tolist()
        else:
            print("⚡ 首次运行，提取 BERT 特征（优化版）...")
            self.features = self._extract_bert_features_fast(
                templates=self.templates,
                bert_path=bert_path,
                max_len=max_len,
                batch_size=batch_size,
                device=self.device,
            )
            # 保存缓存
            np.save(self.feat_cache, self.features.numpy())
            np.save(self.label_cache, np.array(self.labels))

    @staticmethod
    def _template_to_text(value):
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        if pd.isna(value):
            return ""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return " ".join(str(x) for x in parsed)
            except Exception:
                pass
            return text
        return str(value)

    @classmethod
    def _extract_bert_features_fast(cls, templates, bert_path, max_len, batch_size, device):
        # ==========================================
        # 速度优化核心
        # 1. 半精度 FP16
        # 2. 更大 batch
        # 3. tokenizer 并行
        # ==========================================
        tokenizer = AutoTokenizer.from_pretrained(bert_path, use_fast=True)
        model = BertModel.from_pretrained(
            bert_path,
            num_labels=0,
            add_pooling_layer=True
        ).to(device).half()
        model.eval()

        texts = [cls._template_to_text(t) for t in templates]
        features = []

        with torch.no_grad():
            pbar = tqdm(range(0, len(texts), batch_size), desc="提取BERT特征")
            for i in pbar:
                batch_texts = texts[i:i + batch_size]
                enc = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                outputs = model(**enc)
                pooled = outputs.pooler_output
                features.extend(pooled.cpu().numpy())

        return torch.tensor(np.array(features), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def split(self, mode="ordered", train_ratio=0.7, random_seed=42):
        dataset_size = len(self)
        print(f"数据集总样本数: {dataset_size}")
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size

        if mode == "random":
            print(f"🔀 随机划分数据集: train_ratio={train_ratio}, random_seed={random_seed}")
            generator = torch.Generator().manual_seed(int(random_seed))
            return random_split(self, [train_size, test_size], generator=generator)
        elif mode == "ordered":
            print(f"📊 有序划分数据集: train_ratio={train_ratio}")
            train_dataset = torch.utils.data.Subset(self, range(train_size))
            test_dataset = torch.utils.data.Subset(self, range(train_size, dataset_size))
            return train_dataset, test_dataset
        else:
            raise ValueError("mode must be random/ordered")
