import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import ast
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

class LogDataset(Dataset):
    def __init__(self, csv_path, bert_path, max_len=128, batch_size=64, device=None):
        # 读取CSV
        self.df = pd.read_csv(csv_path)

        if "Templates" not in self.df.columns:
            raise ValueError(f"缺少 Templates 列。当前列名: {self.df.columns.tolist()}")
        if "Label" not in self.df.columns:
            raise ValueError(f"缺少 Label 列。当前列名: {self.df.columns.tolist()}")

        self.templates = self.df["Templates"].tolist()
        self.labels = self.df["Label"].tolist()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.features = self._extract_bert_features(
            templates=self.templates,
            bert_path=bert_path,
            max_len=max_len,
            batch_size=batch_size,
            device=device,
        )

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
    def _extract_bert_features(cls, templates, bert_path, max_len, batch_size, device):
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        model = BertModel.from_pretrained(bert_path).to(device)
        model.eval()

        texts = [cls._template_to_text(t) for t in templates]
        features = []

        with torch.no_grad():
            pbar = tqdm(range(0, len(texts), batch_size), desc="提取Templates的BERT特征")
            for i in pbar:
                batch_texts = texts[i:i + batch_size]
                enc = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                outputs = model(**enc)
                pooled = outputs.pooler_output
                features.extend(pooled.cpu().numpy().astype(np.float32))

        return features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]

        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long)
        }

    # 划分训练集 / 测试集（两种模式）
    def split(self, mode="ordered", train_ratio=0.7):
        dataset_size = len(self)
        print(f"数据集总大小: {dataset_size} 条日志")
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size

        if mode == "random":
            print(f"✅ 随机划分：train {train_size} | test {test_size}")
            return random_split(self, [train_size, test_size])

        elif mode == "ordered":
            print(f"划分比例 {train_ratio:.2f}")
            print(f"✅ 有序划分：前 {train_size} 训练，后 {test_size} 测试 ")
            train_dataset = torch.utils.data.Subset(self, range(train_size))
            test_dataset = torch.utils.data.Subset(self, range(train_size, dataset_size))
            return train_dataset, test_dataset

        else:
            raise ValueError("mode 必须是 random / ordered")