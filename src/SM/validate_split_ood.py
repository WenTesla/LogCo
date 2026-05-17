import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

import config


def _templates_to_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value if str(x).strip())
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return " ".join(str(x) for x in parsed if str(x).strip())
    except Exception:
        pass
    return text


def _split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    mode: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode not in {"ordered", "random"}:
        raise ValueError("mode 必须是 ordered 或 random")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("train/val/test ratio 必须非负")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratio 之和必须为 1.0，当前为 {ratio_sum}")

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            f"切分后存在空数据集: train={train_size}, val={val_size}, test={test_size}"
        )

    if mode == "ordered":
        train_idx = np.arange(0, train_size)
        val_idx = np.arange(train_size, train_size + val_size)
        test_idx = np.arange(train_size + val_size, n)
        return train_idx, val_idx, test_idx

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).numpy()
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]
    return train_idx, val_idx, test_idx


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _label_to_binary(series: pd.Series) -> pd.Series:
    """将标签列稳健映射为二值：0=正常, 1=异常。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0) > 0).astype(int)

    s = series.astype(str).str.strip()
    lower = s.str.lower()
    normal_tokens = {"", "-", "0", "normal", "false", "benign", "ok"}
    return (~lower.isin(normal_tokens)).astype(int)


def _resolve_default_csv_path(dataset: str) -> Path:
    candidates = [
        Path("outputs") / dataset / "grouped_logs.csv",
        Path("output") / dataset / "grouped_logs.csv",
        Path("inputs") / dataset / "grouped_logs.csv",
        Path("input") / dataset / "grouped_logs.csv",
        Path("inputs") / dataset / "structured.csv",
        Path("input") / dataset / "structured.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "未找到默认 CSV。已尝试: "
        + ", ".join(str(p) for p in candidates)
    )


def _build_token_counter(texts: List[str]) -> Counter:
    c = Counter()
    for t in texts:
        tokens = t.split()
        c.update(tokens)
    return c


def _js_divergence_from_counters(c1: Counter, c2: Counter) -> float:
    vocab = sorted(set(c1.keys()) | set(c2.keys()))
    if not vocab:
        return 0.0
    p = np.array([c1.get(k, 0) for k in vocab], dtype=np.float64)
    q = np.array([c2.get(k, 0) for k in vocab], dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log2((p + 1e-12) / (m + 1e-12)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log2((q + 1e-12) / (m + 1e-12)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def evaluate_pair(
    df: pd.DataFrame,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    mode: str,
    comparison: str,
) -> dict:
    source_df = df.iloc[source_idx]
    target_df = df.iloc[target_idx]

    source_texts = source_df["log_text"].tolist()
    target_texts = target_df["log_text"].tolist()

    source_token_counter = _build_token_counter(source_texts)
    target_token_counter = _build_token_counter(target_texts)
    source_vocab = set(source_token_counter.keys())
    target_vocab = set(target_token_counter.keys())

    unseen_target_tokens = target_vocab - source_vocab
    unseen_target_token_count = sum(target_token_counter[t] for t in unseen_target_tokens)
    total_target_tokens = sum(target_token_counter.values())

    label_col_exists = "Label" in df.columns
    if label_col_exists:
        source_pos = int(_label_to_binary(source_df["Label"]).sum())
        target_pos = int(_label_to_binary(target_df["Label"]).sum())
        source_pos_rate = _safe_div(source_pos, len(source_df))
        target_pos_rate = _safe_div(target_pos, len(target_df))
    else:
        source_pos_rate = None
        target_pos_rate = None

    return {
        "mode": mode,
        "comparison": comparison,
        "source_size": len(source_df),
        "target_size": len(target_df),
        "token_vocab_overlap_rate": _safe_div(len(source_vocab & target_vocab), len(target_vocab)),
        "target_unseen_token_ratio": _safe_div(unseen_target_token_count, total_target_tokens),
        "token_js_divergence": _js_divergence_from_counters(source_token_counter, target_token_counter),
        "source_pos_rate": source_pos_rate,
        "target_pos_rate": target_pos_rate,
    }


def evaluate_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    mode: str,
    seed: int,
) -> List[dict]:
    train_idx, val_idx, test_idx = _split_indices(
        len(df), train_ratio, val_ratio, test_ratio, mode, seed
    )
    train_val_idx = np.concatenate([train_idx, val_idx])
    return [
        evaluate_pair(df, train_idx, val_idx, mode, "train -> val"),
        evaluate_pair(df, train_idx, test_idx, mode, "train -> test"),
        evaluate_pair(df, train_val_idx, test_idx, mode, "train+val -> test"),
    ]


def _fmt_pct(x):
    if x is None:
        return "N/A"
    return f"{x * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(
        description="按 SM config 的 train/val/test 切分验证 ordered/random 是否引入 OOD"
    )
    parser.add_argument(
        "--dataset",
        default=config.DATASET,
        help="数据集名称，例如 BGL/HDFS/Spirit/Thunderbird；默认读取 config.DATASET",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="CSV 路径；不传时自动检测 output(s)/input(s) 下的 grouped_logs.csv 或 structured.csv",
    )
    parser.add_argument(
        "--split-mode",
        choices=["ordered", "random", "both"],
        default=config.SPLIT_MODE,
        help="切分模式；默认读取 config.SPLIT_MODE",
    )
    parser.add_argument("--train-ratio", type=float, default=config.TRAIN_RATIO, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=config.VAL_RATIO, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=config.TEST_RATIO, help="测试集比例")
    parser.add_argument("--random-seed", type=int, default=config.RANDOM_SEED, help="随机划分种子")
    parser.add_argument(
        "--compare-both",
        action="store_true",
        help="额外同时输出 ordered 和 random，便于比较时间顺序切分与随机切分的 OOD 差异",
    )
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        csv_path = _resolve_default_csv_path(args.dataset)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Templates" not in df.columns and "EventTemplate" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 Templates 或 EventTemplate 列")
    df = df.copy()
    if "Templates" in df.columns:
        df["log_text"] = df["Templates"].apply(_templates_to_text)
    else:
        df["log_text"] = df["EventTemplate"].apply(_templates_to_text)

    if args.compare_both or args.split_mode == "both":
        modes = ["ordered", "random"]
    else:
        modes = [args.split_mode]

    print("\n================ OOD Split Validation ================")
    print(f"CSV            : {csv_path}")
    print(f"Samples        : {len(df)}")
    print(f"Split Mode     : {args.split_mode}")
    print(f"Train/Val/Test : {args.train_ratio} / {args.val_ratio} / {args.test_ratio}")
    print(f"Random Seed    : {args.random_seed}")

    for mode in modes:
        results = evaluate_split(
            df,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            mode,
            args.random_seed,
        )
        print(f"\n[{mode.upper()}]")
        for r in results:
            print(f"\n{r['comparison']}")
            print(f"Source/Target                  : {r['source_size']} / {r['target_size']}")
            print(f"Target词表在Source覆盖率        : {_fmt_pct(r['token_vocab_overlap_rate'])}")
            print(f"OOV Token Ratio                : {_fmt_pct(r['target_unseen_token_ratio'])}")
            print(f"Source/Target词分布JS散度       : {r['token_js_divergence']:.4f}")
            if r["source_pos_rate"] is not None:
                print(
                    "Source异常率 / Target异常率     : "
                    f"{_fmt_pct(r['source_pos_rate'])} / {_fmt_pct(r['target_pos_rate'])}"
                )

    print("\n---------------- Interpretation ----------------")
    print("若 ORDERED 相比 RANDOM 或 train -> test 相比 train -> val 出现：")
    print("1) 更高的 OOV Token Ratio")
    print("2) 更低的 Target词表在Source覆盖率")
    print("3) 更高的 Source/Target词分布JS散度")
    print("则可认为目标切分更可能存在分布外(OOD)或时间漂移问题。")


if __name__ == "__main__":
    main()




# ================ OOD Split Validation ================
# CSV            : outputs/Spirit/grouped_logs.csv
# Samples        : 83333
# Split Mode     : both
# Train/Val/Test : 0.5 / 0.2 / 0.3
# Random Seed    : 42

# [ORDERED]

# train -> val
# Source/Target                  : 41666 / 16666
# Target词表在Source覆盖率        : 87.25%
# OOV Token Ratio                : 1.09%
# Source/Target词分布JS散度       : 0.2563
# Source异常率 / Target异常率     : 51.21% / 49.66%

# train -> test
# Source/Target                  : 41666 / 25001
# Target词表在Source覆盖率        : 74.28%
# OOV Token Ratio                : 0.88%
# Source/Target词分布JS散度       : 0.2183
# Source异常率 / Target异常率     : 51.21% / 2.41%

# train+val -> test
# Source/Target                  : 58332 / 25001
# Target词表在Source覆盖率        : 75.55%
# OOV Token Ratio                : 0.19%
# Source/Target词分布JS散度       : 0.1464
# Source异常率 / Target异常率     : 50.76% / 2.41%

# [RANDOM]

# train -> val
# Source/Target                  : 41666 / 16666
# Target词表在Source覆盖率        : 81.73%
# OOV Token Ratio                : 0.04%
# Source/Target词分布JS散度       : 0.0010
# Source异常率 / Target异常率     : 36.12% / 36.93%

# train -> test
# Source/Target                  : 41666 / 25001
# Target词表在Source覆盖率        : 75.75%
# OOV Token Ratio                : 0.04%
# Source/Target词分布JS散度       : 0.0007
# Source异常率 / Target异常率     : 36.12% / 36.05%

# train+val -> test
# Source/Target                  : 58332 / 25001
# Target词表在Source覆盖率        : 77.50%
# OOV Token Ratio                : 0.04%
# Source/Target词分布JS散度       : 0.0007
# Source异常率 / Target异常率     : 36.35% / 36.05%


# ================ OOD Split Validation ================
# CSV            : outputs/BGL/grouped_logs.csv
# Samples        : 78558
# Split Mode     : both
# Train/Val/Test : 0.5 / 0.2 / 0.3
# Random Seed    : 42

# [ORDERED]

# train -> val
# Source/Target                  : 39279 / 15711
# Target词表在Source覆盖率        : 52.97%
# OOV Token Ratio                : 5.21%
# Source/Target词分布JS散度       : 0.1955
# Source异常率 / Target异常率     : 24.49% / 23.27%

# train -> test
# Source/Target                  : 39279 / 23568
# Target词表在Source覆盖率        : 13.14%
# OOV Token Ratio                : 26.82%
# Source/Target词分布JS散度       : 0.3520
# Source异常率 / Target异常率     : 24.49% / 24.35%

# train+val -> test
# Source/Target                  : 54990 / 23568
# Target词表在Source覆盖率        : 15.93%
# OOV Token Ratio                : 21.04%
# Source/Target词分布JS散度       : 0.2894
# Source异常率 / Target异常率     : 24.14% / 24.35%

# [RANDOM]

# train -> val
# Source/Target                  : 39279 / 15711
# Target词表在Source覆盖率        : 62.19%
# OOV Token Ratio                : 0.34%
# Source/Target词分布JS散度       : 0.0061
# Source异常率 / Target异常率     : 24.26% / 24.02%

# train -> test
# Source/Target                  : 39279 / 23568
# Target词表在Source覆盖率        : 51.37%
# OOV Token Ratio                : 0.37%
# Source/Target词分布JS散度       : 0.0061
# Source异常率 / Target异常率     : 24.26% / 24.23%

# train+val -> test
# Source/Target                  : 54990 / 23568
# Target词表在Source覆盖率        : 53.26%
# OOV Token Ratio                : 0.35%
# Source/Target词分布JS散度       : 0.0059
# Source异常率 / Target异常率     : 24.20% / 24.23%