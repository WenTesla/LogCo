import argparse
import logging
import re


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("LogCo")


def parse_args():
    parser = argparse.ArgumentParser(description="Log grouping preprocess")
    parser.add_argument(
        "--DATASET",
        "--dataset",
        dest="DATASET",
        type=str,
        default="Thunderbird",
        choices=["BGL", "HDFS", "Spirit", "Thunderbird"],
        help="数据集名称，可选: BGL/HDFS/Spirit/Thunderbird",
    )
    parser.add_argument("--input_dir", type=str, default=None, help="输入目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--window_size", type=int, default=60, help="滑动窗口大小")
    parser.add_argument("--step_size", type=int, default=60, help="滑动窗口步长")
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = f"./inputs/{args.DATASET}"
    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.DATASET}"
    return args


def BGL_regex(content: str) -> str:
    normalized = str(content)
    normalized = re.sub(r"0x[0-9a-fA-F]+", "<*>", normalized)
    normalized = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<*>", normalized)
    normalized = re.sub(r"\bblk_-?\d+\b", "<*>", normalized)
    normalized = re.sub(r"\b\d+\b", "<*>", normalized)
    return normalized
