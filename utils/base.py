import logging
import os
import re
from types import SimpleNamespace


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("LogCo")


def parse_args():
    dataset = os.getenv("DATASET", "BGL")
    valid_datasets = {"BGL", "HDFS", "Spirit", "Thunderbird"}
    if dataset not in valid_datasets:
        raise ValueError(
            f"DATASET={dataset} 非法，可选: {sorted(valid_datasets)}"
        )

    input_dir = os.getenv("INPUT_DIR", f"./inputs/{dataset}")
    output_dir = os.getenv("OUTPUT_DIR", f"./outputs/{dataset}")
    window_size = int(os.getenv("WINDOW_SIZE", "60"))
    step_size = int(os.getenv("STEP_SIZE", "60"))

    return SimpleNamespace(
        DATASET=dataset,
        input_dir=input_dir,
        output_dir=output_dir,
        window_size=window_size,
        step_size=step_size,
    )


def BGL_regex(content: str) -> str:
    normalized = str(content)
    normalized = re.sub(r"0x[0-9a-fA-F]+", "<*>", normalized)
    normalized = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<*>", normalized)
    normalized = re.sub(r"\bblk_-?\d+\b", "<*>", normalized)
    normalized = re.sub(r"\b\d+\b", "<*>", normalized)
    return normalized
