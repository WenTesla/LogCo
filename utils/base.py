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


# 在函数外预先编译正则表达式以提高性能
IP_REGEX = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?") # 匹配IP地址，包含可选端口
SERVER_REGEX = re.compile(r"\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[-:])\S+") # 匹配服务器名称
ECID_REGEX = re.compile(r"[A-Z0-9]{28}") # 匹配ECID
SERIAL_REGEX = re.compile(r"[a-zA-Z0-9]{48}")
MEMORY_REGEX = re.compile(r"0[xX][0-9a-fA-F]\S+")
PATH_REGEX = re.compile(r".\S*(?=.[0-9a-zA-Z])(?=[/]).\S*")
IAR_REGEX = re.compile(r"[0-9a-fA-F]{8}")
NUM_REGEX = re.compile(r"(\d+)")

# 优化替换逻辑
def BGL_regex(log)->str:
    # 使用回调函数进行一次替换
    def substitute(match):
        matched_text = match.group(0)
        if IP_REGEX.fullmatch(matched_text):
            return " IP "
        if ECID_REGEX.fullmatch(matched_text):
            return " ECID "
        if MEMORY_REGEX.fullmatch(matched_text):
            return " MEMORY "
        if PATH_REGEX.fullmatch(matched_text):
            return " PATH "
        if NUM_REGEX.fullmatch(matched_text):
            return " NUM "
        return matched_text  # 不替换其他内容

    # 合并所有正则表达式到一个匹配中
    combined_regex = re.compile(rf"{IP_REGEX.pattern}|{ECID_REGEX.pattern}|{MEMORY_REGEX.pattern}|{PATH_REGEX.pattern}|{NUM_REGEX.pattern}")
    optimized_log = combined_regex.sub(substitute, log)
    return optimized_log
