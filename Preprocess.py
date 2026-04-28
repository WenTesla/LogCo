import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from collections import defaultdict
from utils.base import logger, parse_args, BGL_regex

# 加载HDFS标签文件，返回一个字典，键为BlockId，值为标签（0或1）    
def load_label_file(path="../Datasets/HDFS/preprocessed/anomaly_label.csv")->defaultdict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"标签文件 {path} 不存在，请检查路径是否正确。")
    blk_label_dict = defaultdict() 
    blk_df = pd.read_csv(path,engine="c",memory_map=True) 
    for _ , row in tqdm(blk_df.iterrows()): 
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0 
    print(f"加载标签文件 {path} 完成, 共 {len(blk_label_dict)} 个 BlockId 标签.")
    return blk_label_dict

def group_logs(structured_log_file:str, window_size:int, step_size:int, output_dir:str,group_type:str="fixed"):
    # 读取结构化日志文件
    structured_log = pd.read_csv(structured_log_file,engine="c",memory_map=True)
    logger.info(f"已加载结构化日志文件: {structured_log_file}, 共 {len(structured_log)} 条日志")
    # 分组的数据
    grouped_data = []
    total_logs = len(structured_log)
    
    if group_type=="fixed": # 固定的滑动窗口
        logger.info(f"按固定滑动窗口分组日志中,窗口大小: {window_size}, 步长: {step_size}")
        for start_idx in tqdm(range(0, total_logs - window_size + 1, step_size), desc="按滑动窗口分组日志中"):
            window = structured_log.iloc[start_idx:start_idx + window_size]
            template = list(set(window['EventTemplate'].tolist())) # 去重后的模板列表
            label = 1 if window['Label'].apply(lambda x: str(x).strip() not in ['-', '0']).any() else 0 # 窗口内有异常则标记为异常
            contents = window['Content'].tolist() # 窗口内的日志内容列表
            regex_contents = [BGL_regex(content) for content in contents] # 为每一条原始日志进行正则化
            regex_contents = list(set(regex_contents)) # 去重后的正则化日志列表
            grouped_data.append((regex_contents,template, label))
            
    elif group_type=="time": # 时间窗口分组
        logger.info(f"按时间窗口分组日志中,单位为s,窗口大小: {window_size}s")
        current_window = []
        current_start_time = None
        for idx, row in tqdm(structured_log.iterrows(), total=total_logs, desc="按时间窗口分组日志中"):
            # 解析时间字符串为时间戳
            log_time = int(datetime.strptime(row['Time'], '%Y-%m-%d-%H.%M.%S.%f').timestamp())
            if current_start_time is None:
                current_start_time = log_time
            if log_time - current_start_time <= window_size:
                current_window.append(row)
            else:
                if current_window:
                    window_df = pd.DataFrame(current_window)
                    event_ids = window_df['EventId'].tolist()
                    times = window_df['Time'].tolist()
                    components = window_df['Component'].tolist()
                    contents = window_df['Content'].tolist()
                    label = 1 if window_df['Label'].any() else 0
                    grouped_data.append((event_ids, times, components, contents, label))
                current_window = [row]
                current_start_time = log_time
        # 处理最后一个窗口
        if current_window:
            window_df = pd.DataFrame(current_window)
            event_ids = window_df['EventId'].tolist()
            times = window_df['Time'].tolist()
            components = window_df['Component'].tolist()
            contents = window_df['Content'].tolist()
            label = 1 if window_df['Label'].any() else 0
            grouped_data.append((event_ids, times, components, contents, label))
            
    elif group_type=="session": # HDFS 按会话(BlockId)分组
        logger.info(f"按会话分组日志中,根据Session进行分组HDFS日志")
        # 加载标签
        blk_label_dict = load_label_file()
        pattern = r'(blk_-?\d+)'
        df = structured_log.copy()
        df['blkIds'] = df['Content'].str.findall(pattern)
        df_exploded = df.explode('blkIds').dropna(subset=['blkIds'])
        # 去除重复
        df_exploded = df_exploded.drop_duplicates(subset=['EventId', 'blkIds'])
        # 构建字典
        data_dict = defaultdict(list)
        for _, r in df_exploded.iterrows():
            data_dict[r['blkIds']].append((r['Content'],r['EventTemplate']))
        print(f"共识别出 {len(data_dict)} 个唯一的 BlockId")
        for blk_id, events in tqdm(data_dict.items(),desc="构建分组DataFrame中"):
            contents = list(set([BGL_regex(event[0]) for event in events])) # 去重
            templates = list(set([event[1] for event in events])) # 去重后的模板列表
            labels = blk_label_dict.get(blk_id, None) 
            grouped_data.append((contents, templates, labels))

    # 保存分组结果
    grouped_log_df = pd.DataFrame(
        grouped_data,
        columns=['regex_contents','Templates', 'Label']
    )
    output_path = os.path.join(output_dir, "grouped_logs.csv")
    
    grouped_log_df.to_csv(
        output_path,
        index=True,
        chunksize=100_000,
        encoding='utf-8'
    )
    print(f"分组日志已保存至: {output_path}")
    logger.info(f"分组后日志总数: {len(grouped_log_df)}")
    logger.info(f"异常日志数: {grouped_log_df['Label'].sum()}")
    logger.info(f"正常日志数: {len(grouped_log_df)-grouped_log_df['Label'].sum()}")
    logger.info(f"异常比例: {grouped_log_df['Label'].sum()/len(grouped_log_df):.4f}")
    return grouped_log_df

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    structured_file = os.path.join(args.input_dir, "structured.csv")
    if not os.path.isfile(structured_file):
        candidates = sorted(glob.glob(os.path.join(args.input_dir, "*.log_structured.csv")))
        if candidates:
            structured_file = candidates[0]
            logger.warning(f"未找到 structured.csv，改用: {structured_file}")
        else:
            raise FileNotFoundError(f"在 {args.input_dir} 下未找到 structured.csv 或 *.log_structured.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    # 执行日志分组（核心功能）
    group_logs(
        structured_log_file=structured_file,
        window_size=args.window_size,
        step_size=args.step_size,
        output_dir=os.path.join(args.output_dir),
        group_type="fixed"  # 可选：fixed / time / session
    )
