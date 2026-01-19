import tarfile
import os
import json
import io


from tqdm import tqdm
try:
    import soundfile as sf
except ImportError:  # graceful fallback if soundfile not installed
    sf = None
    print("[警告] 未安装 soundfile 库，音频读取功能不可用。请运行: pip install soundfile")
from tqdm import tqdm
import ray


import subprocess
import sys
from pathlib import Path
import json 
import numpy as np

@ray.remote
def process_ray(process_fn, info_dict):
    """
    Ray远程函数：记录每个文件在tar包中的位置，生成一个info.scp文件
    这是get_tar_info的Ray版本，用于并行处理
    """
    return process_fn(info_dict)


def ParallelProcessRay(info_dicts, process_fn, num_cpus=None):
    """
    并行处理多个tar文件，生成info.scp文件
    
    Args:
        info_dicts: info_dict列表
        process_fn: 处理函数，接收info_dict作为参数
        num_cpus: 使用的CPU核心数，None表示使用所有可用核心
    
    Returns:
        Defined by process_fn
    """
    # 初始化Ray，使用所有可用CPU
    if not ray.is_initialized():
        if num_cpus is None:
            ray.init()
        else:
            ray.init(num_cpus=num_cpus)
    
    print(f"使用 Ray 并行处理 {len(info_dicts)} 个info_dict...")
    print(f"可用CPU核心数: {ray.cluster_resources().get('CPU', 0)}")
    
    # 提交所有任务
    futures = [process_ray.remote(process_fn, info_dict) for info_dict in info_dicts]
    
    # 使用tqdm显示进度
    results = []
    for future in tqdm(futures, desc="处理tar文件"):
        try:
            result = ray.get(future)
            results.append(result)
        except Exception as e:
            print(f"处理失败: {e}")
            results.append(None)
    
    return results
