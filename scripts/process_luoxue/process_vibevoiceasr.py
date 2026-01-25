

#!/usr/bin/env python3
"""
按照speaker number将scp文件分成多个scp文件。
每行是一个json文件路径，读取json文件中的segments，
统计unique speaker数量，然后按speaker数量分组输出到不同的scp文件。
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def get_speaker_count(json_path: str, min_seg_count: int = 2, min_seg_ratio: float = 0.1) -> int:
    """
    读取json文件，返回有效speaker数量。
    过滤条件：
    - speaker的segment个数 >= min_seg_count
    - speaker的segment个数占总speaker segment数的比例 >= min_seg_ratio
    
    如果文件不存在或格式错误，返回-1。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('segments', [])
        
        # 统计每个speaker的segment数量
        speaker_seg_counts = defaultdict(int)
        for seg in segments:
            if 'Speaker' in seg:
                speaker_seg_counts[seg['Speaker']] += 1
        
        # 如果没有任何speaker，返回0
        if not speaker_seg_counts:
            return 0
        
        # 计算总的speaker segment数量
        total_speaker_segs = sum(speaker_seg_counts.values())
        
        # 过滤有效的speaker
        valid_speakers = []
        for speaker, count in speaker_seg_counts.items():
            ratio = count / total_speaker_segs if total_speaker_segs > 0 else 0
            # speaker的segment个数 >= min_seg_count 且占比 >= min_seg_ratio
            if count >= min_seg_count and ratio >= min_seg_ratio:
                valid_speakers.append(speaker)
        
        return len(valid_speakers)
    except FileNotFoundError:
        print(f"Warning: File not found: {json_path}")
        return -1
    except json.JSONDecodeError:
        print(f"Warning: JSON decode error: {json_path}")
        return -1
    except Exception as e:
        print(f"Warning: Error reading {json_path}: {e}")
        return -1


def split_scp_by_speaker_num(input_scp: str, output_dir: str = None, prefix: str = None):
    """
    按speaker数量将scp文件分成多个scp文件
    
    Args:
        input_scp: 输入的scp文件路径
        output_dir: 输出目录，默认为输入scp文件所在目录
        prefix: 输出文件前缀，默认为输入scp文件名(不含扩展名)
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_scp)
        if not output_dir:
            output_dir = '.'
    
    # 设置输出文件前缀
    if prefix is None:
        prefix = Path(input_scp).stem
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取scp文件
    print(f"Reading scp file: {input_scp}")
    with open(input_scp, 'r', encoding='utf-8') as f:
        json_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Total {len(json_paths)} json files in scp")
    
    # 过滤掉路径中包含 "error" 的文件
    filtered_paths = []
    skipped_error_paths = []
    for json_path in json_paths:
        if 'error' in json_path.lower():
            skipped_error_paths.append(json_path)
        else:
            filtered_paths.append(json_path)
    
    print(f"Skipped {len(skipped_error_paths)} files with 'error' in path")
    print(f"Processing {len(filtered_paths)} json files")
    
    # 按speaker数量分组
    speaker_groups = defaultdict(list)
    error_files = []
    
    for json_path in tqdm(filtered_paths, desc="Processing"):
        speaker_count = get_speaker_count(json_path)
        if speaker_count >= 0:
            speaker_groups[speaker_count].append(json_path)
        else:
            error_files.append(json_path)
    
    # 输出统计信息
    print("\n=== Statistics ===")
    print(f"{'Speaker Count':<15} {'File Count':<10}")
    print("-" * 30)
    for speaker_count in sorted(speaker_groups.keys()):
        print(f"{speaker_count:<15} {len(speaker_groups[speaker_count]):<10}")
    
    if error_files:
        print(f"{'Read Error':<15} {len(error_files):<10}")
    if skipped_error_paths:
        print(f"{'Skipped(error)':<15} {len(skipped_error_paths):<10}")
    
    print("-" * 30)
    print(f"{'Total':<15} {len(json_paths):<10}")
    
    # 合并成3组: 0, 1, 2+
    group_0 = sorted(speaker_groups.get(0, []))
    group_1 = sorted(speaker_groups.get(1, []))
    group_2_plus = []
    for speaker_count in speaker_groups:
        if speaker_count >= 2:
            group_2_plus.extend(speaker_groups[speaker_count])
    group_2_plus = sorted(group_2_plus)
    
    # 写入3个分组文件
    print("\n=== Writing output files ===")
    
    # speaker0.scp
    output_path_0 = os.path.join(output_dir, f"{prefix}_speaker0.scp")
    with open(output_path_0, 'w', encoding='utf-8') as f:
        for json_path in group_0:
            f.write(json_path + '\n')
    print(f"Written {len(group_0)} files to {output_path_0}")
    
    # speaker1.scp
    output_path_1 = os.path.join(output_dir, f"{prefix}_speaker1.scp")
    with open(output_path_1, 'w', encoding='utf-8') as f:
        for json_path in group_1:
            f.write(json_path + '\n')
    print(f"Written {len(group_1)} files to {output_path_1}")
    
    # speaker2+.scp
    output_path_2_plus = os.path.join(output_dir, f"{prefix}_speaker2+.scp")
    with open(output_path_2_plus, 'w', encoding='utf-8') as f:
        for json_path in group_2_plus:
            f.write(json_path + '\n')
    print(f"Written {len(group_2_plus)} files to {output_path_2_plus}")
    
    # 写入读取错误的文件列表
    if error_files:
        error_files = sorted(error_files)
        error_path = os.path.join(output_dir, f"{prefix}_read_error.scp")
        with open(error_path, 'w', encoding='utf-8') as f:
            for json_path in error_files:
                f.write(json_path + '\n')
        print(f"Written {len(error_files)} read error files to {error_path}")
    
    # 写入被跳过的error路径文件列表
    if skipped_error_paths:
        skipped_path = os.path.join(output_dir, f"{prefix}_skipped_error_path.scp")
        with open(skipped_path, 'w', encoding='utf-8') as f:
            for json_path in skipped_error_paths:
                f.write(json_path + '\n')
        print(f"Written {len(skipped_error_paths)} skipped error path files to {skipped_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    json_vibeasr_path = "/home/jianwei/music/luoxue_20251226/json_vibevocieasr_local.scp"
    split_scp_by_speaker_num(json_vibeasr_path)