"""
统计 scp 文件中所有音频的时长信息
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def calculate_durations(scp_path: str):
    """
    统计 scp 文件中所有音频的时长
    
    Args:
        scp_path: scp 文件路径，每行一个 JSON 文件路径
    """
    with open(scp_path, 'r', encoding='utf-8') as f:
        json_paths = [line.strip().split('\t')[0] for line in f if line.strip()]
    
    print(f'总文件数: {len(json_paths)}')
    
    durations = []
    missing = 0
    errors = 0
    
    for json_path in tqdm(json_paths, desc='Processing'):
        if not Path(json_path).exists():
            missing += 1
            continue
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 优先使用 audio_length 字段，否则使用最后一个 segment 的 end
            audio_length = data.get('audio_length')
            if audio_length is None:
                segments = data.get('segments', [])
                if segments:
                    audio_length = segments[-1].get('end', 0)
                else:
                    audio_length = 0
            
            durations.append(audio_length)
        except Exception as e:
            errors += 1
            print(f'Error: {json_path}: {e}')
    
    print(f'\n缺失文件: {missing}')
    print(f'错误文件: {errors}')
    print(f'有效文件: {len(durations)}')
    
    if durations:
        total = sum(durations)
        avg = total / len(durations)
        min_dur = min(durations)
        max_dur = max(durations)
        
        print(f'\n=== 统计结果 ===')
        print(f'总时长: {total:.2f}s = {total/60:.2f}min = {total/3600:.2f}h')
        print(f'平均时长: {avg:.2f}s = {avg/60:.2f}min')
        print(f'最小时长: {min_dur:.2f}s')
        print(f'最大时长: {max_dur:.2f}s = {max_dur/60:.2f}min')
        
        return {
            'total': total,
            'avg': avg,
            'min': min_dur,
            'max': max_dur,
            'count': len(durations)
        }
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 scp 文件中所有音频的时长")
    parser.add_argument("--scp", type=str, required=True,
                        help="scp 文件路径")
    
    args = parser.parse_args()
    
    calculate_durations(args.scp)
