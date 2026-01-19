"""
处理歌曲文件名，构建 singer dict
文件名格式: 歌手 - 歌曲名.flac.json
歌手可能有多个，用 "、" 分隔
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def parse_singer_from_path(json_path: str) -> Tuple[str, List[str], int]:
    """
    从文件路径解析歌手信息
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        (singer_str, singer_list, singer_number)
        - singer_str: 原始歌手字符串，如 "何鹏、陈玉建"
        - singer_list: 歌手列表，如 ["何鹏", "陈玉建"]
        - singer_number: 歌手数量
    """
    # 获取文件名（不含路径）
    filename = Path(json_path).name
    
    # 去掉 .flac.json 或 .mp3.json 等后缀
    # 格式: 歌手 - 歌曲名.flac.json
    base_name = filename
    for ext in ['.flac.json', '.mp3.json', '.wav.json', '.m4a.json', '.json']:
        if base_name.endswith(ext):
            base_name = base_name[:-len(ext)]
            break
    
    # 按 " - " 分割，取第一部分作为歌手
    if ' - ' in base_name:
        singer_str = base_name.split(' - ')[0].strip()
    else:
        # 没有 " - " 分隔符，整个作为歌手
        singer_str = base_name.strip()
    
    # 歌手可能用 "、" 分隔
    singer_list = [s.strip() for s in singer_str.split('、') if s.strip()]
    singer_number = len(singer_list)
    
    return singer_str, singer_list, singer_number


def build_singer_dict(scp_path: str) -> Dict[str, Dict]:
    """
    从 scp 文件构建 singer dict
    
    Args:
        scp_path: processed.scp 文件路径
        
    Returns:
        singer_dict: {
            singer_str: {
                'singer_list': [...],
                'singer_number': int,
                'json_files': [...]
            }
        }
    """
    singer_dict = defaultdict(lambda: {
        'singer_list': [],
        'singer_number': 0,
        'json_files': []
    })
    
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # scp 格式: path\tseg_number\till_number\tratio
            parts = line.split('\t')
            json_path = parts[0]
            
            singer_str, singer_list, singer_number = parse_singer_from_path(json_path)
            
            if singer_str not in singer_dict:
                singer_dict[singer_str]['singer_list'] = singer_list
                singer_dict[singer_str]['singer_number'] = singer_number
            
            singer_dict[singer_str]['json_files'].append(json_path)
    
    return dict(singer_dict)


def main():
    parser = argparse.ArgumentParser(description="处理歌曲文件名，构建 singer dict")
    parser.add_argument("--scp", type=str,
                        default="/home/jianweiyu/exp/music/luoxue/json_group_lyric_gp5_v1_processed_v2/processed.scp",
                        help="processed.scp 文件路径")
    parser.add_argument("--output", type=str,
                        default="/home/jianweiyu/exp/music/luoxue/json_group_lyric_gp5_v1_processed_v2/singer_dict.json",
                        help="输出 singer_dict.json 文件路径")
    parser.add_argument("--debug", action="store_true", help="Debug 模式，只处理前 100 行")
    
    args = parser.parse_args()
    
    print(f"Reading scp: {args.scp}")
    singer_dict = build_singer_dict(args.scp)
    
    # 统计信息
    total_singers = len(singer_dict)
    total_files = sum(len(v['json_files']) for v in singer_dict.values())
    
    # 按歌曲数量排序
    sorted_singers = sorted(singer_dict.items(), key=lambda x: len(x[1]['json_files']), reverse=True)
    
    print(f"\n统计信息:")
    print(f"  总歌手数: {total_singers}")
    print(f"  总文件数: {total_files}")
    
    # 单人歌手 vs 多人歌手统计
    single_singer_cnt = sum(1 for v in singer_dict.values() if v['singer_number'] == 1)
    multi_singer_cnt = total_singers - single_singer_cnt
    print(f"  单人歌手数: {single_singer_cnt}")
    print(f"  多人歌手/组合数: {multi_singer_cnt}")
    
    # Top 10 歌手
    print(f"\nTop 10 歌手 (按歌曲数量):")
    for i, (singer, info) in enumerate(sorted_singers[:10]):
        print(f"  {i+1}. {singer}: {len(info['json_files'])} 首 (singer_number={info['singer_number']})")
    
    # Top 10 多人歌手
    multi_singer_sorted = [(s, i) for s, i in sorted_singers if i['singer_number'] > 1]
    print(f"\nTop 10 多人歌手 (按歌曲数量):")
    for i, (singer, info) in enumerate(multi_singer_sorted[:10]):
        print(f"  {i+1}. {singer}: {len(info['json_files'])} 首 (singer_number={info['singer_number']}, singers={info['singer_list']})")
    
    # 保存 singer_dict
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(singer_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nsinger_dict 已保存: {output_path}")
    
    # 保存单人歌手和多人歌手的 scp 文件
    single_singer_files = []
    multi_singer_files = []
    
    for singer_str, info in singer_dict.items():
        if info['singer_number'] == 1:
            single_singer_files.extend(info['json_files'])
        else:
            multi_singer_files.extend(info['json_files'])
    
    # 保存单人歌手 scp
    single_scp_path = output_path.parent / "single_singer.scp"
    with open(single_scp_path, 'w', encoding='utf-8') as f:
        for json_file in single_singer_files:
            f.write(f"{json_file}\n")
    print(f"单人歌手 scp 已保存: {single_scp_path} ({len(single_singer_files)} 首)")
    
    # 保存多人歌手 scp
    multi_scp_path = output_path.parent / "multi_singer.scp"
    with open(multi_scp_path, 'w', encoding='utf-8') as f:
        for json_file in multi_singer_files:
            f.write(f"{json_file}\n")
    print(f"多人歌手 scp 已保存: {multi_scp_path} ({len(multi_singer_files)} 首)")


if __name__ == "__main__":
    main()