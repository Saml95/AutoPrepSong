"""
后处理工具集
用于处理已标注的JSON文件
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


def load_scp_file(scp_path: str) -> List[str]:
    """加载scp文件，返回JSON文件路径列表"""
    scp_path = Path(scp_path)
    json_paths = []
    
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # scp格式可能是 "id path" 或只有 "path"
                parts = line.split()
                if len(parts) >= 2:
                    json_paths.append(parts[1])
                else:
                    json_paths.append(parts[0])
    
    return json_paths


def remove_laughter(
    scp_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    移除segments中的Laughter标记（使用正则，不区分大小写）
    
    Args:
        scp_path: SCP文件路径
        output_dir: 输出目录
    
    Returns:
        统计信息
    """
    output_dir = Path(output_dir)
    json_paths = load_scp_file(scp_path)
    
    stats = {
        "total_files": len(json_paths),
        "processed_files": 0,
        "total_segments_before": 0,
        "total_segments_after": 0,
        "removed_segments": 0,
        "modified_segments": 0,
    }
    
    for json_path in tqdm(json_paths, desc="Processing files"):
        json_path = Path(json_path)
        
        if not json_path.exists():
            print(f"Warning: File not found: {json_path}")
            continue
        
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            segments = data.get("segments", [])
            stats["total_segments_before"] += len(segments)
            
            # 处理每个segment
            new_segments = []
            for seg in segments:
                text = seg.get("text", "")
                original_text = text
                
                # 使用正则移除 laughter（不区分大小写，后面可能跟各种标点）
                # 匹配 laughter 后面跟0或多个标点符号
                text = re.sub(r'laughter[.,!?\'"]*', '', text, flags=re.IGNORECASE)
                
                # 清理多余空格
                text = re.sub(r' +', ' ', text).strip()
                
                # 如果text为空，跳过这个segment
                if not text:
                    stats["removed_segments"] += 1
                    continue
                
                # 如果整句只有 noise/Noise 等变体，统一变成 [noise]
                text_lower = text.lower().strip().rstrip('.,!?')
                if text_lower == "noise":
                    text = "[noise]"
                
                # 更新segment
                if text != original_text:
                    stats["modified_segments"] += 1
                    seg = seg.copy()
                    seg["text"] = text
                    seg["text_ori_laughter"] = original_text
                    seg["is_modified_laughter"] = True
                else:
                    seg = seg.copy()
                    seg["is_modified_laughter"] = False
                
                new_segments.append(seg)
            
            stats["total_segments_after"] += len(new_segments)
            
            # 确定输出路径（保留最后两级目录）
            # 例如: /path/to/0049/file.json -> output_dir/0049/file.json
            parent_name = json_path.parent.name
            output_path = output_dir / parent_name / json_path.name
            
            # 更新数据
            data["segments"] = new_segments
            data["laughter_removed"] = True
            
            # 保存
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            stats["processed_files"] += 1
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue
    
    # 打印统计
    print("\n" + "=" * 50)
    print("Remove Laughter Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Processed files: {stats['processed_files']}")
    print(f"  Total segments before: {stats['total_segments_before']}")
    print(f"  Total segments after: {stats['total_segments_after']}")
    print(f"  Removed segments (empty after removal): {stats['removed_segments']}")
    print(f"  Modified segments: {stats['modified_segments']}")
    print("=" * 50)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="JSON后处理工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # remove_laughter 子命令
    laughter_parser = subparsers.add_parser("remove_laughter", help="移除Laughter标记")
    laughter_parser.add_argument("--scp", type=str, required=True, help="SCP文件路径")
    laughter_parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    # patterns 参数已废弃，改用正则匹配
    
    args = parser.parse_args()
    
    if args.command == "remove_laughter":
        remove_laughter(
            scp_path=args.scp,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
