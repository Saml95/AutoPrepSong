"""
歌词处理脚本
1. 如果 is_lyric 是 false，text_ori 放原始 text
2. Function 1: 使用 re 只保留中括号和中括号里的内容
3. Function 2: 计算词/字时长统计
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm


def detect_language(text: str) -> str:
    """
    检测文本语种
    返回: 'cjk' (中日韩泰等字符语系) 或 'western' (西语系)
    """
    if not text:
        return 'western'
    
    # 移除中括号及其内容
    text_clean = re.sub(r'\[[^\]]*\]', '', text).strip()
    if not text_clean:
        return 'western'
    
    # 统计字符类型
    cjk_count = 0
    western_count = 0
    
    for char in text_clean:
        # 中文
        if '\u4e00' <= char <= '\u9fff':
            cjk_count += 1
        # 日文平假名和片假名
        elif '\u3040' <= char <= '\u30ff':
            cjk_count += 1
        # 韩文
        elif '\uac00' <= char <= '\ud7af':
            cjk_count += 1
        # 泰语
        elif '\u0e00' <= char <= '\u0e7f':
            cjk_count += 1
        # 拉丁字母
        elif char.isalpha():
            western_count += 1
    
    # 如果有CJK/泰语字符，判定为CJK语系
    if cjk_count > 0:
        return 'cjk'
    else:
        return 'western'


def keep_only_brackets(text: str) -> str:
    """
    Function 1: 只保留中括号和中括号里的内容
    例如: "[intro] 词:ABC" -> "[intro]"
    """
    # 匹配所有中括号及其内容
    brackets = re.findall(r'\[[^\]]*\]', text)
    return ''.join(brackets)


def get_text_without_brackets(text: str) -> str:
    """
    去掉中括号和里面的内容
    """
    return re.sub(r'\[[^\]]*\]', '', text).strip()


def count_words(text: str) -> int:
    """
    计算西语系的词数
    优先按空格分隔，如果没有空格则计算非空白非标点字符数
    （处理泰语、阿拉伯语等非拉丁字母语言）
    """
    text_clean = get_text_without_brackets(text)
    if not text_clean:
        return 0
    
    # 先尝试按空格分词
    words = text_clean.split()
    if len(words) > 1:
        # 有多个词，返回词数
        return len(words)
    
    # 只有一个词或没有空格，尝试匹配拉丁字母词
    latin_words = re.findall(r"[a-zA-ZàâäéèêëïîôùûüÿœæçÀÂÄÉÈÊËÏÎÔÙÛÜŸŒÆÇäöüÄÖÜßáéíóúñÁÉÍÓÚÑ]+", text_clean)
    if latin_words:
        return len(latin_words)
    
    # 没有拉丁字母，计算非空白非标点的字符数（适用于泰语等）
    # 去除空白和常见标点
    chars = re.sub(r'[\s\.,!?;:\'\"()\[\]{}，。！？；：、""''（）【】]', '', text_clean)
    return len(chars) if chars else 1  # 至少返回 1


def count_characters(text: str) -> int:
    """
    计算字符语系的字符数（中日韩泰字符）
    """
    text_clean = get_text_without_brackets(text)
    if not text_clean:
        return 0
    
    count = 0
    for char in text_clean:
        # 中文
        if '\u4e00' <= char <= '\u9fff':
            count += 1
        # 日文平假名和片假名
        elif '\u3040' <= char <= '\u30ff':
            count += 1
        # 韩文
        elif '\uac00' <= char <= '\ud7af':
            count += 1
        # 泰语
        elif '\u0e00' <= char <= '\u0e7f':
            count += 1
    return count


def process_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个 segment
    - 先处理 is_lyric：如果是 false，text_ori 放原始 text，text 只保留中括号
    - 然后基于处理后的 text 计算 lang_type, lyric_count 和 average_duration
    """
    new_seg = seg.copy()
    
    is_lyric = seg.get('is_lyric', True)
    text_original = seg.get('text', '')
    start = seg.get('start', 0.0)
    end = seg.get('end', 0.0)
    duration = end - start
    
    # Step 1: 先处理 is_lyric
    if not is_lyric:
        new_seg['text_ori'] = text_original
        # Function 1: 只保留中括号内容
        new_seg['text'] = keep_only_brackets(text_original)
        # 使用处理后的 text 进行统计
        text_for_stats = new_seg['text']
    else:
        # is_lyric=True 的使用原始 text
        text_for_stats = text_original
    
    # Step 2: 基于处理后的 text 计算 lang_type 和时长统计
    text_clean = get_text_without_brackets(text_for_stats)
    
    if text_clean:
        lang_type = detect_language(text_for_stats)
        new_seg['lang_type'] = lang_type
        
        if lang_type == 'western':
            # 西语系：计算词数
            lyric_count = count_words(text_for_stats)
            new_seg['lyric_count'] = lyric_count
            if lyric_count > 0 and duration > 0:
                new_seg['average_duration'] = duration / lyric_count
        else:
            # CJK语系：计算字符数
            lyric_count = count_characters(text_for_stats)
            new_seg['lyric_count'] = lyric_count
            if lyric_count > 0 and duration > 0:
                new_seg['average_duration'] = duration / lyric_count
    else:
        # 空文本（如纯 [intro]）
        new_seg['lang_type'] = None
    
    return new_seg


def calculate_duration_stats(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Function 2: 计算时长统计（只考虑 is_lyric=True 的片段）
    返回整首歌的平均时长统计，key 为 lang_type (cjk/western)
    """
    total_western = 0
    total_cjk = 0
    total_duration_western = 0.0
    total_duration_cjk = 0.0
    
    for seg in segments:
        # 只考虑 is_lyric=True 的片段
        if not seg.get('is_lyric', True):
            continue
        
        lang_type = seg.get('lang_type')
        if lang_type is None:
            continue
        
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        duration = end - start
        
        lyric_count = seg.get('lyric_count', 0)
        if lyric_count > 0:
            if lang_type == 'western':
                total_western += lyric_count
                total_duration_western += duration
            else:  # cjk
                total_cjk += lyric_count
                total_duration_cjk += duration
    
    # 整首歌的统计，key 为 lang_type
    song_stats = {
        'cjk': {
            'average_duration': total_duration_cjk / total_cjk if total_cjk > 0 else 0,
            'number': total_cjk,
            'total_dur': total_duration_cjk
        },
        'western': {
            'average_duration': total_duration_western / total_western if total_western > 0 else 0,
            'number': total_western,
            'total_dur': total_duration_western
        }
    }
    
    return song_stats


def process_json_data(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    处理单个 JSON 数据
    返回: (处理后的数据, 统计信息)
    """
    segments = data.get('segments', [])
    
    # 处理每个 segment（Function 1 + 计算每个 seg 的统计）
    processed_segments = [process_segment(seg) for seg in segments]
    
    # 计算整首歌时长统计（Function 2，只考虑 is_lyric=True）
    duration_stats = calculate_duration_stats(processed_segments)
    
    # 检查每个 seg 的时长是否异常（相对于整首歌平均值的 3 倍以上或 1/2 以下）
    is_duration_ill_cnt = 0
    total_seg_num = len(processed_segments)
    
    for seg in processed_segments:
        lang_type = seg.get('lang_type')
        
        if lang_type and lang_type in duration_stats:
            avg_dur = duration_stats[lang_type]['average_duration']
            seg_avg = seg.get('average_duration', 0)
            
            # 计算 segment 时长
            seg_start = seg.get('start', 0.0)
            seg_end = seg.get('end', 0.0)
            seg_duration = seg_end - seg_start
            
            if avg_dur > 0 and seg_avg > 0:
                # 3 倍以上或 1/3 以下视为异常
                is_ill = seg_avg > avg_dur * 3 or seg_avg < avg_dur / 3
                
                # 特殊情况：segment 长度小于 3s 且 average_duration 大于整曲平均
                # 这种情况不视为异常（短片段的单词时长偏长是正常的）
                if is_ill and seg_duration < 3 and (seg_avg > avg_dur or seg_avg > avg_dur / 5):
                    is_ill = False
                
                # 特殊情况：lyric_count * 整曲avg_dur 与 seg 实际时长差距小于 3s
                # 说明该 segment 的总时长是合理的，不应标记为异常
                lyric_count = seg.get('lyric_count', 0)
                expected_duration = lyric_count * avg_dur
                if is_ill and abs(expected_duration - seg_duration) < 3:
                    is_ill = False
                    
                
                seg['is_duration_ill'] = is_ill
                if is_ill:
                    is_duration_ill_cnt += 1
            else:
                seg['is_duration_ill'] = False
        else:
            # lang_type 为 None 或平均值为 0 时，不判断
            seg['is_duration_ill'] = False
    
    # 添加异常统计到 duration_stats
    duration_stats['is_duration_ill_cnt'] = is_duration_ill_cnt
    duration_stats['total_seg_num'] = total_seg_num
    duration_stats['ill_duration_seg_ratio'] = is_duration_ill_cnt / total_seg_num if total_seg_num > 0 else 0
    
    # 更新数据
    new_data = data.copy()
    new_data['segments'] = processed_segments
    new_data['duration_stats'] = duration_stats
    
    return new_data, duration_stats


def process_json_file(json_path: str, output_path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    处理单个 JSON 文件
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data, stats = process_json_data(data)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return processed_data, stats


def process_scp(scp_path: str, output_dir: Optional[str] = None):
    """
    处理 scp 文件中的所有 JSON 文件
    """
    scp_path = Path(scp_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(scp_path, 'r', encoding='utf-8') as f:
        json_paths = [line.strip() for line in f if line.strip()]
    
    all_stats = []
    
    for json_path in tqdm(json_paths, desc="Processing JSON files"):
        if not Path(json_path).exists():
            print(f"File not found: {json_path}")
            continue
        
        try:
            if output_dir:
                output_path = output_dir / Path(json_path).name
            else:
                output_path = None
            
            processed_data, stats = process_json_file(json_path, str(output_path) if output_path else None)
            all_stats.append({
                'json_path': json_path,
                'stats': stats
            })
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
    
    return all_stats


def demo():
    """
    演示功能
    """
    # 测试数据
    test_segments = [
        {"text": "[intro] 词:ABC 曲:XYZ", "start": 0.0, "end": 5.0, "is_lyric": False},
        {"text": "[verse] 喜欢你 仿佛坠入无边海底", "start": 5.0, "end": 10.0, "is_lyric": True},
        {"text": "[chorus] Hello world how are you", "start": 10.0, "end": 15.0, "is_lyric": True},
        {"text": "[outro] 制作人:张三", "start": 15.0, "end": 20.0, "is_lyric": False},
        {"text": "[bridge] こんにちは世界", "start": 20.0, "end": 25.0, "is_lyric": True},
    ]
    
    test_data = {"segments": test_segments}
    
    processed_data, stats = process_json_data(test_data)
    
    print("=" * 50)
    print("处理后的 segments:")
    print("=" * 50)
    for seg in processed_data['segments']:
        print(json.dumps(seg, ensure_ascii=False, indent=2))
        print("-" * 30)
    
    print("\n" + "=" * 50)
    print("整首歌统计 (duration_stats):")
    print("=" * 50)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="处理歌词 JSON 文件")
    parser.add_argument("--scp", type=str, 
                        default="/home/jianwei/music/luoxue/json_group_lyric_gp5_v1/json_group_lyric_gp5_v1_local.scp",
                        help="scp 文件路径")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/jianweiyu/exp/music/luoxue/json_group_lyric_gp5_v1_processed_v2_fixed",
                        help="输出目录（保留倒数两级目录结构）")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    parser.add_argument("--debug", action="store_true", help="Debug 模式，只处理 10 个文件")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    else:
        json_scp = args.scp
        output_base_dir = Path(args.output_dir)
        
        # 读取 scp 文件
        with open(json_scp, 'r', encoding='utf-8') as f:
            json_paths = [line.strip() for line in f if line.strip()]
        
        # Debug 模式：只处理 10 个文件
        if args.debug:
            json_paths = json_paths[:10]
            print("[DEBUG MODE] 只处理前 10 个文件")
        
        print(f"Found {len(json_paths)} JSON files to process")
        print(f"Output dir: {output_base_dir}")
        
        all_stats = []
        for json_path in tqdm(json_paths, desc="Processing"):
            if not Path(json_path).exists():
                print(f"File not found: {json_path}")
                continue
            
            try:
                # 读取 JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理数据
                processed_data, stats = process_json_data(data)
                
                # 保留倒数两级目录结构
                # 例如: /a/b/c/d/file.json -> output_dir/c/d/file.json
                json_path_obj = Path(json_path)
                parts = json_path_obj.parts
                if len(parts) >= 2:
                    # 取倒数两级目录 + 文件名
                    relative_path = Path(parts[-2]) / parts[-1]
                else:
                    relative_path = json_path_obj.name
                
                output_path = output_base_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存到新位置（禁止覆盖源文件）
                if str(output_path.resolve()) == str(json_path_obj.resolve()):
                    print(f"Warning: 跳过，输出路径与源文件相同: {json_path}")
                    continue
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                all_stats.append({
                    'json_path': json_path,
                    'output_path': str(output_path),
                    'duration_stats': stats
                })
                
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
        
        # 汇总统计
        print(f"\n处理完成: {len(all_stats)} / {len(json_paths)} 个文件")
        
        # 计算整体统计（统一格式）
        total_western = sum(s['duration_stats']['western']['number'] for s in all_stats)
        total_cjk = sum(s['duration_stats']['cjk']['number'] for s in all_stats)
        total_western_dur = sum(s['duration_stats']['western']['total_dur'] for s in all_stats)
        total_cjk_dur = sum(s['duration_stats']['cjk']['total_dur'] for s in all_stats)
        
        print(f"\n整体统计:")
        if total_western > 0:
            print(f"  western: 总词数={total_western}, 平均词时长={total_western_dur/total_western:.3f}s")
        if total_cjk > 0:
            print(f"  cjk: 总字数={total_cjk}, 平均字时长={total_cjk_dur/total_cjk:.3f}s")
        
        # 保存新的 scp 文件（格式：path seg_number ill_number ratio）
        new_scp_path = output_base_dir / "processed.scp"
        with open(new_scp_path, 'w', encoding='utf-8') as f:
            for s in all_stats:
                seg_num = s['duration_stats']['total_seg_num']
                ill_cnt = s['duration_stats']['is_duration_ill_cnt']
                ratio = s['duration_stats']['ill_duration_seg_ratio']
                f.write(f"{s['output_path']}\t{seg_num}\t{ill_cnt}\t{ratio:.4f}\n")
        print(f"\n新 scp 文件已保存: {new_scp_path}")