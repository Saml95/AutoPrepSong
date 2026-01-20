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

# 用于绘制直方图
try:
    import matplotlib
    matplotlib.use('Agg')  # 无头模式
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, histogram will not be generated")


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


def reset_ill_duration_segments(segments: List[Dict[str, Any]], duration_stats: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """
    对 is_duration_ill=True 且 average_duration > 整曲平均 的 segment 进行重置
    返回: (处理后的 segments, 是否有重置)
    """
    new_segments = []
    has_reset = False
    
    for seg in segments:
        is_ill = seg.get('is_duration_ill', False)
        lang_type = seg.get('lang_type')
        
        # 检查是否需要重置
        if is_ill and lang_type and lang_type in duration_stats:
            seg_avg = seg.get('average_duration', 0)
            global_avg = duration_stats[lang_type]['average_duration']
            
            # 只处理 average_duration > 整曲平均 的情况
            if seg_avg > global_avg and global_avg > 0:
                lyric_count = seg.get('lyric_count', 0)
                seg_start = seg.get('start', 0.0)
                seg_end = seg.get('end', 0.0)
                
                # 计算新的 end time
                new_end = global_avg * lyric_count * 2 + seg_start
                
                # 如果重置时间和原始 seg end 相差小于 3s，则不重置
                time_diff = seg_end - new_end
                
                if time_diff >= 7.0:
                    # 需要重置
                    has_reset = True
                    
                    # 创建重置后的 segment
                    reset_seg = seg.copy()
                    reset_seg['end'] = new_end
                    reset_seg['reset_end_time'] = True
                    # 重新计算 average_duration
                    if lyric_count > 0:
                        reset_seg['average_duration'] = (new_end - seg_start) / lyric_count
                    new_segments.append(reset_seg)
                    
                    # 创建从重置点到原来 seg end 的新 segment
                    # text 只保留括号里的部分
                    original_text = seg.get('text', '')
                    bracket_text = keep_only_brackets(original_text)
                    
                    filler_seg = {
                        'text': bracket_text,
                        'start': new_end,
                        'end': seg_end,
                        'speaker': None,
                        'is_lyric': False,
                        'lang_type': None,
                        'is_duration_ill': False
                    }
                    new_segments.append(filler_seg)
                else:
                    # 不需要重置，保持原样
                    new_segments.append(seg)
            else:
                # average_duration <= 整曲平均，不处理
                new_segments.append(seg)
        else:
            # 不是异常 segment，保持原样
            new_segments.append(seg)
    
    return new_segments, has_reset


def process_json_data(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    处理单个 JSON 数据
    返回: (处理后的数据, 统计信息, 是否有重置)
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
                if is_ill and abs(expected_duration - seg_duration) < 6:
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
    
    # 对 is_duration_ill 的 segment 进行重置处理
    processed_segments, has_reset = reset_ill_duration_segments(processed_segments, duration_stats)
    
    # 更新数据
    new_data = data.copy()
    new_data['segments'] = processed_segments
    new_data['duration_stats'] = duration_stats
    new_data['has_reset_segments'] = has_reset
    
    return new_data, duration_stats, has_reset


def process_json_file(json_path: str, output_path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    处理单个 JSON 文件
    返回: (处理后的数据, 统计信息, 是否有重置)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data, stats, has_reset = process_json_data(data)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return processed_data, stats, has_reset


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
            
            processed_data, stats, has_reset = process_json_file(json_path, str(output_path) if output_path else None)
            all_stats.append({
                'json_path': json_path,
                'stats': stats,
                'has_reset': has_reset
            })
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
    
    return all_stats


def collect_lyric_segment_durations(json_paths: List[str], bin_size: float = 1.0, max_duration: float = 30.0) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """
    收集所有带歌词的 segment 的时长，直接按区间统计（内存友好）
    
    Args:
        json_paths: JSON 文件路径列表
        bin_size: 区间大小（秒）
        max_duration: 最大统计时长（超过的归入最后一个 bin）
    
    返回: (区间统计字典 {bin_start: count}, 统计信息)
    """
    # 初始化区间统计字典
    num_bins = int(max_duration / bin_size) + 1
    bin_counts = {i: 0 for i in range(num_bins)}  # key 是 bin 的起始秒数
    
    # 统计变量
    total_count = 0
    total_duration = 0.0
    min_duration = float('inf')
    max_dur_seen = 0.0
    
    for json_path in tqdm(json_paths, desc="Collecting segment durations"):
        if not Path(json_path).exists():
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = data.get('segments', [])
            for seg in segments:
                # 只统计 is_lyric=True 且有实际歌词内容的 segment
                is_lyric = seg.get('is_lyric', True)
                lang_type = seg.get('lang_type')
                
                if is_lyric and lang_type is not None:
                    start = seg.get('start', 0.0)
                    end = seg.get('end', 0.0)
                    duration = end - start
                    if duration > 0:
                        # 更新统计
                        total_count += 1
                        total_duration += duration
                        min_duration = min(min_duration, duration)
                        max_dur_seen = max(max_dur_seen, duration)
                        
                        # 计算所属的 bin
                        bin_idx = min(int(duration / bin_size), num_bins - 1)
                        bin_counts[bin_idx] += 1
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
    
    # 计算统计信息
    if total_count > 0:
        stats = {
            'total_segments': total_count,
            'mean_duration': total_duration / total_count,
            'min_duration': min_duration,
            'max_duration': max_dur_seen,
            'total_duration': total_duration
        }
    else:
        stats = {
            'total_segments': 0,
            'mean_duration': 0,
            'min_duration': 0,
            'max_duration': 0,
            'total_duration': 0
        }
    
    return bin_counts, stats


def plot_duration_histogram(bin_counts: Dict[int, int], stats: Dict[str, Any], output_path: str, bin_size: float = 1.0):
    """
    绘制歌词 segment 时长的柱状图（基于预统计的区间数据）
    
    Args:
        bin_counts: 区间统计字典 {bin_start: count}
        stats: 统计信息字典
        output_path: 输出图片路径
        bin_size: 区间大小（秒）
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping histogram generation")
        return
    
    total_count = stats['total_segments']
    if total_count == 0:
        print("No durations to plot")
        return
    
    # 准备绘图数据
    bins = sorted(bin_counts.keys())
    counts = [bin_counts[b] for b in bins]
    
    # 绘图
    plt.figure(figsize=(14, 8))
    
    # 绘制柱状图
    x_positions = [b + bin_size / 2 for b in bins]  # 柱子中心位置
    bars = plt.bar(x_positions, counts, width=bin_size * 0.9, edgecolor='black', alpha=0.7, color='steelblue')
    
    # 添加标签
    plt.xlabel('Segment Duration (seconds)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    mean_dur = stats['mean_duration']
    plt.title(f'Distribution of Lyric Segment Durations\n(Total: {total_count} segments, Mean: {mean_dur:.2f}s)', fontsize=14)
    
    # 添加网格
    plt.grid(axis='y', alpha=0.3)
    
    # 添加均值线
    plt.axvline(x=mean_dur, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dur:.2f}s')
    
    # 在柱状图上添加数字标签
    for bar, count in zip(bars, counts):
        if count > 0:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            plt.annotate(f'{count}', xy=(x, y), ha='center', va='bottom', fontsize=8)
    
    # 设置 x 轴刻度
    x_ticks = list(range(0, int(max(bins)) + 2, int(bin_size)))
    plt.xticks(x_ticks)
    plt.xlim(-0.5, max(bins) + bin_size + 0.5)
    
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Histogram saved to: {output_path}")
    
    # 打印每个 bin 的统计
    print(f"\nDuration distribution ({bin_size}s bins):")
    for b in bins:
        count = bin_counts[b]
        pct = count / total_count * 100
        end_label = f"{b + bin_size:.0f}" if b < max(bins) else "+"
        print(f"  [{b:5.0f}s - {end_label:>5s}s): {count:6d} ({pct:5.2f}%)")


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
    
    processed_data, stats, has_reset = process_json_data(test_data)
    
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
    print(f"\n是否有重置的 segment: {has_reset}")


def get_structure_label(text: str) -> Optional[str]:
    """
    从 text 中提取结构化标签（如 [intro], [verse], [chorus] 等）
    返回第一个中括号里的内容，如果没有则返回 None
    """
    match = re.search(r'\[([^\]]+)\]', text)
    if match:
        return match.group(1).lower().strip()
    return None


def merge_segments(segments: List[Dict[str, Any]], chunk_size: float = 10.0, min_chunk_size: float = 3.0, merge_mode: int = 3) -> List[Dict[str, Any]]:
    """
    合并 segments，遵循以下规则：
    1. chunk_size 为 merge 的最大可接受长度（默认 10s）
    2. 对于无歌词的 seg（is_lyric=False），不需要管 chunksize，相邻的直接 merge，
       直到出现第一个歌词 seg 或者结构化标签切换（如 [intro] 变成 [verse]）
    3. 对于歌词片段，在结构化标签相同的情况下，merge 到 chunksize，然后切换下一个 chunk。
       如果下一个 chunk 和上一个 chunk 结构标签相同，且时间不足 min_chunk_size（默认 3s），
       则和上一个 chunk 合并
    4. merge 完成后，对于上一个片段没有歌词的情况，当前歌词片段开头向前延伸 0.5s，
       上一个片段的结尾减去 0.5s
    
    Args:
        segments: 输入的 segment 列表
        chunk_size: merge 的最大可接受长度（秒）
        min_chunk_size: 最小 chunk 长度（秒），低于此值会尝试与前一个合并
    
    Returns:
        合并后的 segment 列表
    """
    if not segments:
        return []
    
    merged = []
    current_chunk = None
    
    for seg in segments:
        text = seg.get('text', '')
        # 根据 text 去掉结构标签后是否为空来判断是否有歌词
        text_content = get_text_without_brackets(text)
        has_lyric = bool(text_content.strip())
        structure_label = get_structure_label(text)
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        
        if current_chunk is None:
            # 第一个 segment，直接作为当前 chunk
            current_chunk = seg.copy()
            current_chunk['_has_lyric'] = has_lyric
            current_chunk['is_lyric'] = has_lyric  # 更新 is_lyric 字段
            current_chunk['_structure_label'] = structure_label
            current_chunk['_merged_count'] = 1
            continue
        
        current_has_lyric = current_chunk.get('_has_lyric', True)
        current_label = current_chunk.get('_structure_label')
        current_start = current_chunk.get('start', 0.0)
        current_end = current_chunk.get('end', 0.0)
        current_duration = current_end - current_start
        
        # 判断是否可以合并
        can_merge = False
        
        if not current_has_lyric:
            # 当前 chunk 是无歌词片段
            if not has_lyric:
                # 下一个也是无歌词，检查结构标签是否相同
                if structure_label == current_label or structure_label is None or current_label is None:
                    can_merge = True
            # 遇到歌词片段或结构标签切换，停止合并无歌词片段
        else:
            # 当前 chunk 是歌词片段
            if has_lyric:
                # 下一个也是歌词片段，检查结构标签和时长
                if structure_label == current_label:
                    # 结构标签相同
                    new_duration = end - current_start
                    if new_duration <= chunk_size:
                        can_merge = True
                    else:
                        # 超过 chunk_size，不合并，但需要检查当前 seg 是否太短
                        seg_duration = end - start
                        if seg_duration < min_chunk_size and current_duration + seg_duration <= chunk_size * 1.5:
                            # 当前 seg 太短，允许略微超过 chunk_size
                            can_merge = True
        
        if can_merge:
            # 合并到当前 chunk
            current_text = current_chunk.get('text', '')
            new_text_content = get_text_without_brackets(text)
            
            # 合并文本：保留当前的结构标签，追加新的歌词内容
            if new_text_content:
                # 如果新 segment 有实际歌词内容，追加
                if current_has_lyric and has_lyric:
                    # 两个都是歌词，合并歌词部分（用换行符隔开）
                    current_text_content = get_text_without_brackets(current_text)
                    bracket_part = keep_only_brackets(current_text)
                    current_chunk['text'] = f"{bracket_part} {current_text_content}\t{new_text_content}".strip()
                else:
                    # 保留当前文本
                    pass
            
            current_chunk['end'] = end
            current_chunk['_merged_count'] = current_chunk.get('_merged_count', 1) + 1
            
            # 如果合并后变成有歌词的，更新状态
            if has_lyric and not current_has_lyric:
                current_chunk['_has_lyric'] = True
                current_chunk['is_lyric'] = True
        else:
            # 不能合并，保存当前 chunk 并开始新的
            # 检查当前 seg 是否太短需要合并到前一个
            seg_duration = end - start
            if has_lyric and merged and seg_duration < min_chunk_size:
                prev_chunk = merged[-1]
                prev_has_lyric = prev_chunk.get('_has_lyric', bool(get_text_without_brackets(prev_chunk.get('text', '')).strip()))
                prev_label = prev_chunk.get('_structure_label', get_structure_label(prev_chunk.get('text', '')))
                
                # 只有当 current_chunk 已经被保存（即 prev_chunk 就是刚保存的 current_chunk 的前一个），
                # 或者 current_chunk 和 prev_chunk 是同一个结构标签时，才考虑合并到 prev
                # 否则应该先保存 current_chunk，再把当前 seg 作为新的 current_chunk
                if prev_has_lyric and structure_label == prev_label:
                    # 检查 current_chunk 是否与 prev_chunk 首尾相接
                    # 如果不是，说明 current_chunk 还没保存，需要先保存
                    current_chunk_start = current_chunk.get('start', 0.0)
                    prev_chunk_end = prev_chunk.get('end', 0.0)
                    
                    if abs(current_chunk_start - prev_chunk_end) > 0.01:
                        # current_chunk 还没保存，先保存它
                        merged.append(current_chunk)
                    
                    # 现在尝试合并到新的 merged[-1]
                    prev_chunk = merged[-1]
                    prev_has_lyric = prev_chunk.get('_has_lyric', bool(get_text_without_brackets(prev_chunk.get('text', '')).strip()))
                    prev_label = prev_chunk.get('_structure_label', get_structure_label(prev_chunk.get('text', '')))
                    
                    if prev_has_lyric and structure_label == prev_label:
                        prev_start = prev_chunk.get('start', 0.0)
                        new_total_duration = end - prev_start
                        
                        # 如果合并后不超过 chunk_size * 1.5，则合并
                        if new_total_duration <= chunk_size * 1.5:
                            prev_text = prev_chunk.get('text', '')
                            new_text_content = get_text_without_brackets(text)
                            if new_text_content:
                                prev_text_content = get_text_without_brackets(prev_text)
                                bracket_part = keep_only_brackets(prev_text)
                                merged[-1]['text'] = f"{bracket_part} {prev_text_content}\t{new_text_content}".strip()
                            merged[-1]['end'] = end
                            merged[-1]['_merged_count'] = merged[-1].get('_merged_count', 1) + 1
                            # 开始新的 current_chunk（下一个 seg 会初始化）
                            current_chunk = None
                            continue
            
            # 保存当前 chunk
            merged.append(current_chunk)
            
            # 开始新的 chunk
            current_chunk = seg.copy()
            current_chunk['_has_lyric'] = has_lyric
            current_chunk['is_lyric'] = has_lyric  # 更新 is_lyric 字段
            current_chunk['_structure_label'] = structure_label
            current_chunk['_merged_count'] = 1
    
    # 保存最后一个 chunk
    if current_chunk is not None:
        merged.append(current_chunk)
    
    # merge_mode=1: 只执行第一遍，清理临时字段后返回
    if merge_mode == 1:
        result = []
        for seg in merged:
            new_seg = seg.copy()
            new_seg.pop('_has_lyric', None)
            new_seg.pop('_structure_label', None)
            merged_count = new_seg.pop('_merged_count', 1)
            new_seg['merged_count'] = merged_count
            # 根据 text 内容判断是否有歌词
            text_content = get_text_without_brackets(new_seg.get('text', ''))
            has_lyric = bool(text_content.strip())
            new_seg['is_lyric'] = has_lyric
            result.append(new_seg)
        return result
    
    # 第二遍：处理时间调整（歌词片段向前延伸 0.5s）
    EXTEND_TIME = 0.5
    final_merged = []
    
    for i, seg in enumerate(merged):
        new_seg = seg.copy()
        
        # 清理临时字段
        new_seg.pop('_has_lyric', None)
        new_seg.pop('_structure_label', None)
        merged_count = new_seg.pop('_merged_count', 1)
        new_seg['merged_count'] = merged_count
        
        # 根据 text 内容判断是否有歌词
        text_content = get_text_without_brackets(new_seg.get('text', ''))
        has_lyric = bool(text_content.strip())
        new_seg['is_lyric'] = has_lyric  # 确保 is_lyric 与实际内容一致
        
        if has_lyric and i > 0:
            # 当前是歌词片段，检查上一个片段是否是无歌词
            prev_seg = final_merged[-1]
            prev_has_lyric = prev_seg.get('is_lyric', True)
            
            if not prev_has_lyric:
                # 上一个片段没有歌词，当前歌词片段开头向前延伸 0.5s
                current_start = new_seg.get('start', 0.0)
                prev_end = prev_seg.get('end', 0.0)
                prev_start = prev_seg.get('start', 0.0)
                
                # 确保不会超过上一个片段的开始时间
                new_start = max(current_start - EXTEND_TIME, prev_start)
                
                if new_start < current_start:
                    new_seg['start'] = new_start
                    new_seg['extended_start'] = True
                    
                    # 上一个非歌词片段的结尾设为歌词片段的新开始时间
                    # 确保 segments 首尾相接
                    new_prev_end = new_start
                    
                    # 如果上一个片段 trim 后时长 <= 0，删除该片段
                    if new_prev_end <= prev_start:
                        # 删除上一个无效片段，当前歌词片段直接从 prev_start 开始
                        new_seg['start'] = prev_start
                        final_merged.pop()
                    else:
                        final_merged[-1]['end'] = new_prev_end
                        final_merged[-1]['trimmed_end'] = True
        
        final_merged.append(new_seg)
    
    # merge_mode=2: 只执行到第二遍，返回 final_merged
    if merge_mode == 2:
        return final_merged
    
    # 第三遍：处理两个歌词片段之间小于2s的非歌词片段，从中间分开merge进前后歌词片段
    SHORT_NON_LYRIC_THRESHOLD = 2.0
    result_merged = []
    
    for i, seg in enumerate(final_merged):
        seg_start = seg.get('start', 0.0)
        seg_end = seg.get('end', 0.0)
        seg_duration = seg_end - seg_start
        has_lyric = seg.get('is_lyric', True)
        
        # 跳过无效片段（start >= end，可能是第二遍 trim 导致的）
        if seg_duration <= 0:
            continue
        
        if not has_lyric and seg_duration < SHORT_NON_LYRIC_THRESHOLD:
            # 当前是小于2s的非歌词片段
            # 检查前后是否都是歌词片段
            prev_is_lyric = result_merged[-1].get('is_lyric', False) if result_merged else False
            next_is_lyric = final_merged[i+1].get('is_lyric', False) if i + 1 < len(final_merged) else False
            
            if prev_is_lyric and next_is_lyric:
                # 前后都是歌词片段，从中间分开，当前非歌词片段将被移除
                mid_point = (seg_start + seg_end) / 2
                
                # 上一个歌词片段的 end 延伸到 mid_point
                result_merged[-1]['end'] = mid_point
                result_merged[-1]['merged_next_non_lyric'] = True
                
                # 下一个歌词片段的 start 改为 mid_point
                # 注意：需要复制一份再修改，避免影响 final_merged 中后续的判断
                final_merged[i+1] = final_merged[i+1].copy()
                final_merged[i+1]['start'] = mid_point
                final_merged[i+1]['extended_start'] = True
                
                # 当前非歌词片段被合并到前后歌词片段中，不添加到结果（相当于删除）
                continue
        
        # 复制一份添加到结果，避免后续修改影响
        result_merged.append(seg.copy())
    
    # 重新计算统计字段
    for seg in result_merged:
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        duration = end - start
        
        text = seg.get('text', '')
        lang_type = seg.get('lang_type')
        
        if lang_type == 'western':
            lyric_count = count_words(text)
            seg['lyric_count'] = lyric_count
            if lyric_count > 0 and duration > 0:
                seg['average_duration'] = duration / lyric_count
        elif lang_type == 'cjk':
            lyric_count = count_characters(text)
            seg['lyric_count'] = lyric_count
            if lyric_count > 0 and duration > 0:
                seg['average_duration'] = duration / lyric_count
    
    return result_merged


def merge_json_data(data: Dict[str, Any], chunk_size: float = 10.0, json_path: str = None, merge_mode: int = 3) -> Dict[str, Any]:
    """
    对 JSON 数据进行 merge 处理
    
    Args:
        data: 输入的 JSON 数据
        chunk_size: merge 的最大可接受长度（秒）
        json_path: JSON 文件路径（用于 warning 输出）
        merge_mode: merge 模式，1=只第一遍，2=到第二遍，3=完整三遍（默认）
    
    Returns:
        merge 后的 JSON 数据
    """
    segments = data.get('segments', [])
    audio_length = data.get('audio_length', float('inf'))
    
    # 执行 merge
    merged_segments = merge_segments(segments, chunk_size=chunk_size, merge_mode=merge_mode)
    
    # 检查 segment 首尾相接
    has_gap = False
    for i in range(1, len(merged_segments)):
        prev_end = merged_segments[i-1].get('end', 0.0)
        curr_start = merged_segments[i].get('start', 0.0)
        # 允许小于 0.01 的误差
        if abs(curr_start - prev_end) > 0.01:
            has_gap = True
            if json_path:
                print(f"[WARNING] Segment gap detected: seg[{i-1}].end={prev_end:.3f} != seg[{i}].start={curr_start:.3f}, path: {json_path}")
            break
    
    # 检查最后一个 segment 的 end time
    if merged_segments:
        last_end = merged_segments[-1].get('end', 0.0)
        if last_end > audio_length + 0.01:  # 允许小误差
            if json_path:
                print(f"[WARNING] Last segment end ({last_end:.3f}) > audio_length ({audio_length:.3f}), path: {json_path}")
    
    # 创建新的数据
    new_data = data.copy()
    new_data['segments'] = merged_segments
    new_data['merged'] = True
    new_data['merge_chunk_size'] = chunk_size
    new_data['original_segment_count'] = len(segments)
    new_data['merged_segment_count'] = len(merged_segments)
    
    return new_data


def process_merge_from_scp(scp_path: str, output_dir: str, chunk_size: float = 10.0, merge_mode: int = 3):
    """
    从 scp 文件读取 JSON 文件路径，进行 merge 处理并保存
    
    Args:
        scp_path: scp 文件路径（每行格式：path\tseg_num\till_cnt\tratio）
        output_dir: 输出目录
        chunk_size: merge 的最大可接受长度（秒）
        merge_mode: merge 模式，1=只第一遍，2=到第二遍，3=完整三遍（默认）
    """
    scp_path = Path(scp_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取 scp 文件
    json_paths = []
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                json_paths.append(parts[0])
    
    print(f"Found {len(json_paths)} JSON files to merge")
    print(f"Chunk size: {chunk_size}s")
    print(f"Merge mode: {merge_mode}")
    print(f"Output dir: {output_dir}")
    
    merged_stats = []
    
    for json_path in tqdm(json_paths, desc="Merging"):
        if not Path(json_path).exists():
            print(f"File not found: {json_path}")
            continue
        
        try:
            # 读取 JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 执行 merge
            merged_data = merge_json_data(data, chunk_size=chunk_size, json_path=json_path, merge_mode=merge_mode)
            
            # 构建输出路径：XXX.merged.CHUNKSIZE.json
            json_path_obj = Path(json_path)
            # 去掉 .json 后缀，添加 .merged.CHUNKSIZE.json
            base_name = json_path_obj.stem  # 不含 .json 的文件名
            new_name = f"{base_name}.merged.{int(chunk_size)}s.json"
            
            # 保留目录结构
            parts = json_path_obj.parts
            if len(parts) >= 2:
                relative_dir = Path(parts[-2])
                output_path = output_dir / relative_dir / new_name
            else:
                output_path = output_dir / new_name
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            merged_stats.append({
                'input_path': json_path,
                'output_path': str(output_path),
                'original_count': merged_data.get('original_segment_count', 0),
                'merged_count': merged_data.get('merged_segment_count', 0)
            })
            
        except Exception as e:
            print(f"Error merging {json_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存 merged.scp（只保留 JSON 路径）
    merged_scp_path = output_dir / f"merged.{int(chunk_size)}s.scp"
    with open(merged_scp_path, 'w', encoding='utf-8') as f:
        for s in merged_stats:
            f.write(f"{s['output_path']}\n")
    
    print(f"\nMerge 完成: {len(merged_stats)} 个文件")
    print(f"Merged scp 已保存: {merged_scp_path}")
    
    # 统计
    total_original = sum(s['original_count'] for s in merged_stats)
    total_merged = sum(s['merged_count'] for s in merged_stats)
    print(f"总 segment 数: {total_original} -> {total_merged} (压缩率: {total_merged/total_original*100:.1f}%)")
    
    return merged_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="处理歌词 JSON 文件")
    parser.add_argument("--scp", type=str, 
                        default="/home/jianwei/music/luoxue/json_group_lyric_gp5_v1/json_group_lyric_gp5_v1_local.scp",
                        help="scp 文件路径")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/jianweiyu/exp/music/luoxue/json_group_lyric_gp5_v1_processed_v2_resetendtime",
                        help="输出目录（保留倒数两级目录结构）")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    parser.add_argument("--debug", action="store_true", help="Debug 模式，只处理 10 个文件")
    
    # Merge 相关参数
    parser.add_argument("--merge", action="store_true", help="执行 merge 操作")
    parser.add_argument("--merge_scp", type=str, 
                        default=None,
                        help="merge 操作的输入 scp 文件路径（默认使用 output_dir/filtered.scp）")
    parser.add_argument("--merge_output_dir", type=str,
                        default=None,
                        help="merge 输出目录（默认使用 output_dir）")
    parser.add_argument("--chunk_size", type=float, default=10.0,
                        help="merge 的最大 chunk 长度（秒），默认 10s")
    parser.add_argument("--merge_mode", type=int, default=3, choices=[1, 2, 3],
                        help="merge 模式: 1=只第一遍, 2=到第二遍, 3=完整三遍(默认)")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.merge:
        # 执行 merge 操作
        output_base_dir = Path(args.output_dir)
        
        # 确定 merge 的输入 scp
        if args.merge_scp:
            merge_scp = args.merge_scp
        else:
            merge_scp = output_base_dir / "filtered.scp"
        
        # 确定 merge 的输出目录
        if args.merge_output_dir:
            merge_output_dir = args.merge_output_dir
        else:
            merge_output_dir = output_base_dir
        
        print(f"Merge 模式")
        print(f"输入 scp: {merge_scp}")
        print(f"输出目录: {merge_output_dir}")
        print(f"Chunk size: {args.chunk_size}s")
        print(f"Merge mode: {args.merge_mode}")
        
        process_merge_from_scp(str(merge_scp), str(merge_output_dir), chunk_size=args.chunk_size, merge_mode=args.merge_mode)
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
        reset_list = []  # 有重置的文件
        ori_list = []    # 没有重置的文件
        
        for json_path in tqdm(json_paths, desc="Processing"):
            if not Path(json_path).exists():
                print(f"File not found: {json_path}")
                continue
            
            try:
                # 读取 JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理数据
                processed_data, stats, has_reset = process_json_data(data)
                
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
                
                # 检查是否存在 is_duration_ill=True 且未被 reset 的 segment
                has_unresolved_ill = False
                for seg in processed_data.get('segments', []):
                    if seg.get('is_duration_ill', False) and not seg.get('reset_end_time', False):
                        has_unresolved_ill = True
                        break\
                
                all_stats.append({
                    'json_path': json_path,
                    'output_path': str(output_path),
                    'duration_stats': stats,
                    'has_reset': has_reset,
                    'has_unresolved_ill': has_unresolved_ill
                })
                
                # 根据是否有重置分类
                if has_reset:
                    reset_list.append({
                        'output_path': str(output_path),
                        'duration_stats': stats
                    })
                else:
                    ori_list.append({
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
        
        # 保存 processed_reset.scp（有重置的文件）
        reset_scp_path = output_base_dir / "processed_reset.scp"
        with open(reset_scp_path, 'w', encoding='utf-8') as f:
            for s in reset_list:
                seg_num = s['duration_stats']['total_seg_num']
                ill_cnt = s['duration_stats']['is_duration_ill_cnt']
                ratio = s['duration_stats']['ill_duration_seg_ratio']
                f.write(f"{s['output_path']}\t{seg_num}\t{ill_cnt}\t{ratio:.4f}\n")
        print(f"重置文件 scp 已保存: {reset_scp_path} ({len(reset_list)} 个文件)")
        
        # 保存 processed_ori.scp（没有重置的文件）
        ori_scp_path = output_base_dir / "processed_ori.scp"
        with open(ori_scp_path, 'w', encoding='utf-8') as f:
            for s in ori_list:
                seg_num = s['duration_stats']['total_seg_num']
                ill_cnt = s['duration_stats']['is_duration_ill_cnt']
                ratio = s['duration_stats']['ill_duration_seg_ratio']
                f.write(f"{s['output_path']}\t{seg_num}\t{ill_cnt}\t{ratio:.4f}\n")
        print(f"原始文件 scp 已保存: {ori_scp_path} ({len(ori_list)} 个文件)")
        
        # 定义是否通过过滤的函数
        def is_filtered_pass(s):
            """
            通过过滤的条件：
            1. segment数量 > 20
            2. 且满足以下任一条件：
               a) 不存在未解决的 is_duration_ill
               b) 问题seg数量 <= 2 且 占比 < 10%
            """
            seg_num = s['duration_stats']['total_seg_num']
            ill_cnt = s['duration_stats']['is_duration_ill_cnt']
            ratio = s['duration_stats']['ill_duration_seg_ratio']
            has_unresolved = s.get('has_unresolved_ill', False)
            
            if seg_num <= 15:
                return False
            
            # 条件 a: 没有未解决的 ill
            if not has_unresolved:
                return True
            
            # 条件 b: 问题seg数量 <= 2 且占比 < 10%
            if ill_cnt <= 2 and ratio < 0.1:
                return True
            
            return False
        
        # 保存 filtered.scp
        filtered_list = [s for s in all_stats if is_filtered_pass(s)]
        filtered_scp_path = output_base_dir / "filtered.scp"
        with open(filtered_scp_path, 'w', encoding='utf-8') as f:
            for s in filtered_list:
                seg_num = s['duration_stats']['total_seg_num']
                ill_cnt = s['duration_stats']['is_duration_ill_cnt']
                ratio = s['duration_stats']['ill_duration_seg_ratio']
                f.write(f"{s['output_path']}\t{seg_num}\t{ill_cnt}\t{ratio:.4f}\n")
        print(f"过滤后 scp 已保存: {filtered_scp_path} ({len(filtered_list)} 个文件)")
        
        # 保存 filtered_out.scp（不符合过滤条件的文件）
        filtered_out_list = [s for s in all_stats if not is_filtered_pass(s)]
        filtered_out_scp_path = output_base_dir / "filtered_out.scp"
        with open(filtered_out_scp_path, 'w', encoding='utf-8') as f:
            for s in filtered_out_list:
                seg_num = s['duration_stats']['total_seg_num']
                ill_cnt = s['duration_stats']['is_duration_ill_cnt']
                ratio = s['duration_stats']['ill_duration_seg_ratio']
                f.write(f"{s['output_path']}\t{seg_num}\t{ill_cnt}\t{ratio:.4f}\n")
        print(f"过滤掉的 scp 已保存: {filtered_out_scp_path} ({len(filtered_out_list)} 个文件)")
        
        # # 收集所有带歌词 segment 的时长并绘制直方图
        # print("\n" + "=" * 50)
        # print("收集带歌词 segment 时长统计...")
        # output_json_paths = [s['output_path'] for s in all_stats]
        # bin_counts, dur_stats = collect_lyric_segment_durations(output_json_paths, bin_size=1.0, max_duration=30.0)
        
        # print(f"\n带歌词 Segment 时长统计:")
        # print(f"  总数量: {dur_stats['total_segments']}")
        # print(f"  平均时长: {dur_stats['mean_duration']:.3f}s")
        # print(f"  最小时长: {dur_stats['min_duration']:.3f}s")
        # print(f"  最大时长: {dur_stats['max_duration']:.3f}s")
        # print(f"  总时长: {dur_stats['total_duration']:.2f}s ({dur_stats['total_duration']/3600:.2f}h)")
        
        # # 绘制直方图
        # histogram_path = output_base_dir / "segment_duration_histogram.png"
        # plot_duration_histogram(bin_counts, dur_stats, str(histogram_path), bin_size=1.0)