"""
歌词判断处理器
处理音乐文件的segments，判断每个segment是否为歌词内容
针对原始 .lrc文件的
"""

import json
import re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from azure_api import TaskProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_lrc import parse_lrc_with_timestamps

def is_special_text(text: str) -> bool:
    """
    判断text是否为纯特殊标记（如[intro], [music], [verse]等，没有实际歌词）
    返回True表示是纯标记
    """
    if not text:
        return True
    text_stripped = text.strip()
    if not text_stripped:
        return True
    # 移除所有 [xxx] 标记后，检查是否还有其他内容
    text_without_tags = re.sub(r'\[[^\]]+\]', '', text_stripped)
    return len(text_without_tags.strip()) == 0


class LyricProcessor(TaskProcessor):
    """
    歌词判断处理器
    
    输入: List[Dict] - segments列表，每个包含 text, start, end, speaker
    输出: List[Dict] - 添加 is_lyric 字段后的segments列表
    """
    
    def __init__(
        self,
        language: str = "zh",
        prompt_language: str = "zh",
        strict_check: bool = True,
    ):
        self.language = language
        self.prompt_language = prompt_language
        self.strict_check = strict_check
        self.last_error_feedback = None
    
    def prepare_input(self, raw_data: Any) -> Dict[str, Any]:
        """
        准备输入数据
        raw_data: {
            "batch_id": int,
            "segments": List[Dict] - 原始segments (包含text, start, end, speaker等)
        }
        """
        segments = raw_data.get("segments", [])
        
        # 提取用于判断的简化信息
        simplified_segments = []
        for i, seg in enumerate(segments):
            simplified_segments.append({
                "index": i,
                "text": seg.get("text", ""),
                "start": seg.get("start", 0),
                # "end": seg.get("end", 0),
            })
        
        return {
            "batch_id": raw_data.get("batch_id", 0),
            "original_segments": segments,
            "simplified_segments": simplified_segments,
        }
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建提示词"""
        segments = input_data["simplified_segments"]
        segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
        
        prompt = f"""请判断以下音乐文件的segments中，每个segment的text是否是"歌词"内容。

【判断例子】
非歌词 (is_lyric=false)：
- "词：五条悟" → 作词信息，非歌词
- "曲 Composer：刘迦宁 Liu Jianing" → 作曲信息，非歌词
- "人声录音室 Vocal Recording Studio：未来•福录音室" → 录音室信息，非歌词
- "制作人 Producer：关诗敏" → 制作人信息，非歌词
- "All instruments by Chris Wahle" → 乐器演奏信息，非歌词
- "作词：胖小迪Yan" → 作词信息，非歌词
- "作曲：漠规" → 作曲信息，非歌词
- "编曲：萧冷" → 编曲信息，非歌词
- "混音：xxx" → 混音信息，非歌词
- "母带：xxx" → 母带信息，非歌词
- "吉他：xxx / Guitar: xxx" → 乐手信息，非歌词
- "男：“ → 演唱者，非歌词
- "韩红：“ → 演唱者，非歌词

歌词 (is_lyric=true)：
- "喜欢你 仿佛坠入无边海底" → 实际演唱歌词
- "我想要去无人的岛要在海尽头垂钓" → 实际演唱歌词
- " 我扔掉手表走向无人的海岛" → 实际演唱歌词
- "让我们一起飞翔" → 实际演唱歌词

【重要警告】
1. text字段必须完全保持原样，一个字符都不能修改！
2. 只需要添加 is_lyric 字段（true 或 false）
3. 验证时会检查 text 是否完全相同

【输出格式】
直接输出JSON数组，每个元素包含：
- index: 原始索引（不变）
- text: 原始文本（必须完全不变）
- is_lyric: true 或 false

输入数据：
{segments_json}
"""
        
        if self.last_error_feedback:
            prompt += f"""
【上次错误】你之前的输出有问题：
{self.last_error_feedback}
请确保text字段完全不变！
"""
            self.last_error_feedback = None
        
        prompt += "\n请直接输出JSON数组："
        
        return prompt
    
    def check_output(self, output: str, input_data: Dict[str, Any]) -> bool:
        """验证输出：text必须完全不变，必须有is_lyric字段"""
        try:
            output = output.strip()
            if output.startswith("```"):
                lines = output.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or not line.startswith("```"):
                        json_lines.append(line)
                output = "\n".join(json_lines)
            
            parsed_output = json.loads(output)
            
            if not isinstance(parsed_output, list):
                print(f"Output is not a list")
                return False
            
            original_segments = input_data["simplified_segments"]
            
            if len(parsed_output) != len(original_segments):
                print(f"Segment count mismatch: {len(parsed_output)} vs {len(original_segments)}")
                return False
            
            for i, (orig, new) in enumerate(zip(original_segments, parsed_output)):
                orig_text = orig.get("text", "")
                new_text = new.get("text", "")
                
                # text 必须完全相同
                if orig_text != new_text:
                    print(f"Segment {i} text mismatch:")
                    print(f"  Original: {orig_text}")
                    print(f"  New: {new_text}")
                    
                    self.last_error_feedback = f"""Segment {i} text被修改！
原文: "{orig_text}"
你的输出: "{new_text}"
text必须完全保持不变！"""
                    
                    if self.strict_check:
                        return False
                
                # 必须有 is_lyric 字段
                if "is_lyric" not in new:
                    print(f"Segment {i} missing is_lyric field")
                    self.last_error_feedback = f"Segment {i} 缺少 is_lyric 字段！"
                    return False
                
                # is_lyric 必须是 bool
                if not isinstance(new.get("is_lyric"), bool):
                    print(f"Segment {i} is_lyric is not boolean: {new.get('is_lyric')}")
                    self.last_error_feedback = f"Segment {i} is_lyric 必须是 true 或 false！"
                    return False
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse output JSON: {e}")
            print(f"Output: {output[:500]}...")
            return False
        except Exception as e:
            print(f"Check error: {e}")
            return False
    
    def process_output(self, output: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输出，将is_lyric添加到原始segments"""
        output = output.strip()
        if output.startswith("```"):
            lines = output.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not line.startswith("```"):
                    json_lines.append(line)
            output = "\n".join(json_lines)
        
        parsed_output = json.loads(output)
        
        original_segments = input_data["original_segments"]
        updated_segments = []
        
        for orig_seg, new_seg in zip(original_segments, parsed_output):
            updated_seg = orig_seg.copy()
            updated_seg["is_lyric"] = new_seg.get("is_lyric", True)
            updated_segments.append(updated_seg)
        
        return {
            "batch_id": input_data["batch_id"],
            "segments": updated_segments,
        }
    
    def get_system_prompt(self) -> str:
        return """你是一个专业的音乐内容分析助手，擅长区分音乐文件中的实际歌词和制作信息。

【核心原则】
- 绝对不能修改text内容
- 你的唯一任务是判断每个segment是歌词(is_lyric: true)还是非歌词(is_lyric: false)
- 制作信息、版权信息、人员信息等是非歌词
- 实际演唱内容和音乐段落标记是歌词"""


def split_segments_by_duration(
    segments: List[Dict],
    max_duration_seconds: float = 600.0,
) -> List[List[Dict]]:
    """
    按时间长度分割segments为多个batch
    
    Args:
        segments: 原始segments列表
        max_duration_seconds: 每个batch的最大时长（秒）
    
    Returns:
        分割后的batches列表
    """
    if not segments:
        return []
    
    batches = []
    current_batch = []
    current_batch_start = None
    
    for seg in segments:
        seg_start = seg.get("start", seg.get("start_time", 0))
        
        if current_batch_start is None:
            current_batch_start = seg_start
        
        # 计算如果加入这个segment后的总时长
        potential_duration = seg_start - current_batch_start
        
        if potential_duration > max_duration_seconds and current_batch:
            # 超过限制，先保存当前batch，开始新batch
            batches.append(current_batch)
            current_batch = []
            current_batch_start = seg_start
        
        current_batch.append(seg)
    
    # 添加最后一个batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


def process_lrc_file(lrc_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    处理单个JSON文件，返回处理后的数据
    
    Args:
        json_path: 输入LRC文件路径
        output_path: 输出JSON文件路径（可选）
    
    Returns:
        处理后的JSON数据
    """
    lrc_path = Path(lrc_path)
    
    
    segments = parse_lrc_with_timestamps(lrc_path)
    
    # 按时长分割
    batches, err_load_segs = split_segments_by_duration(segments, max_duration_seconds=600.0)
    
    print(f"Split {len(segments)} segments into {len(batches)} batches")
    
    batch_data_list = []
    for i, batch_segments in enumerate(batches):
        batch_data_list.append({
            "batch_id": i,
            "segments": batch_segments,
        })
    
    return {
        "lrc_path": str(lrc_path),
        "batches": batch_data_list,
        "num_batches": len(batches),
    }


if __name__ == "__main__":
    # 测试
    test_segments = [
        {"text": "我扔掉手表走向无人的海岛", "start": 24.18, "end": 29.97, "speaker": None},
        {"text": "词：五条悟", "start": 0, "end": 5, "speaker": None},
        {"text": "曲 Composer：刘迦宁", "start": 0, "end": 5, "speaker": None},
        {"text": "制作人 Producer：关诗敏", "start": 100, "end": 105, "speaker": None},
        {"text": "All instruments by Chris Wahle", "start": 50, "end": 55, "speaker": None},
        {"text": "喜欢你 仿佛坠入无边海底", "start": 30, "end": 35, "speaker": None},
        {"text": "作词：胖小迪Yan", "start": 0, "end": 3, "speaker": None},
        {"text": "作曲：漠规", "start": 3, "end": 6, "speaker": None},
        {"text": "编曲：萧冷", "start": 6, "end": 9, "speaker": None},
    ]
    
    processor = LyricProcessor()
    
    # 测试prepare_input
    input_data = processor.prepare_input({"batch_id": 0, "segments": test_segments})
    print("Input data:", json.dumps(input_data["simplified_segments"], ensure_ascii=False, indent=2))
    
    # 测试build_prompt
    prompt = processor.build_prompt(input_data)
    print("\nPrompt:")
    print(prompt)
