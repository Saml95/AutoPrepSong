"""
数字格式转换处理器
处理ASR输出的segments，将中文数字转换为易读的阿拉伯数字形式
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from azure_api import TaskProcessor


def remove_punctuation(text: str) -> str:
    """移除所有标点符号，只保留文字内容"""
    result = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith('L') or cat.startswith('N') or cat.startswith('M'):
            result.append(char)
    return ''.join(result)


def normalize_text(text: str) -> str:
    """标准化文本用于比较"""
    text = remove_punctuation(text)
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    return text


class NumberProcessor(TaskProcessor):
    """
    数字格式转换处理器
    
    将中文数字转换为阿拉伯数字的易读形式，例如：
    - 一九九四年十月 → 1994年10月
    - 三块六毛八 → 3块6毛8
    - 十五分之七 → 7/15
    - 百分之八十五 → 85%
    - 一千两百三十四 → 1234
    
    输入: List[Dict] - segments列表，每个包含 text, speaker
    输出: List[Dict] - 转换后的segments列表
    """
    
    def __init__(
        self,
        language: str = "zh",
        prompt_language: str = "zh",  # "zh" or "en" for prompt language
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
            "segments": List[Dict] - 原始segments (包含text, speaker, start, end等)
        }
        """
        segments = raw_data.get("segments", [])
        
        # 提取text和speaker用于处理
        simplified_segments = []
        for seg in segments:
            simplified_segments.append({
                "text": seg.get("text", ""),
                "speaker": seg.get("speaker", "unknown"),
            })
        
        return {
            "batch_id": raw_data.get("batch_id", 0),
            "original_segments": segments,
            "simplified_segments": simplified_segments,
        }
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建提示词，根据prompt_language选择中文或英文"""
        if self.prompt_language == "en":
            return self.build_prompt_en(input_data)
        else:
            return self.build_prompt_zh(input_data)
    
    def build_prompt_zh(self, input_data: Dict[str, Any]) -> str:
        """构建中文提示词"""
        segments = input_data["simplified_segments"]
        segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
        
        prompt = f"""请将以下语音识别系统识别的文本中的数字转化为更为易读符合日常表达的格式，注意只允许修改数字的格式，不得改变内容，不修改和数字无关的部分

【转换规则】
1. 年份：一九九四年 → 1994年，二零二三年 → 2023年
2. 月日：十月 → 10月，二十三号 → 23号，三月十五日 → 3月15日
3. 时间：三点半 → 3点半，十二点四十五分 → 12点45分
4. 货币：三块六毛八 → 3块6毛8，一百二十元 → 120元，五毛钱 → 5毛钱
5. 分数：十五分之七 → 7/15，三分之一 → 1/3，四分之三 → 3/4
6. 百分比：百分之八十五 → 85%，百分之五点三 → 5.3%
7. 序数/编号：第一 → 第1，第二十三 → 第23
8. 普通数字：一千两百三十四 → 1234，三万五千 → 35000，两亿 → 2亿
9. 小数：三点一四 → 3.14，零点五 → 0.5
10. 电话号码：一三八零零零零零零零零 → 13800000000
11. 倍数/比例：两倍 → 2倍，三比一 → 3:1
12. 英文数字词：GPT-Five → GPT5，iPhone Fifteen → iPhone15，Windows Eleven → Windows11，version two → version2

【不需要转换的情况】
- 表达情感、夸张、修辞的数字：我爱你三千年、一万个理由、三生三世、九死一生、一心一意
- 成语/固定表达：三心二意、七上八下、五颜六色、一模一样、千方百计
- 诗词/文学表达：三人行必有我师、一日不见如隔三秋
- 约定俗成的表达：二话不说、三言两语
- 一些常用的中文表达，如 今年上大一，小学一年级，一个游戏平均一局下来，这两种等

【重要规则】
- 只转换数字相关的内容，其他文字保持不变
- 保持语义不变，只改变数字的表示形式
- 量词和单位保留中文形式（如：个、块、毛、年、月、日等）
- 如果数字已经是阿拉伯数字形式，保持不变
- speaker字段必须保持不变
- 不要添加或删除任何非数字相关的文字

【示例】
输入: "我一九九四年十月出生的"
输出: "我1994年10月出生的"

输入: "这个要三块六毛八"
输出: "这个要3块6毛8"

输入: "十五分之七加四分之一等于多少"
输出: "7/15加1/4等于多少"

输入: "百分之八十五的人同意"
输出: "85%的人同意"

输入数据：
{segments_json}
"""
        
        if self.last_error_feedback:
            prompt += f"""
【上次错误】你之前的输出有问题：
{self.last_error_feedback}
请仔细检查并修正！
"""
            self.last_error_feedback = None
        
        prompt += "\n请直接输出修改后的JSON数组，不要有任何其他说明："
        
        return prompt
    
    def build_prompt_en(self, input_data: Dict[str, Any]) -> str:
        """构建英文提示词"""
        segments = input_data["simplified_segments"]
        segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
        
        prompt = f"""You are given a JSON array of dialogue segments from a ASR system. Your task is to normalize **English transcription text** by converting **spoken/written-out numbers** into **readable Arabic numerals**, while keeping everything else unchanged.

Return the **same JSON array structure** with only the necessary number normalization applied.

[What to Convert]
Convert English number words and numeric phrases to Arabic numerals, including:

1) Years & decades
- "nineteen ninety four" → "1994"
- "two thousand and six" → "2006"
- "the 1980s" / "nineteen eighties" → "the 1980s"
- "twenty twenty-one" / "two thousand twenty-one" → "2021" (choose the most natural year form)
- "in oh eight" (year context) → "in 2008" when clearly a year; otherwise leave unchanged

2) Ordinals (ranking / sequence)
- "first" → "1st", "second" → "2nd", "twenty third" → "23rd"
- "the first one" → "the 1st one"
- Keep suffixes: st/nd/rd/th

3) Cardinal numbers (general counting)
- "zero"→"0", "ten"→"10", "twenty one"→"21", "one hundred"→"100"
- "one hundred and five" → "105"
- "two thousand five hundred" → "2500"
- "a hundred" → "100"; "a thousand" → "1000" (only if clearly numeric)

4) Decimals
- "three point one four" → "3.14"
- "point five" → "0.5"
- "zero point zero one" → "0.01"

5) Fractions
- "one half" → "1/2"
- "three quarters" → "3/4"
- "two thirds" → "2/3"
- "seven over fifteen" → "7/15" (when clearly a fraction)

6) Percentages
- "eighty five percent" → "85%"
- "five point three percent" → "5.3%"

7) Money (common spoken forms)
- "ten dollars" → "$10"
- "ten bucks" → "$10" (keep "bucks" if you prefer; choose ONE style consistently)
- "five dollars and fifty cents" → "$5.50"
- "a hundred and twenty dollars" → "$120"
- "ten ninety nine" (price context) → "$10.99"

8) Time & dates (only when unambiguous)
- "three thirty" (time context) → "3:30"
- "three thirty p m" / "three thirty pm" → "3:30 pm"
- "twelve oh five" (time context) → "12:05"
- "January fifth" → "January 5th"
- "May twenty third" → "May 23rd"

9) Phone numbers / IDs / digit sequences
When the phrase is clearly a digit-by-digit sequence (phone numbers, codes, IDs), convert each spoken digit:
- "one three eight zero zero zero zero" → "1380000"
- "double three" → "33", "triple seven" → "777"
Keep separators if present/obvious:
- "one two three dash four five six" → "123-456"

10) Versions / model names / product numbers (common ASR patterns)
- "g p t five" / "gpt five" → "GPT-5" or "GPT5" (choose ONE style consistently)
- "iphone fifteen" → "iPhone 15"
- "windows eleven" → "Windows 11"
- "version two point one" → "version 2.1"

[Do NOT Convert]
Do NOT convert when the number words are clearly NOT numeric content:
- Idioms/metaphors: "at sixes and sevens", "a million times", "one in a million" (unless it's clearly literal)
- Fixed phrases or titles where conversion would look unnatural
- When conversion is ambiguous and may change meaning (prefer leaving it unchanged)

[Hard Constraints]
- Only change number-related parts; do not rewrite wording, grammar, punctuation, casing, or spacing outside the converted spans.
- Preserve meaning exactly.
- Do not add/remove fields; do not change key names.
- Do not modify the speaker field.
- If something is already in Arabic numerals, keep it unchanged.
- Output must be valid JSON.

[Examples]
- "I was born in nineteen eighties." → "I was born in 1980s."
- "I moved here in two thousand and six." → "I moved here in 2006."
- "It happened on May twenty third at three thirty pm." → "It happened on May 23rd at 3:30 pm."
- "The success rate is eighty five percent." → "The success rate is 85%."
- "Set version two point one." → "Set version 2.1."
- "Call me at one two three dash four five six dash seven eight nine zero." → "Call me at 123-456-7890."

[Input JSON]
{segments_json}
"""
        
        if self.last_error_feedback:
            prompt += f"""
[Previous Error] Your previous output had issues:
{self.last_error_feedback}
Please check carefully and correct!
"""
            self.last_error_feedback = None
        
        prompt += "\nReturn ONLY the modified JSON array. Do not include any explanation or extra text:"
        
        return prompt
    
    def _parse_output_json(self, output: str) -> Optional[List[Dict]]:
        """解析输出JSON，处理markdown代码块"""
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
        
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return None
    
    def check_output(self, output: str, input_data: Dict[str, Any]) -> bool:
        """验证输出格式是否正确"""
        try:
            parsed_output = self._parse_output_json(output)
            
            if parsed_output is None:
                print(f"Failed to parse output JSON")
                print(f"Output: {output[:500]}...")
                return False
            
            if not isinstance(parsed_output, list):
                print(f"Output is not a list")
                return False
            
            original_segments = input_data["simplified_segments"]
            
            if len(parsed_output) != len(original_segments):
                print(f"Segment count mismatch: {len(parsed_output)} vs {len(original_segments)}")
                return False
            
            for i, (orig, new) in enumerate(zip(original_segments, parsed_output)):
                # 检查是否有text字段
                if "text" not in new:
                    print(f"Segment {i} missing 'text' field")
                    return False
                
                # 检查speaker是否一致
                if orig.get("speaker") != new.get("speaker"):
                    print(f"Segment {i} speaker mismatch: {orig.get('speaker')} vs {new.get('speaker')}")
                    new["speaker"] = orig.get("speaker")
                
                # 基本长度检查：转换后的文本不应该比原文短太多或长太多
                orig_len = len(orig.get("text", ""))
                new_len = len(new.get("text", ""))
                if orig_len > 0:
                    ratio = new_len / orig_len
                    if ratio < 0.3 or ratio > 3.0:
                        print(f"Segment {i} length ratio abnormal: {ratio:.2f}")
                        print(f"  Original ({orig_len}): {orig.get('text', '')}")
                        print(f"  New ({new_len}): {new.get('text', '')}")
                        self.last_error_feedback = f"""Segment {i} 转换后长度异常！
原文: "{orig.get('text', '')}"
你的输出: "{new.get('text', '')}"
请确保只转换数字，不要删除或添加其他内容！"""
                        if self.strict_check:
                            return False
            
            return True
            
        except Exception as e:
            print(f"Check error: {e}")
            return False
    
    def process_output(self, output: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输出，将转换后的文本更新到原始segments，并标记是否修改"""
        parsed_output = self._parse_output_json(output)
        
        original_segments = input_data["original_segments"]
        simplified_segments = input_data["simplified_segments"]
        updated_segments = []
        batch_is_modified = False
        
        for orig_seg, simplified_seg, new_seg in zip(original_segments, simplified_segments, parsed_output):
            updated_seg = orig_seg.copy()
            new_text = new_seg.get("text", orig_seg.get("text", ""))
            old_text = simplified_seg.get("text", "")
            
            # 检查文本是否被修改
            is_modified = (new_text != old_text)
            if is_modified:
                batch_is_modified = True
                # 保留修改前的原始文本
                updated_seg["text_ori"] = old_text
            
            updated_seg["text"] = new_text
            updated_seg["is_modified_number"] = is_modified
            updated_segments.append(updated_seg)
        
        return {
            "batch_id": input_data["batch_id"],
            "segments": updated_segments,
            "is_modified_number": batch_is_modified,  # 整个batch是否有任何修改
        }
    
    def get_system_prompt(self) -> str:
        if self.prompt_language == "en":
            return """You are a professional text processing assistant, skilled at converting Chinese numbers to readable Arabic numeral format.

[Core Principles]
- Only convert number-related content, keep other text unchanged
- Ensure the converted meaning is exactly the same as the original
- Keep measure words and units"""
        else:
            return """你是一个专业的中文文本处理助手，擅长将中文数字转换为阿拉伯数字的易读形式。

【核心原则】
- 只转换数字相关的内容，保持其他文字不变
- 确保转换后的语义与原文完全一致
- 保留中文量词和单位"""


def split_segments_by_duration(
    segments: List[Dict],
    max_duration_seconds: float = 120.0,
) -> List[List[Dict]]:
    """
    按时间长度分割segments为多个batch
    
    规则：
    1. 必须在speaker切换的位置进行分割
    2. 每个batch的时长尽量接近但不超过max_duration_seconds
    3. 最后一个batch如果不足max_duration_seconds的一半，则合并到上一个batch
    
    Args:
        segments: 原始segments列表
        max_duration_seconds: 每个batch的最大时长（秒）
    
    Returns:
        分割后的batches列表
    """
    if not segments:
        return []
    
    speaker_change_indices = [0]
    for i in range(1, len(segments)):
        prev_speaker = segments[i-1].get("speaker", "")
        curr_speaker = segments[i].get("speaker", "")
        if prev_speaker != curr_speaker:
            speaker_change_indices.append(i)
    speaker_change_indices.append(len(segments))
    
    speaker_groups = []
    for i in range(len(speaker_change_indices) - 1):
        start_idx = speaker_change_indices[i]
        end_idx = speaker_change_indices[i + 1]
        group_segments = segments[start_idx:end_idx]
        
        group_start = group_segments[0].get("start", group_segments[0].get("start_time", 0))
        group_end = group_segments[-1].get("end", group_segments[-1].get("end_time", 0))
        group_duration = group_end - group_start
        
        speaker_groups.append({
            "segments": group_segments,
            "start": group_start,
            "end": group_end,
            "duration": group_duration,
        })
    
    batches = []
    current_batch_segments = []
    current_batch_start = None
    
    for group in speaker_groups:
        if current_batch_start is None:
            current_batch_start = group["start"]
        
        potential_duration = group["end"] - current_batch_start
        
        if potential_duration > max_duration_seconds and current_batch_segments:
            batches.append(current_batch_segments)
            current_batch_segments = []
            current_batch_start = group["start"]
        
        current_batch_segments.extend(group["segments"])
    
    if current_batch_segments:
        batches.append(current_batch_segments)
    
    if len(batches) >= 2:
        last_batch = batches[-1]
        last_start = last_batch[0].get("start", last_batch[0].get("start_time", 0))
        last_end = last_batch[-1].get("end", last_batch[-1].get("end_time", 0))
        last_duration = last_end - last_start
        
        if last_duration < max_duration_seconds / 2:
            batches[-2].extend(batches[-1])
            batches.pop()
    
    return batches


def process_json_file(json_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    处理单个JSON文件，返回处理后的数据
    
    Args:
        json_path: 输入JSON文件路径
        output_path: 输出JSON文件路径（可选，不指定则覆盖原文件）
    
    Returns:
        处理后的JSON数据
    """
    json_path = Path(json_path)
    
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    
    batches = split_segments_by_duration(segments, max_duration_seconds=600.0)
    
    print(f"Split {len(segments)} segments into {len(batches)} batches")
    
    batch_data_list = []
    for i, batch_segments in enumerate(batches):
        batch_data_list.append({
            "batch_id": i,
            "segments": batch_segments,
        })
    
    return {
        "json_path": str(json_path),
        "original_data": data,
        "batches": batch_data_list,
        "num_batches": len(batches),
    }


if __name__ == "__main__":
    # 测试
    test_segments = [
        {"text": "我一九九四年十月出生的", "speaker": "O1", "start": 0.7, "end": 3.9},
        {"text": "这个要三块六毛八", "speaker": "O2", "start": 4.5, "end": 6.4},
        {"text": "十五分之七加四分之一等于多少", "speaker": "O1", "start": 6.5, "end": 11.1},
        {"text": "百分之八十五的人同意这个方案", "speaker": "O2", "start": 11.5, "end": 15.0},
    ]
    
    processor = NumberProcessor()
    
    # 测试prepare_input
    input_data = processor.prepare_input({"batch_id": 0, "segments": test_segments})
    print("Input data:", json.dumps(input_data["simplified_segments"], ensure_ascii=False, indent=2))
    
    # 测试build_prompt
    prompt = processor.build_prompt(input_data)
    print("\nPrompt:")
    print(prompt)
    
    # 测试check_output
    test_output = json.dumps([
        {"text": "我1994年10月出生的", "speaker": "O1"},
        {"text": "这个要3块6毛8", "speaker": "O2"},
        {"text": "7/15加1/4等于多少", "speaker": "O1"},
        {"text": "85%的人同意这个方案", "speaker": "O2"},
    ], ensure_ascii=False)
    
    print("\nTest output validation:")
    print(f"Valid: {processor.check_output(test_output, input_data)}")
    print(f"Output: {test_output}")
