"""
中文标点符号添加处理器
处理ASR输出的segments，添加标点符号
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from azure_api import TaskProcessor


def remove_punctuation(text: str, keep_space: bool = False) -> str:
    """移除所有标点符号，只保留文字内容
    
    Args:
        text: 输入文本
        keep_space: 是否保留空格（英文需要保留）
    """
    # 对于英文，先将各种破折号/连字符替换为空格，避免 small-town 变成 smalltown
    if keep_space:
        # 包括: hyphen(-), en-dash(–), em-dash(—), minus(−), 等
        for dash in ['-', '–', '—', '−', '‐', '‑', '‒', '―']:
            text = text.replace(dash, ' ')
    
    # 使用Unicode类别移除所有标点符号
    # P = Punctuation (所有标点), S = Symbol (符号), Z = Separator (分隔符)
    result = []
    for char in text:
        cat = unicodedata.category(char)
        # 保留字母(L)、数字(N)、标记(M)
        if cat.startswith('L') or cat.startswith('N') or cat.startswith('M'):
            result.append(char)
        elif keep_space and char == ' ':
            result.append(char)
    
    text_out = ''.join(result)
    
    # 如果保留空格，将连续空格合并为一个
    if keep_space:
        text_out = re.sub(r' +', ' ', text_out).strip()
    
    return text_out


def normalize_text(text: str, keep_space: bool = False) -> str:
    """标准化文本用于比较
    
    Args:
        text: 输入文本
        keep_space: 是否保留空格（英文需要保留）
    """
    # 移除所有标点和空白
    text = remove_punctuation(text, keep_space=keep_space)
    # 转小写（处理英文）
    text = text.lower()
    # Unicode标准化
    text = unicodedata.normalize('NFKC', text)
    return text


class PunctuationProcessor(TaskProcessor):
    """
    标点符号添加处理器
    
    输入: List[Dict] - segments列表，每个包含 text, speaker
    输出: List[Dict] - 添加标点后的segments列表
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
        self.last_error_feedback = None  # 存储上次错误的反馈信息
    
    def prepare_input(self, raw_data: Any) -> Dict[str, Any]:
        """
        准备输入数据
        raw_data: {
            "batch_id": int,
            "segments": List[Dict] - 原始segments (包含text, speaker, start, end等)
        }
        """
        segments = raw_data.get("segments", [])
        
        # 英文需要保留空格
        keep_space = (self.language == "en" or self.prompt_language == "en")
        
        # 提取text和speaker用于处理，并去掉text中的标点
        simplified_segments = []
        for seg in segments:
            original_text = seg.get("text", "")
            # 去掉输入text中的所有标点符号（英文保留空格）
            cleaned_text = remove_punctuation(original_text, keep_space=keep_space)
            simplified_segments.append({
                "text": cleaned_text,
                "speaker": seg.get("speaker", "unknown"),
            })
        
        return {
            "batch_id": raw_data.get("batch_id", 0),
            "original_segments": segments,  # 保留完整原始数据
            "simplified_segments": simplified_segments,  # 简化版用于API（已去标点）
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
        
        # 构建segments的JSON表示
        segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
        
        # 基础prompt
        prompt = f"""请为以下对话文本添加合适的中文标点符号。

【重要警告】绝对不能修改、删除、添加任何文字！只能插入标点符号！

严格要求：
1. 【禁止】删除任何文字，包括重复的字词（如"请请"、"我我"等口误）
2. 【禁止】修改任何文字的顺序或内容
3. 【禁止】添加原文中没有的文字
4. 【只能】在原文字之间插入标点符号（，。？！、；：""''……等）
5. 每个字符必须原样保留，验证时会逐字对比
6. speaker字段必须保持不变

【标点使用规则】
- 问句必须使用问号"？"，包括：
  - 以"吗"、"呢"、"吧"、"啊"、"呀"结尾的疑问句
  - 以"什么"、"怎么"、"哪"、"谁"、"为什么"等疑问词的句子
  - 选择问句、反问句
- 陈述句使用句号"。"
- 感叹句使用感叹号"！"
- 句中停顿使用逗号"，"
- 如果和下一个segment之前的语义连贯并且speaker相同，则不需要添加标点

输入数据：
{segments_json}
"""
        
        # 如果有上次错误的反馈，添加到prompt中
        if self.last_error_feedback:
            prompt += f"""
【上次错误】你之前的输出有问题：
{self.last_error_feedback}
请仔细检查并确保不删改任何文字！
"""
            self.last_error_feedback = None  # 清除反馈
        
        prompt += "\n请直接输出修改后的JSON数组，不要有任何其他说明："
        
        return prompt
    
    def build_prompt_en(self, input_data: Dict[str, Any]) -> str:
        """构建英文提示词 - 添加标点和正确的大小写"""
        segments = input_data["simplified_segments"]
        
        # 构建segments的JSON表示
        segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
        
        # English prompt
        prompt = f"""Please add appropriate punctuation and fix capitalization for the following conversation transcripts.

【CRITICAL WARNING】You must NOT add, delete, or change any words! Only:
1. Insert punctuation marks (. , ? ! ' " ; : - etc.)
2. Change letter case (capitalize/lowercase)

Strict Requirements:
1. 【FORBIDDEN】Deleting any words and characters, including repeated words, abnormal characters (e.g., "I I", "the the" - speech disfluencies)
2. 【FORBIDDEN】Changing the order of any words and characters      
3. 【FORBIDDEN】Adding words or characters that don't exist in the original
4. 【ALLOWED】Only inserting punctuation and changing capitalization
5. The validation will compare: lowercase(remove_punctuation(output)) == lowercase(original)
6. The speaker field must remain unchanged

【Capitalization Rules】
- First letter of each sentence should be capitalized
- Proper nouns (names, places, organizations) should be capitalized
- "I" as a pronoun should always be capitalized
- Acronyms should be in uppercase (e.g., USA, NASA, FBI)

【Punctuation Rules】
- Questions must use question mark "?"
- Statements end with period "."
- Exclamations use exclamation mark "!"
- Use comma "," for pauses within sentences
- Use apostrophe for contractions (e.g., don't, can't, I'm)
- If the segment is semantically connected to the next one with the same speaker, no ending punctuation is needed

Input data:
{segments_json}
"""
        
        # 如果有上次错误的反馈，添加到prompt中
        if self.last_error_feedback:
            prompt += f"""
【PREVIOUS ERROR】Your previous output had issues:
{self.last_error_feedback}
Please check carefully and ensure no words are added/deleted/changed!
"""
            self.last_error_feedback = None  # 清除反馈
        
        prompt += "\nPlease output the modified JSON array directly, without any other explanation:"
        
        return prompt
    
    def check_output(self, output: str, input_data: Dict[str, Any]) -> bool:
        """验证输出：去掉标点后内容应与原内容完全一致"""
        try:
            # 解析输出JSON
            # 尝试提取JSON部分
            output = output.strip()
            if output.startswith("```"):
                # 去掉markdown代码块
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
            
            # 检查数量是否一致
            if len(parsed_output) != len(original_segments):
                print(f"Segment count mismatch: {len(parsed_output)} vs {len(original_segments)}")
                return False
            
            # 英文需要保留空格
            keep_space = (self.language == "en" or self.prompt_language == "en")
            
            # 检查每个segment
            for i, (orig, new) in enumerate(zip(original_segments, parsed_output)):
                orig_text = orig.get("text", "")
                new_text = new.get("text", "")
                
                # 去掉标点后比较（英文保留空格）
                orig_normalized = normalize_text(orig_text, keep_space=keep_space)
                new_normalized = normalize_text(new_text, keep_space=keep_space)
                
                if orig_normalized != new_normalized:
                    # 如果只是空格差异，认为相同（边缘情况太多）
                    orig_no_space = orig_normalized.replace(' ', '')
                    new_no_space = new_normalized.replace(' ', '')
                    if orig_no_space == new_no_space:
                        # 只有空格差异，认为通过
                        continue
                    
                    # 找出具体差异
                    diff_info = self._find_text_diff(orig_normalized, new_normalized)
                    print(f"Segment {i} text mismatch after removing punctuation:")
                    print(f"  Original: {orig_text}")
                    print(f"  New: {new_text}")
                    print(f"  Original normalized: {orig_normalized}")
                    print(f"  New normalized: {new_normalized}")
                    print(f"  Diff: {diff_info}")
                    
                    # 保存错误反馈供下次重试使用
                    self.last_error_feedback = f"""Segment {i} 文字内容被修改！
原文: "{orig_text}"
你的输出: "{new_text}"
问题: {diff_info}
请保留所有原文字符，包括重复的字！"""
                    
                    if self.strict_check:
                        return False
                
                # 检查speaker是否一致
                if orig.get("speaker") != new.get("speaker"):
                    print(f"Segment {i} speaker mismatch: {orig.get('speaker')} vs {new.get('speaker')}")
                    # return False
                    new["speaker"] = orig.get("speaker")  # 自动修正speaker字段
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse output JSON: {e}")
            print(f"Output: {output[:500]}...")
            return False
        except Exception as e:
            print(f"Check error: {e}")
            return False
    
    def process_output(self, output: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输出，将标点添加回原始segments"""
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
        
        # 将修改后的text更新到原始segments
        original_segments = input_data["original_segments"]
        simplified_segments = input_data["simplified_segments"]
        updated_segments = []
        
        for orig_seg, simplified_seg, new_seg in zip(original_segments, simplified_segments, parsed_output):
            updated_seg = orig_seg.copy()
            new_text = new_seg.get("text", orig_seg.get("text", ""))
            old_text = simplified_seg.get("text", "")
            
            # 检查文本是否被修改
            is_modified = (new_text != old_text)
            if is_modified:
                # 保留修改前的原始文本
                updated_seg["text_ori_punc"] = old_text
            
            updated_seg["text"] = new_text
            updated_seg["is_modified_punc"] = is_modified
            updated_segments.append(updated_seg)
        
        return {
            "batch_id": input_data["batch_id"],
            "segments": updated_segments,
        }
    
    def _find_text_diff(self, orig: str, new: str) -> str:
        """找出文本差异的具体位置和内容"""
        from collections import Counter
        orig_counter = Counter(orig)
        new_counter = Counter(new)
        
        missing = []  # 原文有但新文本没有的
        extra = []    # 新文本有但原文没有的
        
        for char, count in orig_counter.items():
            diff = count - new_counter.get(char, 0)
            if diff > 0:
                missing.extend([char] * diff)
        
        for char, count in new_counter.items():
            diff = count - orig_counter.get(char, 0)
            if diff > 0:
                extra.extend([char] * diff)
        
        parts = []
        if missing:
            parts.append(f"缺少字符: {''.join(missing)}")
        if extra:
            parts.append(f"多余字符: {''.join(extra)}")
        
        return "; ".join(parts) if parts else "字符顺序不同"
    
    def get_system_prompt(self) -> str:
        if self.prompt_language == "en":
            return """You are a professional text processing assistant, skilled at adding appropriate punctuation and fixing capitalization for speech transcripts.

【Core Principles】You must NOT modify any word content!
- Even if the original text appears to have errors or repetitions, you must keep them as is
- Your only tasks are: 1) Insert punctuation marks 2) Fix letter capitalization
- The validation system will compare: lowercase(remove_punctuation(output)) must equal lowercase(original)
- Any word addition/deletion/change will cause validation failure"""
        else:
            return """你是一个专业的中文文本处理助手，擅长为口语转写文本添加合适的标点符号。

【核心原则】你绝对不能修改任何文字内容！
- 即使原文看起来有错误或重复，也必须原样保留
- 你的唯一任务是在文字之间插入标点符号
- 验证系统会逐字对比，任何字符变化都会导致失败"""


def split_segments_by_duration(
    segments: List[Dict],
    max_duration_seconds: float = 120.0,  # 2分钟
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
    
    # 第一步：找出所有speaker切换点的索引
    speaker_change_indices = [0]  # 第一个segment总是一个切换点
    for i in range(1, len(segments)):
        prev_speaker = segments[i-1].get("speaker", "")
        curr_speaker = segments[i].get("speaker", "")
        if prev_speaker != curr_speaker:
            speaker_change_indices.append(i)
    speaker_change_indices.append(len(segments))  # 末尾
    
    # 第二步：按speaker切换点构建候选分割区间
    # 每个区间是连续相同speaker的segments
    speaker_groups = []
    for i in range(len(speaker_change_indices) - 1):
        start_idx = speaker_change_indices[i]
        end_idx = speaker_change_indices[i + 1]
        group_segments = segments[start_idx:end_idx]
        
        # 计算这个group的时长
        group_start = group_segments[0].get("start", group_segments[0].get("start_time", 0))
        group_end = group_segments[-1].get("end", group_segments[-1].get("end_time", 0))
        group_duration = group_end - group_start
        
        speaker_groups.append({
            "segments": group_segments,
            "start": group_start,
            "end": group_end,
            "duration": group_duration,
        })
    
    # 第三步：合并speaker groups成batches，确保不超过max_duration
    batches = []
    current_batch_segments = []
    current_batch_start = None
    
    for group in speaker_groups:
        if current_batch_start is None:
            current_batch_start = group["start"]
        
        # 计算如果加入这个group后的总时长
        potential_duration = group["end"] - current_batch_start
        
        if potential_duration > max_duration_seconds and current_batch_segments:
            # 超过限制，先保存当前batch，开始新batch
            batches.append(current_batch_segments)
            current_batch_segments = []
            current_batch_start = group["start"]
        
        current_batch_segments.extend(group["segments"])
    
    # 添加最后一个batch
    if current_batch_segments:
        batches.append(current_batch_segments)
    
    # 第四步：检查最后一个batch，如果时长不足一半则合并到上一个
    if len(batches) >= 2:
        last_batch = batches[-1]
        last_start = last_batch[0].get("start", last_batch[0].get("start_time", 0))
        last_end = last_batch[-1].get("end", last_batch[-1].get("end_time", 0))
        last_duration = last_end - last_start
        
        if last_duration < max_duration_seconds / 2:
            # 合并到上一个batch
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
    
    # 分割为2分钟的batches
    batches = split_segments_by_duration(segments, max_duration_seconds=600.0)
    
    print(f"Split {len(segments)} segments into {len(batches)} batches")
    
    # 准备batch数据
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
        {"text": "王嘉尔新出的一首歌新歌儿你听了吗。", "speaker": "O1", "start": 0.7, "end": 3.9},
        {"text": "嗯哪一首啊。", "speaker": "O2", "start": 4.5, "end": 6.4},
        {"text": "就是那个就是我爱你三千遍那个你听吗。", "speaker": "O1", "start": 6.5, "end": 11.1},
    ]
    
    processor = PunctuationProcessor()
    
    # 测试prepare_input
    input_data = processor.prepare_input({"batch_id": 0, "segments": test_segments})
    print("Input data:", json.dumps(input_data["simplified_segments"], ensure_ascii=False, indent=2))
    
    # 测试build_prompt
    prompt = processor.build_prompt(input_data)
    print("\nPrompt:")
    print(prompt)
    
    # # 测试check_output
    # test_output = json.dumps([
    #     {"text": "王嘉尔新出的一首歌，新歌儿你听了吗？", "speaker": "O1"},
    #     {"text": "嗯，哪一首啊？", "speaker": "O2"},
    #     {"text": "就是那个，就是我爱你三千遍那个，你听吗？", "speaker": "O1"},
    # ], ensure_ascii=False)
    
    # print("\nTest output validation:")
    # print(f"Valid: {processor.check_output(test_output, input_data)}")
    # print(test_output)
