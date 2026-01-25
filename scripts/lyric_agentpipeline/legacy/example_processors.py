"""
Example Task Processors for Azure API.
Demonstrates how to create custom task processors with input preparation,
prompt building, output validation, and post-processing.
"""

import json
import re
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

# Import base class from azure_api
from azure_api import TaskProcessor


# ========================== Example 1: JSON Output Validator ==========================
class JSONOutputProcessor(TaskProcessor):
    """
    Task processor that expects JSON output from the API.
    Validates that the output is valid JSON and optionally checks for required keys.
    """
    
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant. Always respond in valid JSON format.",
        required_keys: Optional[List[str]] = None,
        prompt_template: str = "{input}",
    ):
        self.system_prompt = system_prompt
        self.required_keys = required_keys or []
        self.prompt_template = prompt_template
    
    def prepare_input(self, raw_data: Any) -> Any:
        return raw_data
    
    def build_prompt(self, input_data: Any) -> str:
        if isinstance(input_data, dict) and "prompt" in input_data:
            return self.prompt_template.format(input=input_data["prompt"])
        return self.prompt_template.format(input=str(input_data))
    
    def check_output(self, output: str, input_data: Any) -> bool:
        try:
            # Try to parse as JSON
            parsed = json.loads(output)
            
            # Check for required keys
            if self.required_keys:
                if not isinstance(parsed, dict):
                    return False
                for key in self.required_keys:
                    if key not in parsed:
                        return False
            
            return True
        except json.JSONDecodeError:
            return False
    
    def process_output(self, output: str, input_data: Any) -> Dict[str, Any]:
        return json.loads(output)
    
    def get_system_prompt(self) -> str:
        return self.system_prompt


# ========================== Example 2: Structured Output Processor ==========================
class SummaryResult(BaseModel):
    """Pydantic model for summary output."""
    summary: str
    key_points: List[str]
    word_count: int


class SummarizationProcessor(TaskProcessor):
    """
    Task processor for text summarization with structured output.
    """
    
    def __init__(
        self,
        max_length: int = 200,
        language: str = "en",
        min_key_points: int = 3,
    ):
        self.max_length = max_length
        self.language = language
        self.min_key_points = min_key_points
    
    def prepare_input(self, raw_data: Any) -> Dict[str, Any]:
        if isinstance(raw_data, str):
            return {"text": raw_data}
        return raw_data
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        text = input_data.get("text", "")
        lang_instruction = "in Chinese" if self.language == "zh" else "in English"
        
        return f"""Summarize the following text {lang_instruction}.

Requirements:
1. Summary should be no more than {self.max_length} words
2. Include at least {self.min_key_points} key points
3. Respond in JSON format with fields: summary, key_points (list), word_count

Text to summarize:
{text}

Output (JSON only):"""
    
    def check_output(self, output: str, input_data: Dict[str, Any]) -> bool:
        try:
            parsed = json.loads(output)
            
            # Check required fields
            if "summary" not in parsed or "key_points" not in parsed:
                return False
            
            # Check key points count
            if len(parsed.get("key_points", [])) < self.min_key_points:
                return False
            
            # Check word count (approximate)
            summary_words = len(parsed["summary"].split())
            if summary_words > self.max_length * 1.5:  # Allow some tolerance
                return False
            
            return True
        except (json.JSONDecodeError, KeyError):
            return False
    
    def process_output(self, output: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = json.loads(output)
        result["original_text_length"] = len(input_data.get("text", ""))
        return result
    
    def get_system_prompt(self) -> str:
        return "You are an expert summarizer. Always output valid JSON."


# ========================== Example 3: ASR Refinement Processor ==========================
class ASRRefinementProcessor(TaskProcessor):
    """
    Task processor for ASR (Automatic Speech Recognition) text refinement.
    Takes noisy ASR output and refines it with GPT.
    """
    
    def __init__(
        self,
        hotwords: Optional[List[str]] = None,
        speaker_map: Optional[Dict[str, str]] = None,
        language: str = "auto",
    ):
        self.hotwords = hotwords or []
        self.speaker_map = speaker_map or {}
        self.language = language
    
    def prepare_input(self, raw_data: Any) -> Dict[str, Any]:
        """Prepare segment data for refinement."""
        if isinstance(raw_data, dict):
            return {
                "segment_id": raw_data.get("segment_id", 0),
                "speaker": raw_data.get("speaker", "unknown"),
                "text_original": raw_data.get("text", raw_data.get("text_original", "")),
                "start_time": raw_data.get("start", raw_data.get("start_time", 0)),
                "end_time": raw_data.get("end", raw_data.get("end_time", 0)),
            }
        return {"text_original": str(raw_data)}
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        text = input_data.get("text_original", "")
        speaker = input_data.get("speaker", "unknown")
        
        hotwords_str = ", ".join(self.hotwords) if self.hotwords else "None"
        speaker_map_str = json.dumps(self.speaker_map, ensure_ascii=False) if self.speaker_map else "{}"
        
        return f"""Refine the following ASR transcription text.

Rules:
1. Fix obvious transcription errors while preserving the original meaning
2. Correct punctuation and formatting
3. Use the provided hotwords for proper noun spelling
4. Do not add or remove content
5. Output in JSON format with fields: text, speaker, confidence

Hotwords: {hotwords_str}
Speaker mapping: {speaker_map_str}

Original text (Speaker {speaker}):
{text}

Output (JSON only):"""
    
    def check_output(self, output: str, input_data: Dict[str, Any]) -> bool:
        try:
            parsed = json.loads(output)
            
            # Must have text field
            if "text" not in parsed:
                return False
            
            # Text should not be empty if original was not empty
            original_text = input_data.get("text_original", "")
            if original_text.strip() and not parsed["text"].strip():
                return False
            
            # Text length should be similar (within 50% tolerance)
            original_len = len(original_text)
            refined_len = len(parsed["text"])
            if original_len > 0:
                ratio = refined_len / original_len
                if ratio < 0.5 or ratio > 2.0:
                    return False
            
            return True
        except (json.JSONDecodeError, KeyError):
            return False
    
    def process_output(self, output: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = json.loads(output)
        # Merge with original data
        return {
            **input_data,
            "text_refined": result.get("text"),
            "speaker_refined": result.get("speaker", input_data.get("speaker")),
            "confidence": result.get("confidence"),
        }
    
    def get_system_prompt(self) -> str:
        return "You are an ASR refinement assistant. Output valid JSON only."


# ========================== Example 4: Code Generation Processor ==========================
class CodeGenerationProcessor(TaskProcessor):
    """
    Task processor for code generation with syntax validation.
    """
    
    def __init__(
        self,
        language: str = "python",
        max_lines: int = 100,
    ):
        self.language = language
        self.max_lines = max_lines
    
    def prepare_input(self, raw_data: Any) -> Dict[str, Any]:
        if isinstance(raw_data, str):
            return {"description": raw_data}
        return raw_data
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        description = input_data.get("description", "")
        context = input_data.get("context", "")
        
        prompt = f"""Write {self.language} code for the following task:

Task Description:
{description}
"""
        if context:
            prompt += f"""
Context/Existing Code:
{context}
"""
        prompt += f"""
Requirements:
1. Code should be no more than {self.max_lines} lines
2. Include comments explaining key parts
3. Follow best practices for {self.language}

Respond with the code only, wrapped in ```{self.language}``` blocks."""
        
        return prompt
    
    def check_output(self, output: str, input_data: Dict[str, Any]) -> bool:
        # Extract code from markdown blocks
        pattern = rf"```{self.language}\n(.*?)```"
        matches = re.findall(pattern, output, re.DOTALL)
        
        if not matches:
            # Try without language specifier
            pattern = r"```\n(.*?)```"
            matches = re.findall(pattern, output, re.DOTALL)
        
        if not matches:
            return False
        
        code = matches[0]
        
        # Check line count
        lines = code.strip().split("\n")
        if len(lines) > self.max_lines:
            return False
        
        # Basic syntax check for Python
        if self.language == "python":
            try:
                compile(code, "<string>", "exec")
            except SyntaxError:
                return False
        
        return True
    
    def process_output(self, output: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract code
        pattern = rf"```{self.language}\n(.*?)```"
        matches = re.findall(pattern, output, re.DOTALL)
        
        if not matches:
            pattern = r"```\n(.*?)```"
            matches = re.findall(pattern, output, re.DOTALL)
        
        code = matches[0] if matches else output
        
        return {
            "code": code.strip(),
            "language": self.language,
            "line_count": len(code.strip().split("\n")),
            "description": input_data.get("description"),
        }
    
    def get_system_prompt(self) -> str:
        return f"You are an expert {self.language} programmer. Write clean, well-documented code."


# ========================== Example 5: Translation Processor ==========================
class TranslationProcessor(TaskProcessor):
    """
    Task processor for text translation with quality checks.
    """
    
    def __init__(
        self,
        source_lang: str = "auto",
        target_lang: str = "en",
        preserve_format: bool = True,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.preserve_format = preserve_format
    
    def prepare_input(self, raw_data: Any) -> Dict[str, Any]:
        if isinstance(raw_data, str):
            return {"text": raw_data}
        return raw_data
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        text = input_data.get("text", "")
        
        source_instruction = f"from {self.source_lang}" if self.source_lang != "auto" else ""
        format_instruction = "Preserve the original formatting (paragraphs, lists, etc.)." if self.preserve_format else ""
        
        return f"""Translate the following text {source_instruction} to {self.target_lang}.

{format_instruction}

Text to translate:
{text}

Output the translation only, without any explanations."""
    
    def check_output(self, output: str, input_data: Dict[str, Any]) -> bool:
        original_text = input_data.get("text", "")
        
        # Output should not be empty if input was not empty
        if original_text.strip() and not output.strip():
            return False
        
        # Output should be different from input (actual translation happened)
        if output.strip() == original_text.strip():
            return False
        
        # Length should be reasonable (within 3x)
        if len(original_text) > 0:
            ratio = len(output) / len(original_text)
            if ratio < 0.1 or ratio > 5.0:
                return False
        
        return True
    
    def process_output(self, output: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "original": input_data.get("text"),
            "translated": output.strip(),
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
        }
    
    def get_system_prompt(self) -> str:
        return f"You are a professional translator. Translate accurately to {self.target_lang}."


# ========================== Usage Examples ==========================
if __name__ == "__main__":
    print("Example Task Processors for Azure API")
    print("=" * 50)
    print()
    print("Available processors:")
    print("  - JSONOutputProcessor: For tasks requiring JSON output")
    print("  - SummarizationProcessor: For text summarization")
    print("  - ASRRefinementProcessor: For ASR text refinement")
    print("  - CodeGenerationProcessor: For code generation")
    print("  - TranslationProcessor: For text translation")
    print()
    print("Usage in YAML config:")
    print("""
# Example: Using SummarizationProcessor
mode: task_ray
api_family: gpt4o
num_workers: 10
max_api_retries: 5
max_check_retries: 3
data_file: "./texts_to_summarize.json"
processor_cls: "example_processors.SummarizationProcessor"
processor_kwargs:
  max_length: 150
  language: "en"
  min_key_points: 3
output_path: "./summaries.json"
""")
