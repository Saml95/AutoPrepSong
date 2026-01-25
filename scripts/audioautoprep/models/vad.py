"""
语音活动检测（Voice Activity Detection）模型
提供了两种不同的VAD实现：
1. Silero VAD
2. FunASR VAD

两者都通过统一的接口提供语音检测功能
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple, Optional
import soundfile
import numpy as np
from audioautoprep.models.utils import read_audio, write_audio
import os

class BaseVAD(ABC):
    """VAD基类，定义统一接口"""
    
    @abstractmethod
    def load_model(self):
        """加载VAD模型"""
        pass
    
    @abstractmethod
    def detect(self, audio_path: str) -> List[Dict[str, float]]:
        """
        检测音频文件中的语音片段
        
        参数:
            audio_path: 音频文件路径
            
        返回:
            语音片段列表，每个片段包含开始和结束时间（秒）
        """
        pass

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载音频文件
        
        参数:
            audio_path: 音频文件路径

        返回:
            音频数据
        """
        return read_audio(audio_path)
    
    def length_detect(self, results: List[Dict[str, float]], threshold: float = 30.0, visualize: bool = False) -> Dict[str, Any]:
        """
        统计检测结果中语音片段的长度信息
        
        参数:
            results: 语音检测结果列表，每个元素包含start和end时间
            threshold: 长片段的阈值（秒），默认为30秒
            visualize: 是否在函数内部展示结果，默认为False
            
        返回:
            包含以下信息的字典：
            - max_length: 最长片段的长度（秒）
            - min_length: 最短片段的长度（秒）
            - max_segment: 最长片段的信息（包含start和end）
            - min_segment: 最短片段的信息（包含start和end）
            - long_segments_count: 长度超过阈值的片段数量
            - long_segments: 长度超过阈值的片段列表
        """
        if not results or len(results) == 0:
            stats = {
                "max_length": 0,
                "min_length": 0,
                "max_segment": None,
                "min_segment": None,
                "long_segments_count": 0,
                "long_segments": [],
                "threshold": threshold
            }
            
            if visualize:
                print(f"\n语音片段长度统计 (阈值{threshold}秒):")
                print("没有检测到语音片段")
                
            return stats
            
        # 计算每个片段的长度
        segments_with_length = []
        for segment in results:
            length = segment["end"] - segment["start"]
            segments_with_length.append({
                "start": segment["start"],
                "end": segment["end"],
                "length": length
            })
            
        # 找出最长和最短的片段
        max_segment = max(segments_with_length, key=lambda x: x["length"])
        min_segment = min(segments_with_length, key=lambda x: x["length"])
        
        # 找出长度超过阈值的片段
        long_segments = [seg for seg in segments_with_length if seg["length"] > threshold]
        
        stats = {
            "max_length": max_segment["length"],
            "min_length": min_segment["length"],
            "max_segment": {"start": max_segment["start"], "end": max_segment["end"]},
            "min_segment": {"start": min_segment["start"], "end": min_segment["end"]},
            "long_segments_count": len(long_segments),
            "long_segments": [{"start": seg["start"], "end": seg["end"], "length": seg["length"]} for seg in long_segments],
            "threshold": threshold
        }
        
        if visualize:
            print(f"\n语音片段长度统计 (阈值{threshold}秒):")
            print(f"最长片段: {stats['max_length']:.2f}秒 (开始: {stats['max_segment']['start']:.2f}, 结束: {stats['max_segment']['end']:.2f})")
            print(f"最短片段: {stats['min_length']:.2f}秒 (开始: {stats['min_segment']['start']:.2f}, 结束: {stats['min_segment']['end']:.2f})")
            print(f"长度超过{stats['threshold']}秒的片段数量: {stats['long_segments_count']}")
            
        return stats
    
    # def merge_segments(self, segments: List[Dict[str, float]], threshold: float = 0.5) -> List[Dict[str, float]]:
    #     """
    #     合并语音片段
        
    #     参数:
    #         segments: 语音片段列表，每个元素包含start和end时间
    #         threshold: 合并阈值（秒），默认为30秒
            
    #     返回:
    #         合并后的语音片段列表
    #     """

        
        


class SileroVAD(BaseVAD):
    """基于Silero的VAD实现"""
    
    def __init__(self):
        self.model = None
    
    def load_model(self):
        """加载Silero VAD模型"""
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()
        return self
    
    def detect(self, audio_path: str, min_silence_duration_ms: int = 100, min_speech_duration_ms: int = 100, threshold: float = 30.0, save: bool = False, output_dir: str = "output") -> List[Dict[str, float]]:
        """
        使用Silero VAD检测语音片段
        
        参数:
            audio_path: 音频文件路径
            
        返回:
            语音片段列表，每个片段包含开始和结束时间（秒）
        """
        # 
        # torch.set_num_threads(1) # JIT
        # session.intra_op_num_threads = 1 # ONNX
        # session.inter_op_num_threads = 1 # ONNX

        
        if self.model is None:
            self.load_model()
            
        from silero_vad import get_speech_timestamps
        
        # 读取音频文件
        wav = read_audio(audio_path)
        
        # 获取语音时间戳
        speech_timestamps = get_speech_timestamps(
            wav,
            self.model,
            return_seconds=True,  # 以秒为单位返回时间戳
            min_silence_duration_ms=min_silence_duration_ms,
            min_speech_duration_ms=min_speech_duration_ms,
        )

        # 检查长片段并进行细分
        if threshold is not None and threshold > 0:
            refined_timestamps = []
            
            for ts in speech_timestamps:
                duration = ts["end"] - ts["start"]
                if duration > threshold:
                    # 对长片段重新进行VAD检测,使用更小的静音间隔
                    # 计算片段的采样点范围
                    start_sample = int(ts["start"] * 16000)  # 16kHz采样率
                    end_sample = int(ts["end"] * 16000)
                    segment_wav = wav[start_sample:end_sample]
                    # 使用更小的静音间隔重新检测
                    refined_segments = get_speech_timestamps(
                        segment_wav,
                        self.model,
                        return_seconds=True,
                        min_silence_duration_ms=20
                    )


                    
                    # 调整时间戳以反映在原始音频中的位置
                    for seg in refined_segments:
                        refined_timestamps.append({
                            "start": ts["start"] + seg["start"],
                            "end": ts["start"] + seg["end"],
                        })
                else:
                    refined_timestamps.append(ts)
                    
            speech_timestamps = refined_timestamps





        
        # 转换为统一格式
        result = []
        for idx, ts in enumerate(speech_timestamps):
            result.append({
                "start": ts["start"],
                "end": ts["end"],
                'audio_path': os.path.abspath(f"{output_dir}/chunk_{idx}.wav"),
                'text': "<None>"    
            })
            if save:
                # import pdb; pdb.set_trace()
                write_audio(f"{output_dir}/chunk_{idx}.wav", wav[int(ts["start"] * 16000):int(ts["end"] * 16000)], 16000)
                
            
        return result


class FunASRVAD(BaseVAD):
    """基于FunASR的VAD实现"""
    
    def __init__(self, model_revision: str = "v2.0.4"):
        self.model = None
        self.model_revision = model_revision
    
    def load_model(self):
        """加载FunASR VAD模型"""
        from funasr import AutoModel
        self.model = AutoModel(model="fsmn-vad", model_revision=self.model_revision)
        return self
    
    def detect(self, audio_path: str) -> List[Dict[str, float]]:
        """
        使用FunASR VAD检测语音片段
        
        参数:
            audio_path: 音频文件路径
            
        返回:
            语音片段列表，每个片段包含开始和结束时间（秒）
        """
        if self.model is None:
            self.load_model()
            
        # 生成VAD结果
        audio_data = self.load_audio(audio_path)
        res = self.model.generate(input=audio_data)
        
        # 解析结果并转换为统一格式
        # 注意：这里假设FunASR的输出格式，可能需要根据实际输出调整
        result = []
        for segment in res[0]['value']:
            result.append({
                "start": float(segment[0])/1000,
                "end": float(segment[1])/1000
            })
                    
        return result
    
    def detect_streaming(self, audio_path: str, chunk_size: int = 200) -> List[Dict[str, float]]:
        """
        使用FunASR VAD的流式模式检测语音片段
        
        参数:
            audio_path: 音频文件路径
            chunk_size: 分块大小（毫秒）
            
        返回:
            语音片段列表，每个片段包含开始和结束时间（秒）
        """
        if self.model is None:
            self.load_model()

        
        # 读取音频文件
        speech, sample_rate = soundfile.read(audio_path)
        chunk_stride = int(chunk_size * sample_rate / 1000)
        
        # 流式处理
        cache = {}
        total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)
        results = []
        
        for i in range(total_chunk_num):
            speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
            is_final = i == total_chunk_num - 1
            
            res = self.model.generate(
                input=speech_chunk, 
                cache=cache, 
                is_final=is_final, 
                chunk_size=chunk_size
            )
            if len(res[0]['value']):
                print(res[0])
            
def create_vad(vad_type: str = "silero") -> BaseVAD:
    """
    创建VAD实例的工厂函数
    
    参数:
        vad_type: VAD类型，可选 "silero" 或 "funasr"
        
    返回:
        VAD实例
    """
    if vad_type.lower() == "silero":
        return SileroVAD().load_model()
    elif vad_type.lower() == "funasr":
        return FunASRVAD().load_model()
    else:
        raise ValueError(f"不支持的VAD类型: {vad_type}，请选择 'silero' 或 'funasr'")


# 使用示例
if __name__ == "__main__":
    # audio_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo2/383137.mp3"
    # audio_path = "/data/jianwei/experiment/DataPropc/Emilia/data/demo/2_104.mp3"
    audio_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo/2.mp3"

    
    output_dir = "output_{}".format(audio_path.split("/")[-1].split(".")[0])
    os.makedirs(output_dir, exist_ok=True)

    # 使用Silero VAD
    silero_vad = create_vad("silero")
    silero_results = silero_vad.detect(audio_path, save=True, output_dir=output_dir)
    print("Silero VAD结果:", silero_results)

    # 保存Silero VAD结果为JSON文件
    import json
    
    # 保存为JSON文件
    with open("{}/silero_vad_results.json".format(output_dir), "w", encoding="utf-8") as f:
        json.dump(silero_results, f, ensure_ascii=True, indent=4)
    
    # 分析Silero VAD结果的长度信息（使用默认阈值30秒，并自动展示结果）
    silero_length_stats = silero_vad.length_detect(silero_results, visualize=True)
    import pdb; pdb.set_trace()
    
    # 使用FunASR VAD
    funasr_vad = create_vad("funasr")
    funasr_results = funasr_vad.detect(audio_path)
    print("\nFunASR VAD结果:", funasr_results)
    
    # 分析FunASR VAD结果的长度信息（使用默认阈值30秒，并自动展示结果）
    funasr_length_stats = funasr_vad.length_detect(funasr_results, visualize=True)
    
    
    