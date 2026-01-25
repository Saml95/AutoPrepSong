from faster_whisper import WhisperModel, BatchedInferencePipeline
import faster_whisper
from audioautoprep.models.utils import read_audio, write_audio
import os
import torch
import json
from tqdm import tqdm
import numpy as np


class WhisperASRInference:
    def __init__(self, model_size="turbo", local_files_only=True, download_root=None,
                 device="cuda", compute_type="float16", 
                 batch_size=16, vad_filter=True, word_timestamps=True, 
                 min_sentence_end_chars=[".", "!", "?"], 
                 context_padding=0.1,
                 log_progress=False):
        """
        初始化Whisper ASR推理类
        
        参数:
            model_size (str): Whisper模型大小，默认为"turbo"
            device (str): 运行设备，默认为"cuda"
            compute_type (str): 计算类型，默认为"float16"
            batch_size (int): 批处理大小，默认为16
            vad_filter (bool): 是否使用VAD过滤，默认为True
            word_timestamps (bool): 是否生成单词时间戳，默认为True
            min_sentence_end_chars (list): 表示句子结束的字符列表，默认为[".", "!", "?"]
            context_padding (float): 音频片段前后的填充时间（秒），默认为0.1
            log_progress (bool): 是否打印日志，默认为False
        """
        # 模型参数
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.log_progress = log_progress
        # 转录参数
        self.batch_size = batch_size
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        
        # 引入标点，但是会提升幻觉
        self.asr_options = {
            "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."
        }
        
        self.vad_parameters={
            "max_speech_duration_s": 30,
            "min_silence_duration_ms": 50,
            'speech_pad_ms': 550,
        }
        
        # 可视化参数
        self.min_sentence_end_chars = min_sentence_end_chars
        self.context_padding = context_padding
        
        # 在初始化时加载模型
        # import pdb; pdb.set_trace()
        if os.path.exists(download_root):
            local_files_only = True
        else:
            local_files_only = False

        # local_files_only = False    
        self.model = WhisperModel(
            self.model_size, 
            device=self.device, 
            compute_type=self.compute_type,
            local_files_only=local_files_only,
            download_root=download_root
        )
        self.batched_model = BatchedInferencePipeline(model=self.model)

        
    
    def transcribe(self, audio, return_list=False):
        """
        转录音频
        
        参数:
            audio (str or numpy.ndarray): 音频文件路径或音频数据
            
        返回:
            list: 转录的segments列表
        """
        # 如果输入是字符串路径，则读取音频文件
        if isinstance(audio, str):
            audio = read_audio(audio)
        
        # 确保输入是numpy数组
        if not isinstance(audio, np.ndarray):
            raise TypeError("音频输入必须是文件路径(str)或numpy数组(numpy.ndarray)")
               
        segments, info = self.batched_model.transcribe(
            audio, 
            batch_size=self.batch_size, 
            vad_filter=self.vad_filter, 
            word_timestamps=self.word_timestamps,
            vad_parameters=self.vad_parameters,
            initial_prompt=self.asr_options["initial_prompt"],
            log_progress=self.log_progress
        )

        vad_list = []
        for item in info.transcription_options.clip_timestamps:
            for vad_segment in item['segments']:
                vad_list.append((vad_segment[0] / 16000, vad_segment[1] / 16000))
        # 转换为可序列化的对象
        new_segments = []
        for segment in segments:
            new_segments.append(segment)

        if return_list:
            new_segments = self.save_visualize(audio, new_segments, output_dir=None, save_audio=False)
        return new_segments, info, vad_list
    

    def process_segments(self, segments):
        output_segments = []
        
        cur_start = None
        cur_end = None
        cur_text = ""
        cut_text_timestamp = []
    
        
        # for segment in tqdm(segments):
        for segment in segments:
            for word in segment.words:
                if cur_start is None:
                    cur_start = word.start
                cur_text += word.word
                cut_text_timestamp.append((word.start, word.end, word.word))
                # 检查是否是句子结束
                if word.word and word.word[-1] in self.min_sentence_end_chars:
                    cur_end = word.end
                    
                    # 添加上下文填充
                    cur_start_ori = cur_start
                    cur_end_ori = cur_end
                    
                    # 记录片段信息
                    output_segments.append({
                        "start_ori": cur_start_ori,
                        "end_ori": cur_end_ori,
                        "text": cur_text,
                        "text_timestamp": cut_text_timestamp
                    })
                    
                    cur_text = ""
                    cur_start = None
                    cur_end = None
                    cut_text_timestamp = []
            if cur_start is not None:
                cur_end = segment.end
                cur_start_ori = cur_start
                cur_end_ori = cur_end
                

                
                output_segments.append({
                    "start_ori": cur_start_ori,
                    "end_ori": cur_end_ori,
                    "text": cur_text,
                    "text_timestamp": cut_text_timestamp
                })

                cur_text = ""
                cur_start = None
                cur_end = None
                cut_text_timestamp = []
        return output_segments
    
    def save_visualize(self, audio, segments=None, output_segments=None, output_dir=None, save_audio=False, vocal=None, bgm=None):
        """
        保存并可视化转录结果
        
        参数:
            audio (numpy.ndarray): 音频数据
            segments (list): 转录的segments列表
            output_dir (str): 输出目录
            save_audio (bool): 是否离线保存音频，默认为False
        返回:
            list: 处理后的音频片段信息列表
        """

        assert segments is not None or output_segments is not None, "segments and output_segments cannot be None at the same time"
        if output_segments is None:
            output_segments = self.process_segments(segments)


        # 过滤掉小于0.3秒的音频片段
        filtered_segments = []
        for segment in output_segments:
            # 计算片段时长
            duration = segment['end_ori'] - segment['start_ori']
            # 只保留时长大于等于0.3秒的片段
            if duration >= 0.3:
                filtered_segments.append(segment)
        
        # 更新输出片段列表
        output_segments = filtered_segments

        for subseg_idx, item in enumerate(output_segments):
            cur_start = max(0, item['start_ori'] - self.context_padding)
            cur_end = min(item['end_ori'] + self.context_padding, len(audio)/16000)
            
            if cur_end - cur_start < 0.5:
                cur_start = max(0, item['start_ori'] - 0.1)
                cur_end = min(item['end_ori'] + 0.1, len(audio)/16000)
            
            audio_subseg = audio[int(cur_start*16000):int(cur_end*16000)]
            if save_audio:
                write_audio(f"{output_dir}/chunk_{subseg_idx}.wav", audio_subseg, 16000)
                item['audio_path'] = os.path.abspath(f"{output_dir}/chunk_{subseg_idx}.wav")
                if vocal is not None:
                    vocal_subseg = vocal[int(cur_start*16000):int(cur_end*16000)]
                    write_audio(f"{output_dir}/chunk_{subseg_idx}_vocal.wav", vocal_subseg, 16000)
                    item['vocal_path'] = os.path.abspath(f"{output_dir}/chunk_{subseg_idx}_vocal.wav")
                if bgm is not None:
                    bgm_subseg = bgm[int(cur_start*16000):int(cur_end*16000)]
                    write_audio(f"{output_dir}/chunk_{subseg_idx}_bgm.wav", bgm_subseg, 16000)
                    item['bgm_path'] = os.path.abspath(f"{output_dir}/chunk_{subseg_idx}_bgm.wav")
            else:
                item['wav'] = audio_subseg

            item['start'] = cur_start
            item['end'] = cur_end
        
        # 保存为JSON文件
        if save_audio:
            with open(f"{output_dir}/audio.json", "w") as f:
                json.dump(output_segments, f, indent=4, ensure_ascii=False)
        
        return output_segments
    
    
    def transcribe_and_visualize(self, audio_path, output_dir=None):
        """
        转录并可视化音频文件
        
        参数:
            audio_path (str): 音频文件路径
            output_dir (str): 输出目录，默认为根据音频文件名自动生成
            
        返回:
            tuple: (segments, output_segments) 转录的segments列表和处理后的音频片段信息列表
        """
        # 读取音频
        audio = read_audio(audio_path)
        
        # 设置输出目录
        if output_dir is None:
            output_dir = "output_whisper_{}".format(audio_path.split("/")[-1].split(".")[0])
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 从缓存加载或重新转录
        if os.path.exists(f"{output_dir}/segments.pt"):
            segments = torch.load(f"{output_dir}/segments.pt")
        else:
            segments = self.transcribe(audio, return_list=False)
            # 保存到缓存
            torch.save(segments, f"{output_dir}/segments.pt")
        
        # 保存并可视化
        output_segments = self.save_visualize(audio, segments, output_dir, save_audio=True)
        
        return segments, output_segments


# 使用示例
if __name__ == "__main__":
    # 初始化ASR推理类，可以在这里设置所有超参数
    asr = WhisperASRInference(
        model_size="turbo", 
        device="cuda", 
        compute_type="float16",
        batch_size=16,
        vad_filter=True,
        word_timestamps=True,
        min_sentence_end_chars=[".", "!", "?"],
        context_padding=0.1
    )


    
    # 转录并可视化音频，不需要再传递超参数
    # audio_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo/2.mp3"
    # audio_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo/1_89.mp3"
    # audio_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo2/383137.mp3"
    audio_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo/camilia_interview.mp3"
    output_dir = 'exp/asr_inference/viz/'
    # segments = asr.transcribe(audio_path, return_list=True)
    # import pdb; pdb.set_trace()
    segments, output_segments = asr.transcribe_and_visualize(audio_path, output_dir=output_dir)
    print(f"已处理 {len(output_segments)} 个音频片段") 