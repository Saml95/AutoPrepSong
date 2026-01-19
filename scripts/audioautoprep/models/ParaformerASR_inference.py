from funasr import AutoModel
import json
import librosa
import re

from audioautoprep.models.utils import read_audio, write_audio
from faster_whisper import WhisperModel, BatchedInferencePipeline
import faster_whisper
from audioautoprep.models.utils import read_audio, write_audio
import os
import torch
import json
from tqdm import tqdm
import numpy as np

'''
{'start_ori': 0.0, 'end_ori': 0.66, 'text': ' Hey, everyone.', 'text_timestamp': [(0.0, 0.36, ' Hey,'), (0.58, 0.66, ' everyone.')], 'wav': array([ 0.0000000e+00, -3.0517578e-05,  0.0000000e+00, ...,
        4.6514893e-01,  4.2932129e-01,  3.5720825e-01], dtype=float32), 'start': 0, 'end': 0.66}
'''


class ParaformerASRInference:
    def __init__(self, 
                 asr_model="paraformer-zh", 
                 vad_model="fsmn-vad", 
                 punc_model="ct-punc", 
                 spk_model="cam++", 
                 disable_pbar=True,
                 load_model=True,
                 device="cuda",
                 context_padding=0.1,
                 ):

        if load_model:  
            self.model = AutoModel(
                        model=asr_model,  
                        vad_model=vad_model,  
                        punc_model=punc_model, 
                        spk_model=spk_model, 
                        disable_pbar=disable_pbar,
                        device=device,
                        disable_update=True
                        
                  )
        else:
            self.model = None
        self.sr = 16000
        self.context_padding = context_padding
        self.device = device
        # self.model.to(self.device)
    def parser_text(self, text):
        '''
        输入: 如果我 have 一个 banana .
        如果英文和汉字中间没有空格，先加空格 
        变成： 如果我 have 一个 banana.
        进一步
        变成一个列表, 每个汉字为一个元素，每个英文单词为一个元素，如果英文和汉字中间没有空格，先加空格
        ['如', '果', '我', 'have', '一', '个', 'banana.']
        '''
        # 定义中英文标点符号
        chinese_punctuation = '，。！？、；：""''（）【】《》'
        # english_punctuation = ',.!?;:"\'()[]{}<>'
        english_punctuation = ',.!?;:"()[]{}<>'
        all_punctuation = chinese_punctuation + english_punctuation
        
        # 去掉标点符号周围的空格
        text = re.sub(r'\s+([{}])'.format(re.escape(all_punctuation)), r'\1', text)  # 去掉标点前的空格
        text = re.sub(r'([{}])\s+'.format(re.escape(all_punctuation)), r'\1', text)  # 去掉标点后的空格
        
        # 在中英文之间添加空格
        text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)
        text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
        
        # 将文本分割成单词
        words = text.split()
        
        # 将每个汉字单独分开，标点优先跟随后面的字符
        result = []
        for word in words:
            if re.match(r'^[\u4e00-\u9fff{}]+$'.format(re.escape(all_punctuation)), word):  # 如果是中文或标点
                chars = list(word)
                # 处理标点符号
                processed_chars = []
                i = 0
                while i < len(chars):
                    if chars[i] in all_punctuation:  # 如果是标点
                        if i + 1 < len(chars):  # 如果后面有字符，将标点附加到后面的字符
                            processed_chars.append(chars[i] + chars[i + 1])
                            i += 2
                        elif processed_chars:  # 如果后面没有字符但前面有，将标点附加到前一个字符
                            processed_chars[-1] = processed_chars[-1] + chars[i]
                            i += 1
                        else:  # 如果前后都没有字符，将标点作为独立元素
                            processed_chars.append(chars[i])
                            i += 1
                    else:
                        processed_chars.append(chars[i])
                        i += 1
                result.extend(processed_chars)
            else:  # 如果是英文单词
                result.append(word)
                
        return result
        
    def transcribe(self, wav, return_list=False, vocal=None, use_vocal=False):
        # import pdb; pdb.set_trace()
        res = self.model.generate(
                    input=wav, 
                    batch_size_s=1200,                
                    )
        if vocal is not None and use_vocal:
            print(f"使用增强后的音频进行diarization!!!!")
            segments = self.reformulate_res(res[0], vocal)
        else:
            segments = self.reformulate_res(res[0], wav)
        return segments, None, None
        

    def reformulate_res(self, res, wav):
        segments = []
        # import pdb; pdb.set_trace()
        for item in res['sentence_info']:
            text_list = self.parser_text(item['text'])
            try:
                assert len(text_list) == len(item['timestamp'])
            except:
                raise ValueError(f"text_list: {text_list} != item['timestamp']: {item['timestamp']}")
            text_timestamp = []
            for char, timestamp in zip(text_list, item['timestamp']):
                text_timestamp.append((timestamp[0]/1000., timestamp[1]/1000., char))
            segments.append({
                'text': item['text'],
                'text_timestamp': text_timestamp,
                'wav': wav[int(item['start'] * self.sr/1000.):int(item['end'] * self.sr/1000.)],
                'start': item['start'] / 1000.,
                'end': item['end'] / 1000.,
                'start_ori': item['start'] / 1000.,
                'end_ori': item['end'] / 1000.,
            })
        return segments
    
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



if __name__ == "__main__":
    # test_unit_list = ['parser_text', 'process']
    test_unit_list = ['process']
    wav_path = "/mnt/conversationhubhot/yaoyaochang/speech/data/xyz/audio_all/5e280b1f418a84a0461f2654.mp3"
    
    if 'parser_text' in test_unit_list:
        model = ParaformerASRInference(load_model=False)
        text = "如果我 have 一个 banana ."
        text_list = model.parser_text(text)
        import pdb; pdb.set_trace()
    if 'process' in test_unit_list:
        wav, sr = librosa.load(wav_path, sr=16000)
        model = ParaformerASRInference(load_model=True)
        segments = model.process(wav)
        import pdb; pdb.set_trace()
