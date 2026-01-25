from fireredasr.models.fireredasr import FireRedAsr
from audioautoprep.models.utils import read_audio, calculate_wer
import os
import torch
from tqdm import tqdm
from funasr import AutoModel
import re
# from audioautoprep.evaluation.utils import calculate_wer 


def parser_text(text):
    # 在中英文之间添加空格
    text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)
    text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
    
    return text

class FireredASRInference:
    def __init__(self, 
                 model_name="aed", 
                 model_path="pretrained_models/FireRedASR-AED-L", 
                 device="cuda", 
                 batch_size=8,
                 use_punc_model=False,
                 max_batch_duration=180,
                 max_single_duration=90,
                 punc_model_name="ct-punc",
                 ):
        self.model = FireRedAsr.from_pretrained(model_name, model_path)
        self.device = device
        self.batch_size = batch_size
        self.use_punc_model = use_punc_model
        self.max_batch_duration = max_batch_duration
        self.max_single_duration = max_single_duration

        if use_punc_model:
            self.punc_model = AutoModel(model=punc_model_name, model_revision="v2.0.4", disable_pbar=True, disable_update=True)

    

    def transcribe(self, wav_list, audio=None, return_wav_list=False):
        # 准备音频数据和时长信息
        audio_list = []
        duration_list = []
        
        for idx, item in enumerate(wav_list):
            if isinstance(item, str):
                audio_data = read_audio(item)
                audio_list.append(audio_data)
                # 计算音频时长（秒）
                duration = len(audio_data) / 16000
                duration_list.append(duration)
            elif isinstance(item, dict):
                st, ed = int(16000 * item['start_ori']), int(16000 * item['end_ori'])
                audio_data = audio[st:ed]
                audio_list.append(audio_data)
                # 计算音频时长（秒）
                duration = (item['end_ori'] - item['start_ori'])
                duration_list.append(duration)
            else:
                raise ValueError(f"不支持的输入类型: {type(item)}")

        # 按时长分批处理
        all_results = []
        total_samples = len(audio_list)
        
        # 初始化结果列表
        all_results = [None] * total_samples
        
        batch_indices = []
        current_batch = []
        current_batch_duration = 0
        # max_batch_duration = 2 * 60  # 8分钟转换为秒
        # max_single_duration = 1.5 * 60  # 2分钟转换为秒
        
        for i in range(total_samples):
            single_duration = duration_list[i]
            
            # 如果单个片段超过2分钟，直接标记为NA
            if single_duration > self.max_single_duration:
                all_results[i] = {'text': 'NA'}
                continue
            
            # 检查加入当前片段后是否超过batch时长限制
            if current_batch_duration + single_duration > self.max_batch_duration and current_batch:
                # 当前batch已满，开始新batch
                batch_indices.append(current_batch)
                current_batch = [i]
                current_batch_duration = single_duration
            else:
                # 加入当前batch
                current_batch.append(i)
                current_batch_duration += single_duration
        
        # 处理最后一个batch
        if current_batch:
            batch_indices.append(current_batch)
        
        # 按批次处理
        for batch_idx_list in tqdm(batch_indices, desc="处理批次"):
            # 当前批次的数据
            batch_audio_list = [audio_list[i] for i in batch_idx_list]
            batch_uttid = batch_idx_list
            
            # 推理当前批次
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    batch_results = self.model.transcribe(
                        batch_uttid,
                        batch_audio_list,
                        {
                            "use_gpu": 1,
                            "beam_size": 3,
                            "nbest": 1,
                            "decode_max_len": 0,
                            "softmax_smoothing": 1.25,
                            "aed_length_penalty": 0.6,
                            "eos_penalty": 1.0
                        }
                    )
            
            # 收集当前批次的结果，按原始索引顺序存储
            for i, result in zip(batch_idx_list, batch_results):
                all_results[i] = result

        assert len(all_results) == len(wav_list)
        for i, item in enumerate(wav_list):
            if isinstance(item, dict):
                # import pdb; pdb.set_trace()
                item['firered_asr_text_ori'] = all_results[i]['text']
                if self.use_punc_model and all_results[i]['text'] != 'NA':
                    # item['firered_asr_text'] = self.punc_model.generate(input=item['firered_asr_text_ori'].replace(" ", ""))[0]['text']
                    item['firered_asr_text'] = self.punc_model.generate(input=parser_text(item['firered_asr_text_ori']).lower())[0]['text']
                else:
                    item['firered_asr_text'] = item['firered_asr_text_ori']
                try:
                    if all_results[i]['text'] != 'NA':
                        item['firered_asr_wer_info'] = calculate_wer(item['firered_asr_text'].strip(), item['text'].strip())
                    else:
                        item['firered_asr_wer_info'] = None
                except Exception as e:
                    item['firered_asr_wer_info'] = None
        if return_wav_list:
            return wav_list
        else:
            return all_results

if __name__ == "__main__":
     pt_model = AutoModel(model="ct-punc", model_revision="v2.0.4", disable_pbar=True, disable_update=True)
     input_text = "对第一支是 VIOLENT MAGIC ULTRA 就 VMO 当时我是拍了一张非常好看的照片也是他们那个女的合嗓的主唱是啊腿是被架起来吧就有点攻壳机动队的会"
     print(pt_model.generate(input=parser_text(input_text).lower()))
