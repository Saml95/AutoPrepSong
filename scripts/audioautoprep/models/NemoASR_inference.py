import nemo.collections.asr as nemo_asr
from audioautoprep.models.utils import read_audio, calculate_wer

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
import os
import torch
class NemoASRInference:
    def __init__(self, model_name="nvidia/parakeet-tdt-1.1b", device="cuda", batch_size=32, verbose=False):

        """
        初始化 NemoASR 推理类
        
        参数:
            model_name: 预训练模型名称
        """
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose  
        # import pdb; pdb.set_trace()
        if os.path.exists(model_name):
            print(f"Loading NemoASR model from local path {model_name}")
            from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
            # save_restore_connector = SaveRestoreConnector()
            save_restore_connector = None
            self.asr_model = EncDecRNNTBPEModel.restore_from(
                restore_path=model_name,
                map_location=device,
                strict=True,
                return_config=False,
                trainer=None,
                save_restore_connector=save_restore_connector,
            )
        else:
            print(f"Loading NemoASR model from hub {model_name}")
            self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=model_name, map_location=device)
    
    def transcribe(self, wav_list, audio = None, return_wav_list = False):
        """
        转录音频文件
        
        参数:
            audio_paths: 音频文件路径列表
            
        返回:
            转录结果
        """

        audio_list = []
        for item in wav_list:
            if isinstance(item, str):
                audio_list.append(read_audio(item))
            elif isinstance(item, dict):
                st, ed = int(16000 * item['start_ori']), int(16000 * item['end_ori'])
                audio_list.append(audio[st:ed])
            else:
                raise ValueError(f"不支持的输入类型: {type(item)}")

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = self.asr_model.transcribe(audio_list, batch_size=self.batch_size, verbose=self.verbose)
        # import pdb; pdb.set_trace()
        assert len(output) == len(wav_list)
        for i, item in enumerate(wav_list):
            if isinstance(item, dict):
                item['nemo_asr_text'] = output[i].text
                try:
                    item['nemo_asr_wer_info'] = calculate_wer(item['nemo_asr_text'].strip(), item['text'].strip())
                except Exception as e:
                    item['nemo_asr_wer_info'] = None
        if return_wav_list:
            return wav_list
        else:
            return output

# 使用示例
if __name__ == "__main__":
    import os

    # 初始化模型
    asr_inference = NemoASRInference()
    
    # 转录音频
    data_dir = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/exp/example_output/camilia_interview/"
    audio_path_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]
    # 按照音频文件名中的数字排序
    audio_path_list = sorted(audio_path_list, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    print(f"共找到 {len(audio_path_list)} 个音频文件")
    output = asr_inference.transcribe(audio_path_list)
    
    # 打印结果
    print(output[0].text)
