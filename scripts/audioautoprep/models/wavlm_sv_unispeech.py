from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import torch.nn.functional as F
from audioautoprep.models.utils import read_audio
from audioautoprep.evaluation.seed_tts_eval.thirdparty.UniSpeech.downstreams.speaker_verification.models.ecapa_tdnn import ECAPA_TDNN_SMALL
import soundfile as sf
import librosa
from torchaudio.transforms import Resample
import numpy as np


class CustomerSV:
    def __init__(self, 
                model_name='wavlm_large', 
                model_path="/mnt/conversationhub/jianweiyu/DataPropc/AudioAutoPrepV2/ckpts/wavlm/wavlm_large_finetune.pth", 
                device="cuda"
                ):
        """
        初始化CustomerSV类
        
        Args:
            model_name: 模型名称，默认为'wavlm_large'
            checkpoint: 模型检查点路径
            device: 设备，默认为"cuda"
        """
        self.sample_rate = 16000
        self.device = device
        self.model_name = model_name
        self.model = self.init_model(model_name, model_path)
        self.model.eval()
        self.model.to(device)
    
    def init_model(self, model_name, checkpoint=None):
        """
        初始化模型
        
        Args:
            model_name: 模型名称
            checkpoint: 模型检查点路径
            
        Returns:
            model: 初始化后的模型
        """

        if model_name == 'unispeech_sat':
            config_path = 'config/unispeech_sat.th'
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
        elif model_name == 'wavlm_base_plus':
            config_path = None
            model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
        elif model_name == 'wavlm_large':
            config_path = None
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
        elif model_name == 'hubert_large':
            config_path = None
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
        elif model_name == 'wav2vec2_xlsr':
            config_path = None
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
        else:
            model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict['model'], strict=False)
        return model
    
    def process(self, audio_list, audio=None, prompt=None):
        """
        处理音频列表并返回平均相似度
        
        Args:
            audio_list: 列表，每个元素是包含音频段信息的字典
            audio: 完整音频数据
            prompt: 提示音频列表
            
        Returns:
            average_similarity: 平均相似度
        """
        # 处理提示音频
        prompt_embeddings = []
        if prompt is not None:
            for item in prompt:
                if isinstance(item['audio_path'], str):
                    wav = read_audio(item['audio_path'], sr=self.sample_rate)
                else:
                    wav = item['audio_path']
                embedding = self.compute_embedding(wav)
                prompt_embeddings.append(embedding)

        # 处理音频列表
        similarities = []
        for item in audio_list:
            # 准备音频数据
            if 'wav' not in item and audio is not None:
                item['wav'] = audio[int(item['start_ori'] * self.sample_rate): int(item['end_ori'] * self.sample_rate)]
            
            # 计算嵌入向量
            wav = item['wav']

            # 计算嵌入向量

            embedding = self.compute_embedding(wav)
            # chunk_size = 16000 * 10
            # chunk_embeddings = []
            # for i in range(0, len(wav), chunk_size):
            #     chunk = wav[i:i + chunk_size]
            #     chunk_embedding = self.compute_embedding(chunk)
            #     chunk_embeddings.append(chunk_embedding)

            # # embedding = self.compute_embedding(wav)
            # embedding = torch.mean(torch.cat(chunk_embeddings, dim=0), dim=0, keepdim=True)
            item['simo_embedding'] = embedding
            
            # 计算与提示音频的相似度
            if prompt is not None:
                if embedding is None:
                    similarity = 0
                else:
                    prompt_idx = len(similarities) % len(prompt)
                    similarity = []
                    for prompt_embedding in prompt_embeddings:
                        similarity.append(self.compute_similarity(embedding, prompt_embedding))
                    similarity = max(similarity)

                    similarities.append(similarity)
                    # prompt_embedding = prompt_embeddings[prompt_idx]
                    # similarity = self.compute_similarity(embedding, prompt_embedding)
                item['simo_similarity'] = similarity
                # similarities.append(similarity)

        # 计算平均相似度
        if similarities:
            average_similarity = sum(similarities) / len(similarities)
            return average_similarity, similarities
        else:
            return None
    
    def compute_embedding(self, wav):
        """
        计算单个音频的嵌入向量
        
        Args:
            wav: 音频数据
            
        Returns:
            embedding: 嵌入向量
        """
        # 检查音频长度是否超过10秒、
        if isinstance(wav, str):
            wav = read_audio(wav, sr=self.sample_rate)

        
        max_duration = 10.0  # 最大处理时长（秒）
        max_samples = int(max_duration * self.sample_rate)
        min_segment_duration = 3.0  # 最小片段时长（秒）
        min_segment_samples = int(min_segment_duration * self.sample_rate)
        
        if len(wav) > max_samples:
            # 先正常分段
            segments = []
            for i in range(0, len(wav), max_samples):
                segment = wav[i:i + max_samples]
                if len(segment) > 0:  # 确保片段不为空
                    segments.append(segment)
            
            # 检查最后一个片段，如果剩余部分小于最小片段时长，则拼接到最后一个片段
            if len(segments) > 1:  # 确保有多个片段
                last_segment = segments[-1]
                if len(last_segment) < min_segment_samples:
                    # 将最后一个片段拼接到倒数第二个片段
                    segments[-2] = np.concatenate([segments[-2], last_segment])
                    segments.pop()  # 移除最后一个片段
            
            # 计算每个片段的嵌入向量
            embeddings = []
            for segment in segments:
                # 转换为torch张量
                wav_tensor = torch.from_numpy(segment).unsqueeze(0).float()
                
                # 移动到设备
                wav_tensor = wav_tensor.to(self.device)
                
                # 获取嵌入向量
                self.model.eval()
                with torch.no_grad():
                    try:
                        emb = self.model(wav_tensor)
                        # emb = F.normalize(emb, dim=-1).cpu()
                        embeddings.append(emb.cpu())
                    except Exception as e:
                        import pdb; pdb.set_trace()
                torch.cuda.empty_cache()
            
            # 计算平均嵌入向量
            if embeddings:
                avg_embedding = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)
                # 再次归一化
                avg_embedding = F.normalize(avg_embedding, dim=-1)
                return avg_embedding
            else:
                return None
        else:
            # 转换为torch张量
            wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()
            
            # 移动到设备
            wav_tensor = wav_tensor.to(self.device)
            
            # 获取嵌入向量
            self.model.eval()
            with torch.no_grad():
                try:
                    emb = self.model(wav_tensor)
                    emb = F.normalize(emb, dim=-1).cpu()
                except Exception as e:
                    # import pdb; pdb.set_trace()
                    return None
            
            return emb

    def compute_similarity(self, emb1, emb2):
        """
        计算两个嵌入向量之间的余弦相似度
        
        Args:
            emb1: 第一个嵌入向量
            emb2: 第二个嵌入向量
            
        Returns:
            similarity: 相似度得分
        """
        sim = F.cosine_similarity(emb1, emb2)
        return sim[0].item()

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    sv_model = CustomerSV()

    wav_path1 = "/mnt/conversationhub/zhiliang/exp/podcast_eval/select_mosset/prompts/AzureAva_long_v1.wav"
    wav_path2 = "/mnt/conversationhub/zhiliang/exp/podcast_eval/select_mosset/prompts/AzureAndrew_long_v2.wav"
    wav_path3 = "/mnt/conversationhub/zhiliang/exp/podcast_eval/select_mosset/prompts/AzureAva.wav"

    emb1 = sv_model.compute_embedding(wav_path1)
    emb2 = sv_model.compute_embedding(wav_path2)
    emb3 = sv_model.compute_embedding(wav_path3)

    sim12 = sv_model.compute_similarity(emb1, emb2)
    sim13= sv_model.compute_similarity(emb1, emb3)
    sim23 = sv_model.compute_similarity(emb2, emb3)

    print(f"sim12: {sim12}")
    print(f"sim13: {sim13}")
    print(f"sim23: {sim23}")
    
