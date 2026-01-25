from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import torch.nn.functional as F
from audioautoprep.models.utils import read_audio


class CustomerSV:
    def __init__(self, 
                model_path='microsoft/wavlm-base-plus-sv', 
                use_local_files=True, device="cuda",
                batch_size=16
                ):
        """
        初始化CustomerSV类
        
        Args:
            model_path: 模型路径，默认使用'microsoft/wavlm-base-plus-sv'
            use_local_files: 是否使用本地文件
            device: 设备，默认为"cuda"
            batch_size: 批处理大小，默认为16
        """
        self.sample_rate = 16000
        self.device = device
        self.batch_size = batch_size
        if use_local_files:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                local_files_only=True,
                pretrained_model_name_or_path=model_path
            )
            self.model = WavLMForXVector.from_pretrained(
                local_files_only=True,
                pretrained_model_name_or_path=model_path
            )
        else:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            self.model = WavLMForXVector.from_pretrained(model_path)
        self.model.to(device)
    
    def process(self, audio_list, audio=None, prompt=None):
        """
        批处理音频列表并返回嵌入向量
        
        Args:
            audio_list: 列表，每个元素是包含音频段信息的字典
            audio: 完整音频数据
            
        Returns:
            audio_list: 更新后的音频列表，每个元素中添加了嵌入向量
        """
        wav_list = []
        if prompt is not None:
            for item in prompt:
                wav_list.append(read_audio(item['audio_path'], sr=self.sample_rate))
            embeddings = self.compute_embedding(wav_list)
            assert len(embeddings) == len(prompt), "嵌入向量数量与prompt数量不一致"
            for i, item in enumerate(prompt):
                item['simo_embedding'] = embeddings[i]

        # 为每个项目准备音频数据
        for item in audio_list:
            if 'wav' not in item and audio is not None:
                item['wav'] = audio[int(item['start_ori'] * self.sample_rate): int(item['end_ori'] * self.sample_rate)]
        
        # 批处理
        total_items = len(audio_list)
        processed_items = []
        
        for i in range(0, total_items, self.batch_size):
            batch = audio_list[i:i+self.batch_size]
            batch_processed = self._process_batch(batch)
            processed_items.extend(batch_processed)


        # Compute similarity
        for idx, item in enumerate(audio_list):
            gen_embedding = item['simo_embedding']
            prompt_idx = idx % len(prompt)
            prompt_embedding = prompt[prompt_idx]['simo_embedding']
            item['simo_similarity'] = self.compute_similarity(gen_embedding, prompt_embedding)

        average_similarity = sum(item['simo_similarity'] for item in audio_list) / len(audio_list)
        
        # return audio_list, average_similarity
        return average_similarity
    
    def _process_batch(self, batch):
        """
        处理单个批次的音频数据
        
        Args:
            batch: 音频项列表
            
        Returns:
            batch: 添加了嵌入向量的批次
        """
        # 提取音频数据
        wav_list = [item['wav'] for item in batch]
        
        # 准备输入
        with torch.no_grad():
            inputs = self.feature_extractor(wav_list, padding=True, return_tensors="pt")
            
            # 将输入移至指定设备
            for key, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    inputs[key] = tensor.to(self.device)
            
            # 获取嵌入向量
            embeddings = self.model(**inputs).embeddings
            
            # 归一化嵌入向量
            embeddings = F.normalize(embeddings, dim=-1).cpu()
        
        # 确保嵌入向量数量与音频数量一致
        assert len(embeddings) == len(batch), "嵌入向量数量与音频批次数量不一致"
        
        # 将嵌入向量添加到每个项目中
        for i, item in enumerate(batch):
            item['simo_embedding'] = embeddings[i]
        
        return batch
    
    def compute_embedding(self, wav_list):
        inputs = self.feature_extractor(wav_list, padding=True, return_tensors="pt")
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                inputs[key] = tensor.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
            embeddings = F.normalize(embeddings, dim=-1).cpu()
        return embeddings


    def compute_similarity(self, emb1, emb2):
        """
        计算两个嵌入向量之间的余弦相似度
        
        Args:
            emb1: 第一个嵌入向量
            emb2: 第二个嵌入向量
            
        Returns:
            similarity: 相似度得分
        """
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(emb1, emb2)        
        return similarity.item()


# 使用示例
if __name__ == "__main__":
    from datasets import load_dataset
    import numpy as np
    
    # 加载测试数据
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
    
    # 准备完整音频
    full_audio = dataset[0]["audio"]["array"]
    
    # 创建测试段
    segment_length = 1.0  # 每段1秒
    sample_rate = 16000
    segments = []
    for i in range(5):  # 创建5个测试段
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        segments.append({
            "start_ori": start_time,
            "end_ori": end_time
        })
    
    # 初始化模型
    sv_model = CustomerSV(batch_size=2)  # 使用较小的批次大小进行测试
    
    # 处理音频段并获取嵌入向量
    processed_segments = sv_model.process(segments, full_audio)
    
    # 计算第一个段和第二个段之间的相似度
    similarity = sv_model.compute_similarity(
        processed_segments[0]['simo_embedding'], 
        processed_segments[1]['simo_embedding']
    )
    
    print(f"段1和段2的相似度: {similarity}")