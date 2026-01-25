import torch
import torchaudio.compliance.kaldi as kaldi
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import wespeaker
from wespeaker.diar.umap_clusterer import cluster as umap_cluster
from wespeaker.diar.spectral_clusterer import cluster as spectral_cluster
from tqdm import tqdm
import os
import copy
import torch.nn.functional as F
from copy import deepcopy
class CustomBaseDiarization:
    def __init__(self, segment_length=1.5, segment_shift=0.75, sample_rate=16000):
        self.segment_length = segment_length
        self.segment_shift = segment_shift
        self.sample_rate = sample_rate

    def diarize(self, audio_path):
        pass

    def pad_wav_to_multiple(self, wav):
         # 计算每个segment的样本数
        chunk_size = int(self.segment_length * self.sample_rate)
        # 计算需要填充的样本数，使wav的长度可以被chunk_size整除
        remainder = len(wav) % chunk_size
        if remainder != 0:
            padding_size = chunk_size - remainder
            if padding_size <= len(wav):
                # 如果padding_size小于或等于wav的长度，直接从末尾截取
                wav = np.concatenate([wav, wav[-padding_size:]])
            else:
                # 如果padding_size大于wav的长度，循环填充
                num_repeats = (padding_size // len(wav)) + 1
                repeated_wav = np.tile(wav, num_repeats)
                wav = np.concatenate([wav, repeated_wav[:padding_size]])
        return wav

    def segment_wav(self, wav):
        # 首先通过复制末尾使得wav长度可以被segment_length整除
        # import pdb; pdb.set_trace()
        wav = self.pad_wav_to_multiple(wav)
        
        chunk_size = int(self.segment_length * self.sample_rate)  # 每个segment的样本数
        shift_size = int(self.segment_shift * self.sample_rate)   # 窗移的样本数
        num_samples = wav.shape[-1]
        segments = []

        for start_idx in range(0, num_samples - chunk_size + 1, shift_size):
            wav_seg = torch.from_numpy(wav[start_idx:start_idx + chunk_size]).unsqueeze(0)
            segments.append(wav_seg)

        return segments
    

class CustomWespeakerDiarization(CustomBaseDiarization):
    def __init__(self, segment_length=1.5, segment_shift=0.75, sample_rate=16000, \
                 model_name='vblinkp_large', device='cuda', num_thread=4, batch_size=64, \
                 cluster_merge_threshold=0.67, segment_merge_threshold=0.67, max_segment_duration=60.0, use_chunk_cluster=True, chunk_cluster_merge_threshold=0.65, time_gap_threshold=1.0,
                 cluster_chunk_size=960, min_cluster_size=480):
        super().__init__(segment_length, segment_shift, sample_rate)

        self.model_name = model_name
        self.device = device
        if os.path.exists(model_name):
            print(f"Loading wespeaker model from local path {model_name}")
            self.wespeaker_model = wespeaker.cli.speaker.Speaker(model_name)
        else:
            print(f"Loading wespeaker model from hub {model_name}")
            self.wespeaker_model = wespeaker.load_model(model_name)
        self.wespeaker_model.set_device(device)
        self.num_thread=num_thread
        self.batch_size = batch_size
        self.cluster_merge_threshold = cluster_merge_threshold
        self.segment_merge_threshold = segment_merge_threshold
        self.max_segment_duration = max_segment_duration
        self.use_chunk_cluster = use_chunk_cluster
        self.chunk_cluster_merge_threshold = chunk_cluster_merge_threshold
        self.cluster_chunk_size = cluster_chunk_size
        self.min_cluster_size = min_cluster_size
        self.time_gap_threshold = time_gap_threshold
        assert self.cluster_chunk_size > self.min_cluster_size, "cluster_chunk_size must be greater than min_cluster_size"

    def compute_fbank_list(self, wav_list, cmn=False):
        def process_wav(wav_seg):
            wav_seg_list = self.segment_wav(wav_seg['wav'])
            feats_list = []
            for seg in wav_seg_list:
                feat = self.wespeaker_model.compute_fbank(seg, cmn=cmn)
                feats_list.append(feat)
            feats_list = torch.stack(feats_list)
            wav_seg['feats'] = feats_list
            return wav_seg
        with ThreadPoolExecutor(max_workers=min(self.num_thread, len(wav_list))) as executor:
            wav_list = list(executor.map(process_wav, wav_list))
        
        return wav_list

    def compute_embeddings(self, wav_list, audio=None):
        embeddings = [] 
        for wav in wav_list:
            # if audio is not None and not isinstance(wav_list[0], str):
            #     wav = audio[int(wav['start_ori'] * self.sample_rate): int(wav['end_ori'] * self.sample_rate)]
            #     embedding = self.wespeaker_model.extract_embedding(wav)
            # else:
            #     embedding = self.wespeaker_model.extract_embedding_from_pcm(torch.from_numpy(wav).unsqueeze(0), 16000)

            if isinstance(wav, str):
                embedding = self.wespeaker_model.extract_embedding(wav)
            else:
                embedding = self.wespeaker_model.extract_embedding_from_pcm(torch.from_numpy(wav).unsqueeze(0), 16000)

            embeddings.append(embedding)
        return embeddings
    
    def compute_neighbor_cos_similarity(self, embeddings):
        embeddings_prev = embeddings[:-1]  # 除最后一个外的所有嵌入向量
        embeddings_next = embeddings[1:]   # 除第一个外的所有嵌入向量
        
        # 对嵌入向量进行L2归一化
        embeddings_prev = embeddings_prev / np.linalg.norm(embeddings_prev, axis=1, keepdims=True)
        embeddings_next = embeddings_next / np.linalg.norm(embeddings_next, axis=1, keepdims=True)
        return np.sum(embeddings_prev * embeddings_next, axis=1)
    

    
    

    def merge_cluster_centers(self, embeddings, labels, merge_threshold=0.7):
        # 初始化合并标记
        need_merge = True
        if isinstance(labels, list):
            labels = np.array(labels)
        while need_merge:
            need_merge = False
            # 获取当前所有标签
            unique_labels = np.unique(labels)
            
            # 计算当前的聚类中心
            current_centers = {}
            for label in unique_labels:
                current_centers[label] = np.mean(embeddings[labels == label], axis=0)
            
            # 归一化聚类中心
            center_vectors = np.array(list(current_centers.values()))
            center_vectors = center_vectors / np.linalg.norm(center_vectors, axis=1, keepdims=True)
            
            # 计算相似度矩阵
            similarity_matrix = np.dot(center_vectors, center_vectors.T)
            
            # 检查是否需要合并
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    if similarity_matrix[i][j] > merge_threshold:
                        # 找到需要合并的标签
                        label1, label2 = unique_labels[i], unique_labels[j]
                        
                        labels[labels == label2] = label1
                        
                        need_merge = True
                        break
                if need_merge:
                    break
        return labels

    def get_cluster_centers(self, embeddings, labels, merge_threshold=0.67):

        labels = self.merge_cluster_centers(embeddings, labels, merge_threshold)
        cluster_centers = {}
        for label in np.unique(labels):
            cluster_centers[label] = np.mean(embeddings[labels == label], axis=0)

        # 计算每个聚类中心与其他聚类中心的余弦相似度
        center_similarities = {}
        center_vectors = np.array(list(cluster_centers.values()))
        
        # L2归一化
        center_vectors = center_vectors / np.linalg.norm(center_vectors, axis=1, keepdims=True)
        
        # 计算余弦相似度矩阵
        similarity_matrix = np.dot(center_vectors, center_vectors.T)

        # 将相似度矩阵四舍五入到三位小数
        similarity_matrix = np.round(similarity_matrix, decimals=3).tolist()
        
        # 为每个聚类中心存储与其他中心的相似度
        for i, label in enumerate(cluster_centers.keys()):
            similarities = {}
            for j, other_label in enumerate(cluster_centers.keys()):
                similarities[other_label] = similarity_matrix[i][j]
            center_similarities[label] = similarities
            
        # 将相似度信息添加到cluster_centers字典中
        for label in cluster_centers.keys():
            cluster_centers[label] = {
                'embedding': cluster_centers[label],
                'similarities': center_similarities[label]
            }
        return cluster_centers, similarity_matrix, labels

    def get_segment_labels(self, wav_list, embeddings, cluster_method=['umap', 'spectral'], num_spks=None):
        embedding_idx = 0
        seg_sv_embeddings = []
        for item in wav_list:
            num_feats = item['feats'].shape[0]
            item['sv_emb'] = embeddings[embedding_idx:embedding_idx + num_feats]
            item['seg_sv_emb'] = np.mean(embeddings[embedding_idx:embedding_idx + num_feats], axis=0)
            seg_sv_embeddings.append(item['seg_sv_emb'])
            embedding_idx += num_feats
        seg_sv_embeddings = np.array(seg_sv_embeddings)
        seg_sv_neighbor_cos_similarity = self.compute_neighbor_cos_similarity(seg_sv_embeddings)

        cluster_label_dict_seg = {}
        if 'umap' in cluster_method:
            seg_sv_labels_umap = umap_cluster(seg_sv_embeddings)
            cluster_centers_umap_seg, similarity_matrix_umap_seg, labels_umap_seg = self.get_cluster_centers(seg_sv_embeddings, seg_sv_labels_umap, merge_threshold=self.cluster_merge_threshold)
            cluster_label_dict_seg['umap'] = {
                'labels': labels_umap_seg,
                'cluster_centers': cluster_centers_umap_seg,
                'similarity_matrix': similarity_matrix_umap_seg
            }
        if 'spectral' in cluster_method:
            seg_sv_labels_spectral = spectral_cluster(seg_sv_embeddings, num_spks=num_spks)
            cluster_centers_spectral_seg, similarity_matrix_spectral_seg, labels_spectral_seg = self.get_cluster_centers(seg_sv_embeddings, seg_sv_labels_spectral, merge_threshold=self.cluster_merge_threshold)
            cluster_label_dict_seg['spectral'] = {
                'labels': labels_spectral_seg,
                'cluster_centers': cluster_centers_spectral_seg,
                'similarity_matrix': similarity_matrix_spectral_seg
            }

        # 计算每个 seg_sv_emb 和其对应聚类中心的相似度（使用矩阵乘法）
        for method in cluster_method:
            # 提取所有聚类中心的嵌入向量
            centers = []
            center_labels = []
            for label, center_info in cluster_label_dict_seg[method]['cluster_centers'].items():
                centers.append(center_info['embedding'])
                center_labels.append(label)
            
            # 将聚类中心转换为矩阵
            centers = np.array(centers)
            
            # 使用矩阵乘法计算所有片段与所有聚类中心的余弦相似度
            # 归一化嵌入向量
            norm_centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
            norm_embeddings = seg_sv_embeddings / np.linalg.norm(seg_sv_embeddings, axis=1, keepdims=True)
            
            # 计算余弦相似度矩阵
            similarity_matrix = np.dot(norm_embeddings, norm_centers.T)
            
            # 为每个片段分配相似度
            for i, item in enumerate(wav_list):
                # 获取当前片段的聚类标签
                label_idx = cluster_label_dict_seg[method]['labels'][i]
                # 找到标签对应的中心索引
                center_idx = center_labels.index(label_idx)
                # 保存当前片段与所有聚类中心的相似度
                all_center_similarities = {}
                for j, center_label in enumerate(center_labels):
                    all_center_similarities[str(int(center_label))] = float(similarity_matrix[i, j])
                
                # 将所有相似度添加到 wav_list 中
                item[f'{method}_all_center_similarities'] = all_center_similarities
                # 从相似度矩阵中获取相似度
                similarity = similarity_matrix[i, center_idx]
                
                # 将标签和相似度添加到 wav_list 中
                if similarity > 0.35 or (item['end_ori'] - item['start_ori']) < 1.0:
                    item[f'{method}_segment_labels'] = int(label_idx)
                    item[f'{method}_center_similarity'] = float(similarity)
                else:
                    item[f'{method}_segment_labels'] = str(label_idx)+'_OUTLIER'
                    item[f'{method}_center_similarity'] = float(similarity)


        return cluster_label_dict_seg, seg_sv_neighbor_cos_similarity

    def extract_embeddings(self, wav_list, subseg_cmn=True):
        wav_list = self.compute_fbank_list(wav_list, cmn=~subseg_cmn)

        feats_list = []
        for item in wav_list:
            feats_list.append(item['feats'])
        feats = torch.cat(feats_list, 0).float()
        
        if subseg_cmn:
            feats = feats - torch.mean(feats, axis=1, keepdims=True)
        
        embeddings = []
        for i in range(0, len(feats), self.batch_size):
            batch_feats = feats[i:i + self.batch_size].to(self.device)
            with torch.no_grad():
                batch_embedding = self.wespeaker_model.model(batch_feats).detach().cpu().numpy()
            embeddings.append(batch_embedding)
        embeddings = np.concatenate(embeddings, axis=0)

        return wav_list, embeddings
    

    def chunk_cluster(self, wav_list, embeddings, cluster_method=['umap', 'spectral'], num_spks=None):
        """
        对大规模嵌入向量进行分块聚类，并合并聚类中心
        
        参数:
            wav_list (list): 包含音频片段信息的列表
            embeddings (numpy.ndarray): 所有音频片段的嵌入向量
            cluster_method (list): 聚类方法列表，可以包含'umap'和'spectral'
            
        返回:
            dict: 包含聚类结果的字典
        """
        # 初始化结果字典
        cluster_label_dict = {}
        for method in cluster_method:
            cluster_label_dict[method] = {
                'labels': None,
                'cluster_centers': None,
                'similarity_matrix': None
            }
        
        # 步骤1: 直接在循环wav_list时完成分块，每块累积超过1000个特征
        chunks = []  # 存储每个chunk的起始和结束索引
        current_chunk_start_idx = 0
        current_chunk_start_item = 0
        current_chunk_size = 0
        embedding_idx = 0
        
        for i, item in enumerate(wav_list):
            num_feats = item['feats'].shape[0]
            
            # 如果当前块加上这个item会超过1000，且当前块不为空，则结束当前块
            if current_chunk_size + num_feats > self.cluster_chunk_size and current_chunk_size > 0:
                chunks.append({
                    'start_idx': current_chunk_start_idx,
                    'end_idx': embedding_idx,
                    'start_item': current_chunk_start_item,
                    'end_item': i,
                    'size': current_chunk_size
                })
                current_chunk_start_idx = embedding_idx
                current_chunk_start_item = i
                current_chunk_size = 0
            
            current_chunk_size += num_feats
            embedding_idx += num_feats
        
        # 添加最后一个块（如果有）
        if current_chunk_start_item < len(wav_list):
            chunks.append({
                'start_idx': current_chunk_start_idx,
                'end_idx': embedding_idx,
                'start_item': current_chunk_start_item,
                'end_item': len(wav_list),
                'size': current_chunk_size
            })
        
        # 步骤2: 如果最后一个chunk不足500，则与倒数第二个chunk合并
        if len(chunks) > 1:
            if chunks[-1]['size'] < self.min_cluster_size:
                chunks[-2]['end_idx'] = chunks[-1]['end_idx']
                chunks[-2]['end_item'] = chunks[-1]['end_item']
                chunks[-2]['size'] += chunks[-1]['size']
                chunks = chunks[:-1]
        
        # 对每个块进行聚类
        chunk_results = []
        # for chunk_idx, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        for chunk_idx, chunk in enumerate(chunks):
            # 直接使用索引范围获取嵌入向量
            start_idx = chunk['start_idx']
            end_idx = chunk['end_idx']
            chunk_embeddings = embeddings[start_idx:end_idx]
            
            # 对当前块进行聚类
            chunk_cluster_dict = {}
            for method in cluster_method:
                if method == 'umap':
                    labels = umap_cluster(chunk_embeddings)
                elif method == 'spectral':
                    labels = spectral_cluster(chunk_embeddings, num_spks=num_spks)
                else:
                    continue
                
                # 获取聚类中心和相似度矩阵
                cluster_centers, similarity_matrix, labels = self.get_cluster_centers(
                    chunk_embeddings, labels, merge_threshold=self.chunk_cluster_merge_threshold
                )
                
                chunk_cluster_dict[method] = {
                    'labels': labels,
                    'cluster_centers': cluster_centers,
                    'similarity_matrix': similarity_matrix,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
            
            chunk_results.append(chunk_cluster_dict)
        
        # 步骤3: 合并所有块的聚类中心
        for method in cluster_method:
            # 初始化全局聚类结果
            all_labels = np.array([])
            all_cluster_centers = {}
            label_offset = 0
            
            # 收集所有块的聚类结果
            for chunk_idx, chunk_result in enumerate(chunk_results):
                if method not in chunk_result:
                    continue
                
                # 获取当前块的标签和聚类中心
                chunk_labels = chunk_result[method]['labels']
                chunk_centers = chunk_result[method]['cluster_centers']
                start_idx = chunk_result[method]['start_idx']
                end_idx = chunk_result[method]['end_idx']
                
                # 应用偏移量到标签
                if isinstance(chunk_labels, list):
                    adjusted_labels = np.array(chunk_labels) + label_offset
                else:
                    adjusted_labels = chunk_labels + label_offset
                
                # 将调整后的标签添加到全局结果
                all_labels = np.concatenate([all_labels, adjusted_labels.astype(int)])
                
                # 将聚类中心添加到全局聚类中心，并应用偏移量
                for label, center in chunk_centers.items():
                    all_cluster_centers[label + label_offset] = center
                
                # 更新偏移量，为下一个块准备
                if chunk_centers:
                    label_offset += len(chunk_centers)
            
            assert len(all_labels) == embeddings.shape[0]

            # 获取最终的聚类中心和相似度矩阵
            cluster_centers, similarity_matrix, merged_labels = self.get_cluster_centers(
                embeddings, all_labels, merge_threshold=self.chunk_cluster_merge_threshold
            )
            merged_labels = merged_labels.astype(int)

            # 存储最终结果
            cluster_label_dict[method] = {
                'labels': merged_labels,
                'cluster_centers': cluster_centers,
                'similarity_matrix': similarity_matrix
            }
            
        
        return cluster_label_dict

    def diarize(self, wav_list, subseg_cmn=True, cluster_method=['umap', 'spectral'], vad_list=None, \
                merge_speaker=False, keep_wav=False, keep_seg_emb=False, num_spks=None, prompts=None, merge_segments=True, compute_chunk_cluster=False):

        ori_wav_list_length = len(wav_list)
        wav_list, embeddings = self.extract_embeddings(wav_list, subseg_cmn=subseg_cmn)

        if compute_chunk_cluster:
            if self.use_chunk_cluster and embeddings.shape[0] > 1500 and self.use_chunk_cluster:
                cluster_label_dict = self.chunk_cluster(wav_list, embeddings, cluster_method, num_spks=num_spks)
            else:
                cluster_label_dict = {}
                if 'umap' in cluster_method:    
                    labels_umap = umap_cluster(embeddings)
                    cluster_centers_umap, similarity_matrix_umap, labels_umap = self.get_cluster_centers(embeddings, labels_umap)
                    cluster_label_dict['umap'] = {
                        'labels': labels_umap,
                        'cluster_centers': cluster_centers_umap,
                        'similarity_matrix': similarity_matrix_umap
                    }
                if 'spectral' in cluster_method:
                    labels_spectral = spectral_cluster(embeddings, num_spks=num_spks)
                    cluster_centers_spectral, similarity_matrix_spectral, labels_spectral = self.get_cluster_centers(embeddings, labels_spectral)
                    cluster_label_dict['spectral'] = {
                        'labels': labels_spectral,
                        'cluster_centers': cluster_centers_spectral,
                        'similarity_matrix': similarity_matrix_spectral
                    }
        else:
            cluster_label_dict = {}

        # import pdb; pdb.set_trace()
        # compute segment labels
        cluster_label_dict_seg, seg_sv_neighbor_cos_similarity = self.get_segment_labels(wav_list, embeddings, cluster_method, num_spks=num_spks)
        
        # Compute_final_labels 
        embedding_idx = 0
        for seg_idx, item in enumerate(wav_list):
            num_feats = item['feats'].shape[0]
            # for cluster_method in cluster_label_dict_seg.keys():
            #     item['{}_chunk_labels'.format(cluster_method)] = cluster_label_dict[cluster_method]['labels'][embedding_idx:embedding_idx + num_feats].tolist()
            #     assert len(item['{}_chunk_labels'.format(cluster_method)]) == num_feats
                # item['{}_segment_labels'.format(cluster_method)] = cluster_label_dict_seg[cluster_method]['labels'][seg_idx].tolist()
            del item['feats']
            del item['sv_emb']
            if keep_seg_emb:
                item['seg_sv_emb'] = item['seg_sv_emb']
            else:
                del item['seg_sv_emb']
            if keep_wav:
                item['wav'] = item['wav'] 
            else:
                del item['wav']
            embedding_idx += num_feats

        # import pdb; pdb.set_trace()   
        diarz_info = {
            'cluster_label_dict': cluster_label_dict,
            'cluster_label_dict_seg': cluster_label_dict_seg,
            'seg_sv_neighbor_cos_similarity': seg_sv_neighbor_cos_similarity
        }

        assert len(wav_list) == ori_wav_list_length, "wav_list的长度应该与ori_wav_list_length相同"

        
        if merge_segments:
            wav_list_merged = self.merge_segments(deepcopy(wav_list), seg_sv_neighbor_cos_similarity, vad_list, cluster_method)
        else:
            wav_list_merged = deepcopy(wav_list)
            assert len(wav_list_merged) == len(wav_list), "如果不进行merge_segments，wav_list_merged的长度应该与wav_list相同"
        if merge_speaker:
            wav_list_merged = self.merged_speaker(wav_list_merged, cluster_method)

        # import pdb; pdb.set_trace()
        # embedding_idx = 0
        if prompts is not None:
            embedding_idx = 0
            similarities = []
            prompt_embeddings = self.compute_embeddings([item['audio_path'] for item in prompts])
            for idx, item in enumerate(wav_list_merged):
                seg_embedding = np.mean(embeddings[embedding_idx:embedding_idx + len(item['spectral_chunk_labels'])], axis=0)
                embedding_idx += len(item['spectral_chunk_labels'])
                prompt_idx = idx % len(prompts)
                # prompt_embedding = prompt_embeddings[prompt_idx]
                similarity = []
                for prompt_embedding in prompt_embeddings:
                    similarity.append(F.cosine_similarity(torch.from_numpy(seg_embedding)[None], prompt_embedding[None])[0].item())
                item['wespeaker_similarity'] = max(similarity)
                # item['wespeaker_similarity'] = self.wespeaker_model.compute_similarity(seg_embedding, prompt_embedding)
                # item['wespeaker_similarity'] = F.cosine_similarity(torch.from_numpy(seg_embedding)[None], prompt_embedding[None])[0].item()
                
                similarities.append(item['wespeaker_similarity'])
            average_similarity = sum(similarities) / len(similarities)
            assert embedding_idx == len(embeddings) , "embedding_idx应该等于embeddings的长度"

            return wav_list, wav_list_merged, diarz_info

        return wav_list, wav_list_merged, diarz_info

    def merge_segments(self, wav_list, seg_sv_neighbor_cos_similarity, vad_list, cluster_method=['umap', 'spectral']):
        wav_list_merged = []
        """
        合并相邻的相同说话人片段
        
        参数:
            wav_list (list): 包含音频片段信息的列表
            seg_sv_neighbor_cos_similarity (numpy.ndarray): 片段间的余弦相似度矩阵
            
        返回:
            list: 合并后的音频片段列表
        """
        # 如果列表为空，直接返回
        if not wav_list:
            return wav_list
        # 初始化合并后的列表，从第一个片段开始
        # import pdb; pdb.set_trace()
        # wav_list_merged = [copy.deepcopy(wav_list[0])]
        wav_list_merged = [wav_list[0]]
        
        # 遍历剩余片段
        for i in range(1, len(wav_list)):
            # current_segment = copy.deepcopy(wav_list[i])
            current_segment = wav_list[i]
            previous_segment = wav_list_merged[-1]
            
            # 检查是否满足合并条件:
            # 1. 相同的说话人标签
            # 2. 相似度大于阈值
            # 3. 时间间隔小于1.0秒
            # 4. 合并后的总长度不超过60秒
            assert 'umap' in cluster_method or 'spectral' in cluster_method, "cluster_method必须包含'umap'或'spectral'"
            
            if 'umap' in cluster_method:
                same_speaker_umap = (current_segment['umap_segment_labels'] == previous_segment['umap_segment_labels'])
            else:
                same_speaker_umap = True
            if 'spectral' in cluster_method:
                same_speaker_spectral = (current_segment['spectral_segment_labels'] == previous_segment['spectral_segment_labels'])
            else:
                same_speaker = True
            
            # 获取相似度
            similarity = seg_sv_neighbor_cos_similarity[i-1]
            
            # 计算时间间隔
            time_gap = current_segment['start_ori'] - previous_segment['end_ori']
            
            # 计算合并后的总长度
            merged_duration = current_segment['end_ori'] - previous_segment['start_ori']
            
            # 如果满足合并条件
            # if ((same_speaker_umap and same_speaker_spectral) and similarity > self.segment_merge_threshold and time_gap < 1.0 and merged_duration <= self.max_segment_duration)\
            #     or (
            #         (same_speaker_umap and same_speaker_spectral) and time_gap < 1.0 and similarity > 0.55 and (previous_segment['end_ori'] - previous_segment['start_ori'] < 0.5 or current_segment['end_ori'] - current_segment['start_ori'] < 0.5)
            #     ) or (
            #         time_gap < 1.0 and (previous_segment['end_ori'] - previous_segment['start_ori'] < 0.75 or current_segment['end_ori'] - current_segment['start_ori'] < 0.75)
            #     ):
            if ((same_speaker_umap or same_speaker_spectral) and (similarity > self.segment_merge_threshold) and time_gap < 1.0 and merged_duration <= self.max_segment_duration) \
                or (
                    time_gap < self.time_gap_threshold and (previous_segment['end_ori'] - previous_segment['start_ori'] < 0.75 or current_segment['end_ori'] - current_segment['start_ori'] < 0.75)
                ):
                # 合并文本
                previous_segment['text'] += " " + current_segment['text']
                
                # 合并时间戳
                if 'text_timestamp' in previous_segment and 'text_timestamp' in current_segment:
                    previous_segment['text_timestamp'].extend(current_segment['text_timestamp'])
                
                # 更新结束时间
                previous_segment['end'] = current_segment['end']
                previous_segment['end_ori'] = current_segment['end_ori']
                
                # 合并其他标签
                if 'umap_chunk_labels' in previous_segment and 'umap_chunk_labels' in current_segment:
                    previous_segment['umap_chunk_labels'] += current_segment['umap_chunk_labels']
                
                if 'spectral_chunk_labels' in previous_segment and 'spectral_chunk_labels' in current_segment:
                    previous_segment['spectral_chunk_labels'] += current_segment['spectral_chunk_labels']
            else:
                # 如果不满足合并条件，添加到合并列表中
                wav_list_merged.append(current_segment)

        # 检查并调整相邻片段之间的时间间隔

        # 计算每个片段的时长
        # 首尾扩增0.2秒
        # import pdb; pdb.set_trace()
        wav_list_merged[0]['start_ori'] = max(0, wav_list_merged[0]['start_ori'] - 0.2)
        wav_list_merged[-1]['end_ori'] = wav_list_merged[-1]['end_ori'] + 0.2

        for i in range(1, len(wav_list_merged)):
            current_segment = wav_list_merged[i]
            previous_segment = wav_list_merged[i-1]
            
            # 计算时间间隔
            time_gap = current_segment['start_ori'] - previous_segment['end_ori']

            # 检查previous_segment和current_segment之间的间隔是否在vad_list中有重叠
            # 如果在vad_list中有重叠，则将间隔平分；否则按照下面的规则处理
            has_overlap_with_vad = False
            
            # 这里假设vad_list是可用的，如果不可用，需要在类初始化或其他地方设置
            # if i >= 9:
            #     import pdb; pdb.set_trace()
            if vad_list is not None:
                for vad_segment in vad_list:
                    # 检查vad段是否与previous_segment和current_segment之间的间隔重叠
                    if (vad_segment[0] < previous_segment['end_ori'] and 
                        vad_segment[1] > current_segment['start_ori']):
                        has_overlap_with_vad = True
                        break
            
            # 如果与vad有重叠，则将间隔平分
            if has_overlap_with_vad and time_gap > 0.0:
                half_gap = time_gap / 2
                previous_segment['end_ori'] += half_gap 
                current_segment['start_ori'] -= half_gap
                # 这里提前返回，不执行下面的代码
            # 根据时间间隔大小调整起止时间
            elif time_gap > 1.0:
                # 如果间隔大于0.4秒，前一段结束时间向后延长0.2秒，当前段开始时间向前提前0.2秒
                previous_segment['end_ori'] += 0.5
                current_segment['start_ori'] -= 0.5
            elif time_gap >= 0.0:
                half_gap = time_gap / 2
                previous_segment['end_ori'] += half_gap
                current_segment['start_ori'] -= half_gap
            else:
                previous_segment['end_ori'] = current_segment['start_ori']
        
        return wav_list_merged
    
    def merged_speaker(self, segments_merged, cluster_method="spectral"):
        """
        合并说话人
        """
        segments_merged_speaker =[segments_merged[0]]
        for current_segment in segments_merged[1:]:
            previous_segment = segments_merged_speaker[-1]
            
            # import pdb; pdb.set_trace()
            if cluster_method == "spectral":
                same_speaker_umap = (str(current_segment['spectral_segment_labels']).split('_')[0] == str(previous_segment['spectral_segment_labels']).split('_')[0])
            
            if same_speaker_umap:
                previous_segment['text'] += " " + current_segment['text']
            
                if 'text_timestamp' in previous_segment and 'text_timestamp' in current_segment:
                    previous_segment['text_timestamp'].extend(current_segment['text_timestamp'])
                    
                # 更新结束时间
                previous_segment['end'] = current_segment['end']
                previous_segment['end_ori'] = current_segment['end_ori']
                
                # 合并其他标签
                if 'umap_chunk_labels' in previous_segment and 'umap_chunk_labels' in current_segment:
                    previous_segment['umap_chunk_labels'] += current_segment['umap_chunk_labels']
                
                if 'spectral_chunk_labels' in previous_segment and 'spectral_chunk_labels' in current_segment:
                    previous_segment['spectral_chunk_labels'] += current_segment['spectral_chunk_labels']
            else:
                segments_merged_speaker.append(current_segment)
        for seg in segments_merged_speaker:
            seg["speaker"] = str(seg["spectral_segment_labels"]).split('_')[0]

        return segments_merged_speaker
