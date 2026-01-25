"""
对歌词 JSON 进行 speaker diarization
只对 is_lyric=True 的 segments 进行 diarization
"""

import json
import os
import torch
from tqdm import tqdm

from audioautoprep.models.diarization import CustomWespeakerDiarization
from audioautoprep.models.utils import load_audio_use_ffmpeg


class LyricDiarization:

    def __init__(self):
        diarz_config = {
            "segment_length": 1.5,
            "segment_shift": 0.75,
            "sample_rate": 16000,
            "model_name": "/mnt/conversationhub/jianweiyu/DataPropc/AudioAutoPrepV2/ckpts/wespeaker/vblinkp_large",
            "num_thread": 4,
            "batch_size": 64,
            "cluster_merge_threshold": 0.70,
            "segment_merge_threshold": 0.60,
            "max_segment_duration": 40.0,
            "use_chunk_cluster": False,
            "chunk_cluster_merge_threshold": 0.65,
            "cluster_chunk_size": 960,
            "min_cluster_size": 480,
        }
        device = "cuda"
        self.diarization = CustomWespeakerDiarization(device=device, **diarz_config)
    
    def compute_similarity(self, e1, e2):
        cosine_score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
        cosine_score = cosine_score.item()
        return (cosine_score + 1.0) / 2  # normalize: [-1, 1] => [0, 1]

    def process(self, json_path, resume=False):
        output_json_path = json_path.replace('.json', '.diarized.json')
        assert json_path != output_json_path, "输入输出路径不能相同"
        
        if resume and os.path.exists(output_json_path):
            print(f"{output_json_path} 已存在，跳过处理")
            return
        
        # 从文件名提取歌手列表，确定 speaker 数量
        # 格式：XXX、YYY - SongName.mp3.merged.5s.json
        filename = json_path.split("/")[-1]
        singer_part = filename.split(" - ")[0] if " - " in filename else filename.split("-")[0]
        singer_list = [s.strip() for s in singer_part.split("、") if s.strip()]
        
        # if len(singer_list) <= 1:
        #     num_speakers = None
        # else:
        #     num_speakers = len(singer_list)
        num_speakers = None
        
        print(f"文件: {filename}, 歌手: {singer_list}, num_speakers: {num_speakers}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 使用 audio_vocal_path 读取音频
        audio_path = data.get('audio_vocal_path') or data.get('audio_path')
        if not audio_path or not os.path.exists(audio_path):
            print(f"音频文件不存在: {audio_path}, 跳过 {json_path}")
            return
        
        audio, sr = load_audio_use_ffmpeg(audio_path, resample=True, target_sr=16000)
        assert sr == 16000, "音频采样率不等于16000"
        
        all_segments = data.get('segments', [])
        
        # 只提取 is_lyric=True 的 segments 进行 diarization
        lyric_segments = []
        for idx, item in enumerate(all_segments):
            is_lyric = item.get('is_lyric', False)
            if not is_lyric:
                continue
            
            start = float(item.get('start', 0))
            end = float(item.get('end', 0))
            duration = end - start
            
            if duration < 0.3:
                continue
            
            # 创建 segment 字典用于 diarization
            segment = {
                'order': idx,  # 保存原始索引
                'start': start,
                'end': end,
                'start_ori': start,
                'end_ori': end,
                'wav': audio[int(start * sr):int(end * sr)],
            }
            
            # 如果片段太短，重复音频
            if duration < 1:
                repeat = int(1.5 / duration) + 1
                segment['wav'] = segment['wav'].repeat(int(repeat))
            
            lyric_segments.append(segment)
        
        if not lyric_segments:
            print(f"没有 is_lyric=True 的 segments: {json_path}")
            # 直接保存原始数据
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        
        # 执行 diarization
        segments_diarized, segments_merged, diarz_info = self.diarization.diarize(
            lyric_segments, merge_segments=False, num_spks=num_speakers
        )


        if max(len(diarz_info['cluster_label_dict_seg']['umap']['cluster_centers']), len(diarz_info['cluster_label_dict_seg']['umap']['cluster_centers'])) >= 5:
            if len(diarz_info['cluster_label_dict_seg']['umap']['cluster_centers']) <= len(diarz_info['cluster_label_dict_seg']['spectral']['cluster_centers']): 
                cluster_method = 'umap'
            else:
                cluster_method = 'spectral'
        else:
            if len(diarz_info['cluster_label_dict_seg']['umap']['cluster_centers']) >= len(diarz_info['cluster_label_dict_seg']['spectral']['cluster_centers']): 
                cluster_method = 'umap'
            else:
                cluster_method = 'spectral'
        
    

        # 将 diarization 结果转换为以 order 为 key 的字典
        diarized_dict = {item['order']: item for item in segments_diarized}
        # import pdb; pdb.set_trace()
        
        mapped_speaker_dict = {}
        # 更新原始 segments 的 speaker 字段
        for idx, seg in enumerate(all_segments):
            if idx in diarized_dict:
                # 获取 speaker label，去掉 _OUTLIER 后缀
                label = str(diarized_dict[idx].get(f'{cluster_method}_segment_labels', ''))
                label = int(label.replace('_OUTLIER', ''))
                if label not in mapped_speaker_dict:
                    mapped_speaker_dict[label] = len(mapped_speaker_dict)
                seg['speaker'] = mapped_speaker_dict[label]
            else:
                seg['speaker'] = None

                
            # 其他 segments (is_lyric=False) 保持原样的 speaker
        
        # 打印 speaker 分布
        speakers = [seg.get('speaker') for seg in all_segments if seg.get('is_lyric', False)]
        print(f"Speakers: {speakers}")
        
        # 保存结果
        data['segments'] = all_segments
        data['diarization_info'] = {
            'selected_cluster_method': cluster_method,
            'lyric_segments_count': len(lyric_segments),
            'diarized_count': len(segments_diarized),
            'singer_list': singer_list,
            'num_speakers': num_speakers,
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存: {output_json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对歌词 JSON 进行 speaker diarization")
    parser.add_argument('--job_id', type=int, default=0, help='Job ID (0-indexed)')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of jobs')
    parser.add_argument('--json_scp', type=str, 
                        default='/home/jianweiyu/exp/AutoPrepSongV2/example/example.scp',
                        help='Path to scp file')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Skip if output file exists')
    args = parser.parse_args()
    
    with open(args.json_scp, 'r', encoding='utf-8') as f:
        json_list = [line.strip() for line in f if line.strip()]
    
    # 均分成 world_size 份，只处理 job_id 那份
    total_count = len(json_list)
    chunk_size = (total_count + args.world_size - 1) // args.world_size  # 向上取整
    start_idx = args.job_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_count)
    json_list = json_list[start_idx:end_idx]
    
    print(f"Job {args.job_id}/{args.world_size}: processing {len(json_list)} files (index {start_idx} to {end_idx - 1})")
    
    args.resume = False
    processor = LyricDiarization()
    for json_path in tqdm(json_list, desc=f"Job {args.job_id}"):
        try:
            processor.process(json_path, resume=args.resume)
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            import traceback
            traceback.print_exc()
