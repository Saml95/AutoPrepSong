"""
对歌词 JSON 进行 speaker diarization
只对 is_lyric=True 的 segments 进行 diarization

支持参数：
- data_yaml: 包含 input_jsonl 和 output_base_dir 的 YAML 文件
- input_jsonl: 输入 JSONL 文件路径（每行格式：{"json_path": "xxx"}）
- output_base_dir: 输出基础目录
- start_idx: 起始索引
- chunk_size: 处理数量
"""

import json
import os
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from audioautoprep.models.diarization import CustomWespeakerDiarization
from audioautoprep.models.utils import load_audio_use_ffmpeg


class LyricDiarization:

    def __init__(self, output_dir: str = None, resume: bool = True):
        self.output_dir = output_dir
        self.resume = resume
        
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

    def get_output_path(self, json_path):
        """获取输出文件路径
        
        输出路径为: output_dir / json_path倒数第二级目录 / 文件名.diarized.json
        例如输入: /xxx/luoxue_batch5/xxx.mp3.merged.5s.json
        输出: output_dir/luoxue_batch5/xxx.mp3.merged.5s.diarized.json
        """
        if self.output_dir is not None:
            # 使用指定的 output_dir
            json_path_obj = Path(json_path)
            # 获取倒数第二级目录名
            parent_dir_name = json_path_obj.parent.name
            # 替换后缀
            output_name = json_path_obj.name.replace('.json', '.diarized.json')
            # 组合路径: output_dir / 倒数第二级目录 / 文件名
            output_path = os.path.join(self.output_dir, parent_dir_name, output_name)
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            return output_path
        else:
            # 默认保存在原文件同目录
            return json_path.replace('.json', '.diarized.json')

    def process(self, json_path, resume=None):
        if resume is None:
            resume = self.resume
            
        output_json_path = self.get_output_path(json_path)
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
        return output_json_path


def load_jsonl(input_jsonl: str, start_idx: int = 0, chunk_size: int = None):
    """Load json paths from jsonl file.
    
    Each line in the jsonl file should be a JSON object with key:
    - json_path: path to the json file
    
    Args:
        input_jsonl: Path to the input JSONL file
        start_idx: Starting index (0-based) for processing. Default is 0.
        chunk_size: Number of lines to process from start_idx. 
                    If None, process all lines from start_idx to the end.
    """
    json_paths = []
    line_idx = 0
    end_idx = start_idx + chunk_size if chunk_size is not None else float('inf')
    
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines before start_idx
            if line_idx < start_idx:
                line_idx += 1
                continue
            
            # Stop if we've reached the end of the chunk
            if line_idx >= end_idx:
                break
            
            try:
                record = json.loads(line)
                json_path = record.get("json_path")
                if json_path:
                    json_paths.append(json_path)
                else:
                    print(f"[Warning] Missing json_path: {line}")
            except json.JSONDecodeError as e:
                print(f"[Warning] Invalid JSON line: {line}, error: {e}")
            
            line_idx += 1
    return json_paths


def parse_args():
    parser = argparse.ArgumentParser(description="对歌词 JSON 进行 speaker diarization")
    parser.add_argument(
        "--data_yaml", "-d",
        type=str,
        default=None,
        help="Path to a YAML file containing 'input_jsonl' and 'output_base_dir' keys."
    )
    parser.add_argument(
        "--input_jsonl", "-i",
        type=str,
        default=None,
        help="Path to the input JSONL file containing json_path list"
    )
    parser.add_argument(
        "--output_base_dir", "-o",
        type=str,
        default=None,
        help="Base directory for all output files"
    )
    parser.add_argument(
        "--start_idx", "-s",
        type=int,
        default=0,
        help="Starting index (0-based) for processing JSONL lines (default: 0)"
    )
    parser.add_argument(
        "--chunk_size", "-n",
        type=int,
        default=None,
        help="Number of lines to process from start_idx. If not specified, process all remaining lines"
    )
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=True,
        help='Skip if output file exists'
    )
    return parser.parse_args()


def load_config(args) -> dict:
    """Load and merge configuration from args."""
    # Validate: cannot pass both --input_jsonl/--output_base_dir and --data_yaml
    has_direct_args = args.input_jsonl is not None or args.output_base_dir is not None
    has_data_yaml = args.data_yaml is not None
    
    if has_direct_args and has_data_yaml:
        raise ValueError(
            "Cannot use both --data_yaml and --input_jsonl/--output_base_dir simultaneously. "
            "Choose one method: either provide --data_yaml OR --input_jsonl and --output_base_dir."
        )
    
    # Determine input_jsonl and output_base_dir
    if has_data_yaml:
        # Load from data_yaml
        data_cfg = OmegaConf.load(open(args.data_yaml, 'r'))
        input_jsonl = data_cfg.get("input_jsonl", None)
        output_base_dir = data_cfg.get("output_base_dir", None)
    else:
        # Use direct args
        input_jsonl = args.input_jsonl
        output_base_dir = args.output_base_dir
    
    # Validate that we have the required parameters
    if input_jsonl is None:
        raise ValueError("input_jsonl is required. Provide via --input_jsonl or --data_yaml")
    if output_base_dir is None:
        raise ValueError("output_base_dir is required. Provide via --output_base_dir or --data_yaml")
    
    # Create output_dir with secondary directory
    chunk_suffix = args.chunk_size if args.chunk_size is not None else 'all'
    # output_dir = os.path.join(output_base_dir, f"{args.start_idx}_{chunk_suffix}")
    output_dir = output_base_dir
    
    return {
        "input_jsonl": input_jsonl,
        "output_base_dir": output_base_dir,
        "output_dir": output_dir,
        "start_idx": args.start_idx,
        "chunk_size": args.chunk_size,
        "resume": args.resume
    }


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    
    # Create output directory
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # Load json paths from jsonl
    json_paths = load_jsonl(
        cfg["input_jsonl"], 
        start_idx=cfg["start_idx"], 
        chunk_size=cfg["chunk_size"]
    )
    
    chunk_info = f"start_idx={cfg['start_idx']}, chunk_size={cfg['chunk_size']}" if cfg['chunk_size'] else f"start_idx={cfg['start_idx']}, all remaining"
    print(f"Loaded {len(json_paths)} json paths from {cfg['input_jsonl']} ({chunk_info})")
    print(f"Output directory: {cfg['output_dir']}")
    
    # Initialize processor
    processor = LyricDiarization(
        output_dir=cfg["output_dir"],
        resume=cfg["resume"]
    )
    
    # Process and collect output paths
    output_paths = []
    for json_path in tqdm(json_paths, desc="Diarizing"):
        try:
            output_path = processor.process(json_path)
            if output_path:
                output_paths.append(output_path)
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # # Save output jsonl
    # output_jsonl_path = os.path.join(cfg["output_dir"], "output.jsonl")
    # with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    #     for path in output_paths:
    #         item = {"json_path": path}
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # print(f"Saved {len(output_paths)} paths to {output_jsonl_path}")
