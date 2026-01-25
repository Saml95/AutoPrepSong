import math
import os, sys
import argparse
from pathlib import Path
import json
from typing import Optional
import librosa
import torch
from tqdm import tqdm
import importlib
from omegaconf import OmegaConf, DictConfig, MISSING
from muq import MuQ
from ema_pytorch import EMA
import scipy
import numpy as np
from torchcodec.decoders import AudioDecoder
from dataclasses import dataclass, field
import soundfile as sf  
import run_separation
import run_struct_anal
import run_sentence_vad
from load_lrc import parse_lrc_with_timestamps
from run_merge import merge_segments
from process_lyrics import process_single_segment_list

scipy.inf = np.inf


@dataclass
class SongFormerConfig:
    output_dir: str = MISSING
    model: str = MISSING
    config_path: str = MISSING

    no_rule_post_processing: bool = False
    win_size: int = 420
    hop_size: int = 420
    num_classes: int = 128

    CHECKPOINT_PATH: str = os.path.join(run_struct_anal.BASE_PATH, "ckpts", "SongFormer.safetensors")
    MUSICFM_HOME_PATH : str = os.path.join(run_struct_anal.BASE_PATH, 'ckpts', "MusicFM")
    MUQ_PATH : str = os.path.join(run_struct_anal.BASE_PATH, 'ckpts', "MuQ-large-msd-iter")
    
    BEFORE_DOWNSAMPLING_FRAME_RATES : float = 25
    AFTER_DOWNSAMPLING_FRAME_RATES : float = 8.333
    DATASET_LABEL : str = "SongForm-HX-8Class"
    DATASET_IDS : list = field(default_factory=lambda: [5])
    TIME_DUR : int = 420
    INPUT_SAMPLING_RATE : int = 24000


@dataclass
class SeparationConfig:
    store_dir: str = MISSING
    model_type: str = MISSING
    config_path: str = MISSING
    start_check_point: str = MISSING
    filename_template: str = '{file_name}/{instr}'
    pcm_type: str = "FLOAT" #['PCM_16', 'PCM_24', 'FLOAT']
    # flac_file: bool = False
    flac_file: bool = True
    use_tta: bool = False
    extract_instrumental: bool = True
    draw_spectro: int = 0
    lora_checkpoint_loralib: str = ''

@dataclass
class VADConfig:
    output_dir: str = MISSING
    frame_length: int = 1024
    hop_length: int = 256
    min_silence_duration: float = 1.0
    energy_threshold: float = 0.01
    fmin: int = 200
    fmax: int = 4000



class AutoPrepSong:
    def __init__(self, config: DictConfig = None, **kwargs):
        # Support both config object and kwargs
        if config is not None:
            self.config = OmegaConf.create(config) if isinstance(config, dict) else config
        else:
            self.config = OmegaConf.create(kwargs)
        
        # Get basic paths from config
        self.input_jsonl = self.config.get("input_jsonl", None)
        self.output_dir = self.config.get("output_dir", None)
        
        # Get do_xxx flags from config (default to True if corresponding init_args exist)
        self.do_songformer = self.config.get("do_songformer", self.config.get("songformer_init_args") is not None)
        self.do_separation = self.config.get("do_separation", self.config.get("separator_init_args") is not None)
        self.do_vad = self.config.get("do_vad", self.config.get("vad_init_args") is not None)
        
        # Resume option: skip if output already exists
        self.resume = self.config.get("resume", True)

        self.file_split_chunk = self.config.get("split_chunk", None)
        
        # Get start_idx and chunk_size from config
        self.start_idx = self.config.get("start_idx", 0)
        self.chunk_size = self.config.get("chunk_size", None)

        self.merge_seconds = self.config.get("merge_seconds", None)

        # Load audio-lyric pairs from jsonl file
        if self.input_jsonl is not None:
            self.audio_lyric_pairs = self.load_jsonl(start_idx=self.start_idx, chunk_size=self.chunk_size)
            chunk_info = f"start_idx={self.start_idx}, chunk_size={self.chunk_size}" if self.chunk_size else f"start_idx={self.start_idx}, all remaining"
            print(f"Loaded {len(self.audio_lyric_pairs)} audio-lyric pairs from {self.input_jsonl} ({chunk_info}).")
        else:
            self.audio_lyric_pairs = []

        self.device = f"cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize songformer if do_songformer is True
        self.songformer_init_args = None
        if self.do_songformer and self.config.get("songformer_init_args") is not None:
            self.songformer_init_args = OmegaConf.merge(SongFormerConfig, self.config.songformer_init_args)
            self.init_struct_analyzer()

        # Initialize separator if do_separation is True
        self.separator_init_args = None
        if self.do_separation and self.config.get("separator_init_args") is not None:
            self.separator_init_args = OmegaConf.merge(SeparationConfig, self.config.separator_init_args)
            self.separator_init_args.pcm_type = run_separation.validate_sndfile_subtype(self.separator_init_args)
            self.init_separator()

        # Initialize vad config if do_vad is True
        self.vad_init_args = None
        if self.do_vad and self.config.get("vad_init_args") is not None:
            self.vad_init_args = OmegaConf.merge(VADConfig, self.config.vad_init_args)

    def load_jsonl(self, start_idx: int = 0, chunk_size: int = None):
        """Load audio-lyric pairs from jsonl file.
        
        Each line in the jsonl file should be a JSON object with keys:
        - audio_path: path to the audio file
        - lyric_path: path to the lyric file
        
        Args:
            start_idx: Starting index (0-based) for processing. Default is 0.
            chunk_size: Number of lines to process from start_idx. 
                        If None, process all lines from start_idx to the end.
        """
        pairs = []
        line_idx = 0
        end_idx = start_idx + chunk_size if chunk_size is not None else float('inf')
        
        with open(self.input_jsonl, "r", encoding="utf-8") as f:
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
                    audio_path = record.get("audio_path")
                    lyric_path = record.get("lyric_path")
                    if audio_path and lyric_path:
                        pairs.append({"audio_path": audio_path, "lyric_path": lyric_path})
                    else:
                        print(f"[Warning] Missing audio_path or lyric_path: {line}")
                except json.JSONDecodeError as e:
                    print(f"[Warning] Invalid JSON line: {line}, error: {e}")
                
                line_idx += 1
        return pairs


    def init_struct_analyzer(self):
        # MuQ model loading (this will automatically fetch the checkpoint from huggingface)
        self.muq = MuQ.from_pretrained(self.songformer_init_args.MUQ_PATH)
        self.muq = self.muq.to(self.device).eval()

        # MusicFM model loading
        self.musicfm = run_struct_anal.MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join(self.songformer_init_args.MUSICFM_HOME_PATH, "msd_stats.json"),
            model_path=os.path.join(self.songformer_init_args.MUSICFM_HOME_PATH, "pretrained_msd.pt"),
        )
        self.musicfm = self.musicfm.to(self.device)
        self.musicfm.eval()

        # Custom model loading based on the config
        module = importlib.import_module("models." + str(self.songformer_init_args.model))
        Model = getattr(module, "Model")
        self.songformer_hp = OmegaConf.load(os.path.join(run_struct_anal.BASE_PATH, "configs", self.songformer_init_args.config_path))
        model = Model(self.songformer_hp)


        """Load checkpoint from path"""
        checkpoint_path=self.songformer_init_args.CHECKPOINT_PATH
        if checkpoint_path.endswith(".pt"):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        elif checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            checkpoint = {"model_ema": load_file(checkpoint_path, device=self.device)}
        else:
            raise ValueError("Unsupported checkpoint format. Use .pt or .safetensors")


        if checkpoint.get("model_ema", None) is not None:
            print("Loading EMA model parameters")
            model_ema = EMA(model, include_online_model=False)
            model_ema.load_state_dict(checkpoint["model_ema"])
            model.load_state_dict(model_ema.ema_model.state_dict())
        else:
            print("No EMA model parameters found, using original model")
            model.load_state_dict(checkpoint["model"])

        model.to(self.device)
        model.eval()
        self.songformer = model

        self.songformer_dataset_id2label_mask = {}

        for key, allowed_ids in run_struct_anal.DATASET_ID_ALLOWED_LABEL_IDS.items():
            self.songformer_dataset_id2label_mask[key] = np.ones(self.songformer_init_args.num_classes, dtype=bool)
            self.songformer_dataset_id2label_mask[key][allowed_ids] = False


    @torch.no_grad()
    def struct_analyze(self, audio_path, output_path):
        """Run inference on the input audio"""
        num_classes = self.songformer_init_args.num_classes

        with torch.no_grad():
            try:
                # Loading the audio file
                wav, sr = librosa.load(audio_path, sr=self.songformer_init_args.INPUT_SAMPLING_RATE)
                audio = torch.tensor(wav).to(self.device)

                win_size = self.songformer_init_args.win_size
                hop_size = self.songformer_init_args.hop_size
                total_len = (
                    (audio.shape[0] // self.songformer_init_args.INPUT_SAMPLING_RATE) // self.songformer_init_args.TIME_DUR
                ) * self.songformer_init_args.TIME_DUR + self.songformer_init_args.TIME_DUR
                total_frames = math.ceil(total_len * self.songformer_init_args.AFTER_DOWNSAMPLING_FRAME_RATES)

                logits = {
                    "function_logits": np.zeros([total_frames, num_classes]),
                    "boundary_logits": np.zeros([total_frames]),
                }
                logits_num = {
                    "function_logits": np.zeros([total_frames, num_classes]),
                    "boundary_logits": np.zeros([total_frames]),
                }

                lens = 0
                i = 0
                while True:
                    start_idx = i * self.songformer_init_args.INPUT_SAMPLING_RATE
                    end_idx = min((i + win_size) * self.songformer_init_args.INPUT_SAMPLING_RATE, audio.shape[-1])
                    if start_idx >= audio.shape[-1]:
                        break
                    if end_idx - start_idx <= 1024:
                        continue
                    audio_seg = audio[start_idx:end_idx]

                    # MuQ embedding
                    muq_output = self.muq(audio_seg.unsqueeze(0), output_hidden_states=True)
                    muq_embd_420s = muq_output["hidden_states"][10]
                    del muq_output
                    torch.cuda.empty_cache()

                    # MusicFM embedding
                    _, musicfm_hidden_states = self.musicfm.get_predictions(
                        audio_seg.unsqueeze(0)
                    )
                    musicfm_embd_420s = musicfm_hidden_states[10]
                    del musicfm_hidden_states
                    torch.cuda.empty_cache()

                    wraped_muq_embd_30s = []
                    wraped_musicfm_embd_30s = []

                    for idx_30s in range(i, i + hop_size, 30):
                        start_idx_30s = idx_30s * self.songformer_init_args.INPUT_SAMPLING_RATE
                        end_idx_30s = min(
                            (idx_30s + 30) * self.songformer_init_args.INPUT_SAMPLING_RATE,
                            audio.shape[-1],
                            (i + hop_size) * self.songformer_init_args.INPUT_SAMPLING_RATE,
                        )
                        if start_idx_30s >= audio.shape[-1]:
                            break
                        if end_idx_30s - start_idx_30s <= 1024:
                            continue
                        wraped_muq_embd_30s.append(
                            self.muq(
                                audio[start_idx_30s:end_idx_30s].unsqueeze(0),
                                output_hidden_states=True,
                            )["hidden_states"][10]
                        )
                        torch.cuda.empty_cache()
                        wraped_musicfm_embd_30s.append(
                            self.musicfm.get_predictions(
                                audio[start_idx_30s:end_idx_30s].unsqueeze(0)
                            )[1][10]
                        )
                        torch.cuda.empty_cache()

                    wraped_muq_embd_30s = torch.concatenate(wraped_muq_embd_30s, dim=1)
                    wraped_musicfm_embd_30s = torch.concatenate(
                        wraped_musicfm_embd_30s, dim=1
                    )
                    all_embds = [
                        wraped_musicfm_embd_30s,
                        wraped_muq_embd_30s,
                        musicfm_embd_420s,
                        muq_embd_420s,
                    ]

                    if len(all_embds) > 1:
                        embd_lens = [x.shape[1] for x in all_embds]
                        max_embd_len = max(embd_lens)
                        min_embd_len = min(embd_lens)
                        if abs(max_embd_len - min_embd_len) > 4:
                            raise ValueError(
                                f"Embedding shapes differ too much: {max_embd_len} vs {min_embd_len}"
                            )

                        for idx in range(len(all_embds)):
                            all_embds[idx] = all_embds[idx][:, :min_embd_len, :]

                    embd = torch.concatenate(all_embds, axis=-1)

                    dataset_label = self.songformer_init_args.DATASET_LABEL
                    dataset_ids = torch.Tensor(self.songformer_init_args.DATASET_IDS).to(self.device, dtype=torch.long)
                    msa_info, chunk_logits = self.songformer.infer(
                        input_embeddings=embd,
                        dataset_ids=dataset_ids,
                        label_id_masks=torch.Tensor(
                            self.songformer_dataset_id2label_mask[
                                run_struct_anal.DATASET_LABEL_TO_DATASET_ID[dataset_label]
                            ]
                        )
                        .to(self.device, dtype=bool)
                        .unsqueeze(0)
                        .unsqueeze(0),
                        with_logits=True,
                    )

                    start_frame = int(i * self.songformer_init_args.AFTER_DOWNSAMPLING_FRAME_RATES)
                    end_frame = start_frame + min(
                        math.ceil(hop_size * self.songformer_init_args.AFTER_DOWNSAMPLING_FRAME_RATES),
                        chunk_logits["boundary_logits"][0].shape[0],
                    )

                    logits["function_logits"][start_frame:end_frame, :] += (
                        chunk_logits["function_logits"][0].detach().cpu().numpy()
                    )
                    logits["boundary_logits"][start_frame:end_frame] = (
                        chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                    )
                    logits_num["function_logits"][start_frame:end_frame, :] += 1
                    logits_num["boundary_logits"][start_frame:end_frame] += 1
                    lens += end_frame - start_frame

                    i += hop_size
                logits["function_logits"] /= logits_num["function_logits"]
                logits["boundary_logits"] /= logits_num["boundary_logits"]

                logits["function_logits"] = torch.from_numpy(
                    logits["function_logits"][:lens]
                ).unsqueeze(0)
                logits["boundary_logits"] = torch.from_numpy(
                    logits["boundary_logits"][:lens]
                ).unsqueeze(0)

                msa_infer_output = run_struct_anal.postprocess_functional_structure(logits, self.songformer_hp)

                assert msa_infer_output[-1][-1] == "end"
                if not self.songformer_init_args.no_rule_post_processing:
                    msa_infer_output = run_struct_anal.rule_post_processing(msa_infer_output)
                msa_json = []
                for idx in range(len(msa_infer_output) - 1):

                    # combine the same labels CYY
                    if idx > 0 and msa_infer_output[idx][1] == msa_json[-1]["label"]:
                        msa_json[-1]["end"] = msa_infer_output[idx + 1][0]
                        continue

                    msa_json.append(
                        {
                            "label": msa_infer_output[idx][1],
                            "start": msa_infer_output[idx][0],
                            "end": msa_infer_output[idx + 1][0],
                        }
                    )
                
                Path(output_path).parent.mkdir(exist_ok=True, parents=True)
                json.dump(
                    msa_json,
                    open(output_path, "w"),
                    indent=4,
                    ensure_ascii=False,
                )

            except Exception as e:
                print(f"process {audio_path} error\n{e}")


    def init_separator(self):
        torch.backends.cudnn.benchmark = True

        self.separator_model, self.separate_config = run_separation.get_model_from_config(self.separator_init_args.model_type, self.separator_init_args.config_path)
        if 'model_type' in self.separate_config.training:
            self.separator_init_args.model_type = self.separate_config.training.model_type
        if self.separator_init_args.start_check_point:
            checkpoint = torch.load(self.separator_init_args.start_check_point, weights_only=False, map_location='cpu')
            run_separation.load_start_checkpoint(self.separator_init_args, self.separator_model, checkpoint, type_='inference')

        print("Instruments: {}".format(self.separate_config.training.instruments))

        self.separator_model = self.separator_model.to(self.device)
        self.separator_model.eval()



    def separate(self, path, output_dir):
        instruments = run_separation.prefer_target_instrument(self.separate_config)[:]

        print(f"Processing track: {path}")
        try:
            mix, sr = librosa.load(path, sr=getattr(self.separate_config.audio, 'sample_rate', 44100), mono=False)
        except Exception as e:
            print(f'Cannot read track: {format(path)}')
            print(f'Error message: {str(e)}')
            return 

        # If mono audio we must adjust it depending on model
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in self.separate_config.audio:
                if self.separate_config.audio['num_channels'] == 2:
                    print(f'Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in self.separate_config.inference:
            if self.separate_config.inference['normalize'] is True:
                mix, norm_params = run_separation.normalize_audio(mix)

        waveforms_orig = run_separation.demix(self.separate_config, self.separator_model, mix, self.device, model_type=self.separator_init_args.model_type, pbar=False)
        if self.separator_init_args.use_tta:
            waveforms_orig = run_separation.apply_tta(self.separate_config, self.separator_model, mix, waveforms_orig, self.device, self.separator_init_args.model_type)

        if self.separator_init_args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        file_name = os.path.basename(path)# 保留后缀

        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in self.separate_config.inference:
                if self.separate_config.inference['normalize'] is True:
                    estimates = run_separation.denormalize_audio(estimates, norm_params)

            codec = 'flac' if getattr(self.separator_init_args, 'flac_file', False) else 'wav'
            subtype = self.separator_init_args.pcm_type

            dirnames, fname = run_separation.format_filename(
                self.separator_init_args.filename_template,
                instr=instr,
                # start_time=int(start_time),
                file_name=file_name,
                dir_name=os.path.dirname(path),
                model_type=self.separator_init_args.model_type,
                model=os.path.splitext(os.path.basename(self.separator_init_args.start_check_point))[0]
            )

            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{fname}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)
            print("Wrote file:", output_path)
            if self.separator_init_args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                run_separation.draw_spectrogram(estimates.T, sr, self.separator_init_args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)



    def process_all(self):
        """Process all audio files, each audio goes through all steps before moving to the next."""
        # Create output directories
        if self.do_songformer and self.songformer_init_args is not None:
            os.makedirs(self.songformer_init_args.output_dir, exist_ok=True)
        if self.do_separation and self.separator_init_args is not None:
            os.makedirs(self.separator_init_args.store_dir, exist_ok=True)
        if self.do_vad and self.vad_init_args is not None:
            os.makedirs(self.vad_init_args.output_dir, exist_ok=True)
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        # Process each audio-lyric pair through all steps
        for idx_offset, pair in tqdm(enumerate(self.audio_lyric_pairs), total=len(self.audio_lyric_pairs), desc="Processing"):
            try:
                if self.file_split_chunk is not None:
                    store_chunk_id = (self.start_idx + idx_offset) // self.file_split_chunk
                else:
                    store_chunk_id = None
                self.process(wav_path=pair["audio_path"], lrc_path=pair["lyric_path"], output_dir=self.output_dir, store_chunk_id=store_chunk_id)
            except Exception as e:
                raise e
                print(f"[Error] Failed to process {pair['audio_path']}: {e}")
                continue

    def process(self, wav_path: str, lrc_path: str = None, output_dir: str = None, store_chunk_id: int = None):
        """
        Process a single audio file through all steps: struct_analyze -> separate -> vad -> combine
        
        Args:
            wav_path: Path to the audio file (string)
            lrc_path: Path to the lyric file
            output_dir: Output directory for final results
        """
        if store_chunk_id is None:
            basename = Path(wav_path).name
        else:
            basename = f"{store_chunk_id}/{Path(wav_path).name}"
        codec = 'flac' if getattr(self.separator_init_args, 'flac_file', True) else 'wav'
        
        # Check if final output already exists (resume mode)
        if self.resume and output_dir is not None:
            final_output_path = Path(output_dir) / f"{basename}.json"
            if final_output_path.exists():
                print(f"Skipping (already exists): {wav_path}")
                return
        
        print(f"Processing: {wav_path}")

        other_info = {}
        # Step 1: Remove MetaInfo
        if Path(lrc_path).suffix == ".lrc":
            unsorted_lyric_with_starts, wrong_segments = parse_lrc_with_timestamps(lrc_path)
            other_info["perhaps_translation"] = wrong_segments
            print(f"Warning: the .lrc file has not been checked by GPT")
        elif Path(lrc_path).name.endswith('.lyric.json'): # After GPT check
            js_entry = json.load(open(lrc_path, 'r'))
            unsorted_lyric_with_starts = [i for i in js_entry['segments'] if i['is_lyric']]
            other_info["perhaps_translation"]= js_entry['error_loading']
            other_info["GPT_not_lyric"]= [i for i in js_entry['segments'] if not i['is_lyric']]
        else:
            raise NotImplementedError(f"Unrecognized lyric file type: {lrc_path}")

        lyric_with_starts = unsorted_lyric_with_starts
        if lyric_with_starts != []:
            lyric_with_starts.sort(key=lambda x:x['start'])

        # Step 1: Structural Analysis
        if self.do_songformer and self.songformer_init_args is not None:
            struct_output = Path(self.songformer_init_args.output_dir) / f"{basename}.json"
            if self.resume and struct_output.exists():
                print(f"  [1/4] Skipping structural analysis (exists)...")
            else:
                print(f"  [1/4] Running structural analysis...")
                self.struct_analyze(wav_path, struct_output)

        # Step 2: Source Separation
        if self.do_separation and self.separator_init_args is not None:
            sep_output = Path(self.separator_init_args.store_dir) / basename 
            if self.resume and (sep_output / f"vocals.{codec}").exists():
                print(f"  [2/4] Skipping source separation (exists)...")
            else:
                print(f"  [2/4] Running source separation...")
                self.separate(wav_path, sep_output)

        # Step 3: VAD Processing
        if self.do_vad and self.vad_init_args is not None and lrc_path is not None:
            vad_output = Path(self.vad_init_args.output_dir) / f"{basename}.json"
            if self.resume and vad_output.exists():
                print(f"  [3/4] Skipping VAD processing (exists)...")
            else:
                print(f"  [3/4] Running VAD processing...")
                vad_kwargs = OmegaConf.to_container(self.vad_init_args, resolve=True)
                vad_kwargs.pop("output_dir", None)
                result = run_sentence_vad.process(
                    audio_path=Path(self.separator_init_args.store_dir) / basename / f"vocals.{codec}",
                    lyric_starts=lyric_with_starts,
                    vad_kwargs=vad_kwargs,
                )
                vad_output.parent.mkdir(exist_ok=True)
                json.dump(result, open(vad_output, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

        # Step 4: Combine Results
        if output_dir is not None and lrc_path is not None:
            print(f"  [4/4] Combining results...")
            self.combine_single_result(wav_path=wav_path, output_dir=output_dir, store_chunk_id=store_chunk_id, other_info=other_info)

        print(f"  Done: {wav_path}")

    def combine_single_result(self, wav_path: str, output_dir: str, store_chunk_id: int = None, other_info: dict = {}):
        """Combine results for a single audio file."""
        struct_dir = Path(self.songformer_init_args.output_dir)
        sep_dir = Path(self.separator_init_args.store_dir)
        vad_dir = Path(self.vad_init_args.output_dir)
        output_dir = Path(output_dir)

        if store_chunk_id is None:
            basename = Path(wav_path).name
        else:
            basename = f"{store_chunk_id}/{Path(wav_path).name}"

        struct_json = json.load(open(struct_dir / f"{basename}.json", 'r', encoding='utf-8'))
        total_len = AudioDecoder(wav_path).metadata.duration_seconds

        segments, extra, wrong_t = [], [], []
        vad_json = json.load(open(vad_dir / f"{basename}.json", 'r', encoding='utf-8'))

        max_sec, filtered_vad_json = 0, []
        for idx, seg in enumerate(vad_json):
            seg.pop('vad_res', None)
            if seg['end'] - seg['start'] < 0.3:
                extra.append(seg)
                continue

            if seg['start'] - max_sec >= 1.5:
                filtered_vad_json.append({
                    "text": "",
                    "start": max_sec,
                    "end": seg['start'],
                })
            elif filtered_vad_json != []:
                # merge with previous segment
                while filtered_vad_json != [] and seg['start'] < filtered_vad_json[-1]['start']:
                    wrong_t.append(filtered_vad_json.pop(-1)) #TODO 应该前面VAD之前就check完 

                if filtered_vad_json != []:
                    filtered_vad_json[-1]['end'] = seg['start']


            max_sec = seg['end']
            filtered_vad_json.append(seg.copy())

        if total_len - max_sec >= 1.5:
            filtered_vad_json.append({
                "text": "",
                "start": max_sec,
                "end": total_len,
            })
        elif filtered_vad_json != []:
            filtered_vad_json[-1]['end'] = total_len

        # for i in range(filtered_vad_json):
        #     filtered_vad_json[i]['is_lyric'] = filtered_vad_json[i]['text'] != ''
        filtered_vad_json, duration_stats, has_reset = process_single_segment_list(filtered_vad_json)

        struct_ptr = 0
        for seg in filtered_vad_json:
            while seg['start'] >= struct_json[struct_ptr]['end']:
                struct_ptr += 1
                if struct_ptr >= len(struct_json):
                    raise ValueError(f"Structure pointer out of range {seg}, {struct_json}")

            overlaped_ptr = 0
            overlaps = {}
            while struct_ptr + overlaped_ptr < len(struct_json) and struct_json[struct_ptr + overlaped_ptr]['start'] < seg['end']:
                overlap_len = min(seg['end'], struct_json[struct_ptr + overlaped_ptr]['end']) - max(seg['start'], struct_json[struct_ptr + overlaped_ptr]['start'])
                assert overlap_len > 0, f"No overlap found between {seg} and {struct_json[struct_ptr + overlaped_ptr]}"
                overlaps[struct_json[struct_ptr + overlaped_ptr]['label']] = overlaps.get(struct_json[struct_ptr + overlaped_ptr]['label'], 0) + overlap_len
                overlaped_ptr += 1
            most_possible_label = max(overlaps.items(), key=lambda x: x[1])[0]

            new_seg = seg.copy()
            new_seg['text'] = f"[{most_possible_label}] " + new_seg['text']
            new_seg['speaker'] = None


            segments.append(new_seg)

        # 根据 separator 配置决定后缀
        codec = 'flac' if getattr(self.separator_init_args, 'flac_file', True) else 'wav'
        
        if self.merge_seconds is not None:
            segments = merge_segments(segments, self.merge_seconds)

        other_info.update({"extra_segments": extra, "wrong_time_segments": wrong_t})
        output_json = {
            "audio_path": str(Path(wav_path).resolve()),
            "audio_length": total_len,
            "audio_vocal_path": str((sep_dir / basename / f"vocals.{codec}").resolve()),
            "audio_bgm_path": str((sep_dir / basename / f"instrumental.{codec}").resolve()),
            "structures": struct_json,
            "segments": segments,
            "info": other_info
        }
        (output_dir / f"{basename}.json").parent.mkdir(parents=True, exist_ok=True)
        json.dump(output_json,
                  open(output_dir / f"{basename}.json", 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)




def parse_args():
    parser = argparse.ArgumentParser(description="AutoPrepSong: Audio preprocessing pipeline for song data")
    parser.add_argument(
        "--config_path", "-c",
        type=str,
        default=None,
        help="Path to the configuration YAML file"
    )
    parser.add_argument(
        "--data_yaml", "-d",
        type=str,
        default=None,
        help="Path to a YAML file containing 'input_jsonl' and 'output_base_dir' keys. "
             "Alternative to providing --input_jsonl and --output_base_dir directly."
    )
    parser.add_argument(
        "--input_jsonl", "-i",
        type=str,
        default=None,
        help="Path to the input JSONL file containing audio-lyric pairs"
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
    return parser.parse_args()


def load_config(args) -> dict:
    # Load config from file if provided
    if args.config_path is not None:
        file_cfg = OmegaConf.load(open(args.config_path, 'r'))
    else:
        file_cfg = OmegaConf.create()
    
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
    else:
        # Use direct args
        data_cfg = OmegaConf.create()
    

    # Create config from command line args
    cli_cfg = OmegaConf.create(
        {k: v for k, v in vars(args).items() if v is not None}
    )
    

    # Merge configs (CLI args override file config)
    cfgs = OmegaConf.merge(file_cfg, data_cfg, cli_cfg)
    OmegaConf.resolve(cfgs)

    # Validate that we have the required parameters
    if cfgs.input_jsonl is None:
        raise ValueError("input_jsonl is required. Provide via --input_jsonl or --data_yaml")
    if cfgs.output_base_dir is None:
        raise ValueError("output_base_dir is required. Provide via --output_base_dir or --data_yaml")
    
    
    cfgs = OmegaConf.to_container(cfgs, resolve=True)

    return cfgs


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    print(cfg)

    autoprep = AutoPrepSong(config=cfg)
    autoprep.process_all()