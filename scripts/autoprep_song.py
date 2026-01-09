import math
import os, sys
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

scipy.inf = np.inf




@dataclass
class SongFormerConfig:
    output_dir: str = MISSING
    model: str = MISSING
    checkpoint: str  = MISSING
    config_path: str = MISSING
    no_rule_post_processing: bool = False
    win_size: int = 420
    hop_size: int = 420
    num_classes: int = 128
    MUSICFM_HOME_PATH : str = os.path.join(run_struct_anal.BASE_PATH, 'ckpts', "MusicFM")
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
    flac_file: bool = False
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
    def __init__(self,
                input_scp: str,
                intermediate_scp_dir: str = None,
                output_dir: str = None,

                songformer_init_args: Optional[SongFormerConfig] = None,
                separator_init_args: Optional[SeparationConfig] = None,
                vad_init_args: Optional[VADConfig] = None,

                ):
        self.input_scp = input_scp
        self.intermediate_scp_dir = intermediate_scp_dir
        self.output_dir = output_dir

        self.lyric2wavs = self.load_and_category()
        self.lyrics = list(self.lyric2wavs.keys())
        self.wavs = set([w for wav_list in self.lyric2wavs.values() for w in wav_list])
        print(f"Loaded {len(self.lyrics)} lyrics and {len(self.wavs)} audio files from {self.input_scp}.")

        self.device = f"cuda:0" if torch.cuda.is_available() else "cpu"


        self.songformer_init_args = songformer_init_args
        if songformer_init_args is not None:
            self.songformer_init_args = OmegaConf.merge(SongFormerConfig, self.songformer_init_args)
            self.init_struct_analyzer()

        self.separator_init_args = separator_init_args
        if separator_init_args is not None:
            self.separator_init_args = OmegaConf.merge(SeparationConfig, self.separator_init_args)
            self.separator_init_args.pcm_type = run_separation.validate_sndfile_subtype(self.separator_init_args)
            self.init_separator()

        self.vad_init_args = vad_init_args
        if self.vad_init_args is not None:
            self.vad_init_args = OmegaConf.merge(VADConfig, self.vad_init_args)
    pass


    def load_and_category(self):
        with open(self.input_scp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        idx2lrc, idx2wav = {}, {}
        for l in lines:
            l = Path(l.strip())
            if l.suffix == '.lrc':
                idx2lrc[l.stem] = str(l)
            elif l.suffix in ['.wav', '.flac', '.mp3']:
                idx2wav[l.stem] = idx2wav.get(l.stem, []) + [str(l)]
            else:
                print(f"Unknown file type: {l}")
        
        lrc2wavs = {}
        for lrc_key, lrc_path in idx2lrc.items():
            wav_paths = idx2wav.get(lrc_key, [])
            if wav_paths:
                lrc2wavs[lrc_path] = wav_paths
            else:
                print(f"No corresponding WAV files found for LRC: {lrc_path}")
        
        if self.intermediate_scp_dir is not None:
            output_dir = Path(self.intermediate_scp_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            f = open(output_dir / "lrc2wav.scp", "w", encoding="utf-8")
            fw = open(output_dir / "wav.scp", "w", encoding="utf-8")
            fl = open(output_dir / "lrc.scp", "w", encoding="utf-8")
            for lrc, wavs in lrc2wavs.items():
                fl.write(f"{lrc}\n")
                for wav in wavs:
                    f.write(f"{lrc}\t{wav}\n")
                    fw.write(f"{wav}\n")
        
        return lrc2wavs


    def init_struct_analyzer(self):
        # MuQ model loading (this will automatically fetch the checkpoint from huggingface)
        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
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
        checkpoint_path=os.path.join(run_struct_anal.BASE_PATH, "ckpts", self.songformer_init_args.checkpoint)
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
    def struct_analyze(self, audio_path):
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
                
                json.dump(
                    msa_json,
                    open(os.path.join(self.songformer_init_args.output_dir, f"{Path(audio_path).name}.json"), "w"),
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



    def separate(self, path):
        instruments = run_separation.prefer_target_instrument(self.separate_config)[:]
        os.makedirs(self.separator_init_args.store_dir, exist_ok=True)

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

            output_dir = os.path.join(self.separator_init_args.store_dir, *dirnames)
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{fname}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)
            print("Wrote file:", output_path)
            if self.separator_init_args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                run_separation.draw_spectrogram(estimates.T, sr, self.separator_init_args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)



    def process_all(self):

        if self.songformer_init_args is not None:
            os.makedirs(self.songformer_init_args.output_dir, exist_ok=True)
            for wav in tqdm(self.wavs, total=len(self.wavs), desc="Structural Analysis"):
                self.struct_analyze(wav)
        
        if self.separator_init_args is not None:
            os.makedirs(self.separator_init_args.store_dir, exist_ok=True)
            for wav in tqdm(self.wavs, total=len(self.wavs), desc="Separation"):
                self.separate(wav)

        if self.vad_init_args is not None:
            os.makedirs(self.vad_init_args.output_dir, exist_ok=True)
            for lrc, wavs in tqdm(self.lyric2wavs.items(), total=len(self.lyric2wavs), desc="VAD Processing"):
                for w in wavs:
                    vad_kwargs = OmegaConf.to_container(self.vad_init_args, resolve=True)
                    vad_kwargs.pop("output_dir", None)
                    result = run_sentence_vad.process(
                        audio_path=Path(self.separator_init_args.store_dir) / Path(w).name / "vocals.wav",
                        lyric_path=lrc,
                        vad_kwargs=vad_kwargs,
                    )
                    output_path = os.path.join(self.vad_init_args.output_dir, f"{Path(w).name}.json")
                    json.dump(result, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

        self.combine_results(self.output_dir)

    def combine_results(self, output_dir: str):
        struct_dir, sep_dir, vad_dir, output_dir = Path(self.songformer_init_args.output_dir), Path(self.separator_init_args.store_dir), Path(self.vad_init_args.output_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        lrc2wav = [[k,v] for k,vs in self.lyric2wavs.items() for v in vs]

        for lrc_path, wav_path in lrc2wav:
            basename = Path(wav_path).name
            struct_json = json.load(open(struct_dir / f"{basename}.json", 'r', encoding='utf-8'))
            total_len = AudioDecoder(wav_path).metadata.duration_seconds

            segments, extra, wrong_t = [], [], []
            vad_json = json.load(open(vad_dir / f"{basename}.json", 'r', encoding='utf-8'))

            max_sec, filtered_vad_json = 0, []
            for seg in vad_json:
                # print(seg)
                # if seg['text'] == "At least just say": breakpoint()
                if seg['end'] - seg['start'] < 0.2:
                    extra.append(seg)
                    continue
                if seg['start'] < max_sec-1: # allow 1 sec mistake due to precision
                    wrong_t.append(seg)
                    break
                

                if seg['start'] - max_sec >= 1.5:
                    filtered_vad_json.append({
                        "text": "",
                        "start": max_sec,
                        "end": seg['start'],
                    })
                elif filtered_vad_json!= []:
                    # merge with previous segment
                    filtered_vad_json[-1]['end'] = seg['start']
                

                max_sec = seg['end']
                filtered_vad_json.append(seg.copy())

            if total_len - max_sec >= 1.5:
                filtered_vad_json.append({
                    "text": "",
                    "start": max_sec,
                    "end": total_len,
                })
            elif filtered_vad_json!= []:
                filtered_vad_json[-1]['end'] = total_len



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
                # print(overlaps, struct_json, seg)
                most_possible_label = max(overlaps.items(), key=lambda x: x[1])[0]

                segments.append({
                        "text": f"[{most_possible_label}] " + seg['text'],
                        "start": seg['start'], 
                        "end": seg['end'],
                        "speaker": None,
                    })

            output_json = {
                "audio_path": wav_path,
                "audio_length": total_len,
                "audio_vocal_path": str(sep_dir / basename / "vocals"),
                "audio_bgm_path": str(sep_dir / basename / "instrumental"),
                "structures": struct_json,
                "segments": segments,
                "info": {"extra_segments": extra, "wrong_time_segments": wrong_t}
            }
            # print(output_json)
            json.dump(output_json, 
                    open(output_dir / f"{basename}.json", 'w', encoding='utf-8'), 
                    ensure_ascii=False, indent=4)
            

            
def load_config() -> DictConfig:
    cmd_cfg = OmegaConf.from_cli()
    
    cfg_file_path = cmd_cfg.pop("cfg_file", None) 
    file_cfg = OmegaConf.load(open(cfg_file_path, 'r')) if cfg_file_path is not None \
                else OmegaConf.create()
    
    cfgs = OmegaConf.merge(file_cfg, cmd_cfg)
    OmegaConf.resolve(cfgs)
    
    cfgs = OmegaConf.to_container(cfgs, resolve=True)

    return cfgs

if __name__ == "__main__":
    cfg = load_config()

    autoprep = AutoPrepSong(**cfg)
    autoprep.process_all()