
import numpy as np
import librosa
import re
import sys
from pathlib import Path
import librosa
import json
from load_lrc import parse_lrc_with_timestamps

def energy_vad_from_start(
    audio,
    sr,
    frame_length=1024,
    hop_length=256,
    min_silence_duration=1.0, # 不能太短！
    energy_threshold = 0.01,
    fmin=200,
    fmax=4000,
):
    """
    Music-aware energy-based VAD:
    - 自适应能量阈值
    - 低频抑制（语音频带）
    - 忽略非常短的 non-speech
    - 只保留从开头开始的完整语音段
    - 输出时间戳（秒）

    Returns:
        start_time (float): 语音开始时间（始终为 0.0）
        end_time (float): 语音结束时间（秒）
        vad_mask (np.ndarray): frame-level speech mask (bool)
    """

    # ===== 1. STFT =====
    stft = librosa.stft(
        audio,
        n_fft=frame_length,
        hop_length=hop_length,
        center=True
    )
    magnitude = np.abs(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # ===== 2. 低频抑制：仅保留语音频带 =====
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    band_energy = magnitude[freq_mask, :] ** 2
    frame_energy = np.sqrt(np.mean(band_energy, axis=0) + 1e-8)

    vad_mask = frame_energy > energy_threshold

    # ===== 4. 忽略短 non-speech，只保留开头语音 =====
    frames_per_second = sr / hop_length
    min_silence_frames = int(min_silence_duration * frames_per_second)

    silence_count = 0
    end_frame = 0

    for i in range(len(vad_mask)):
        if vad_mask[i]:
            silence_count = 0
            end_frame = i + 1
        else:
            silence_count += 1
            if silence_count >= min_silence_frames:
                break


    # ===== 5. 时间戳 =====
    start_time = 0.0
    end_time = min(len(audio) / sr, end_frame * hop_length / sr)

    return start_time, end_time, vad_mask






def process(audio_path, lyric_path, vad_kwargs={}):
    lyric_starts = parse_lrc_with_timestamps(lyric_path)

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    # print(audio.shape)
    
    result = []
    for i in range(len(lyric_starts)):
        start_frame = int(lyric_starts[i]['start'] * sr)
        end_frame = int(lyric_starts[i+1]['start'] * sr) if i+1 < len(lyric_starts) else audio.shape[0]
        segment_audio = audio[...,start_frame:end_frame]
        vad_start , vad_end , vad_mask = energy_vad_from_start(segment_audio, sr, **vad_kwargs)
        # print(vad_start+lyric_starts[i]['start'], vad_end+lyric_starts[i]['start'], lyric_starts[i])

        result.append({
            "text": lyric_starts[i]['text'],
            "start": vad_start + lyric_starts[i]['start'],
            "end": vad_end + lyric_starts[i]['start'],
            "vad_res": str(vad_mask.astype(int).tolist())
        })
    return result


def process_with_seg(audio_path, segments, vad_kwargs={}):

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    # print(audio.shape)
    
    result = []
    for i in range(len(segments)):
        start_frame = int(segments[i]['start'] * sr)
        end_frame = int(segments[i]['end'] * sr)
        segment_audio = audio[...,start_frame:end_frame]
        vad_start , vad_end , vad_mask = energy_vad_from_start(segment_audio, sr, **vad_kwargs)
        # print(vad_start+lyric_starts[i]['start'], vad_end+lyric_starts[i]['start'], lyric_starts[i])

        result.append({
            "text": segments[i]['text'],
            "start": segments[i]['start'],
            "end": segments[i]['end'],
            "vad_res": str(vad_mask.astype(int).tolist())
        })
    # breakpoint()
    return result

if __name__ == "__main__":
    lrc2wav_scp, sep_dir, output_dir = sys.argv[1:]
    output_dir, sep_dir = Path(output_dir), Path(sep_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in open(lrc2wav_scp, "r", encoding="utf-8"):
        f = f.strip()
        print(f)
        if not f:
            continue
        lrc_path, audio_path = f.split('\t')
        basename = Path(audio_path).name
        result = process(sep_dir / basename / "vocals.wav", lrc_path)
        with open(str(output_dir/ basename) + ".json", "w", encoding="utf-8") as fout:
            json.dump(result, fout, ensure_ascii=False, indent=2)   

    # print(process("local/separation_output/bs_roformer/luoxue_20251226_all/ - 在夏天为你写的情诗.flac/vocals.wav", "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue/ - 在夏天为你写的情诗.lrc"))
    
    # x = json.load(open("/mnt/conversationhubhot/yaoyaochang/speech/data/music/muse20260112/jsons/en_part16_of_35/suno_en_015653_0.json"))
    # process_with_seg(x['audio_path'], x['segments'])