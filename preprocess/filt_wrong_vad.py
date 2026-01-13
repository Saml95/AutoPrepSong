import json
import re
from pathlib import Path
from tqdm import tqdm

LYRIC_ROOT = Path("/data/jianwei/data/music/muse20260112/jsons")
VAD_ROOT = Path("/data/jianwei/data/music/muse20260112/AutoPrepSongV2/20260113/intermediate/vad_output")

def lyric_json_to_vad_json(lyric_json_path: str) -> Path:
    lyric_json_path = Path(lyric_json_path)

    stem = lyric_json_path.stem   # suno_en_015653_0
    return VAD_ROOT / f"{stem}.mp3.json"

def parse_vad(vad_str):
    """
    "[1, 0, 1]" -> list[int]
    """
    return eval(vad_str)

def vad_ratio(vad_list):
    if len(vad_list) == 0:
        return 0.0
    return sum(vad_list) / len(vad_list)

def has_real_lyrics(text):
    """
    判断是否是真正的歌词：
    - 去掉 [xxx] 标签
    - 剩下是否还有可读字符
    """
    if text is None:
        return False
    # 去掉 [intro] [verse] 等
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = text.strip()
    return len(text) > 0


def check_json(json_path, final_json_path):
    """
    返回 True / False
    True  -> correct
    False -> wrong
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        with open(final_json_path, "r", encoding="utf-8") as f:
            final_json = json.load(f)
    except Exception as e:
        print(f"[ERROR] load failed: {json_path}, {e}")
        return False
    
    
    assert len(final_json['segments']) == len(segments)
    for seg_idx, seg in enumerate(segments):
        text = seg.get("text", "")
        vad_str = seg.get("vad_res", "")

        vad_list = parse_vad(vad_str)
        ratio = vad_ratio(vad_list)

        total_correct = True
        if has_real_lyrics(text):
            # 有歌词
            if ratio <= 0.5:
                final_json['segments'][seg_idx]['vad_anomaly'] = True
                total_correct = False
        else:
            # 无歌词
            if ratio >= 0.3:
                final_json['segments'][seg_idx]['vad_anomaly'] = True
                total_correct = False
    with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
    return total_correct

def main(
    scp_path,
):
    correct = []
    wrong = []

    with open(scp_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=116384):
            json_path = line.strip()
            # /mnt/conversationhubhot/yaoyaochang/speech/data/music/muse20260112/jsons/en_part16_of_35/suno_en_015653_0.json
            # /mnt/conversationhubhot/yaoyaochang/speech/data/music/muse20260112/AutoPrepSongV2/20260113/intermediate/vad_output/suno_en_015653_0.mp3.json
            vad_path = lyric_json_to_vad_json(json_path)
            if not (vad_path and vad_path.exists()):
                continue

            if check_json(vad_path, json_path):
                correct.append(json_path)
            else:
                wrong.append(json_path)



    print(f"Correct: {len(correct)}")
    print(f"Wrong:   {len(wrong)}")

if __name__ == "__main__":

    main("/data/jianwei/data/music/muse20260112/Muse_jsons_local.scp")
