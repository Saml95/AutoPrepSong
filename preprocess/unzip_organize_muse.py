import os
from pathlib import Path
import tarfile
import json
from glob import glob
import re
from torchcodec.decoders import AudioDecoder

ROOT_DIR = "/data/jianwei/data/music/muse20260112/bolshyC_Muse"
JSON_OUT_DIR = "/data/jianwei/data/music/muse20260112/jsons"


def extract_all_tars(root_dir):
    """解压所有 tar 文件到同名目录"""
    tar_files = glob(os.path.join(root_dir, "*.tar"))

    for tar_path in tar_files:
        name = os.path.splitext(os.path.basename(tar_path))[0]
        extract_dir = os.path.join(root_dir, name)

        if os.path.exists(extract_dir):
            print(f"[SKIP] {name} already extracted")
            continue

        print(f"[EXTRACT] {tar_path} -> {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir)


def build_audio_index(root_dir):
    """
    建立 audio 文件名 -> 完整相对路径 的索引
    用于快速匹配 audio_path
    """
    index = {}
    for part_dir in os.listdir(root_dir):
        part_path = os.path.join(root_dir, part_dir)
        if not os.path.isdir(part_path):
            continue
        if not (part_dir.startswith("cn_part") or part_dir.startswith("en_part")):
            continue

        for root, _, files in os.walk(part_path):
            for f in files:
                if f.endswith((".mp3", ".wav", ".flac")):
                    abs_path = os.path.abspath(
                        os.path.join(root, f)
                    )
                    index[f] = abs_path
    return index


def process_jsonl_files(root_dir, json_out_dir):
    jsonl_files = glob(os.path.join(root_dir, "*_*.jsonl"))
    audio_index = build_audio_index(root_dir)

    cnt, wrong_seg_cnt, wrong_end_cnt = 0, 0, 0
    for jsonl_path in jsonl_files:
        print(f"[PROCESS] {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line)
                song_id = obj["song_id"]
                old_audio_path = obj["audio_path"]
                audio_name = os.path.basename(old_audio_path)
                total_len = AudioDecoder(audio_index[audio_name]).metadata.duration_seconds

                if audio_name not in audio_index:
                    print(f"[WARN] audio not found: {audio_name}")
                    continue
                
                fout = (Path(json_out_dir) / "/".join(Path(audio_index[audio_name]).parts[-2:])).with_suffix(".json")
                fout.parent.mkdir(exist_ok=True, parents=True)
                fout = open(fout, 'w')
                new_obj = {
                    "song_id": song_id,
                    "audio_path": os.path.join("/mnt/conversationhubhot/yaoyaochang/speech/data/music/muse20260112/bolshyC_Muse", \
                                            os.path.relpath(audio_index[audio_name], root_dir)), #audio_index[audio_name],
                    "audio_length": total_len,
                    "segments": [{'text': f"[{re.subn(r"\d+", "", i['section'].lower())[0].strip()}] {i['text']}", 'start': i['startS'], 'end': i['endS'], 'speaker': None} for i in obj['sections'] if i['startS'] < total_len],
                    "info": json.dumps(obj)
                }

                if len(new_obj['segments']) < len(obj['sections']):
                    wrong_seg_cnt += 1
                
                if new_obj['segments'][-1] > total_len:
                    wrong_end_cnt += 1


                fout.write(json.dumps(new_obj, ensure_ascii=False, indent=2))
                fout.close()
                cnt += 1

    print(f"[DONE] saved {cnt} entries to {json_out_dir}")
    print(f"wrong segment: {wrong_seg_cnt} # of segs | {wrong_end_cnt} time of ends")



if __name__ == "__main__":
    # extract_all_tars(ROOT_DIR)
    process_jsonl_files(ROOT_DIR, JSON_OUT_DIR)
