import os
import tarfile
import json
from glob import glob

ROOT_DIR = "/data/jianwei/data/music/muse20260112/bolshyC_Muse"
OUT_JSONL = os.path.join(ROOT_DIR, "merged_with_new_path.jsonl")


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
                    rel_path = os.path.relpath(
                        os.path.join(root, f), root_dir
                    )
                    index[f] = rel_path
    return index


def process_jsonl_files(root_dir, out_jsonl):
    jsonl_files = glob(os.path.join(root_dir, "*_*.jsonl"))
    audio_index = build_audio_index(root_dir)

    cnt = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for jsonl_path in jsonl_files:
            print(f"[PROCESS] {jsonl_path}")
            with open(jsonl_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line)
                    song_id = obj["song_id"]
                    old_audio_path = obj["audio_path"]
                    audio_name = os.path.basename(old_audio_path)

                    if audio_name not in audio_index:
                        print(f"[WARN] audio not found: {audio_name}")
                        continue

                    new_obj = {
                        "song_id": song_id,
                        "audio_path": audio_index[audio_name]
                    }
                    fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                    cnt += 1

    print(f"[DONE] saved {cnt} entries to {out_jsonl}")


if __name__ == "__main__":
    extract_all_tars(ROOT_DIR)
    process_jsonl_files(ROOT_DIR, OUT_JSONL)
