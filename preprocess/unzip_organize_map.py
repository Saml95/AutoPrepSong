import os
from pathlib import Path
import zipfile
import json
from glob import glob
import re
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm
import tarfile

ROOT_DIR = "/data/jianwei/music/a50w/m-a-p_a50w/audio/"
JSON_OUT_DIR = "/data/jianwei/music/a50w/jsons"
LYRICS_DIR = "/data/jianwei/music/a50w/lyrics/"

import zipfile
from pathlib import Path

import subprocess
from pathlib import Path
import getpass

def unzip_all_7z(
    root="/data/jianwei/music/a50w/m-a-p_a50w/audio/",
    password = 'academiconly'
):
    # password = getpass.getpass("Zip password: ")

    root = Path(root)
    zip_files = sorted(root.glob("*.zip"))

    for zip_path in zip_files:
        out_dir = zip_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "7z", "x",
            f"-p{password}",
            "-y",                 # 自动 yes
            str(zip_path),
            f"-o{out_dir}"
        ]

        ret = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if ret.returncode == 0:
            print(f"[OK] {zip_path.name}")
        else:
            print(f"[FAIL] {zip_path.name}")
            print(ret.stderr)


def extract_all_tars(root_dir):
    """解压所有 tar 文件到同名目录，解压成功后删除原 tar"""
    tar_files = glob(os.path.join(root_dir, "**/*.tar"), recursive=True)

    for tar_path in tar_files:
        name = os.path.splitext(os.path.basename(tar_path))[0]
        extract_dir = os.path.join(root_dir, name)

        # if os.path.exists(extract_dir): # TODO 部分目录重合了然后跳过了，一会儿需要重跑一遍
        #     print(f"[SKIP] {name} already extracted")
        #     continue

        print(f"[EXTRACT] {tar_path} -> {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_dir)

            # 只有解压成功才删除
            os.remove(tar_path)
            print(f"[DELETE] {tar_path}")

        except Exception as e:
            print(f"[FAIL] {tar_path}: {e}")
            # 可选：失败时清理解压到一半的目录
            # shutil.rmtree(extract_dir, ignore_errors=True)



def load_all_jsonl(jsonl_dir):
    all_data = {}
    jsonl_files = glob(str(Path(jsonl_dir) / "*.jsonl"))
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line = json.loads(line)
                    all_data[line['id']] = line
    return all_data


def build_audio_index(audio_dir):
    audio_files = glob(str(Path(audio_dir) / "**/*.mp3"), recursive=True)
    audio_map = {}
    for ap in audio_files:
        p = Path(ap)
        if p.is_file():
            audio_map[p.stem] = str(p)
    return audio_map


def convert_segmented_lyrics(segmented_lyrics):
    new_segments = []
    for seg in segmented_lyrics:
        start = float(seg["offset"])
        duration = float(seg["duration"])
        end = start + duration
        new_segments.append({
            "start": start,
            "end": end,
            "text": seg["line_content"].strip().replace(']\n', '] ').replace('\n', '. '),
            "speaker": None
        })
    return new_segments


def get_output_json_path(audio_path):
    """
    根据 audio_path 生成 json 输出路径
    """
    audio_path = Path(audio_path)
    rel_path = audio_path.relative_to(ROOT_DIR)
    parts = rel_path.parts

    # 深度为 1：xxx.wav
    if len(parts) == 1:
        out_rel = Path("others") / audio_path.stem
    else:
        out_rel = Path(*parts[:-1]) / audio_path.stem

    return Path(JSON_OUT_DIR) / out_rel.with_suffix(".json")


def process():
    all_lyrics = load_all_jsonl(LYRICS_DIR)
    print(f"load {len(all_lyrics)} lyric lines!")
    audio_map = build_audio_index(ROOT_DIR)
    print(f"load {len(audio_map)} audios!")

    saved_cnt = 0

    for idx in tqdm(audio_map, total=len(audio_map)):
        if idx not in audio_map:
            print(f"fail to find lyrics ({audio_map[idx]})")
            continue
        data = all_lyrics[idx]
        audio_path = audio_map[idx]
        
        segmented_lyrics = data["splitted_lyrics"]["segmented_lyrics"]

        new_data = {
            "idx": idx,
            "audio_path": audio_path.replace("/data/jianwei/music/a50w", "/mnt/conversationhubhot/yaoyaochang/speech/data/music/a50w"),
            "audio_length": data["audio_length_in_sec"],
            "segments": convert_segmented_lyrics(segmented_lyrics),
            "info": str({
                k: v for k, v in data.items()
                if k not in ["idx", "audio_length_in_sec"]
            })
        }
        # breakpoint()
        # 6. 保存到对应路径
        out_path = get_output_json_path(audio_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        saved_cnt += 1

    print(f"Saved JSON files: {saved_cnt}")


if __name__ == "__main__":
    # unzip_all_7z(ROOT_DIR)
    # extract_all_tars(ROOT_DIR)
    results = process()



