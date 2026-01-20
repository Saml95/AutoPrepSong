import argparse
import re
from pathlib import Path
from omegaconf import OmegaConf
from loguru import logger
import json
import os
from tqdm import tqdm

def extract_label(text):
    """
    从 '[LABEL] xxx' 中提取 LABEL
    """
    m = re.match(r"\[(.*?)\]\s*(.*)", text)
    if m is None:
        return None, text
    return m.group(1), m.group(2)


def ends_with_punctuation(text):
    return text.rstrip().endswith(('.', '!', '?', '。', '！', '？', ',', '，', ';', '；'))

def merge_segments(segments, max_duration=15.0):
    """
    合并连续 segments:
    - label 相同
    - 合并后总时长 < max_duration
    """
    if not segments:
        return []

    merged = []

    cur = segments[0].copy()
    cur_label, cur_text = extract_label(cur["text"])
    cur["text"] = cur_text

    for seg in segments[1:]:
        label, text = extract_label(seg["text"])

        assert seg["start"] >= cur["end"]

        can_merge = (
            label == cur_label
            and (seg["end"] - cur["start"]) <= max_duration
        )

        if can_merge:
            # 合并
            cur["end"] = seg["end"]
            if text:
                if not ends_with_punctuation(cur["text"]) and cur["text"].strip() != '':
                    cur["text"] += "."
                cur["text"] += " " + text
        else:
            # 结束当前段
            cur["text"] = f"[{cur_label}] {cur['text']}"
            merged.append(cur)

            # 新起一段
            cur = seg.copy()
            cur_label, cur_text = extract_label(cur["text"])
            cur["text"] = cur_text

    # 收尾
    cur["text"] = f"[{cur_label}] {cur['text']}"
    merged.append(cur)

    return merged



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_scp", "-i",
        type=str,
        default=None,
        help="Path to the input JSONL file list"
    )
    parser.add_argument(
        "--output_suffix", "-o",
        type=str,
        default='_merge',
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
        "--merge_seconds",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--remove_old",
        action='store_true',
    )

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    jsonl_list = [i.strip() for i in open(args.input_scp, 'r').readlines()]

    if args.chunk_size is None:
        end = None
    else:
        end = args.start_idx+args.chunk_size
    lines = jsonl_list[args.start_idx: end]
    for line in tqdm(lines, total=len(lines)):
        try:
            jsd = json.load(open(line, 'r'))
            old_seg = jsd['segments']
            new_seg = merge_segments(old_seg, args.merge_seconds)
            jsd['segments'] = new_seg
            new_save_dir = Path(line).with_stem(Path(line).stem+args.output_suffix)
            json.dump(jsd, open(new_save_dir, 'w'), ensure_ascii=False, indent=2)
            if new_save_dir != line and args.remove_old:
                os.remove(line)
        except Exception as e:
            raise e
            logger.warning(f"[FAIL] merge {line} failed!\n{e}")