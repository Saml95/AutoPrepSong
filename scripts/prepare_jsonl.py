#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare JSONL files from scp file.
Each line in the output jsonl is a dict with keys: audio_path, lyric_path
"""

import argparse
import json
from pathlib import Path

# Supported audio formats with priority (lower index = higher priority)
AUDIO_FORMATS_PRIORITY = ['.flac', '.wav', '.mp3', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.aiff', '.ape', '.alac', '.webm']
AUDIO_FORMATS = set(AUDIO_FORMATS_PRIORITY)


def load_and_category(input_scp: str, output_dir: str = None):
    """
    Load scp file and categorize audio/lyric files, then output jsonl file.
    
    Args:
        input_scp: Path to the input scp file (each line is a file path)
        output_dir: Directory to save output jsonl file. If None, save to the same directory as input_scp
    """
    with open(input_scp, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    idx2lrc, idx2wav = {}, {}
    for l in lines:
        l = Path(l.strip())
        if not l.suffix:
            continue
        if l.suffix == '.lrc':
            idx2lrc[l.stem] = str(l)
        elif l.suffix.lower() in AUDIO_FORMATS:
            # Store as list of (priority, path) tuples for later sorting
            priority = AUDIO_FORMATS_PRIORITY.index(l.suffix.lower()) if l.suffix.lower() in AUDIO_FORMATS_PRIORITY else len(AUDIO_FORMATS_PRIORITY)
            idx2wav[l.stem] = idx2wav.get(l.stem, []) + [(priority, str(l))]
        else:
            print(f"Unknown file type: {l}")
    
    # Sort by priority and keep only the highest priority audio for each stem
    for stem in idx2wav:
        idx2wav[stem].sort(key=lambda x: x[0])  # Sort by priority
        idx2wav[stem] = idx2wav[stem][0][1]  # Keep only the best one (as string, not list)
    
    # Collect all stems from both lrc and audio
    all_stems = set(idx2lrc.keys()) | set(idx2wav.keys())
    
    # Check for missing pairs and print warnings
    paired_records = []
    missing_lyric_count = 0
    missing_audio_count = 0
    
    for stem in all_stems:
        lrc_path = idx2lrc.get(stem)
        wav_path = idx2wav.get(stem)
        
        if lrc_path is None:
            print(f"[Warning] Missing lyric file for audio: {wav_path}")
            missing_lyric_count += 1
            continue
        if wav_path is None:
            print(f"[Warning] Missing audio file for lyric: {lrc_path}")
            missing_audio_count += 1
            continue
        
        # Both exist, add to paired records
        paired_records.append({
            "audio_path": wav_path,
            "lyric_path": lrc_path
        })
    
    # Print statistics
    total_audio = len(idx2wav)
    total_lyric = len(idx2lrc)
    print(f"Audio files: {total_audio} total, {missing_lyric_count} missing lyric")
    print(f"Lyric files: {total_lyric} total, {missing_audio_count} missing audio")
    
    # Determine output path
    input_path = Path(input_scp)
    output_filename = input_path.stem + ".jsonl"
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
    else:
        output_path = input_path.parent / output_filename
    
    # Write jsonl file (audio_path, lyric_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in paired_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Total: {len(paired_records)} paired records")
    print(f"Saved to: {output_path}")
    
    return paired_records


def main():
    parser = argparse.ArgumentParser(description="Prepare JSONL files from scp file")
    parser.add_argument("--input_scp", "-i", type=str, required=True,
                        help="Path to input scp file")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory for jsonl file. Default: same directory as input_scp")
    args = parser.parse_args()
    
    load_and_category(args.input_scp, args.output_dir)


if __name__ == "__main__":
    main()
