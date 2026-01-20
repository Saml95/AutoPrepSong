import sys
from pathlib import Path
import json
from torchcodec.decoders import AudioDecoder

raise NotImplementedError("Out of Date")
if __name__=='__main__':
    lrc2wav_path, struct_dir, sep_dir, vad_dir, output_dir = sys.argv[1:]
    struct_dir, sep_dir, vad_dir, output_dir = Path(struct_dir), Path(sep_dir), Path(vad_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lrc2wav = open(lrc2wav_path).read().strip().splitlines()

    for line in lrc2wav:
        lrc_path, wav_path = line.split('\t')
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
                    "speaker": "0",
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
        
