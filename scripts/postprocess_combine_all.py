import sys
from pathlib import Path
import json
from torchcodec.decoders import AudioDecoder

if __name__=='__main__':
    lrc2wav_path, struct_dir, sep_dir, output_dir = sys.argv[1:]
    struct_dir, sep_dir, output_dir = Path(struct_dir), Path(sep_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lrc2wav = open(lrc2wav_path).read().strip().splitlines()

    for line in lrc2wav:
        lrc_path, wav_path = line.split('\t')
        basename = Path(wav_path).name
        struct_json = json.load(open(struct_dir / f"{basename}.json", 'r', encoding='utf-8'))
        for k in struct_json:
            k['text'] = k.pop('label')
            k['speaker'] = "structure"

        local_sep_dirs = [str(sep_dir / basename / d) for d in ['vocals', 'instrumental']]

        output_json = {
            "audio_path": wav_path,
            "audio_length": AudioDecoder(wav_path).metadata.duration_seconds,
            "segments": struct_json,
            "separation_results":local_sep_dirs,
            "info": {}
        }
        # print(output_json)
        json.dump(output_json, 
                  open(output_dir / f"{basename}.json", 'w', encoding='utf-8'), 
                  ensure_ascii=False, indent=4)
        
