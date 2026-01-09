from pathlib import Path
import sys




def load_and_category(scp_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(scp_path, "r", encoding="utf-8") as f:
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
    
    f = open(output_dir / "lrc2wav.scp", "w", encoding="utf-8")
    fw = open(output_dir / "wav.scp", "w", encoding="utf-8")
    fl = open(output_dir / "lrc.scp", "w", encoding="utf-8")
    for lrc, wavs in lrc2wavs.items():
        fl.write(f"{lrc}\n")
        for wav in wavs:
            f.write(f"{lrc}\t{wav}\n")
            fw.write(f"{wav}\n")
            
    

if __name__ == "__main__":

    load_and_category(*sys.argv[1:])