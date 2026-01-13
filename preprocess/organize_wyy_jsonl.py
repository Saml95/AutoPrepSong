import json
from pathlib import Path
from tqdm import tqdm
import librosa

# /mnt/conversationhubhot/yaoyaochang/speech/data/music/jianwei_raw_kuwo/
# scp_path: /mnt/conversationhubhot/yaoyaochang/speech/data/music/kuwo_raw_disk_all_music_file.scp

jsl = json.load(open("/data/jianwei/music/netease/results.json"))
id2meta = {i['id']: i for i in jsl if i['歌词'] != ''}

print(f"{len(id2meta)} / {len(jsl)} have lyrics")
outdir = Path("/data/jianwei/music/netease")
(outdir / 'lyrics').mkdir(exist_ok=True, parents=True)

save_jsonl = open(outdir / "wyy_valid.jsonl", 'w')

process_files = open("/data/jianwei/music/netease/songs.scp").readlines()
for l in tqdm(process_files, total=len(process_files)):
    audio_path = Path(l.strip())
    idx = audio_path.stem
    if idx not in id2meta:
        # print(f"{audio_path} does not find lyrics!")
        continue

    try:
        aa = librosa.load(audio_path)
    except:
        continue
    
    meta = id2meta[idx]
    lrc = meta['歌词']
    with open(outdir/'lyrics'/f"{idx}.lrc", 'w') as f:
        f.write(lrc)

    save_jsonl.write(json.dumps({'audio_path': str(audio_path), "lyric_path": str(outdir/'lyrics'/f"{idx}.lrc"), "language": meta['语种'], "url": meta["详情url"], "name": meta["名称"]}, ensure_ascii=False)+'\n')
