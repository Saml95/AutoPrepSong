from faster_whisper import WhisperModel, BatchedInferencePipeline
from models.utils import read_audio, save_audio
import os
import torch
import json

from tqdm import tqdm

audio_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo/2.mp3"
audio = read_audio(audio_path)

output_dir = "output_whisper_{}".format(audio_path.split("/")[-1].split(".")[0])
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(f"{output_dir}/segments.pt"):
    segments = torch.load(f"{output_dir}/segments.pt")
else:
    model = WhisperModel("turbo", device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)
    segments, info = batched_model.transcribe(audio, batch_size=16, vad_filter=True, word_timestamps=True)
    new_segments = []
    for segment in segments:
        new_segments.append(segment)
    torch.save(new_segments, f"{output_dir}/segments.pt")
    segments = new_segments

output_segments = []

cur_start = None
cur_end = None
cur_text = ""

subseg_idx = 0

for segment in tqdm(segments):
    for word in segment.words:
        if cur_start is None:
            cur_start = word.start
        cur_text += word.word
        if word.word[-1] in [".", "!", "?"]:
            cur_end = word.end
            

            cur_start = max(0, cur_start - 0.1)
            cur_end = min(cur_end + 0.1, len(audio)/16000)
            audio_subseg = audio[int(cur_start*16000):int(cur_end*16000)]
            save_audio(f"{output_dir}/chunk_{subseg_idx}.wav", audio_subseg, 16000)


            output_segments.append({
                "start": cur_start,
                "end": cur_end,
                "text": cur_text,
                "audio_path": os.path.abspath(f"{output_dir}/chunk_{subseg_idx}.wav")
            })
            
            subseg_idx += 1
            cur_text = ""
            cur_start = None
            cur_end = None

with open(f"{output_dir}/audio.json", "w") as f:
    json.dump(output_segments, f, indent=4, ensure_ascii=False)


    #     print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
    # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))