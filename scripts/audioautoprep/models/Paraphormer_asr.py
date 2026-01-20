from funasr import AutoModel
import json
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
# wav_path = f"{model.model_path}/example/asr_example.wav"
wav_path = "/mnt/conversationhubhot/yaoyaochang/speech/data/xyz/audio_all/5e280b1f418a84a0461f2654.mp3"

model = AutoModel(model="paraformer-zh",  vad_model="fsmn-vad",  punc_model="ct-punc", 
                  spk_model="cam++", 
                  )
res = model.generate(input=wav_path, 
                     batch_size_s=300, 
                     hotword='魔搭')
print(res)
# import pdb; pdb.set_trace()
with open("res.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=4)