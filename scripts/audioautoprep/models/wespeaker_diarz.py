import wespeaker
# wav1_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/output_whisper_2/chunk_0.wav"
# wav2_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/output_whisper_2/chunk_5.wav"
# wav_path = '/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo_processed_turbo/2/2_0.mp3'

# model = wespeaker.load_model('vblinkp')
# embedding1 = model.extract_embedding(wav1_path)
# embedding2 = model.extract_embedding(wav2_path)

# similarity = model.compute_similarity(wav1_path, wav2_path)
# diar_result = model.diarize(wav1_path)

# print(similarity)
# import pdb; pdb.set_trace()

import wespeaker
model = wespeaker.load_model('campplus')
model.set_diarization_params(
    min_duration=0.255, 
    window_secs=0.75, 
    period_secs=0.5, 
    frame_shift=10, 
    batch_size=32, 
    subseg_cmn=True)
model.set_vad(False)
wav_path = '/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo_processed_turbo/2/2_0.mp3'
diar_result = model.diarize(wav_path)
print(diar_result)
import pdb; pdb.set_trace()