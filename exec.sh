INPUT_SCP=/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue_20251226_all.scp
META_DIR=local/luoxue_20251226_all
# $SEP_MODEL=bs_roformer
SONGFORMER_OUTPUT_DIR=local/songformer_output/luoxue_20251226_all
SEPARATION_OUTPUT_DIR=local/separation_output/bs_roformer/luoxue_20251226_all
VAD_OUTPUT_DIR=local/vad_output/luoxue_20251226_all
OUTPUT_DIR=local/final/luoxue_20251226_all



## STEP 1: Prepare Metadata and Features ###
python3 scripts/preprocess_scp.py $INPUT_SCP $META_DIR


WAV_SCP=$META_DIR/wav.scp
LRC_SCP=$META_DIR/lrc.scp
LRCWAV_SCP=$META_DIR/lrc2wav.scp

# ### For TEST Purpose Only: Use a small subset of data ###
# head -n 100 $WAV_SCP > $WAV_SCP.tmp
# head -n 100 $LRC_SCP > $LRC_SCP.tmp
# head -n 100 $LRCWAV_SCP > $LRCWAV_SCP.tmp
# WAV_SCP=$WAV_SCP.tmp
# LRC_SCP=$LRC_SCP.tmp
# LRCWAV_SCP=$LRCWAV_SCP.tmp



### STEP 2: Structural Analysis ###

python3 scripts/run_struct_anal.py \
    -i $WAV_SCP \
    -o $SONGFORMER_OUTPUT_DIR \
    --model SongFormer \
    --checkpoint SongFormer.safetensors \
    --config_path SongFormer.yaml 


### STEP 3: Vocal/Accmp Separation ###
python3 scripts/run_separation.py \
    --model_type bs_roformer \
    --config_path thirdparty/music_Source_Separation_Training/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
    --start_check_point thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
    --extract_instrumental \
    --input_folder $WAV_SCP \
    --store_dir $SEPARATION_OUTPUT_DIR


### STEP 4: Sentence VAD ###
python3 scripts/run_sentence_vad.py $LRCWAV_SCP $SEPARATION_OUTPUT_DIR $VAD_OUTPUT_DIR

### STEP 5: Post-processing and Save Results ###
python3 scripts/postprocess_combine_all.py $LRCWAV_SCP $SONGFORMER_OUTPUT_DIR $SEPARATION_OUTPUT_DIR $VAD_OUTPUT_DIR $OUTPUT_DIR

ls /mnt/chenyuyang/AutoPrepSongV2/local/final/luoxue_20251226_all/*.json > /mnt/chenyuyang/AutoPrepSongV2/local/luoxue_test_final.scp

# python3 vis/vis_struct.py --share