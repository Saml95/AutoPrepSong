INPUT_SCP=/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue_20251226_all.scp
META_DIR=local/luoxue_20251226_all
# $SEP_MODEL=bs_roformer
SONGFORMER_OUTPUT_DIR=local/songformer_output/luoxue_20251226_all
SEPARATION_OUTPUT_DIR=local/separation_output/bs_roformer/luoxue_20251226_all
VAD_OUTPUT_DIR=local/vad_output/luoxue_20251226_all
OUTPUT_DIR=local/final/luoxue_20251226_all


# git submodule update --init --recursive

# pip uninstall flash-attn # requirement自动装的flash-atn会有适配问题，推荐自己按照环境去找编译好的版本装



### STEP 1: Prepare Metadata and Features ###
python3 scripts/preprocess_scp.py $INPUT_SCP $META_DIR


WAV_SCP=$META_DIR/wav.scp
LRC_SCP=$META_DIR/lrc.scp
LRCWAV_SCP=$META_DIR/lrc2wav.scp

### For TEST Purpose Only: Use a small subset of data ###
head -n 100 $WAV_SCP > $WAV_SCP.tmp
head -n 100 $LRC_SCP > $LRC_SCP.tmp
head -n 100 $LRCWAV_SCP > $LRCWAV_SCP.tmp
WAV_SCP=$WAV_SCP.tmp
LRC_SCP=$LRC_SCP.tmp
LRCWAV_SCP=$LRCWAV_SCP.tmp



### STEP 2: Structural Analysis ###
{
    cd thirdparty/SongFormer/src/SongFormer
    python utils/fetch_pretrained.py
    
#     export PYTHONPATH=../third_party:$PYTHONPATH
#     export OMP_NUM_THREADS=1
#     export MPI_NUM_THREADS=1
#     export NCCL_P2P_DISABLE=1
#     export NCCL_IB_DISABLE=1

#     python infer/infer.py \
#     -i /mnt/chenyuyang/AutoPrepSongV2/test.scp \
#     -o /mnt/chenyuyang/AutoPrepSongV2/local/test_SongFormer_output \
#     --model SongFormer \
#     --checkpoint SongFormer.safetensors \
#     --config_path SongFormer.yaml \
#     -gn 8 \
#     -tn 1

    cd ../../../../
}
python3 scripts/run_struct_anal.py \
    -i $WAV_SCP \
    -o $SONGFORMER_OUTPUT_DIR \
    --model SongFormer \
    --checkpoint SongFormer.safetensors \
    --config_path SongFormer.yaml 


### STEP 3: Vocal/Accmp Separation ###

if [ ! -f "thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt" ]; then
    echo "Downloading pre-trained BS Roformer..."
    mkdir thirdparty/music_Source_Separation_Training/ckpts
    wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt -O thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt
fi
# # pip install loralib ml_collections pytorch_optimizer rotary_embedding_torch

python3 scripts/run_separation_new.py \
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

python3 vis/vis_struct.py --share