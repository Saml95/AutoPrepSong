git submodule update --init --recursive

# pip uninstall flash-attn


# {
#     cd thirdparty/SongFormer/src/SongFormer
#     python utils/fetch_pretrained.py
    
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

#     cd ../../../../
# }

python3 scripts/run_struct_anal.py \
    -i /mnt/chenyuyang/AutoPrepSongV2/test.scp \
    -o /mnt/chenyuyang/AutoPrepSongV2/local/test_SongFormer_output \
    --model SongFormer \
    --checkpoint SongFormer.safetensors \
    --config_path SongFormer.yaml 

if [ ! -f "thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt" ]; then
    echo "Downloading pre-trained BS Roformer..."
    mkdir thirdparty/music_Source_Separation_Training/ckpts
    wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt -O thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt
fi
# pip install loralib ml_collections pytorch_optimizer rotary_embedding_torch

python3 scripts/run_separation_new.py \
    --model_type bs_roformer \
    --config_path thirdparty/music_Source_Separation_Training/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
    --start_check_point thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
    --extract_instrumental \
    --input_folder test.scp \
    --store_dir local/test_separation_output/bs_roformer