git submodule update --init --recursive

# pip uninstall flash-attn

cd thirdparty/SongFormer/src/SongFormer
{
    python utils/fetch_pretrained.py
    
    export PYTHONPATH=../third_party:$PYTHONPATH
    export OMP_NUM_THREADS=1
    export MPI_NUM_THREADS=1
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1

    python infer/infer.py \
    -i /mnt/chenyuyang/AutoPrepSongV2/test.scp \
    -o /mnt/chenyuyang/AutoPrepSongV2/local/test_SongFormer_output \
    --model SongFormer \
    --checkpoint SongFormer.safetensors \
    --config_path SongFormer.yaml \
    -gn 1 \
    -tn 1
}