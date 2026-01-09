# pip install -r requirements.txt #--use-deprecated=legacy-resolver

# apt-get install ffmpeg -y

# pip uninstall flash-attn -y

# git config --global --add safe.directory `pwd`/AutoPrepSongV2
# git submodule update --init --recursive


# if [ ! -f "thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt" ]; then
#     echo "Downloading pre-trained BS Roformer..."
#     mkdir thirdparty/music_Source_Separation_Training/ckpts
#     wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt -O thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt
# fi

# python fetch_pretrained.py ./ckpts

gits="/home/jianweiyu/exp/AutoPrepSongV2/thirdparty/music_Source_Separation_Training/.git
/home/jianweiyu/exp/AutoPrepSongV2/thirdparty/SongFormer/.git
/home/jianweiyu/exp/AutoPrepSongV2/thirdparty/SongFormer/src/third_party/MuQ/.git
/home/jianweiyu/exp/AutoPrepSongV2/thirdparty/SongFormer/src/third_party/MuQ/src/recipes/pretrain/fairseq/.git
/home/jianweiyu/exp/AutoPrepSongV2/thirdparty/SongFormer/src/third_party/MuQ/src/recipes/pretrain/fairseq/fairseq/model_parallel/megatron/.git
/home/jianweiyu/exp/AutoPrepSongV2/thirdparty/SongFormer/src/third_party/musicfm/.git
"
for item in $gits; do
    rm -rf $item
done