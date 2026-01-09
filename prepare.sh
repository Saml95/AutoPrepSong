pip install -r requirements.txt #--use-deprecated=legacy-resolver

apt-get install ffmpeg -y

pip uninstall flash-attn -y

git submodule update --init --recursive


if [ ! -f "thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt" ]; then
    echo "Downloading pre-trained BS Roformer..."
    mkdir thirdparty/music_Source_Separation_Training/ckpts
    wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt -O thirdparty/music_Source_Separation_Training/ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt
fi

python fetch_pretrained.py ./ckpts