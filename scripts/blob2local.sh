# blob_cmd="/root/.local/bin/addfblob"
blob_cmd="/home/jianweiyu/.local/bin/addfblob"

TARGET="exp/ray_eval_date/"
mkdir -p ${TARGET}


subset="20251222_vllm"
subset=20251223
subset=20251225_1
# subset=20251226_1
# subset=20251227
# subset=20251229
# subset=20251231
subset=20260101
blob_cluster=conversationhub
# INPUT=https://conversationhub.blob.core.windows.net/unilm/jianweiyu/VibeVoice/VibeASR/v1/exp/ray_eval

# INPUT=https://conversationhub.blob.core.windows.net/unilm/jianweiyu/VibeVoice/VibeASR/v1/exp/yujie_eval


# INPUT=https://conversationhub.blob.core.windows.net/unilm/jianweiyu/VibeVoice/VibeASR/v1/exp/ray_eval_480s
INPUT=https://conversationhub.blob.core.windows.net/unilm/jianweiyu/VibeVoice/VibeASR/v1/exp/ray_eval_date/${subset}
TARGET=`pwd`/exp/ray_eval_date/


# TARGET="/mnt/jianwei/VibeASR/exp/yujie_eval/dataset_drop"
# INPUT=https://conversationhub.blob.core.windows.net/unilm/jianweiyu/VibeVoice/VibeASR/v1/exp/yujie_eval/dataset_drop/MLC

# /mnt/msranlp/zliang/hf_ckpt/vibevoice_asr_init_ckpt/qwen25_7/initial_model

# TARGET="/data/jianwei/VibeASRTraining/exp"
# INPUT=https://msranlp.blob.core.windows.net/unilm/zliang/hf_ckpt/vibevoice_asr_init_ckpt/qwen25_7/initial_model
# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/music2025"
# TARGET="/mnt/jianwei/data/music/CY"

# blob_cluster=conversationhubhot
# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/raw_disk/20251229_4"
# TARGET="/mnt/jianwei/data/speech/DC20251229"

# blob_cluster=conversationhubhot
# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/multilingual/japanese/jianwei_raw"
# TARGET="/mnt/jianwei/data/speech/BJ20251230"


# blob_cluster=conversationhubhot
# INPUT=https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/music2025/meta/20251227/
# TARGET="/mnt/jianwei/data/music2025/meta/20251227/"


blob_cluster=conversationhubhot
INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/raw_disk/20251229_4/APY190411001_1351小时普通话自然对话语音数据/json_fix"
TARGET="/datadisk/data/speech/DC20251229"


azcopy copy \
    "${INPUT}$(${blob_cmd}   token -a http://135.149.113.42:5950/api -k CsOG9vleDpcc-AqQcTmJlKw4zxrR3aMsWTvTSGv1GVY= -n ${blob_cluster} -c unilm )"\
    "${TARGET}" \
    --recursive --log-level=WARNING