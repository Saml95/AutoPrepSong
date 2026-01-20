blob_cmd="/root/.local/bin/addfblob"


TARGET="exp/ray_eval_date/"
mkdir -p ${TARGET}


subset="20251222_vllm"
subset=20251223
subset=20251225_1
# subset=20251226_1
# subset=20251227
# subset=20251229
# subset=20251231
subset=20260104
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
# TARGET="/data/jianwei/speech/DC20251229"

# blob_cluster=conversationhubhot
# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/multilingual/japanese/jianwei_raw"
# TARGET="/mnt/jianwei/data/speech/BJ20251230"


# blob_cluster=conversationhubhot
# INPUT=https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/music2025/meta/20251227/
# TARGET="/mnt/jianwei/data/music2025/meta/20251227/"

# INPUT=https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/MiHoYO/
# TARGET="/data/jianwei/speech/MiHoYO"



# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/zhiliang/speech/data/yt/labels_20250108/v1_partial_gemini/"
# TARGET="/data/jianwei/speech/"

# INPUT=https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/a50w
# INPUT=https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/netease
# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/yan/meta"
# TARGET="/data/jianwei/music/luoxue"
# blob_cluster=conversationhubhot

# INPUT=https://conversationhub.blob.core.windows.net/unilm/jianweiyu/data/music/acadamic/yue-labels/a50w
# TARGET="/data/jianwei/music/a50w/lyrics/"
# blob_cluster=conversationhub

# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/muse20260112_concat/concat_json_group"
# TARGET="/data/jianwei/data/music/muse20260112_concat"



# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/muse20260112/jsons"
# TARGET="/home/jianwei/data/music/muse20260112/"


# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/zhiliang/speech/data/gpt_refine_asr/xyz_long_v5.0_iter3_v3.0_speakkermapped_merged0.01_v0.7/"
# TARGET="/home/jianwei/data/speech/xyz_gpt/"

# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/zhiliang/speech/data/gpt_refine_asr/xyz_pt_long_v4.6_all_speakermapped_dehallu_aligned_zh_checked2_iter3_v3.4_merged0.01to0.03_filter_v0.7/"
# TARGET="/data/jianwei/speech/longaudio/"
# INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/yan/meta/luoxue_20251226/AutoPrepSongV2/20260110/final_output/"
# TARGET="/home/jianwei/music/luoxue_20251226/AutoPrepSongV2/20260110/"
# blob_cluster=conversationhubhot



INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/yan/meta/luoxue_20251226/vibevoice-asr/"
TARGET="/home/jianwei/music/luoxue_20251226/"
blob_cluster=conversationhubhot

AZCOPY_BUFFER_GB=300 AZCOPY_CONCURRENCY_VALUE=AUTO AZCOPY_CONCURRENT_FILES=1024 \
    azcopy copy \
    "${INPUT}$(${blob_cmd}   token -a http://135.149.113.42:5950/api -k CsOG9vleDpcc-AqQcTmJlKw4zxrR3aMsWTvTSGv1GVY= -n ${blob_cluster} -c unilm )"\
    "${TARGET}" \
    --recursive --log-level=WARNING



# for item in luoxue_batch2 luoxue_batch3 luoxue_batch4 luoxue_batch5; do
#     INPUT="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/yan/meta/${item}/AutoPrepSongV2/20260110/final_output/"
#     TARGET="/home/jianwei/music/luoxue/json_group/${item}/"
#     mkdir -p ${TARGET}
#     blob_cluster=conversationhubhot

#     AZCOPY_BUFFER_GB=300 AZCOPY_CONCURRENCY_VALUE=AUTO AZCOPY_CONCURRENT_FILES=1024 \
#     azcopy sync \
#     "${INPUT}$(${blob_cmd}   token -a http://135.149.113.42:5950/api -k CsOG9vleDpcc-AqQcTmJlKw4zxrR3aMsWTvTSGv1GVY= -n ${blob_cluster} -c unilm )"\
#     "${TARGET}" \
#     --recursive --log-level=WARNING
# done