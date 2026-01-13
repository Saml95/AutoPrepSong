# blob_cmd="/root/.local/bin/addfblob"
blob_cmd="/home/zilongwang/.local/bin/addfblob"


# INPUT="/data/jianwei/VibeASR"
# TARGET=https://conversationhub.blob.core.windows.net/unilm/jianweiyu/VibeVoice/VibeASR_Eval/backup

# INPUT="/data/yujietu/J-CHAT"
# TARGET="https://conversationhub.blob.core.windows.net/unilm/jianweiyu/datasets/vibevoice/sft/"

# INPUT="/data/jianwei/data/music/CY/meta"
# TARGET="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/music2025/"

# INPUT="/mnt/jianwei/data/speech/DC20251229/20251229_4/APY190411001_1351小时普通话自然对话语音数据/json"
# TARGET="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/raw_disk/20251229_4/APY190411001_1351小时普通话自然对话语音数据"

# INPUT="/mnt/jianwei/data/MiHoYO/2"
# TARGET="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/MiHoYO/20251231"

INPUT="/data/jianwei/music/a50w"
TARGET="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/a50w"

# blob_cluster=conversationhubhot
# TARGET="https://conversationhubhot.blob.core.windows.net/unilm/jianweiyu/"
# INPUT="/home/jianweiyu/exp/AutoPrepSongV2"

# # 同步Muse数据
# INPUT="/data/jianwei/data/music/muse20260112/"
# TARGET="https://conversationhubhot.blob.core.windows.net/unilm/yaoyaochang/speech/data/music/muse20260112"

# AZCOPY_BUFFER_GB=300 AZCOPY_CONCURRENCY_VALUE=AUTO AZCOPY_CONCURRENT_FILES=1024 azcopy copy \
#     "${INPUT}" \
#     "${TARGET}$(${blob_cmd}   token -a http://135.149.113.42:5950/api -k CsOG9vleDpcc-AqQcTmJlKw4zxrR3aMsWTvTSGv1GVY= -n conversationhubhot -c unilm)" \
#     --recursive --log-level=WARNING 



# # 数据集名称到语言的映射
# declare -A dataset_to_lang=(
#     ["德语MDT-AJ297"]="german"
#     ["韩语MDT-AE067"]="korean"
#     ["葡萄牙MDT-AF027"]="portuguese"
#     ["日语MDT-AF008"]="japanese_1"
#     ["日语MDT-AJ039"]="japanese_2"
#     ["意大利语MDT-AJ110"]="italian"
# )

# for dataset in "${!dataset_to_lang[@]}"; do 
#     lang="${dataset_to_lang[$dataset]}"
#     azcopy sync "https://conversationhubhot.blob.core.windows.net/unilm/jianweiyu/datasets/vibevoice/sft/$dataset/data20251230/json/$(${blob_cmd} token -a http://135.149.113.42:5950/api -k CsOG9vleDpcc-AqQcTmJlKw4zxrR3aMsWTvTSGv1GVY= -n conversationhubhot -c unilm)" "https://conversationhubhot.blob.core.windows.net/unilm/jianweiyu/datasets/vibevoice/sft/combine_multilingual_100h/$lang/$(${blob_cmd} token -a http://135.149.113.42:5950/api -k CsOG9vleDpcc-AqQcTmJlKw4zxrR3aMsWTvTSGv1GVY= -n conversationhubhot -c unilm)" --recursive 
# done




AZCOPY_BUFFER_GB=300 AZCOPY_CONCURRENCY_VALUE=AUTO AZCOPY_CONCURRENT_FILES=1024 azcopy sync \
    "${INPUT}" \
    "${TARGET}$(${blob_cmd}   token -a http://135.149.113.42:5950/api -k CsOG9vleDpcc-AqQcTmJlKw4zxrR3aMsWTvTSGv1GVY= -n conversationhubhot -c unilm)" \
    --recursive --log-level=WARNING