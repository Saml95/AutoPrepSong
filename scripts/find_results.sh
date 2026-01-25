inputs="
/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch2/AutoPrepSongV2/20260110
/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch3/AutoPrepSongV2/20260110
/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_20251226/AutoPrepSongV2/20260110
"

for input in $inputs; do
    echo "Processing directory: $input"
    # find $input/final_output -name "*.json" > $input/json.scp
    echo `cat $input/json.scp | wc -l` "json files found."
done