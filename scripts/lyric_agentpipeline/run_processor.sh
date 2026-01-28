# scp_path=/datadisk/data/speech/fisher_long/data20251225/json_local.scp 
# yaml_path=yamls/example_number_en.yaml

# scp_path=/datadisk/data/speech/fisher_long/data20251225/json_fix_local.scp 
# yaml_path=yamls/punctuation_en_gpt5.yaml


# scp_path="/datadisk/data/speech/DC20251229/json_local_fix_punc.scp"
# yaml_path="yamls/example_number_zh.yaml"

# scp_path="/home/jianwei/music/luoxue_20251226/AutoPrepSongV2/20260110/json.scp"
# scp_path="/home/jianwei/music/luoxue/json_local.scp"
# scp_path="/home/jianwei/music/luoxue/2.scp"  
yaml_path="yamls/lyric_gpt5.yaml"
# /data/yan/meta/luoxue_batch6/ -> /mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6
output_dir="/mnt/conversationhubhot/yaoyaochang/speech/data/netease/lyric_after_gpt5"
lrc_dir=/mnt/conversationhubhot/yaoyaochang/speech/data/netease/lyrics
scp_path="/mnt/conversationhubhot/yaoyaochang/speech/data/netease/origin_lyrics.scp"

# mkdir -p $output_dir

# find $lrc_dir -name "*.lrc" > $scp_path

# echo "=============================================="
# echo "SCP: $scp_path"
# echo "len(SCP): $(wc -l < "$scp_path")"
# echo "YAML: $yaml_path"
# echo "OUTPUT: $output_dir"
# echo "=============================================="
# sleep 5

# python run_processor.py --scp $scp_path --yaml "$yaml_path" --output_dir $output_dir

fail_scp="${scp_path%.*}_first_fail.${scp_path##*.}"
python check_fail.py $scp_path $output_dir $fail_scp
python run_processor.py --scp $fail_scp --yaml "$yaml_path" --output_dir $output_dir