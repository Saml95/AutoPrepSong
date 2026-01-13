basedir="../../"

cd $basedir


inputs="
/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue_batch4
/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue_batch5
"

output_base_dir="/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta"

for input in $inputs; do
    output_dir="$output_base_dir/$(basename $input)"
    mkdir -p "$output_dir"
    find $input -name "*.*" > "$output_dir/files.scp"
    echo "Prepared scp for $input, saved to $output_dir/files.scp"

    python scripts/prepare_jsonl.py \
        -i "$output_dir/files.scp" 
done