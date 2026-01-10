# path=/mnt/conversationhubhot/unilm/yaoyaochang/speech/data/podcast/all_audio_index.txt
# path=/mnt/conversationhubhot/jianweiyu/datasets/sdrv1/podcast/jwyu/wav.scp
output_dir=/mnt/conversationhubhot/jianweiyu/datasets/sdrv1/podcast/jwyu/AutoPrepV2_results_noenhance


path=/mnt/conversationhubhot/yaoyaochang/speech/data/gemini/sft_20250628_wav.scp
output_dir=/mnt/conversationhubhot/yaoyaochang/speech/data/gemini/AutoPrepV2_results_noenhance


path="/data/jianwei/experiment/DataPropc/AudioAutoPrep/audioautoprep/evaluation/DiariZen/test_AMI_AliMeeting_AISHELL4.scp"
output_dir="/data/jianwei/experiment/DataPropc/AudioAutoPrep/exp/debug/diarizen/test_AMI_AliMeeting_AISHELL4"
export PYTHONPATH=$PYTHONPATH:$PWD/..
bash multiple_gpu_multiple_process.sh \
    --index_file $path \
    --base_dir "" \
    --start_idx 0 \
    --total_files 80 \
    --num_gpus 1 \
    --processes_per_gpu 1 \
    --buffer_size 20 \
    --output_dir  $output_dir