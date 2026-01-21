# python process_lyrics.py \
#     --merge --merge_scp /home/jianweiyu/exp/music/luoxue/json_group_lyric_gp5_v1_processed_v2_resetendtime/filtered.scp \
#     --merge_output_dir /home/jianweiyu/exp/music/luoxue/json_group_lyric_gp5_v1_processed_v2_resetendtime_5s \
#     --chunk_size 5


# python process_lyrics.py \
#     --merge --merge_scp /home/jianweiyu/exp/music/luoxue/json_group_lyric_gp5_v1_processed_v2_resetendtime/filtered.scp \
#     --merge_output_dir /home/jianweiyu/exp/music/luoxue/json_group_lyric_gp5_v1_processed_v2_resetendtime_5s \
#     --chunk_size 5

python process_lyrics.py \
    --merge_mode 4 \
    --merge --merge_scp /home/jianwei/music/luoxue/diariz_v1/diariz_v1_local.scp  \
    --merge_output_dir /home/jianwei/music/luoxue/diariz_v1_15s \
    --chunk_size 15