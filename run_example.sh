# Old style (deprecated):
# python3 scripts/autoprep_song.py cfg_file=config/example.yaml
# python3 scripts/autoprep_song_jwyu.py cfg_file=config/example_jsonl.yaml


python3 scripts/autoprep_song_jwyu.py \
    --config_path config/amlt_basic.yaml \
    --data_yaml config/example_data.yaml

# # New style with argparse:
# python3 scripts/autoprep_song_jwyu.py \
#     --config_path config/amlt_basic.yaml \
#     --data_yaml /home/jianweiyu/exp/AutoPrepSongV2/config/example_data.yaml \
#     --start_idx 1 \
#     --chunk_size 2

# With start_idx and chunk_size:
# python3 scripts/autoprep_song_jwyu.py \
#     --config_path config/amlt_basic.yaml \
#     --input_jsonl example/test.jsonl \
#     --output_base_dir exp/example_jwyu_jsonl \
#     --start_idx 0 \
#     --chunk_size 100

