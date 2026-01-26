from glob import glob
import os, sys
from pathlib import Path
import json


def check_unfinished(root_dir, vad_dir, sep_dir, struct_dir):
    finished_ids =  [Path(i).stem for i in glob(os.path.join(root_dir, "**", "*.json"), recursive=True)]
    vaded_ids = glob(os.path.join(vad_dir, "**", "*.json"), recursive=True)
    seped_ids = glob(os.path.join(sep_dir, "*/*"), recursive=True)
    structed_ids = glob(os.path.join(struct_dir, "**", "*.json"), recursive=True)
    
    fa_vad = [i for i in vaded_ids if Path(i).stem not in finished_ids]
    fa_sep = [i for i in seped_ids if Path(i).stem not in finished_ids]
    fa_struct = [i for i in structed_ids if Path(i).stem not in finished_ids]
    breakpoint()


def check_missed(output_scp, input_jsonl):
    out_ids = [Path(l.strip()).stem for l in open(output_scp, 'r').readlines()]

    input_entries = [json.loads(l) for l in open(input_jsonl, 'r').readlines()]
    id2entries = {Path(l['audio_path']).name:l for l in input_entries}
    
    missed_entries = []
    for i in id2entries:
        if i not in out_ids:
            missed_entries.append(id2entries[i])
    
    return missed_entries

if __name__=='__main__':
    # final_output = "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6/AutoPrepSongV2/20260125/final_output"
    # vad_dir = "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6/AutoPrepSongV2/20260125/intermediate/vad_output"
    # sep_dir = "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6/AutoPrepSongV2/20260125/intermediate/separation_output"
    # struct_dir = "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6/AutoPrepSongV2/20260125/intermediate/songformer_output"
    # check_unfinished(final_output, vad_dir, sep_dir, struct_dir)


    # missed_list = check_missed(
    #     "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6/jsons.scp",
    #     "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6/files_gpt_filt.jsonl"
    # )
    # with open("/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6/try_again.jsonl", 'w') as f:
    #     for l in missed_list:
    #          f.write(json.dumps(l,ensure_ascii=False)+'\n')

    pass