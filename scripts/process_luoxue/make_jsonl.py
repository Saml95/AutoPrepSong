import json

if __name__ == "__main__":
    json_scp = "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/json_group_v1/luoxue_batch1-5_merged_5s.v1.remove_0_speaker.remove_live.scp"
    output_jsonl = json_scp.replace('.scp', '.jsonl')
    
    with open(json_scp, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for line in lines:
            # 如果行包含 tab，只取第一列
            json_path = line.split('\t')[0]
            item = {"json_path": json_path}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存 {len(lines)} 行到 {output_jsonl}")
