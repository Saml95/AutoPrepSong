import json
import tqdm
import re
from pathlib import Path
import time

def extract_label(text):
    """
    从 '[LABEL] xxx' 中提取 LABEL
    """
    m = re.match(r"\[(.*?)\]\s*(.*)", text)
    if m is None:
        return None, text
    return m.group(1), m.group(2)

label_a = ['intro', 'outro', 'inst', 'silence']
label_b = ['verse', 'chorus', 'pre-chorus', 'bridge']


def detect_language(text: str) -> str:
    """
    检测文本语种
    返回: 'cjk' (中日韩泰等字符语系) 或 'western' (西语系)
    """
    if not text:
        return 'western'
    
    # 移除中括号及其内容
    text_clean = re.sub(r'\[[^\]]*\]', '', text).strip()
    if not text_clean:
        return 'western'
    
    # 统计字符类型
    cjk_count = 0
    western_count = 0
    
    for char in text_clean:
        # 中文
        if '\u4e00' <= char <= '\u9fff':
            cjk_count += 1
        # 日文平假名和片假名
        elif '\u3040' <= char <= '\u30ff':
            cjk_count += 1
        # 韩文
        elif '\uac00' <= char <= '\ud7af':
            cjk_count += 1
        # 泰语
        elif '\u0e00' <= char <= '\u0e7f':
            cjk_count += 1
        # 拉丁字母
        elif char.isalpha():
            western_count += 1
    
    # 如果有CJK/泰语字符，判定为CJK语系
    if cjk_count > 0:
        return 'cjk'
    else:
        return 'western'


# for f in tqdm.tqdm(open("local/luoxue_batch1-5_merged_10s.v1.scp").readlines()):
#     js = json.load(open(f.strip(), 'r'))
#     segments = js['segments']
#     for seg in segments:
#         label, text = extract_label(seg['text'])
#         if label not in label_a+label_b:
#             breakpoint()

#         fa = 0
#         if label in label_a and text.strip() != '':
#             fa += 1
#             breakpoint()

#         miss = 0
#         if label in label_b and text.strip() == '':
#             miss += 1
#             breakpoint()

import re

from scripts.load_lrc import parse_lrc_with_timestamps





# paths = "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/meta/luoxue_batch6/files.jsonl"

# for l in tqdm.tqdm(open(paths, 'r').readlines()[87:]):
#     lrc = json.loads(l)['lyric_path']
#     if Path(lrc).exists():
#         lrc = parse_lrc_with_timestamps(lrc)
#     # breakpoint()



print(Path("/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/HITA、音频怪物 - 剑胆琴心(长歌门) (剧情版).lyric.json").stem)
exit()



fail_ls = [
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/HITA、音频怪物 - 剑胆琴心(长歌门) (剧情版).lyric.json",                                                                            
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/R.I.O. - R.I.O. Megamix (Continuous DJ Mix).lyric.json",                                                                           
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/Vicetone - End Of The Year Mix 2016.lyric.json",                                                                                   
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/deadmau5 - 4x4=12 (Continuous Mix).lyric.json",                                                                                    
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/严凤英、王少舫 - 夫妻观灯 (黄梅戏版).lyric.json",                                                                                  
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/主播凯紫 - 暗恋.lyric.json",                                                                                                       
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/古巨基 - 劲歌金曲 (Live).lyric.json",                                                                                              
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/古巨基 - 劲歌金曲.lyric.json",                                                                                                     
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/将离 - 【剧情版】如愿长生.lyric.json",                                                                                             
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/少徒 - Avicii-UMF2016 REMAKE（少徒 remix）.lyric.json",                                                                            
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/张新尚、杨荣荣 - 张连卖布 (秦腔版).lyric.json",                                                                                    
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/群星 - 伤感2011(Dj径舟首张串烧大碟).lyric.json",                                                                                   
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/群星 - 孤城闭.lyric.json",                                                                                                         
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/音频怪物、五色石南叶 - 圣焰暝影(陆危楼&阿萨辛).lyric.json",                                                                        
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/Queen - Queen at Live Aid 1985.lyric.json",                                                                                        
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/王贰浪 - 爱你.lyric.json",                                                                                                         
"/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6/程砚秋 - 【京剧】窦娥冤.lyric.json",
] 

lines = open("/data/yan/meta/luoxue_batch6/lyric_after_gpt5/origin_lyrics.scp").readlines()
first_fail = open("/data/yan/meta/luoxue_batch6/lyric_after_gpt5/origin_lyrics_fails.scp", 'w')
for l in tqdm.tqdm(lines):
    out_path = l.strip().replace("/home/v-chenyuyang/luoxue_batch6", "/data/yan/meta/luoxue_batch6/lyric_after_gpt5/luoxue_batch6").replace(".lrc", ".lyric.json")
    if Path(out_path).exists() and out_path not in fail_ls:
        continue

    first_fail.write(l)


# import os
# for l in fail_ls:
#     os.remove(l)
#     print(l)