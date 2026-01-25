import re


time_pattern = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")

# 不太能救得了的黑名单
qustion_lrc_records = ['/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue_batch6/Duca - Brand-New World.lrc',# 错拍对唱
                        "/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue_batch6/Avril Lavigne - Hello Kitty.lrc",
]
def parse_lrc_with_timestamps(path):
    # 输出的歌词unsorted / unfiltered
    lyrics, strange_lyrics = [], []
    with open(path, "r", encoding="utf-8") as f:
        has_multiple, has_translate = False, False
        first_occur_time = []
        for line in f:
            line = line.strip().replace("\xa0", " ").replace("\ufeff", "")

            times = time_pattern.findall(line)
            text = time_pattern.sub("", line).strip()
            if text == '':
                continue # empty lines may have strange usage(eg. occupying position for translation)

            times = [int(mm) * 60 + float(ss) for mm, ss in times]

            if len(times) > 1:
                has_multiple = True

            for t in times:
                lyrics.append({
                    "start": t,
                    "text": text
                })
                first_occur_time.append(min(times))
                

    has_translate, translate_ptr = detect_restart_with_threshold(first_occur_time, 10)

    if has_multiple or has_translate:
        # print("repeat:",has_multiple, "\ttranslate: ",has_translate)
        pass

    if has_translate:
        strange_lyrics +=[lyrics[translate_ptr:]]
        lyrics = lyrics[:translate_ptr]
        

    return lyrics, strange_lyrics


def detect_restart_with_threshold(nums, reset_threshold=10):
    """
    reset_threshold: 如果当前值 < 比前一个小，认为是重新开始
    """
    for i in range(1, len(nums)):
        if nums[i] < nums[i - 1] - reset_threshold:
            return True, i
    return False, None





# def parse_lrc_with_timestamps(lrc_path):
#     """
#     读取 LRC 歌词时间戳文件

#     Args:
#         lrc_path (str): lrc 文件路径
#         keep_metadata (bool): 是否保留 作词/作曲 等信息

#     Returns:
#         List[dict]: [
#             {
#                 "text": str,
#                 "start": float
#             },
#             ...
#         ]
#     """

#     pattern = re.compile(
#         r"\[(\d+):(\d+(?:\.\d+)?)\](.*)"
#     )

#     results = []

#     with open(lrc_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             match = pattern.match(line)
#             if not match:
#                 continue

#             minute = int(match.group(1))
#             second = float(match.group(2))
#             text = match.group(3).strip()

#             start_time = minute * 60 + second

#             results.append({
#                 "text": text,
#                 "start": start_time
#             })

#     return results


if __name__=='__main__':
    lrc_res = parse_lrc_with_timestamps('/mnt/conversationhubhot/yaoyaochang/speech/data/music/yan/luoxue_batch6/Alex Goot、Michael Henry & Justin Robinett - Adele Medley.lrc')
    breakpoint()