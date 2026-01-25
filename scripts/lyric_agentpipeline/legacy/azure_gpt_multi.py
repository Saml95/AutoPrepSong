import argparse
import copy
import json
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import requests
import traceback
from collections import Counter
import feedparser

import openai
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI
from tqdm import tqdm

import signal, time

class TimeoutException(Exception): pass

def handler(signum, frame):
    raise TimeoutException("worker timeout ")

import tiktoken

try:
    from langdetect import detect
except ImportError:
    detect = None

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


import datetime    
def get_beijing_time():
    bj_time = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    return bj_time
# 替换全局 print 函数
import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    original_print(get_beijing_time(), *args, **kwargs)
builtins.print = custom_print

TIME_OUT_SECONDS = 36000
SEGMENT_BATCH_SIZE = 100
STAGE2_BATCH_RETRIES = 10
API_EVENT_LOOKBACK_SECONDS = 600
API_MINIMAL_RATIO = 0.01
CLIENT_REFRESH_SECONDS = 3600
DEBUG_MODE = False
API_EVENT_MANAGER: Optional[Manager] = None
API_EVENT_HISTORY: Optional[Any] = None  # Manager().dict proxy
API_EVENT_LOCK: Optional[Any] = None     # Manager().RLock proxy
WORKER_CLIENTS: Optional[List[Tuple["ApiConfig", AzureOpenAI]]] = None
WORKER_CLIENTS_TS: float = 0.0
LANG_DETECT_SAMPLE_CHARS = 3000


@dataclass
class ApiConfig:
    scope: str
    endpoint: str
    model: str
    api_version: str
    
def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)


def _api_key(api_config: ApiConfig) -> Tuple[str, str, str]:
    return (api_config.scope, api_config.endpoint, api_config.model)


def _prune_events(events: List[Tuple[float, bool]]) -> List[Tuple[float, bool]]:
    cutoff = time.time() - API_EVENT_LOOKBACK_SECONDS
    return [event for event in events if event[0] >= cutoff]


def _ensure_shared_event_state() -> bool:
    return API_EVENT_HISTORY is not None and API_EVENT_LOCK is not None


def init_shared_event_history() -> Tuple[Any, Any]:
    """
    Initialize cross-process shared state for API event tracking.
    """
    global API_EVENT_MANAGER, API_EVENT_HISTORY, API_EVENT_LOCK
    if API_EVENT_MANAGER is not None:
        return API_EVENT_HISTORY, API_EVENT_LOCK
    API_EVENT_MANAGER = Manager()
    API_EVENT_HISTORY = API_EVENT_MANAGER.dict()
    API_EVENT_LOCK = API_EVENT_MANAGER.RLock()
    return API_EVENT_HISTORY, API_EVENT_LOCK


def _pool_initializer(shared_history: Any, shared_lock: Any) -> None:
    """
    Set shared proxies inside worker processes.
    """
    global API_EVENT_HISTORY, API_EVENT_LOCK
    API_EVENT_HISTORY = shared_history
    API_EVENT_LOCK = shared_lock


def _record_api_event(api_config: ApiConfig, success: bool) -> None:
    key = _api_key(api_config)
    if not _ensure_shared_event_state():
        return
    with API_EVENT_LOCK:
        events = list(API_EVENT_HISTORY.get(key, []))
        events.append((time.time(), success))
        events = _prune_events(events)
        API_EVENT_HISTORY[key] = events


def _compute_api_weight(api_config: ApiConfig) -> float:
    key = _api_key(api_config)
    if not _ensure_shared_event_state():
        return 1.0
    with API_EVENT_LOCK:
        events = _prune_events(list(API_EVENT_HISTORY.get(key, [])))
        API_EVENT_HISTORY[key] = events
    if not events:
        return 1.0
    successes = sum(1 for _, ok in events if ok)
    total = len(events)
    ratio = successes / total if total > 0 else 0.0
    return ratio + API_MINIMAL_RATIO  # ensure non-zero, bias towards high success ratios


def _select_api_client(api_clients: List[Tuple[ApiConfig, AzureOpenAI]]) -> Tuple[ApiConfig, AzureOpenAI]:
    if not api_clients:
        raise RuntimeError("No API clients available for selection.")
    weights = []
    for api_config, _ in api_clients:
        weight = max(_compute_api_weight(api_config), API_MINIMAL_RATIO)
        weights.append(weight)
    weights_and_api_config_str = ", ".join(
        f"{api_config.endpoint}({api_config.model}):{weight:.3f}"
        for (api_config, _), weight in zip(api_clients, weights)
    )
    # debug_print(f"API client weights: {weights_and_api_config_str}")
    total = sum(weights)
    if total <= 0:
        return random.choice(api_clients)
    r = random.random() * total
    cumulative = 0.0
    for (api_config, api_client), weight in zip(api_clients, weights):
        cumulative += weight
        if r <= cumulative:
            return api_config, api_client
    return api_clients[-1]


def ensure_worker_clients(api_configs: List[ApiConfig]) -> List[Tuple[ApiConfig, AzureOpenAI]]:
    global WORKER_CLIENTS, WORKER_CLIENTS_TS
    now = time.time()
    if WORKER_CLIENTS is not None and now - WORKER_CLIENTS_TS <= CLIENT_REFRESH_SECONDS:
        return WORKER_CLIENTS

    clients: List[Tuple[ApiConfig, AzureOpenAI]] = []
    for cfg in api_configs:
        try:
            client = build_openai_client(cfg)
            clients.append((cfg, client))
        except Exception as exc:
            print(f"Failed to initialize client for {cfg.endpoint} ({cfg.model}): {exc}")

    if not clients:
        raise RuntimeError("Unable to initialize any Azure OpenAI clients.")

    WORKER_CLIENTS = clients
    WORKER_CLIENTS_TS = now
    debug_print(f"Worker refreshed {len(clients)} API clients.")
    return WORKER_CLIENTS

API_CONFIGS_GPT41: List[ApiConfig] = [
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinnorthcentralus.openai.azure.com/",
        model="gpt-4.1-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinnorthcentralus.openai.azure.com/",
        model="gpt-4.1-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubnorthcentralus.openai.azure.com/",
        model="gpt-4.1-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubsouthcentralus.openai.azure.com/",
        model="gpt-4.1-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubsouthcentralus.openai.azure.com/",
        model="gpt-4.1-DZS",
        api_version="2025-04-01-preview",
    ),
    # ApiConfig(
    #     scope="https://cognitiveservices.azure.com/.default",
    #     endpoint="https://readinswedencentral.openai.azure.com/",
    #     model="gpt-4.1-DZS",
    #     api_version="2025-04-01-preview",
    # ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubswedencentral.openai.azure.com/",
        model="gpt-4.1-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-4.1_2025-04-14",
        api_version="2025-04-01-preview",
    )
]


API_CONFIGS_GPT5: List[ApiConfig] = [
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus2.openai.azure.com/",
        model="gpt-5-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus2.openai.azure.com/",
        model="gpt-5-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubswedencentral.openai.azure.com/",
        model="gpt-5-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubswedencentral.openai.azure.com/",
        model="gpt-5-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus2.openai.azure.com/",
        model="gpt-5-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus2.openai.azure.com/",
        model="gpt-5-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinswedencentral.openai.azure.com/",
        model="gpt-5-2025-08-07-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinswedencentral.openai.azure.com/",
        model="gpt-5-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-5_2025-08-07",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-5-chat_2025-08-07",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-5-chat_2025-10-03",
        api_version="2025-04-01-preview",
    ),
    # ApiConfig(
    #     scope="api://trapi/.default",
    #     endpoint="https://trapi.research.microsoft.com/gcr/shared",
    #     model="gpt-5-pro_2025-10-06",
    #     api_version="2025-04-01-preview",
    # ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-5.1_2025-11-13",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-5.1-chat_2025-11-13",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-5.2_2025-12-11",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-5.2-chat_2025-12-11",
        api_version="2025-04-01-preview",
    ),
]



API_CONFIGS_GPT4O: List[ApiConfig] = [
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus2.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubnorthcentralus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubsouthcentralus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    # ApiConfig(
    #     scope="https://cognitiveservices.azure.com/.default",
    #     endpoint="https://conversationhubwestus.openai.azure.com/",
    #     model="gpt-4o",
    #     api_version="2025-04-01-preview",
    # ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus2.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinwestus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),

    # gpt-4o-global group
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus2.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubnorthcentralus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubsouthcentralus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubwestus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus2.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinnorthcentralus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinwestus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
]


API_CONFIG_GROUPS = {
    "gpt5": API_CONFIGS_GPT5,
    "gpt4o": API_CONFIGS_GPT4O,
    "gpt41": API_CONFIGS_GPT41,
    "gpt5+gpt4o": API_CONFIGS_GPT5 + API_CONFIGS_GPT4O,
    "gpt5+gpt41": API_CONFIGS_GPT5 + API_CONFIGS_GPT41,
    "gpt4o+gpt41": API_CONFIGS_GPT4O + API_CONFIGS_GPT41,
    "gpt5+gpt4o+gpt41": API_CONFIGS_GPT5 + API_CONFIGS_GPT4O + API_CONFIGS_GPT41,
}

CHINESE_METADATA_DIR = Path("/mnt/conversationhubhot/yaoyaochang/speech/data/xyz/episode_details_all_0424")
ENGLISH_RSS_DIR = Path("/mnt/conversationhubhot/yaoyaochang/speech/data/podcast/rss_all")
ENGLISH_AUDIO_DESC_DIR = Path("/mnt/conversationhubhot/yaoyaochang/speech/data/podcast/audio_description_all")
LOCAL_PREFIX = Path("/data/yaoyaochang")


PROMPT_STAGE1 = """
# Instruction:
你要帮我把ASR任务做得更好。一个音频按顺序切分为多个 segment。每个 segment 含有几个不同 ASR 模型识别到的版本，你需要综合它们，汇总整段音频的 hotwords 以及可识别的 speaker id 与真实名字/称谓的对应关系。
以及标注这个音频的ASR难度，可以是easy,medium,hard。
* hotwords：包含所有需要统一写法的专有名词、人名、地名等，保持与音频语言一致。
* speaker_map：一个 list，列表中每个元素是一个 JSON 对象：
  - original_id: 原本的speaker id
  - speaker_name: 该 id 对应的真实名字/称谓。如果无法确定真实名字，就保持原样，例如 "1"->"1", "3"->"3"。
* speaker_name_source：标注 speaker_map 中 name 的依据来源，可取值包括：metadata：背景信息中明确给出的人名或称谓；context：可从音频或转写内容中的明确上下文判断得到（如自我介绍“我是 Jack”）；inferred：由 GPT 基于对话结构或角色假设生成的名称（如 teacher、interviewer）；unknown：无法从任何信息中可靠判断说话人名称。
* difficulty: 该音频的ASR难度，取值为easy, medium, hard之一。
* 如果背景信息里有明确的人名或专有名词，你也可以参考。

# Data:
* 当前音频的背景信息如下：
{metadata}

# 当前音频的speaker id和count如下：
{speaker_counter_str}

* 当前音频的所有 segment 内容如下：
{segments}

# Output Format:
{{
  "hotwords": ["HotwordA", "HotwordB"],
  "speaker_map": [
    {{"id": "1", "name": "John"}},
    {{"id": "2", "name": "小明"}},
    {{"id": "3", "name": "3"}}
  ],
  "speaker_name_source": "metadata",
  "difficulty": "medium"
}}
"""

PROMPT_STAGE2 = """
# Instruction:
你要帮我把ASR任务做得更好。一个音频按顺序切分为多个 segment。每个 segment 含有几个不同 ASR 模型识别到的版本，你需要综合它们，生成最合理、最符合发音顺序的最终文本，作为最终的 ASR 结果。音频大多来自播客。
* 多个版本都来自同一个音频，请以发音为准，不要删除或大改原本的发音内容。请严格 follow 原文，尤其是声音上原文是不会出大错的。比如原文里有个“there”，那么那个地方就是有个发音像“there”的东西，可以改成“their”这种发音接近的内容，但不要把“there”删掉或者改成其他发音完全不同的词，或者调换位置。
* 词汇顺序不能打乱，因为不能改变发音内容。
* 你要确保输出的文字和原始文字数量基本一致，不要出现缺句、多句的情况。
* 不需要你来对文本进行总结或者名字解释。因为，你要始终记住，你是在帮忙做ASR任务，修改ASR的文字，要始终忠于原始发音内容。而不是和我聊天、解释或者总结，擅自修改发音内容。
* segment的顺序更不能打乱，要保留原始segment的顺序。
* 可以做同音词替换，但要保留原音信息，例如“there”与“their”、“smell”与“smile”、“write”与“right”等等。
* 中文里面可能夹杂英文，请保持英文内容不变，不要私自添加中文翻译甚至直接改为中文。因为这违背了“不改变发音内容”的原则。
* 可以对标点符号进行修改，补充缺失的标点符号，删除多余的标点符号。
* 语法稍有问题是允许的，因为口语就是这样。
* 保留口语感，例如“you know, you know”、“I I think”、“and and then”这种重复的词是允许的，因为原音频就是这么说的。
* 专有名词或者人名地名，ASR 可能识别不准确，或者在多个 segment 之间可能识别结果不一致，你需要统一写法。
* 如果 speaker 是 unknown，那么你要根据上下文判断这个 segment 的 speaker id，你有信心的话就输出一个数字，无法判断的话还是输出 unknown。
* 对于中文，判断全文主体是普通话还是粤语/台湾话，如果是前者就用简体字输出，如果是后者就用繁体字输出，注意保持全文一致性。
* 如果“text(qwen omni)”给了 [Speech][Music][Human Sounds][Environmental Sounds][Noise][Silence] 等 tag，那么你也给这样一个 tag 即可。
* “text(qwen asr)”在 segment 内部用了换行的话，你也可以考虑这么做。
* 如果前面 segment 或者后面 segment 是 [Music]，那么当前 segment 有可能是歌词，你要通过文本判断，如果是歌词就在 text 前面加上 [Lyric] 标签。
* 少用中文破折号。
* 你要沿用下面的 hotwords 和 speaker 对应关系，保证所有写法在所有segment里一致。
* ASR结果里和hotwords里发音很接近的词，有可能就是hotwords里的词，你根据上下文看看是什么情况，合理的话就改成hotwords里的写法。
* 输出中 `speaker` 字段是string而不是int，参考speaker_map中的name。
* 可以谨慎修改speaker：有几种speaker可能错误的情况，1. segment里的原始speaker id是错误的，尤其在一个音频的开始和结尾有music的时候经常会出现错误；2. speaker_map也可能是错误的；3. 有些speaker id对应的count很少，那么这个speaker有可能是被判断错误的；4. 前后都是A说话中间是B说话，但其实从内容和全文来看B就是A；所以对于每个segment的实际speaker，你可以根据上下文思考这个segment到底应该是谁说的。
* 有的segment可能包含了多个speaker在说话，如果你从文本上判断怀疑segment是这样，就标注may_have_multiple_speaker=true，否则标注may_have_multiple_speaker=false。
* 你要确保输出的 segment 数量和输入的 segment 数量完全一致。
* 输出格式：json。字段 data 是一个 list，每个 item 代表一个 segment，包含 thinking、speaker、text、may_have_multiple_speaker 字段；thinking 是你对该 segment 如何综合生成结果的中文说明，speaker 是一个数字表示 speaker id，text 是你修改后的文本，may_have_multiple_speaker 是一个布尔值，表示该 segment 是否包含多个说话人。
* 输出务必是 json 格式。



# Data:
* 当前音频的背景信息如下：
{metadata}

* 现在仅处理 Segment {batch_start} 到 {batch_end}（共 {batch_count} 条，全部 {total_segments} 条）。
* 已知 hotwords：{hotwords}
* 已知 speaker_map (数字 id -> 名称/称谓)：{speaker_map}
# 已知 speaker id和count如下：{speaker_counter_str}

* 本批次的 segment 内容如下：
{segments}
"""


EN_PROMPT_STAGE1 = """
# Instruction

You must help improve the ASR (Automatic Speech Recognition) task. The audio is split sequentially into multiple segments. Each segment contains outputs from several different ASR models. You need to consolidate them by synthesizing the most accurate hotwords for the entire audio and constructing a speaker mapping between the original speaker IDs and their real names or titles.

hotwords: Include all proper nouns that must follow a unified writing format, such as personal names, locations, or other domain-specific terms. The hotwords must remain consistent with the audio’s original language and pronunciation (do not translate or change language).

speaker_map: Output as a list. Each element in the list must be a JSON object with the following fields:

original_id: The original speaker ID from the ASR outputs.

speaker_name: The real name or title that corresponds to this speaker. If the real name cannot be confidently determined, keep it unchanged, such as "1" → "1" or "3" → "3".

speaker_name_source: The source basis for the names in speaker_map. Allowed values: metadata (explicitly provided in background info), context (clearly stated in transcript, e.g., “I am Jack”), inferred (model-inferred role/title), unknown (cannot be determined).

difficulty: ASR difficulty for this audio. Use one of easy, medium, hard.

You may also reference background information when extracting speaker names or hotwords.

# Data

Background information for the current audio:
{metadata}

Original speaker IDs and their counts for this audio:
{speaker_counter_str}

All sequential segments and their ASR model outputs:
{segments}


# Output Format:
{{
  "hotwords": ["HotwordA", "HotwordB"],
  "speaker_map": [
    {{"id": "1", "name": "John"}},
    {{"id": "2", "name": "小明"}},
    {{"id": "3", "name": "3"}}
  ],
  "speaker_name_source": "context",
  "difficulty": "easy"
}}
"""

EN_PROMPT_STAGE2 = """
# Instruction

You must help improve the ASR (Automatic Speech Recognition) task. The audio is split sequentially into multiple segments. Each segment contains recognition results from several different ASR models, all derived from the same audio. You need to consolidate these model outputs and generate the most reasonable final text, ensuring it strictly follows the original pronunciation sequence. Most audio sources are podcasts.

Rules and constraints you must follow:
* Multiple ASR model outputs come from the same audio. Always rely on true pronunciation, and do not delete or heavily rewrite phonetic content. Audio speech rarely contains major phonetic errors. 
    * For example: if the original audio has a pronunciation like "there", that means a word that sounds like "there" truly exists at that position. You may replace it with a close homophone like "their", but you must never remove, reorder, relocate, or replace it with a word that has completely different pronunciation**, or shuffle the spoken order**.
* The word order must not be altered, as the phonetic sequence in the original audio cannot be changed.
* Keep the amount of text aligned with the original; avoid missing sentences or introducing extra sentences.
* The segment order must also remain intact, preserving the original sequence of all segments.
* Homophone replacement is allowed, but phonetic information must be preserved, such as: "there" ↔ "their"; "smell" ↔ "smile"; "write" ↔ "right";
* Do not summarize the content or explain names/titles—you are refining ASR text and must stay faithful to the original audio.
* The original speech may mix multiple languages; preserve all languages exactly as spoken and do not translate, normalize, or convert any content into another language, as this would violate the principle of not altering phonetic content or pronunciation sequence.
* You may modify punctuation to: add missing punctuation; remove excessive or incorrect punctuation
* Minor grammar imperfections are allowed, because natural spoken language may include self-correction, repetition, and informal grammar.
* You must preserve the original audio’s spoken feel, cadence, and acoustic realism, including confirmed repeated tokens and verbal loops. These repetitions are not errors—they are part of the true phonetic trace and conversational character, especially in long-form podcast speech. You must retain them verbatim, in their original positions, and in the original order, without deletion, compression, rearrangement, or over-smoothing.  This includes but is not limited to: "you know, you know", "I I I think", "and and then", "like, like, like", "yeah yeah yeah", "I just— I just feel like", etc.
* Proper nouns (names, locations, organizations, or domain terms) may be recognized inconsistently or incorrectly across segments or between different ASR model outputs. You must unify their final writing format across all segments.
* If the segment speaker is labeled unknown, you must infer the correct speaker id from context if confident:
    * If you can determine the speaker confidently, output a number
    * Otherwise, keep it "unknown"
* If the Qwen omni uses tags like [Speech][Music][Human Sounds][Environmental Sounds][Noise][Silence], you may output the same tags.
* If text breaks (new lines) are used in text(qwen asr) inside a segment, you may also consider preserving or generating line breaks where appropriate.
* If a segment may actually contain lyrics (typically when [Music] appears nearby), you must infer carefully using content. If confirmed as lyrics, you must prepend [Lyric] before the text field.
* You must reuse the given hotwords and speaker mapping exactly to ensure global consistency across all segments, unless you have strong confidence the mapping itself is incorrect, in which case careful correction is allowed.
* Words that sound extremely close to a hotword may actually refer to that hotword. If contextually reasonable, convert it to the unified hotword spelling from the list.
* In the output, the speaker field must be a string, not an integer, following the name form from speaker_map.
* You may carefully correct speaker labeling errors, especially when:
    * The speaker label is incorrect at the beginning or end due to surrounding music
    * speaker_map itself may contain wrong mappings
    * Speaker IDs with very low occurrence count might be misclassified
    * The segment shows a different speaker label interleaved between the same person (A → B → A), but B is actually still A from context
    So you must reason about the correct speaker for each segment if confident.
* Some segments may contain multiple speakers; if the text indicates that, set may_have_multiple_speaker=true, otherwise set it to false.
* You must ensure the number of output segments exactly matches the number of input segments.
* Output format: JSON. The field data must be a list. Each list item represents one segment and must include the fields thinking, speaker, text, may_have_multiple_speaker.
    * `thinking` is an English explanation of how the segment results were consolidated.
    * `speaker` is the speaker ID (expressed as a number).
    * `text` is the revised transcript for that segment.
    * `may_have_multiple_speaker` is a boolean flag indicating whether multiple speakers appear in the segment.
* The final output must strictly follow JSON format. Ensure the output is valid JSON only.

# Data

Background information for the current audio:
{metadata}

Processing only Segment {batch_start} to {batch_end} (this batch contains {batch_count} segments, {total_segments} total segments overall).

Known hotwords for this audio:
{hotwords}

Known speaker map (original numeric ID → real name/title):
{speaker_map}

Known speaker IDs and their counts:
{speaker_counter_str}

Segment contents in this batch to process:
{segments}
"""

from pydantic import BaseModel
from typing import List

class Segment(BaseModel):
    thinking: str
    speaker: str
    text: str
    may_have_multiple_speaker: bool

class ASRResult(BaseModel):
    data: List[Segment]

class SpeakerItem(BaseModel):
    id: str   # 原来 dict 的 key
    name: str # 原来 dict 的 value

class StageOneResult(BaseModel):
    hotwords: List[str]
    speaker_map: List[SpeakerItem]
    speaker_name_source: str
    difficulty: str

@dataclass
class ProcessingConfig:
    local_dir: Optional[Path]
    gpt_dir: Optional[Path]
    remote_dir: Optional[Path]
    stage1_template: str = PROMPT_STAGE1
    stage2_template: str = PROMPT_STAGE2


def build_openai_client(api_config: ApiConfig) -> AzureOpenAI:
    print(
        f"Initializing client for endpoint {api_config.endpoint} model {api_config.model} "
        f"(api_version={api_config.api_version})"
    )
    credential = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    )
    token_provider = get_bearer_token_provider(credential, api_config.scope)
    client = AzureOpenAI(
        azure_endpoint=api_config.endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_config.api_version,
        max_retries=0,
    )
    return client


def load_index_paths(index_path: Path) -> List[str]:
    with index_path.open("r", encoding="utf-8") as f:
        return [line.strip().split(" ")[0] for line in f if line.strip()]


def collect_text_sample(segments: List[dict], max_chars: int = LANG_DETECT_SAMPLE_CHARS) -> str:
    """
    Pull a slice of text from the earliest segments for language detection.
    Preference order: text_checked > text > text_original > paraformer_asr_text.
    """
    sample_parts: List[str] = []
    total_chars = 0
    for segment in segments:
        val = segment.get("text")
        if not val:
            continue
        val_str = str(val)
        sample_parts.append(val_str)
        total_chars += len(val_str)
        if total_chars >= max_chars:
            break
    return "\n".join(sample_parts)


def infer_is_chinese_from_segments(segments: List[dict], json_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Infer whether the audio is Chinese by running language detection on a text sample.
    """
    sample_text = collect_text_sample(segments)
    detected_lang: Optional[str] = None

    if sample_text.strip() and detect:
        try:
            detected_lang = detect(sample_text)
        except Exception as exc:
            debug_print(f"langdetect failed for {json_path}: {exc}")
    else:
        debug_print(f"No text sample available for language detection for {json_path}.")

    is_chinese = bool(detected_lang and detected_lang.lower().startswith("zh"))

    debug_print(
        f"Language detection for {json_path}: sample_len={len(sample_text)}, "
        f"lang={detected_lang}, is_chinese={is_chinese}"
    )
    return is_chinese, detected_lang


def format_segments_for_prompt(segments: List[dict], start_idx: int = 0) -> str:
    lines: List[str] = []
    for idx, segment in enumerate(segments):
        global_idx = start_idx + idx
        speaker = segment.get("umap_segment_labels")
        if 'outlier' in str(speaker).lower():
            speaker = "unknown"
        start = float(segment.get("start", segment.get("start_time", 0.0)))
        end = float(segment.get("end", segment.get("end_time", 0.0)))
        # text_checked
        lines.append(f"Segment {global_idx}:")
        lines.append(f"Speaker {speaker} from {start:.2f} to {end:.2f}")
        if "text_original" in segment and segment["text_original"]:
            lines.append(f"text(firered): {segment['text_original']}")
        elif 'firered_asr_text' in segment and segment['firered_asr_text']:
            lines.append(f"text(firered): {segment['firered_asr_text']}")
        if "paraformer_asr_text" in segment and segment["paraformer_asr_text"]:
            lines.append(f"text(paraformer): {segment['paraformer_asr_text']}")
        if "text" in segment and segment["text"]:
            lines.append(f"text(qwen omni asr): {segment['text']}")
        if "text_checked" in segment and segment["text_checked"]:
            lines.append(f"text(qwen omni refine): {segment['text_checked']}")
        if "text_gpt_refine" in segment and segment["text_gpt_refine"]:
            lines.append(f"text(high priority): {segment['text_checked']}")
    return "\n".join(lines)


def relative_output_path(json_path: Path, target_root: Path) -> Path:
    """
    Build the target path by taking the final rss_id/episode_id.json components.
    """
    json_path = Path(json_path)
    parts = json_path.parts
    if len(parts) >= 2:
        relative = Path(*parts[-2:])
    else:
        relative = Path(parts[-1])
    return target_root / relative


def apply_local_prefix(path: Optional[Path], enabled: bool) -> Optional[Path]:
    """
    If enabled, prefix absolute/relative paths with LOCAL_PREFIX while avoiding double-prefixing.
    """
    if not enabled or path is None:
        return path
    path = Path(path)
    if str(path).startswith(str(LOCAL_PREFIX)):
        return path
    if path.is_absolute():
        try:
            relative = path.relative_to("/")
        except ValueError:
            relative = path
        return LOCAL_PREFIX / relative
    return LOCAL_PREFIX / path


def apply_local_prefix_str(path_str: str, enabled: bool) -> str:
    return str(apply_local_prefix(Path(path_str), enabled))


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def build_local_segment_dump(segments: Iterable[dict]) -> List[dict]:
    output: List[dict] = []
    for segment in segments:
        output.append(
            {
                "speaker": segment.get("umap_segment_labels", segment.get("speaker", "N/A")),
                "text": segment.get("text", ""),
            }
        )
    return output


def load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Failed to load metadata json {path}: {exc}")
        return None


def build_metadata_for_episode(json_path: Path, is_chinese: bool) -> Tuple[Dict[str, object], str]:
    episode_id = json_path.stem
    rss_id = json_path.parent.name
    metadata_lines: List[str] = []
    metadata_dict: Dict[str, object] = {}

    if is_chinese:
        details_path = CHINESE_METADATA_DIR / f"{episode_id}.json"
        details = load_json_if_exists(details_path) or {}
        if 'data' in details and 'data' in details['data']:
            details = details['data']['data']
        for key in ("author", "title", "description", "summary"):
            val = details.get(key)
            if val:
                metadata_dict[f"podcast_{key}"] = val
                metadata_lines.append(f"Podcast {key}: {val}")
        podcast_info = details.get("podcast")
        if isinstance(podcast_info, dict):
            for key in ("author", "title", "description", "summary"):
                val = podcast_info.get(key)
                if val:
                    metadata_dict[f"episode_{key}"] = val
                    metadata_lines.append(f"Episode {key}: {val}")
            for podcaster in podcast_info.get("podcasters", []):
                name = podcaster.get("nickname")
                if name:
                    metadata_dict.setdefault("podcasters", []).append(name)
            if "podcasters" in metadata_dict:
                metadata_lines.append(f"Podcaster names: {', '.join(metadata_dict.get('podcasters', []))}")
        if details:
            metadata_dict.setdefault("other_details", json.dumps(details, ensure_ascii=False))
            metadata_lines.append(f"Other details: {json.dumps(details, ensure_ascii=False)}")
    else:
        rss_path = ENGLISH_RSS_DIR / f"{rss_id}.xml"
        if rss_path.exists():
            try:
                feed = feedparser.parse(str(rss_path))
                feed_data = getattr(feed, "feed", {}) or {}
                for key in ("author", "title", "description", "summary"):
                    val = feed_data.get(key)
                    if val:
                        metadata_dict[f"podcast_{key}"] = val
                        metadata_lines.append(f"Podcast {key}: {val}")
                if feed_data:
                    metadata_dict["other_details"] = str(feed_data)
                    metadata_lines.append(f"Other details: {feed_data}")
            except Exception as exc:
                print(f"Failed reading RSS {rss_path}: {exc}")
        description_path = ENGLISH_AUDIO_DESC_DIR / f"{rss_id}.json"
        desc_data = load_json_if_exists(description_path) or {}
        episode_desc = desc_data.get(episode_id, "")
        if episode_desc:
            metadata_dict["episode_description"] = episode_desc
            metadata_lines.append(f"Episode description: {episode_desc}")

    metadata_text = "\n".join(metadata_lines) if metadata_lines else "N/A"
    return metadata_dict, metadata_text


def update_segments_with_gpt(segments: List[dict], response_json: List[dict], source: Path) -> bool:
    gpt_segments = response_json.get("data")
    if not gpt_segments:
        print(f"GPT output missing 'data' field for {source}")
        return False
    updated = False
    if len(gpt_segments) != len(segments):
        print(
            f"Warning: segment count mismatch for {source} (original={len(segments)}, gpt={len(gpt_segments)})"
        )
        return False
    
    for idx, segment in enumerate(segments):
        if idx >= len(gpt_segments):
            break
        gpt_seg = gpt_segments[idx] or {}
        speaker = gpt_seg.get("speaker")
        text = gpt_seg.get("text")
        thinking = gpt_seg.get("thinking")
        merge_with_previous = gpt_seg.get("merge_with_previous")
        may_have_multiple_speaker = gpt_seg.get("may_have_multiple_speaker")
        if merge_with_previous is not None:
            segment["merge_with_previous"] = merge_with_previous
            updated = True
        if thinking is not None:
            segment["thinking"] = thinking
            updated = True
        if speaker is not None:
            segment["speaker_gpt"] = speaker
            updated = True
        if text is not None:
            segment["text_gpt"] = text
            updated = True
        if may_have_multiple_speaker is not None:
            segment["may_have_multiple_speaker"] = bool(may_have_multiple_speaker)
            updated = True
    return updated


def get_azure_openai_response(
    client: AzureOpenAI,
    model_name: str,
    prompt: str,
    response_format: Type[BaseModel],
) -> str:
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format=response_format,
    )
    # print(f"estimated_token {count_tokens(response.choices[0].message.content)} raw response:", response)
    return response.choices[0].message.content or ""


def call_model_with_retry(
    api_clients: List[Tuple[ApiConfig, AzureOpenAI]],
    prompt: str,
    max_retries: int,
    json_path_str: str,
    response_format: Type[BaseModel] = ASRResult,
) -> Optional[str]:
    attempt = 0

    while attempt <= max_retries:
        if not api_clients:
            print("❌ No available API clients.")
            return None
        api_config, api_client = _select_api_client(api_clients)
        try:
            response = get_azure_openai_response(api_client, api_config.model, prompt, response_format)
            debug_print(f"✅ {api_config.scope} {api_config.endpoint} {api_config.model} API Success on attempt #{attempt+1}. {json_path_str}")
            _record_api_event(api_config, True)
            return response

        except Exception as exc:
            attempt += 1
            _record_api_event(api_config, False)
            if attempt > max_retries:
                print(f"❌ Repeated failures ({exc}), giving up.")
                return None

            sleep_seconds = random.randint(60, 180)
            print(
                f"⚠️ {api_config.scope} {api_config.endpoint} {api_config.model} Retry error ({type(exc).__name__}): {exc}. "
                f"Sleeping {sleep_seconds}s before retry #{attempt}... {json_path_str}"
            )
            time.sleep(sleep_seconds)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def process_file(
    json_path_str: str,
    api_configs: List[ApiConfig],
    config: ProcessingConfig,
    max_retries: int,
    process_retries: int = 5,
) -> bool:
    try: 
        start_time = time.time()
        json_path = Path(json_path_str)
        
        if config.remote_dir:
            remote_path = relative_output_path(json_path, config.remote_dir)
            # print("json_path_str:", json_path_str)
            # print("remote_path:", remote_path)
            if remote_path.exists():
                
                try:
                    data = load_json(remote_path)
                    print(f"Skipping {json_path} as remote output already exists.")
                    return False
                except Exception:
                    print(f"Failed to load existing remote output {remote_path}, reprocessing {json_path}.")
                    pass
        try:
            with json_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {json_path}")
            return False
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON in {json_path}: {exc}")
            return False
        # print(f"{json_data['audio_length']}")
        
        
        if "segments_relabel" in json_data:
            segments = json_data.get("segments_relabel")
            segment_key = "segments_relabel"
        elif "segments_merged" in json_data:
            segments = json_data.get("segments_merged")
            segment_key = "segments_merged"
        elif "segments" in json_data:
            segments = json_data.get("segments")
            segment_key = "segments"
        else:
            raise ValueError("No segments found in JSON data.")
        if not segments:
            print(f"No segments found in {json_path}")
            return False
        
        is_chinese, detected_lang = infer_is_chinese_from_segments(segments, json_path)
        json_data["detected_language"] = detected_lang

        counter: Counter[str] = Counter()
        for seg in segments:
            label = seg.get("umap_segment_labels")
            if "outlier" in str(label).lower():
                continue
            if label is None:
                continue
            counter[str(label)] += 1
        counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        speaker_counter_str = ", ".join([f"Speaker {k}: {v}" for k, v in counter.items()])
            
        try:
            api_clients = ensure_worker_clients(api_configs)
        except RuntimeError as exc:
            print(f"{exc} -- unable to process {json_path}")
            return False
        metadata_dict, metadata_text = build_metadata_for_episode(json_path, is_chinese)
        json_data["metadata"] = metadata_dict
        base_json = copy.deepcopy(json_data)
        original_segments_dump = build_local_segment_dump(copy.deepcopy(segments))
        debug_print(f"Loaded {json_path} with {len(segments)} segments. key = {segment_key}")

        for attempt in range(process_retries + 1):
            if time.time() - start_time > TIME_OUT_SECONDS:
                print(f"Timeout reached for {json_path_str}, aborting processing.")
                return False

            json_data = copy.deepcopy(base_json)
            segments = json_data.get(segment_key, []) if segment_key else json_data
            segments_text = format_segments_for_prompt(segments)
            stage1_template = config.stage1_template if is_chinese else EN_PROMPT_STAGE1
            stage2_template = config.stage2_template if is_chinese else EN_PROMPT_STAGE2
            stage1_prompt = stage1_template.format(speaker_counter_str=speaker_counter_str, metadata=metadata_text, segments=segments_text)
            stage1_input_tokens = count_tokens(stage1_prompt)
            debug_print(
                f"[Stage1] {json_path}: prepared prompt for {len(segments)} segments "
                f"(tokens≈{stage1_input_tokens})."
            )
            stage1_response_text = call_model_with_retry(
                api_clients,
                stage1_prompt,
                max_retries,
                json_path_str,
                response_format=StageOneResult,
            )
            if not stage1_response_text:
                return False
            stage1_output_tokens = count_tokens(stage1_response_text)
            try:
                stage1_json = json.loads(stage1_response_text)
            except json.JSONDecodeError as exc:
                if attempt < process_retries:
                    print(f"Retrying ({attempt+1}/{process_retries}) due to stage1 JSON parse failure: {exc}")
                    continue
                print(f"Failed to parse stage1 response for {json_path}: {exc}")
                return False

            debug_print(f"[Stage1] {json_path}: response tokens≈{stage1_output_tokens}.")

            hotwords = stage1_json.get("hotwords") or []
            if not isinstance(hotwords, list):
                hotwords = []
            else:
                hotwords = [str(item) for item in hotwords]
            speaker_map_raw = stage1_json.get("speaker_map") or []
            if not isinstance(speaker_map_raw, list):
                speaker_map_raw = []

            speaker_map: Dict[str, str] = {}
            for item in speaker_map_raw:
                try:
                    k = str(item.get("id"))
                    v = str(item.get("name"))
                    if k:
                        speaker_map[k] = v
                except Exception:
                    continue
            debug_print(f"[Stage1] {json_path}: speaker_map={speaker_map}, original_speaker_counter_str={speaker_counter_str}, hotwords={hotwords}, .")
            speaker_name_source = stage1_json.get("speaker_name_source")
            if speaker_name_source is not None:
                speaker_name_source = str(speaker_name_source)
            difficulty = stage1_json.get("difficulty")
            if difficulty is not None:
                difficulty = str(difficulty)

            combined_segments: List[Optional[dict]] = [None] * len(segments)
            stage2_batches_log: List[Dict[str, object]] = []
            stage2_input_tokens = 0
            stage2_output_tokens = 0
            stage2_success = True

            for batch_start in range(0, len(segments), SEGMENT_BATCH_SIZE):
                batch_end = min(len(segments), batch_start + SEGMENT_BATCH_SIZE)
                batch_segments = segments[batch_start:batch_end]
                batch_success = False
                last_error: Optional[str] = None

                for batch_attempt in range(1, STAGE2_BATCH_RETRIES + 1):
                    batch_segments_text = format_segments_for_prompt(batch_segments, start_idx=batch_start)
                    stage2_prompt = stage2_template.format(
                        hotwords=json.dumps(hotwords, ensure_ascii=False),
                        speaker_map=json.dumps(speaker_map, ensure_ascii=False),
                        speaker_counter_str=speaker_counter_str,
                        batch_start=batch_start,
                        batch_end=batch_end - 1,
                        batch_count=len(batch_segments),
                        total_segments=len(segments),
                        metadata=metadata_text,
                        segments=batch_segments_text,
                    )
                    batch_prompt_tokens = count_tokens(stage2_prompt)
                    debug_print(
                        f"[Stage2] {json_path}: batch {batch_start}-{batch_end - 1} attempt {batch_attempt}/"
                        f"{STAGE2_BATCH_RETRIES} with {len(batch_segments)}/{len(segments)} segments "
                        f"(prompt tokens≈{batch_prompt_tokens})."
                    )
                    batch_response_text = call_model_with_retry(
                        api_clients,
                        stage2_prompt,
                        max_retries,
                        json_path_str,
                        response_format=ASRResult,
                    )
                    if not batch_response_text:
                        last_error = "API call failed or returned empty response."
                        debug_print(
                            f"[Stage2] {json_path}: batch {batch_start}-{batch_end - 1} "
                            f"attempt {batch_attempt} failed: {last_error}"
                        )
                        continue
                    batch_response_tokens = count_tokens(batch_response_text)
                    debug_print(
                        f"[Stage2] {json_path}: batch {batch_start}-{batch_end - 1} "
                        f"attempt {batch_attempt} response tokens≈{batch_response_tokens}."
                    )

                    try:
                        batch_response_json = json.loads(batch_response_text)
                    except json.JSONDecodeError as exc:
                        last_error = f"Stage2 JSON parse error: {exc}"
                        print(
                            f"Stage2 JSON parse error for {json_path} batch {batch_start}-{batch_end - 1}: {exc}"
                        )
                        continue

                    gpt_segments_batch = batch_response_json.get("data")
                    if not gpt_segments_batch or len(gpt_segments_batch) != len(batch_segments):
                        last_error = (
                            f"Batch length mismatch: expected {len(batch_segments)}, "
                            f"got {len(gpt_segments_batch) if gpt_segments_batch else 0}"
                        )
                        print(
                            f"Stage2 batch length mismatch for {json_path} batch {batch_start}-{batch_end - 1}: "
                            f"expected {len(batch_segments)}, got {len(gpt_segments_batch) if gpt_segments_batch else 0}"
                        )
                        continue

                    stage2_batches_log.append(
                        {
                            "prompt": stage2_prompt,
                            "response": batch_response_json,
                            "batch_start": batch_start,
                            "batch_end": batch_end - 1,
                            "attempt": batch_attempt,
                        }
                    )
                    for idx, gpt_seg in enumerate(gpt_segments_batch):
                        combined_segments[batch_start + idx] = gpt_seg
                    debug_print(
                        f"[Stage2] {json_path}: batch {batch_start}-{batch_end - 1} merged on attempt {batch_attempt}."
                    )
                    stage2_input_tokens += batch_prompt_tokens
                    stage2_output_tokens += batch_response_tokens
                    batch_success = True
                    break

                if not batch_success:
                    stage2_success = False
                    print(
                        f"Stage2 failed for {json_path} batch {batch_start}-{batch_end - 1} after "
                        f"{STAGE2_BATCH_RETRIES} attempts. Last error: {last_error}"
                    )
                    break

            if not stage2_success or any(seg is None for seg in combined_segments):
                print(f"Stage2 processing failed for {json_path_str}, rerunning...")
                continue

            combined_response_json = {"hotwords": hotwords, "data": combined_segments}

            segments_updated = update_segments_with_gpt(segments, combined_response_json, json_path)
            if not segments_updated:
                if attempt < process_retries:
                    print(f"Retrying ({attempt+1}/{process_retries}) due to no segments updated...")
                    continue
                return False

            json_data["hotwords"] = hotwords
            json_data["speaker_map_gpt"] = speaker_map
            json_data['speaker_counter_str'] = speaker_counter_str
            if speaker_name_source is not None:
                json_data["speaker_name_source"] = speaker_name_source
            if difficulty is not None:
                json_data["difficulty"] = difficulty
            json_data["prompt_gpt_stage1"] = stage1_prompt
            json_data["prompt_gpt_stage2"] = stage2_batches_log[-1]["prompt"] if stage2_batches_log else stage1_prompt

            if config.local_dir:
                local_copy_path = relative_output_path(json_path, config.local_dir)
                save_json(original_segments_dump, local_copy_path)

            if config.gpt_dir:
                gpt_output_path = relative_output_path(json_path, config.gpt_dir)
                raw_json = {
                    "stage1": {"prompt": stage1_prompt, "response": stage1_json},
                    "stage2": stage2_batches_log,
                }
                save_json(raw_json, gpt_output_path)

            if config.remote_dir:
                remote_path = relative_output_path(json_path, config.remote_dir)
                save_json(json_data, remote_path)

            duration_seconds = int(base_json.get("audio_length", 0))
            print(
                f"----------------- Time {time.time() - start_time:.1f}, len(segments) {len(segments)}, "
                f"stage1 tokens in/out {stage1_input_tokens}/{stage1_output_tokens}, "
                f"stage2 tokens in/out {stage2_input_tokens}/{stage2_output_tokens}, "
                f"duration {duration_seconds}s, retried {attempt}, segment_key {segment_key}, Processed {json_path}"
            )
            return True

        return False
    except Exception as e:
        print("found error:", json_path_str)
        print(type(e).__name__, e, traceback.format_exc())
    finally:
        pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine ASR results with Azure OpenAI")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_30k_podcast_json_v2.scp"),
        help="Path to index file that lists JSON files to process",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None, # Path("/data/yaoyaochang/code/speech/data/gpt_refine_asr/relabeled_filtered_30k_podcast_json")
        help="Directory to store local copies of raw segments",
    )
    parser.add_argument(
        "--gpt-dir",
        type=Path,
        default=None, # Path("/data/yaoyaochang/code/speech/data/gpt_refine_asr/any_question3")
        help="Directory to store GPT outputs",
    )
    parser.add_argument(
        "--output-remote-dir",
        type=Path,
        default=Path("/mnt/conversationhubhot/zhiliang/speech/data/gpt_refine_asr/en_long_v3.0_all"),
        help="Optional directory to store JSON files augmented with GPT results",
    )
    parser.add_argument("--count", type=int, default=100, help="Number of files to process (0 = all)")
    parser.add_argument("--workers", type=int, default=200, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=1, help="Shuffle seed for sampling files")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries for rate-limit handling")
    parser.add_argument(
        "--api-family",
        type=str,
        default="gpt5",
        choices=list(API_CONFIG_GROUPS.keys()),
        help="Which API family to use for Azure OpenAI endpoints (default: gpt5)",
    )
    parser.add_argument(
        "--process-retries",
        type=int,
        default=5,
        help="Number of retries for processing a single JSON file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--segment-batch-size",
        type=int,
        default=SEGMENT_BATCH_SIZE,
        help="Number of segments to send per Stage 2 batch",
    )
    parser.add_argument(
        "--use-local-file",
        action="store_true",
        help="Prefix all file paths with /data/yaoyaochang to use local mounted files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DEBUG_MODE, SEGMENT_BATCH_SIZE
    DEBUG_MODE = args.debug
    SEGMENT_BATCH_SIZE = max(100, args.segment_batch_size)
    print("Arguments:", args)
    # if args.gpt_dir is None:
    #     args.gpt_dir = Path(str(args.output_remote_dir) + "_raw_gpt")
    selected_configs = API_CONFIG_GROUPS.get(args.api_family)
    if not selected_configs:
        raise ValueError(f"No API configs defined for family '{args.api_family}'")
    print(f"Using {len(selected_configs)} API configurations.")

    if args.use_local_file:
        global CHINESE_METADATA_DIR, ENGLISH_RSS_DIR, ENGLISH_AUDIO_DESC_DIR
        CHINESE_METADATA_DIR = apply_local_prefix(CHINESE_METADATA_DIR, True)
        ENGLISH_RSS_DIR = apply_local_prefix(ENGLISH_RSS_DIR, True)
        ENGLISH_AUDIO_DESC_DIR = apply_local_prefix(ENGLISH_AUDIO_DESC_DIR, True)
        args.index_path = apply_local_prefix(args.index_path, True)
        args.local_dir = apply_local_prefix(args.local_dir, True)
        args.gpt_dir = apply_local_prefix(args.gpt_dir, True)
        args.output_remote_dir = apply_local_prefix(args.output_remote_dir, True)

    shared_history, shared_lock = init_shared_event_history()

    config = ProcessingConfig(
        local_dir=args.local_dir,
        gpt_dir=args.gpt_dir,
        remote_dir=args.output_remote_dir,
    )

    index_paths = load_index_paths(args.index_path)
    if args.use_local_file:
        index_paths = [apply_local_prefix_str(p, True) for p in index_paths]
    if not index_paths:
        print("Index file is empty, nothing to process.")
        return
    print(f"Loaded {len(index_paths)} paths from index.")

    shuffle_seed = args.seed
    if shuffle_seed == -1:
        shuffle_seed = random.randint(0, 2**32 - 1)
        print(f"Randomizing seed, using {shuffle_seed}")
    random.seed(shuffle_seed)
    random.shuffle(index_paths)
    if args.count > 0:
        target_paths = index_paths[: args.count]
    else:
        target_paths = index_paths
    insert_paths: List[str] = []
    # insert_paths = [
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000091/62c855491e4e54330c8d5e18.json", # 一个segment多个speaker，segment 247
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000119/64c2a1ab1cef0544b75f2d63.json", # 一个segment多个speaker，57-93秒
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000000/5e280faa418a84a0461f9cb2.json", # 中英混杂 （actor network theory）
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000070/61199b19d0030c451ff3bb26.json", # 声动早咖啡，喜茶go
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000068/60daea0303c9bf1da959e163.json", # 蓬皮杜
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000110/642f8ee366e4c00c6a6c9774.json", # 有几句文言文。
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000086/626eabf1e2870ce7771302b0.json", # 莫名的speaker识别错误
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000159/66a7927b33ddcbb53ce8bd59.json", # speaker 1, 有BGM导致speaker识别错误
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000156/668272a1077b88831b745e2b.json", # speaker 0, 有BGM导致speaker识别错误
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_10k_xyz_json_v2/00000055/5fe3fbd6dee9c1e16d53cfd1.json", # 歌词
    #     "/mnt/conversationhubhot/zhiliang/speech/data/relabeled_xyz_20250610_json_sim/00000053/5fb74769dee9c1e16d88b431.json", # 中国人教韩语
    #     "/mnt/conversationhubhot/zhiliang/speech/data/relabeled_enpod_short_trim_json/1489712/206.json",
    #     # "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_30k_podcast_json_v2/1483200/19.json", # 8小时
    #     "/mnt/conversationhubhot/zhiliang/speech/data/checked_relabeled_filtered_30k_podcast_json_v2/1459554/66.json" # 歌词
    #     ]
    if args.use_local_file:
        insert_paths = [apply_local_prefix_str(p, True) for p in insert_paths]
    target_paths = insert_paths + target_paths
    print("first 5 paths:", target_paths[:5])
    print(f"Processing {len(target_paths)} files with {args.workers} workers...")
    worker = partial(
        process_file,
        api_configs=selected_configs,
        config=config,
        max_retries=args.max_retries,
        process_retries=args.process_retries,
    )

    with Pool(
        processes=args.workers,
        initializer=_pool_initializer,
        initargs=(shared_history, shared_lock),
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(worker, target_paths),
            total=len(target_paths),
            desc="Processing files",
        ):
            pass


if __name__ == "__main__":
    main()