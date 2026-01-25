# 标点符号添加 Pipeline 架构文档

## 📋 概述

本 Pipeline 用于为 ASR（语音识别）输出的 JSON 文件添加中文标点符号。使用 Ray 实现并发处理，支持多个 Azure OpenAI API endpoint 负载均衡。

---

## 🔄 整体架构流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              run_punctuation.py 主流程                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: 数据准备阶段                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   json_local.scp ──────► load_scp_file() ──────► [json_path1, json_path2, ...]     │
│                                                                                     │
│   对每个 JSON 文件:                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  cel_phone.json                                                              │   │
│   │  ├── audio_path: "xxx.wav"                                                   │   │
│   │  └── segments: [                                                             │   │
│   │        {text: "...", speaker: "O1", start: 0.7, end: 3.9},                   │   │
│   │        {text: "...", speaker: "O2", start: 4.5, end: 6.4},                   │   │
│   │        ... (239个segments, 总时长约20分钟)                                     │   │
│   │      ]                                                                       │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                          │
│                                          ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  split_segments_by_duration(segments, max_duration=120s)                     │   │
│   │                                                                              │   │
│   │  规则：                                                                        │   │
│   │  1. 必须在 speaker 切换点分割                                                  │   │
│   │  2. 每个 batch 不超过 120 秒                                                   │   │
│   │  3. 最后一个 batch 如果 < 60秒，合并到上一个                                     │   │
│   │                                                                              │   │
│   │  结果: batches = [batch0, batch1, batch2, ...]                               │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                          │
│                                          ▼                                          │
│   all_batch_tasks = [                                                               │
│     {json_path: "file1.json", batch_id: 0, segments: [...]},                        │
│     {json_path: "file1.json", batch_id: 1, segments: [...]},                        │
│     {json_path: "file2.json", batch_id: 0, segments: [...]},                        │
│     ...                                                                             │
│   ]                                                                                 │
│                                                                                     │
│   例如: 2个JSON文件 × 每个4个batch = 8个任务                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Ray 并发处理                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   API_CONFIGS_GPT5 = [config0, config1, config2, ..., config15]  (16个endpoint)     │
│                                                                                     │
│   num_workers = min(args.num_workers, len(api_configs), len(tasks))                 │
│              = min(16, 16, 8) = 8                                                   │
│                                                                                     │
│   ┌─────────────── Worker 初始化 (并行) ───────────────┐                             │
│   │                                                    │                            │
│   │   Worker 0 ──► api_configs[0] ──► build_client()  │                            │
│   │   Worker 1 ──► api_configs[1] ──► build_client()  │                            │
│   │   Worker 2 ──► api_configs[2] ──► build_client()  │                            │
│   │   ...                                              │                            │
│   │   Worker 7 ──► api_configs[7] ──► build_client()  │                            │
│   │                                                    │                            │
│   │   每个 Worker 只初始化 1 个 API client！             │                            │
│   └────────────────────────────────────────────────────┘                            │
│                                          │                                          │
│                                          ▼                                          │
│   ┌─────────────── 任务分发 (轮询) ───────────────┐                                   │
│   │                                               │                                 │
│   │   Task 0 ──► Worker 0                         │                                 │
│   │   Task 1 ──► Worker 1                         │                                 │
│   │   Task 2 ──► Worker 2                         │                                 │
│   │   ...                                         │                                 │
│   │   Task 7 ──► Worker 7                         │                                 │
│   │   Task 8 ──► Worker 0  (如果有更多任务)        │                                 │
│   │   ...                                         │                                 │
│   │                                               │                                 │
│   │   所有任务同时提交！Ray 自动调度执行             │                                 │
│   └───────────────────────────────────────────────┘                                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  每个 Worker 的处理逻辑 (process_batch)                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         PunctuationProcessor                                 │   │
│   │                                                                              │   │
│   │   1. prepare_input(batch_data)                                               │   │
│   │      └─► 去掉原文标点: "你听了吗？" → "你听了吗"                                 │   │
│   │                                                                              │   │
│   │   2. build_prompt(input_data)                                                │   │
│   │      └─► 构建 GPT prompt，包含严格的标点规则                                    │   │
│   │                                                                              │   │
│   │   3. 调用 API                                                                 │   │
│   │      └─► get_chat_response(client, model, prompt, system_prompt)             │   │
│   │                                                                              │   │
│   │   4. check_output(output_text, input_data)                                   │   │
│   │      └─► 验证: 去标点后的输出 == 去标点后的输入                                  │   │
│   │      └─► 如果失败，记录错误反馈，重新 build_prompt                              │   │
│   │                                                                              │   │
│   │   5. process_output(output_text, input_data)                                 │   │
│   │      └─► 将添加标点后的 text 更新到原始 segments                                │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: 结果聚合与保存                                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   batch_results = [result0, result1, result2, ...]                                  │
│                                                                                     │
│   按 json_path 分组:                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  file1.json:                                                                 │   │
│   │    batch_0 结果 + batch_1 结果 + batch_2 结果 + batch_3 结果                   │   │
│   │    └─► 合并所有 segments                                                      │   │
│   │    └─► 保存到 file1.punctuated.json                                          │   │
│   │                                                                              │   │
│   │  file2.json:                                                                 │   │
│   │    batch_0 结果 + batch_1 结果                                                │   │
│   │    └─► 合并所有 segments                                                      │   │
│   │    └─► 保存到 file2.punctuated.json                                          │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│   失败的 segment 会被标记: {"is_success": "fail"}                                    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 并发实现详解

### Ray Actor Pool 模型

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Ray Actor Pool 并发模型                                   │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│   主进程                                                                            │
│      │                                                                             │
│      ├──► BatchWorker.remote(0, cfg0) ──► Worker 0 (独立进程)                       │
│      ├──► BatchWorker.remote(1, cfg1) ──► Worker 1 (独立进程)                       │
│      ├──► BatchWorker.remote(2, cfg2) ──► Worker 2 (独立进程)                       │
│      └──► BatchWorker.remote(3, cfg3) ──► Worker 3 (独立进程)                       │
│                                                                                    │
│   任务提交 (同时提交，不等待):                                                        │
│      │                                                                             │
│      ├──► worker0.process_batch.remote(task0) ──► future0                          │
│      ├──► worker1.process_batch.remote(task1) ──► future1                          │
│      ├──► worker2.process_batch.remote(task2) ──► future2                          │
│      └──► worker3.process_batch.remote(task3) ──► future3                          │
│                                                                                    │
│   并行执行:                                                                          │
│                                                                                    │
│   时间 ─────────────────────────────────────────────────────────────────────►       │
│                                                                                    │
│   Worker 0: ████████████████████ task0 完成 ████████████████████ task4 完成         │
│   Worker 1: ██████████████████████████ task1 完成 ██████████████ task5 完成         │
│   Worker 2: ████████████████ task2 完成 ████████████████████████ task6 完成         │
│   Worker 3: ██████████████████████ task3 完成 ████████████████████ task7 完成       │
│                                                                                    │
│   结果收集 (ray.wait):                                                               │
│      └──► 任何一个完成就立即返回，更新进度条                                           │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### 重试机制

```python
for check_attempt in range(max_check_retries):    # 验证重试 3 次
    for api_attempt in range(max_api_retries):    # API重试 5 次
        try:
            output = call_api()
            break
        except:
            sleep(5-15s)

    if check_output(output):
        return success
    else:
        rebuild_prompt_with_error_feedback()
```

---

## 📊 关键组件说明

| 组件 | 文件 | 功能 |
|------|------|------|
| **API_config.py** | API endpoint 配置 | 定义 GPT4O/GPT41/GPT5 的多个 endpoint |
| **azure_api.py** | API 客户端 | `build_openai_client()`, `get_chat_response()`, `ClientManager` |
| **punctuation_processor.py** | 业务逻辑 | `prepare_input()`, `build_prompt()`, `check_output()`, `process_output()` |
| **run_punctuation.py** | 主入口 | 数据分片、Ray 调度、结果聚合 |

---

## 📁 文件结构

```
agentpipeline/
├── API_config.py              # API endpoint 配置
├── azure_api.py               # Azure OpenAI 客户端封装
├── punctuation_processor.py   # 标点处理器 (TaskProcessor 实现)
├── run_punctuation.py         # 主入口脚本
├── run_punctuation.sh         # 运行脚本
└── README_PIPELINE.md         # 本文档
```

---

## 🛠 使用方法

### 基本用法

```bash
python run_punctuation.py \
    --scp /path/to/json_local.scp \
    --api-family gpt5 \
    --num-workers 16
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scp` | 必填 | SCP 文件路径，包含 JSON 文件列表 |
| `--api-family` | gpt4o | API 系列 (gpt4o, gpt5, gpt41, gpt5+gpt4o 等) |
| `--num-workers` | 4 | Ray 并发 worker 数量 |
| `--batch-duration` | 120.0 | 每个 batch 的最大时长（秒） |
| `--max-api-retries` | 5 | API 最大重试次数 |
| `--max-check-retries` | 3 | 验证最大重试次数 |
| `--output-dir` | None | 输出目录（默认生成 .punctuated.json 后缀） |
| `--overwrite` | False | 是否覆盖原文件 |
| `--count` | 0 | 只处理前 N 个文件（0 表示全部） |

---

## ⚙️ 数据处理流程

### 1. 输入数据格式

```json
{
  "audio_path": "/path/to/audio.wav",
  "audio_length": 1216.248,
  "segments": [
    {
      "text": "王嘉尔新出的一首歌新歌儿你听了吗",
      "speaker": "O1",
      "start": 0.700607532348,
      "end": 3.96849318496
    },
    ...
  ]
}
```

### 2. 输出数据格式

```json
{
  "audio_path": "/path/to/audio.wav",
  "audio_length": 1216.248,
  "segments": [
    {
      "text": "王嘉尔新出的一首歌，新歌儿你听了吗？",
      "speaker": "O1",
      "start": 0.700607532348,
      "end": 3.96849318496
    },
    ...
  ],
  "punctuation_added": true
}
```

### 3. 分片规则 (split_segments_by_duration)

1. **必须在 speaker 切换点分割** - 不能把同一个 speaker 的连续发言切断
2. **每个 batch 不超过 max_duration** - 默认 120 秒
3. **最后一个 batch 合并规则** - 如果最后一个 batch 时长 < max_duration/2，则合并到上一个

---

## 🔍 验证机制

### check_output 逻辑

```python
def check_output(output_text, input_data):
    """验证输出：去掉标点后内容应与原内容完全一致"""
    
    for orig_seg, new_seg in zip(original_segments, parsed_output):
        # 去掉标点后比较
        orig_normalized = normalize_text(orig_seg["text"])  # 去标点
        new_normalized = normalize_text(new_seg["text"])    # 去标点
        
        if orig_normalized != new_normalized:
            # 记录错误，用于下次重试时的反馈
            self.last_error_feedback = f"Segment {i} 文字内容被修改！"
            return False
    
    return True
```

### 错误反馈机制

如果验证失败，会在下次 prompt 中包含错误信息：

```
【上次错误】你之前的输出有问题：
Segment 5 文字内容被修改！
原文: "就是请每一期都请不同的明星来"
你的输出: "就是每一期都请不同的明星来"
问题: 缺少字符: 请
请仔细检查并确保不删改任何文字！
```

---

## ⚡ 性能优化

### 已实现的优化

1. **每个 Worker 只初始化 1 个 client** - 避免重复初始化所有 endpoint
2. **任务并行提交** - 所有任务同时提交给 Ray
3. **失败快速切换** - API 失败后短暂等待即切换到其他 client
4. **权重负载均衡** - 根据历史成功率调整 client 选择权重

### 可能的瓶颈

1. **验证失败重试** - 如果 GPT 经常删改文字，会导致多次重试
2. **API 响应时间** - GPT-5 响应时间较长
3. **batch 太大** - 600 秒的 batch 可能导致 API 处理时间长

---

## 📈 监控与调试

### 日志关键信息

```
[Step 1] Collecting all batches from JSON files...
Total: 10 JSON files, 45 batches to process

[Step 2] Processing 45 batches with Ray (16 workers)...
Creating 16 workers (api_configs: 16, tasks: 45)...
Worker 0: Initializing client for https://xxx.openai.azure.com/ / gpt-5-DZS
Worker 0: Ready
...
Submitted 45 tasks to 16 workers
Processing batches: 100%|██████████| 45/45 [01:23<00:00, 1.85s/it]

[Step 3] Aggregating results and saving...
Processing complete:
  JSON files: 10
  Total batches: 45
  Success batches: 43
  Failed batches: 2
```

### 错误排查

- `Validation failed` - GPT 修改了原文，检查 prompt 是否足够强调不能删改
- `API error` - 检查网络连接和 API endpoint 可用性
- `All clients failed` - 所有 endpoint 都不可用，检查 Azure 服务状态
