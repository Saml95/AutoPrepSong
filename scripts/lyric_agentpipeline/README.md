# AudioAutoPrepV2/agentpipeline 统一文本处理流水线

## 目录结构

- run_processor.py         # 通用主入口，支持多种processor
- run_processor.sh         # 通用shell入口，支持processor参数
- processor.py             # 统一注册/集成所有processor（如标点、数字等）
- azure_api.py             # 统一API调用与管理
- yamls/                   # 推荐所有任务配置yaml放这里
    - example_number_en.yaml
    - example_punctuation_zh.yaml
- legacy/                  # 历史/不用的脚本归档
- number_processor.py      # 数字归一化处理器（可合并进processor.py）
- punctuation_processor.py # 标点处理器（可合并进processor.py）

## 推荐用法

1. 配置任务参数：
   - 在 yamls/ 下新建任务配置 yaml
2. 运行任务：
   ```bash
   ./run_processor.sh number --yaml yamls/example_number_en.yaml
   # 或
   ./run_processor.sh punctuation --yaml yamls/example_punctuation_zh.yaml
   ```

3. 新增处理器：
   - 在 processor.py 中注册/实现新处理器
   - 在 PROCESSOR_REGISTRY 中添加配置

## 说明
- 只需维护 processor.py、azure_api.py、run_processor.py/sh，结构极简
- 任务参数全部可 yaml 配置，便于批量和复现
- legacy/ 目录仅存放历史/不用的脚本，主目录保持清爽
