"""
通用文本处理任务主脚本
处理scp文件中的所有JSON，支持多种processor（标点、数字转换等）
使用Ray并发处理提高效率
"""

import argparse
import json
import os
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from tqdm import tqdm
from collections import defaultdict
import yaml  # 在顶部导入yaml

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from azure_api import (
    process_tasks_with_ray,
    process_tasks_sequential,
    ClientManager,
    process_single_task,
    get_api_configs,
    load_config,
    TaskProcessor,
)
from API_config import get_api_configs

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_lrc import parse_lrc_with_timestamps

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False



# ============ Processor 注册表 ============
PROCESSOR_REGISTRY = {
    "punctuation": {
        "module": "punctuation_processor",
        "class": "PunctuationProcessor",
        "output_suffix": ".punctuated.json",
        "data_field": "punctuation_added",
        "description": "添加标点符号",
    },
    "number": {
        "module": "number_processor",
        "class": "NumberProcessor",
        "output_suffix": ".number_normalized.json",
        "data_field": "number_normalized",
        "description": "数字格式转换（中文数字转阿拉伯数字）",
    },
    "lyric": {
        "module": "lyric_processor",
        "class": "LyricProcessor",
        "output_suffix": ".lyric.json",
        "data_field": "lyric_detected",
        "description": "判断segment是否为歌词",
    },
}


def get_processor_class(processor_type: str) -> Tuple[Type[TaskProcessor], Dict[str, Any]]:
    """
    根据processor类型获取对应的类和配置
    
    Args:
        processor_type: processor类型 (punctuation, number等)
    
    Returns:
        (ProcessorClass, config_dict)
    """
    if processor_type not in PROCESSOR_REGISTRY:
        available = ", ".join(PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown processor type: {processor_type}. Available: {available}")
    
    config = PROCESSOR_REGISTRY[processor_type]
    module = importlib.import_module(config["module"])
    processor_class = getattr(module, config["class"])
    
    return processor_class, config


def get_split_function(processor_type: str):
    """获取对应processor的分割函数"""
    config = PROCESSOR_REGISTRY[processor_type]
    module = importlib.import_module(config["module"])
    return getattr(module, "split_segments_by_duration")


def load_scp_file(scp_path: str) -> List[str]:
    """加载scp文件，返回JSON文件路径列表"""
    scp_path = Path(scp_path)
    lrc_paths = []
    
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # 直接使用整行作为路径（支持带空格的文件名）
                lrc_paths.append(line)
    
    return lrc_paths

def main():
    parser = argparse.ArgumentParser(
        description="通用ASR文本处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_processor.py --scp data.scp --yaml yamls/punctuation_gpt5.yaml
        """
    )
    parser.add_argument("--scp", type=str, required=True, help="SCP文件路径，包含JSON文件列表")
    parser.add_argument("--yaml", type=str, required=True, help="YAML配置文件路径")
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="输出路径，覆盖原始yaml中的output_dir")
    args = parser.parse_args()

    # 从YAML加载所有配置
    with open(args.yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {args.yaml}")

    # 必须参数检查
    if "processor" not in cfg:
        raise ValueError("YAML配置必须包含 'processor' 字段 (punctuation/number)")

    # 提取配置（带默认值）
    processor_type = cfg["processor"]
    api_family = cfg.get("api_family", "gpt4o")
    num_workers = cfg.get("num_workers", 4)
    language = cfg.get("language", "zh")
    prompt_language = cfg.get("prompt_language", "zh")
    strict_check = cfg.get("strict_check", True)
    max_api_retries = cfg.get("max_api_retries", 5)
    max_check_retries = cfg.get("max_check_retries", 3)
    batch_duration = cfg.get("batch_duration", 600.0)
    output_dir = cfg.get("output_dir", None)
    overwrite = cfg.get("overwrite", False)
    debug = cfg.get("debug", False)
    count = cfg.get("count", 0)

    if args.output_dir is not None:
        output_dir = args.output_dir

    # 获取processor配置
    processor_class, processor_config = get_processor_class(processor_type)
    split_segments_by_duration = get_split_function(processor_type)
    output_suffix = processor_config["output_suffix"]
    data_field = processor_config["data_field"]
    
    print(f"Processor: {processor_type} - {processor_config['description']}")
    print(f"Prompt language: {prompt_language}")
    
    # 加载SCP文件
    lrc_paths = load_scp_file(args.scp)
    print(f"Loaded {len(lrc_paths)} JSON files from {args.scp}")
    
    if count > 0:
        lrc_paths = lrc_paths[:count]
        print(f"Processing first {count} files")
    
    # 检查Ray是否可用
    if not RAY_AVAILABLE:
        print("Warning: Ray not available, using sequential processing (slow)")
        use_ray = False
    else:
        use_ray = True
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    # ============ 第一步：收集所有batch任务 ============
    print("\n[Step 1] Collecting all batches from JSON files...")
    all_batch_tasks = []
    json_metadata = {}
    skipped_count = 0
    
    for lrc_path in tqdm(lrc_paths, desc="Loading JSON files"):
        lrc_path = Path(lrc_path)
        
        if not lrc_path.exists():
            print(f"Warning: File not found: {lrc_path}")
            continue
        
        # 确定输出路径
        if output_dir:
            # 保留倒数两级目录: parent_dir/filename
            output_path = Path(output_dir) / lrc_path.parent.name / lrc_path.with_suffix(output_suffix).name
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
        elif overwrite:
            output_path = lrc_path
        else:
            # 使用processor特定的后缀
            output_path = lrc_path.with_suffix(output_suffix)
        
        # 检查是否已处理（支持resume）
        if output_path.exists() and not overwrite:
            skipped_count += 1
            continue
        
        try:            
            segments, wrong_segs = parse_lrc_with_timestamps(lrc_path)
            if not segments:
                continue

            # 分割为batches
            batches = split_segments_by_duration(segments, max_duration_seconds=batch_duration)
            
            json_metadata[str(lrc_path)] = {
                "output_path": str(output_path),
                "num_batches": len(batches),
                "num_segments": len(segments),
                "error_loading_segments": wrong_segs
            }
            
            # 添加所有batch任务
            for batch_id, batch_segments in enumerate(batches):
                all_batch_tasks.append({
                    "lrc_path": str(lrc_path),
                    "batch_id": batch_id,
                    "segments": batch_segments,
                })
        except Exception as e:
            print(f"Error loading {lrc_path}: {e}")
            continue
    
    print(f"Total: {len(json_metadata)} JSON files to process, {len(all_batch_tasks)} batches")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} already processed files (resume mode)")
    
    if not all_batch_tasks:
        print("No tasks to process. All files already done!")
        return
    
    # ============ 第二步：以JSON文件为单位并发处理 ============
    print(f"\n[Step 2] Processing {len(json_metadata)} JSON files with Ray ({num_workers} workers)...")
    
    # print(all_batch_tasks)

    # 初始化API配置
    api_configs = get_api_configs(api_family)
    
    # 使用字典按 lrc_path 分组 batches（O(n) 而不是 O(n*m)）
    batches_by_json = defaultdict(list)
    for task in all_batch_tasks:
        batches_by_json[task["lrc_path"]].append(task)
    
    # 构建JSON文件任务列表
    json_tasks = []
    for lrc_path, metadata in json_metadata.items():
        file_batches = batches_by_json.get(lrc_path, [])
        file_batches.sort(key=lambda x: x["batch_id"])
        
        json_tasks.append({
            "lrc_path": lrc_path,
            "output_path": metadata["output_path"],
            "batches": file_batches,
            "num_segments": metadata["num_segments"],
            "error_loading": metadata["error_loading_segments"],
        })
    
    if use_ray:
        # 使用Ray并发处理
        @ray.remote
        class JsonFileWorker:
            def __init__(self, worker_id: int, all_api_config_dicts: List[Dict], 
                         processor_type: str, language: str, prompt_language: str, strict_check: bool):
                """
                每个worker持有所有client，遇到429时快速切换
                """
                from azure_api import build_openai_client
                from API_config import ApiConfig
                
                self.worker_id = worker_id
                self.processor_type = processor_type
                
                # 动态导入processor
                processor_class, self.processor_config = get_processor_class(processor_type)
                
                # 初始化processor（支持不同的参数）
                try:
                    self.processor = processor_class(
                        language=language, 
                        prompt_language=prompt_language,
                        strict_check=strict_check
                    )
                except TypeError:
                    # 兼容不支持prompt_language参数的processor
                    self.processor = processor_class(language=language, strict_check=strict_check)
                
                # 初始化所有clients
                self.clients = []
                print(f"Worker {worker_id}: Initializing {len(all_api_config_dicts)} clients...")
                
                for i, cfg_dict in enumerate(all_api_config_dicts):
                    try:
                        api_config = ApiConfig(**cfg_dict)
                        client = build_openai_client(api_config)
                        self.clients.append((api_config, client))
                    except Exception as e:
                        print(f"Worker {worker_id}: Failed to init client {i}: {e}")
                
                self.current_idx = worker_id % len(self.clients) if self.clients else 0
                print(f"Worker {worker_id}: Ready with {len(self.clients)} clients, starting at idx {self.current_idx}")
            
            def _get_next_client(self):
                """获取下一个client（轮询）"""
                if not self.clients:
                    return None, None
                self.current_idx = (self.current_idx + 1) % len(self.clients)
                return self.clients[self.current_idx]
            
            def _get_current_client(self):
                """获取当前client"""
                if not self.clients:
                    return None, None
                return self.clients[self.current_idx]
            
            def _process_single_batch(self, batch_segments: List[Dict], batch_id: int, json_name: str,
                                      max_api_retries: int, max_check_retries: int) -> Dict:
                """处理单个batch"""
                from azure_api import get_chat_response, _event_tracker
                import time
                import random
                
                batch_data = {
                    "batch_id": batch_id,
                    "segments": batch_segments,
                }
                
                # 准备输入
                input_data = self.processor.prepare_input(batch_data)
                prompt = self.processor.build_prompt(input_data)
                system_prompt = self.processor.get_system_prompt()
                # print(prompt)
                # API调用 + 验证循环
                success = False
                output_text = None
                error_msg = None
                
                for check_attempt in range(max_check_retries):
                    output_text = None
                    total_endpoints = len(self.clients)
                    
                    for round_num in range(max_api_retries):
                        for endpoint_attempt in range(total_endpoints):
                            api_config, client = self._get_current_client()
                            if api_config is None:
                                error_msg = "No available clients"
                                break
                            
                            try:
                                output_text, usage = get_chat_response(
                                    client,
                                    api_config.model,
                                    prompt,
                                    system_prompt,
                                )
                                _event_tracker.record_event(api_config, True)
                                break
                                
                            except Exception as exc:
                                _event_tracker.record_event(api_config, False)
                                exc_str = str(exc)
                                is_rate_limit = "429" in exc_str or "NoCapacity" in exc_str or "rate" in exc_str.lower()
                                
                                if is_rate_limit:
                                    print(f"Worker {self.worker_id}: 429 on {api_config.model}, switching endpoint...")
                                else:
                                    print(f"Worker {self.worker_id}: Error ({type(exc).__name__}) on {api_config.model}, switching...")
                                self._get_next_client()
                                error_msg = f"API error: {exc}"
                        
                        if output_text is not None:
                            break
                        
                        if round_num < max_api_retries - 1:
                            sleep_sec = random.randint(30, 60)
                            print(f"Worker {self.worker_id}: All {total_endpoints} endpoints failed, waiting {sleep_sec}s before round {round_num + 2}...")
                            time.sleep(sleep_sec)
                    
                    if output_text is None:
                        error_msg = f"API failed after {max_api_retries} rounds × {total_endpoints} endpoints"
                        break
                    
                    if self.processor.check_output(output_text, input_data):
                        success = True
                        break
                    else:
                        error_msg = f"Validation failed on attempt {check_attempt + 1}"
                        prompt = self.processor.build_prompt(input_data)
                
                # print(output_text)
                # 处理结果
                if success and output_text:
                    try:
                        processed = self.processor.process_output(output_text, input_data)
                        segments = processed["segments"]
                    except Exception as e:
                        success = False
                        error_msg = f"Process output error: {e}"
                        segments = None
                else:
                    segments = None
                
                if not success:
                    print(f"❌ Batch {batch_id} in {json_name} failed: {error_msg}")
                    segments = []
                    for seg in batch_segments:
                        seg_copy = seg.copy()
                        seg_copy["is_success"] = "fail"
                        segments.append(seg_copy)
                
                return {
                    "batch_id": batch_id,
                    "success": success,
                    "segments": segments,
                    "error": error_msg,
                }
            
            def process_json_file(self, task: Dict, max_api_retries: int, max_check_retries: int) -> Dict:
                """处理整个JSON文件（包含所有batches），处理完后立即保存"""
                from pathlib import Path
                import json
                
                lrc_path = task["lrc_path"]
                output_path = task["output_path"]
                batches = task["batches"]
                json_name = Path(lrc_path).name
                
                
                print(f"Worker {self.worker_id}: Processing {json_name} ({len(batches)} batches)...")
                
                # 处理所有batches
                all_segments = []
                failed_batches = []
                success_batches = 0
                
                for batch_info in batches:
                    batch_result = self._process_single_batch(
                        batch_segments=batch_info["segments"],
                        batch_id=batch_info["batch_id"],
                        json_name=json_name,
                        max_api_retries=max_api_retries,
                        max_check_retries=max_check_retries,
                    )
                    
                    all_segments.extend(batch_result["segments"])
                    if batch_result["success"]:
                        success_batches += 1
                    else:
                        failed_batches.append(batch_result["batch_id"])
                
                # 如果所有batch都失败，不保存文件（方便resume重试）
                if failed_batches: #success_batches == 0:
                    print(f"❌ Worker {self.worker_id}: All {len(batches)} batches failed for {json_name}, NOT saving (will retry on resume)")
                    return {
                        "lrc_path": lrc_path,
                        "output_path": str(output_path),
                        "success": False,
                        "saved": False,
                        "total_batches": len(batches),
                        "success_batches": 0,
                        "failed_batches": failed_batches,
                    }
                
                # 更新数据
                results = {}
                results["segments"] = all_segments
                results[self.processor_config["data_field"]] = True
                results["error_loading"] = task["error_loading"]
                if failed_batches:
                    results["failed_batches"] = failed_batches
                
                # 立即保存
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with output_path.open("w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                except:
                    print(f"❌ Worker {self.worker_id}: {output_path} might have Output Error")
                    return {
                        "lrc_path": lrc_path,
                        "output_path": str(output_path),
                        "success": False,
                        "saved": False,
                        "total_batches": len(batches),
                        "success_batches": 0,
                        "failed_batches": failed_batches,
                    }
                
                status = "✅" if not failed_batches else "⚠️"
                print(f"{status} Worker {self.worker_id}: Saved {json_name} ({success_batches}/{len(batches)} batches success)")
                
                return {
                    "lrc_path": lrc_path,
                    "output_path": str(output_path),
                    "success": len(failed_batches) == 0,
                    "saved": True,
                    "total_batches": len(batches),
                    "success_batches": success_batches,
                    "failed_batches": failed_batches,
                }
        
        # 创建workers
        actual_workers = min(num_workers, len(json_tasks))
        print(f"Creating {actual_workers} workers (api_configs: {len(api_configs)}, json_files: {len(json_tasks)})...")
        
        # 将所有api_configs转换为dict列表
        all_cfg_dicts = []
        for cfg in api_configs:
            all_cfg_dicts.append({
                "scope": cfg.scope,
                "endpoint": cfg.endpoint,
                "model": cfg.model,
                "api_version": cfg.api_version,
            })
        
        # 每个worker持有所有clients
        workers = []
        for i in range(actual_workers):
            workers.append(JsonFileWorker.remote(
                i, all_cfg_dicts, 
                processor_type, language, prompt_language, strict_check
            ))
        
        print(f"All {actual_workers} workers initialized.")
        
        # 分发任务
        pending_futures = []
        task_map = {}
        
        for i, task in enumerate(json_tasks):
            worker = workers[i % actual_workers]
            future = worker.process_json_file.remote(task, max_api_retries, max_check_retries)
            pending_futures.append(future)
            task_map[future] = i
        
        print(f"Submitted {len(pending_futures)} JSON files to {actual_workers} workers")
        
        # 收集结果
        json_results = [None] * len(json_tasks)
        pbar = tqdm(total=len(json_tasks), desc="Processing JSON files")
        
        while pending_futures:
            done, pending_futures = ray.wait(pending_futures, num_returns=1)
            for future in done:
                result = ray.get(future)
                idx = task_map[future]
                json_results[idx] = result
                pbar.update(1)
        
        pbar.close()
    else:
        # 顺序处理（降级模式）
        processor = processor_class(
            language=language, 
            prompt_language=prompt_language, 
            strict_check=strict_check
        )
        
        client_manager = ClientManager(api_configs)
        
        json_results = []
        for task in tqdm(json_tasks, desc="Processing JSON files"):
            lrc_path = task["lrc_path"]
            output_path = task["output_path"]
            batches = task["batches"]
            json_name = Path(lrc_path).name
            

            
            print(f"Processing {json_name} ({len(batches)} batches)...")
            
            all_segments = []
            failed_batches = []
            success_batches = 0
            
            for batch_info in batches:
                batch_data = {
                    "batch_id": batch_info["batch_id"],
                    "segments": batch_info["segments"],
                }
                
                result = process_single_task(
                    task_processor=processor,
                    raw_data=batch_data,
                    client_manager=client_manager,
                    max_api_retries=max_api_retries,
                    max_check_retries=max_check_retries,
                    task_id=f"{Path(lrc_path).stem}_batch_{batch_info['batch_id']}",
                )
                
                if result["success"]:
                    segments = result["processed_output"]["segments"]
                    success_batches += 1
                else:
                    print(f"❌ Batch {batch_info['batch_id']} in {json_name} failed: {result.get('error')}")
                    segments = []
                    for seg in batch_info["segments"]:
                        seg_copy = seg.copy()
                        seg_copy["is_success"] = "fail"
                        segments.append(seg_copy)
                    failed_batches.append(batch_info["batch_id"])
                
                all_segments.extend(segments)
            
            # 如果所有batch都失败，不保存文件
            if failed_batches: #success_batches == 0:
                print(f"❌ All {len(batches)} batches failed for {json_name}, NOT saving (will retry on resume)")
                json_results.append({
                    "lrc_path": lrc_path,
                    "output_path": str(output_path),
                    "success": False,
                    "saved": False,
                    "total_batches": len(batches),
                    "success_batches": success_batches,
                    "failed_batches": failed_batches,
                })
                continue
            
            # 更新数据并保存
            results = {}
            results["segments"] = all_segments
            results[data_field] = True
            results["error_loading"] = task["error_loading"]
            if failed_batches:
                results["failed_batches"] = failed_batches
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except:
                print(f"❌ {output_path} might have Output Error")
                json_results.append({
                    "lrc_path": lrc_path,
                    "output_path": str(output_path),
                    "success": False,
                    "saved": False,
                    "total_batches": len(batches),
                    "success_batches": 0,
                    "failed_batches": failed_batches,
                })
            
            status = "✅" if not failed_batches else "⚠️"
            print(f"{status} Saved {json_name} ({success_batches}/{len(batches)} batches success)")
            
            json_results.append({
                "lrc_path": lrc_path,
                "output_path": str(output_path),
                "success": len(failed_batches) == 0,
                "saved": True,
                "total_batches": len(batches),
                "success_batches": success_batches,
                "failed_batches": failed_batches,
            })
    
    # ============ 第三步：打印统计 ============
    print(f"\n[Step 3] Processing complete, files already saved.")
    
    success_count = sum(1 for r in json_results if r["success"])
    fail_count = len(json_results) - success_count
    total_batches = sum(r["total_batches"] for r in json_results)
    success_batches = sum(r["success_batches"] for r in json_results)
    
    print("\n" + "=" * 50)
    print(f"Processing complete ({processor_type}):")
    print(f"  Total LRC files in scp: {len(lrc_paths)}")
    print(f"  Skipped (already done): {skipped_count}")
    print(f"  Processed this run: {len(json_results)}")
    print(f"  Total batches: {total_batches}")
    print(f"  Success batches: {success_batches}")
    print(f"  Failed batches: {total_batches - success_batches}")
    
    saved_count = sum(1 for r in json_results if r.get("saved", True))
    not_saved_count = len(json_results) - saved_count
    print(f"  Files saved: {saved_count}")
    print(f"  Files NOT saved (all failed, will retry): {not_saved_count}")
    print(f"  Files with all success: {success_count}")
    print(f"  Files with partial failures: {fail_count - not_saved_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
