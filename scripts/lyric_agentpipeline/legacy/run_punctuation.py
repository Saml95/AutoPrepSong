"""
标点符号添加任务主脚本
处理scp文件中的所有JSON，为segments添加标点符号
使用Ray并发处理提高效率
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from punctuation_processor import (
    PunctuationProcessor,
    split_segments_by_duration,
    normalize_text,
)
from azure_api import (
    process_tasks_with_ray,
    process_tasks_sequential,
    ClientManager,
    process_single_task,
    get_api_configs,
    load_config,
)
from API_config import get_api_configs

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def load_scp_file(scp_path: str) -> List[str]:
    """加载scp文件，返回JSON文件路径列表"""
    scp_path = Path(scp_path)
    json_paths = []
    
    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # scp格式可能是 "id path" 或只有 "path"
                parts = line.split()
                if len(parts) >= 2:
                    json_paths.append(parts[1])
                else:
                    json_paths.append(parts[0])
    
    return json_paths


def process_single_json(
    json_path: str,
    processor: PunctuationProcessor,
    client_manager: ClientManager,
    max_api_retries: int = 5,
    max_check_retries: int = 3,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    处理单个JSON文件
    
    Args:
        json_path: JSON文件路径
        processor: 标点处理器
        client_manager: API客户端管理器
        max_api_retries: API最大重试次数
        max_check_retries: 验证最大重试次数
        output_dir: 输出目录（可选）
        overwrite: 是否覆盖原文件
    
    Returns:
        处理结果字典
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        return {"success": False, "error": f"File not found: {json_path}", "json_path": str(json_path)}
    
    # 确定输出路径
    if output_dir:
        output_path = Path(output_dir) / json_path.name
    elif overwrite:
        output_path = json_path
    else:
        output_path = json_path.with_suffix(".punctuated.json")
    
    # 检查是否已处理
    if output_path.exists() and not overwrite:
        return {"success": True, "skipped": True, "json_path": str(json_path), "output_path": str(output_path)}
    
    try:
        # 加载JSON
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        segments = data.get("segments", [])
        if not segments:
            return {"success": True, "json_path": str(json_path), "message": "No segments found"}
        
        # 分割为2分钟的batches
        batches = split_segments_by_duration(segments, max_duration_seconds=600)
        print(f"Processing {json_path.name}: {len(segments)} segments -> {len(batches)} batches")
        
        # 处理每个batch
        all_updated_segments = []
        failed_batches = []
        
        for batch_id, batch_segments in enumerate(batches):
            batch_data = {
                "batch_id": batch_id,
                "segments": batch_segments,
            }
            
            result = process_single_task(
                task_processor=processor,
                raw_data=batch_data,
                client_manager=client_manager,
                max_api_retries=max_api_retries,
                max_check_retries=max_check_retries,
                task_id=f"{json_path.stem}_batch_{batch_id}",
            )
            
            if result["success"]:
                processed_output = result["processed_output"]
                all_updated_segments.extend(processed_output["segments"])
            else:
                print(f"  Batch {batch_id} failed: {result.get('error')}")
                failed_batches.append(batch_id)
                # 保留原始segments
                all_updated_segments.extend(batch_segments)
        
        # 更新data
        data["segments"] = all_updated_segments
        data["punctuation_added"] = True
        data["failed_batches"] = failed_batches if failed_batches else None
        
        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "json_path": str(json_path),
            "output_path": str(output_path),
            "num_segments": len(segments),
            "num_batches": len(batches),
            "failed_batches": failed_batches,
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "json_path": str(json_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description="为ASR输出添加标点符号")
    parser.add_argument(
        "--scp",
        type=str,
        required=True,
        help="SCP文件路径，包含JSON文件列表",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（可选，默认生成.punctuated.json后缀文件）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖原文件",
    )
    parser.add_argument(
        "--api-family",
        type=str,
        default="gpt4o",
        help="API系列 (gpt4o, gpt5, gpt41等)",
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=5,
        help="API最大重试次数",
    )
    parser.add_argument(
        "--max-check-retries",
        type=int,
        default=3,
        help="验证最大重试次数",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=120.0,
        help="每个batch的最大时长（秒），默认120秒（2分钟）",
    )
    parser.add_argument(
        "--strict-check",
        action="store_true",
        default=True,
        help="严格检查模式（去标点后必须完全一致）",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="只处理前N个文件（0表示全部）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Ray并发worker数量（默认4）",
    )
    
    args = parser.parse_args()
    
    # 加载SCP文件
    json_paths = load_scp_file(args.scp)
    # json_paths = json_paths[4:8]
    print(f"Loaded {len(json_paths)} JSON files from {args.scp}")
    
    if args.count > 0:
        json_paths = json_paths[:args.count]
        print(f"Processing first {args.count} files")
    
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
    all_batch_tasks = []  # [(json_path, batch_id, batch_segments, original_data), ...]
    json_metadata = {}  # json_path -> {"data": original_data, "num_batches": N}
    skipped_count = 0  # 已处理跳过的文件数
    
    for json_path in tqdm(json_paths, desc="Loading JSON files"):
        json_path = Path(json_path)
        
        if not json_path.exists():
            print(f"Warning: File not found: {json_path}")
            continue
        
        # 确定输出路径
        if args.output_dir:
            output_path = Path(args.output_dir) / json_path.name
        elif args.overwrite:
            output_path = json_path
        else:
            output_path = json_path.with_suffix(".punctuated.json")
        
        # 检查是否已处理（支持resume）
        if output_path.exists() and not args.overwrite:
            skipped_count += 1
            continue
        
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            segments = data.get("segments", [])
            if not segments:
                continue
            
            # 分割为batches
            batches = split_segments_by_duration(segments, max_duration_seconds=600)
            
            json_metadata[str(json_path)] = {
                "data": data,
                "output_path": str(output_path),
                "num_batches": len(batches),
                "num_segments": len(segments),
            }
            
            # 添加所有batch任务
            for batch_id, batch_segments in enumerate(batches):
                all_batch_tasks.append({
                    "json_path": str(json_path),
                    "batch_id": batch_id,
                    "segments": batch_segments,
                })
                
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
    
    print(f"Total: {len(json_metadata)} JSON files to process, {len(all_batch_tasks)} batches")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} already processed files (resume mode)")
    
    if not all_batch_tasks:
        print("No tasks to process. All files already done!")
        return
    
    # ============ 第二步：以JSON文件为单位并发处理 ============
    print(f"\n[Step 2] Processing {len(json_metadata)} JSON files with Ray ({args.num_workers} workers)...")
    
    # 初始化处理器和API配置
    api_configs = get_api_configs(args.api_family)
    
    # 构建JSON文件任务列表（每个任务包含该文件的所有batches）
    json_tasks = []
    for json_path, metadata in json_metadata.items():
        # 获取该JSON文件的所有batch
        file_batches = [t for t in all_batch_tasks if t["json_path"] == json_path]
        file_batches.sort(key=lambda x: x["batch_id"])
        
        json_tasks.append({
            "json_path": json_path,
            "output_path": metadata["output_path"],
            "data": metadata["data"],
            "batches": file_batches,
            "num_segments": metadata["num_segments"],
        })
    
    if use_ray:
        # 使用Ray并发处理（以JSON文件为单位）
        @ray.remote
        class JsonFileWorker:
            def __init__(self, worker_id: int, all_api_config_dicts: List[Dict]):
                """
                每个worker持有所有client，遇到429时快速切换
                
                Args:
                    worker_id: worker编号
                    all_api_config_dicts: 所有API配置（字典形式列表）
                """
                from punctuation_processor import PunctuationProcessor
                from azure_api import build_openai_client
                from API_config import ApiConfig
                
                self.worker_id = worker_id
                self.processor = PunctuationProcessor(language="zh", strict_check=True)
                
                # 初始化所有clients
                self.clients = []  # [(ApiConfig, client), ...]
                print(f"Worker {worker_id}: Initializing {len(all_api_config_dicts)} clients...")
                
                for i, cfg_dict in enumerate(all_api_config_dicts):
                    try:
                        api_config = ApiConfig(**cfg_dict)
                        client = build_openai_client(api_config)
                        self.clients.append((api_config, client))
                    except Exception as e:
                        print(f"Worker {worker_id}: Failed to init client {i}: {e}")
                
                # 当前使用的client索引（每个worker从不同位置开始，分散负载）
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
                
                json_path = task["json_path"]
                output_path = task["output_path"]
                data = task["data"]
                batches = task["batches"]
                json_name = Path(json_path).name
                
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
                if success_batches == 0:
                    print(f"❌ Worker {self.worker_id}: All {len(batches)} batches failed for {json_name}, NOT saving (will retry on resume)")
                    return {
                        "json_path": json_path,
                        "output_path": str(output_path),
                        "success": False,
                        "saved": False,
                        "total_batches": len(batches),
                        "success_batches": 0,
                        "failed_batches": failed_batches,
                    }
                
                # 更新数据
                data["segments"] = all_segments
                data["punctuation_added"] = True
                if failed_batches:
                    data["failed_batches"] = failed_batches
                
                # 立即保存
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                status = "✅" if not failed_batches else "⚠️"
                print(f"{status} Worker {self.worker_id}: Saved {json_name} ({success_batches}/{len(batches)} batches success)")
                
                return {
                    "json_path": json_path,
                    "output_path": str(output_path),
                    "success": len(failed_batches) == 0,
                    "saved": True,
                    "total_batches": len(batches),
                    "success_batches": success_batches,
                    "failed_batches": failed_batches,
                }
        
        # 创建workers
        num_workers = min(args.num_workers, len(json_tasks))
        print(f"Creating {num_workers} workers (api_configs: {len(api_configs)}, json_files: {len(json_tasks)})...")
        
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
        for i in range(num_workers):
            workers.append(JsonFileWorker.remote(i, all_cfg_dicts))
        
        print(f"All {num_workers} workers initialized.")
        
        # 分发任务 - 以JSON文件为单位
        pending_futures = []
        task_map = {}  # future -> task index
        
        for i, task in enumerate(json_tasks):
            worker = workers[i % num_workers]
            future = worker.process_json_file.remote(task, args.max_api_retries, args.max_check_retries)
            pending_futures.append(future)
            task_map[future] = i
        
        print(f"Submitted {len(pending_futures)} JSON files to {num_workers} workers")
        
        # 收集结果（使用tqdm显示真实进度）
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
        processor = PunctuationProcessor(language="zh", strict_check=args.strict_check)
        client_manager = ClientManager(api_configs)
        
        json_results = []
        for task in tqdm(json_tasks, desc="Processing JSON files"):
            json_path = task["json_path"]
            output_path = task["output_path"]
            data = task["data"]
            batches = task["batches"]
            json_name = Path(json_path).name
            
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
                    max_api_retries=args.max_api_retries,
                    max_check_retries=args.max_check_retries,
                    task_id=f"{Path(json_path).stem}_batch_{batch_info['batch_id']}",
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
            
            # 如果所有batch都失败，不保存文件（方便resume重试）
            if success_batches == 0:
                print(f"❌ All {len(batches)} batches failed for {json_name}, NOT saving (will retry on resume)")
                json_results.append({
                    "json_path": json_path,
                    "output_path": str(output_path),
                    "success": False,
                    "saved": False,
                    "total_batches": len(batches),
                    "success_batches": 0,
                    "failed_batches": failed_batches,
                })
                continue
            
            # 更新数据并立即保存
            data["segments"] = all_segments
            data["punctuation_added"] = True
            if failed_batches:
                data["failed_batches"] = failed_batches
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            status = "✅" if not failed_batches else "⚠️"
            print(f"{status} Saved {json_name} ({success_batches}/{len(batches)} batches success)")
            
            json_results.append({
                "json_path": json_path,
                "output_path": str(output_path),
                "success": len(failed_batches) == 0,
                "saved": True,
                "total_batches": len(batches),
                "success_batches": success_batches,
                "failed_batches": failed_batches,
            })
    
    # ============ 第三步：打印统计（文件已在处理时保存） ============
    print(f"\n[Step 3] Processing complete, files already saved.")
    
    # 统计
    success_count = sum(1 for r in json_results if r["success"])
    fail_count = len(json_results) - success_count
    total_batches = sum(r["total_batches"] for r in json_results)
    success_batches = sum(r["success_batches"] for r in json_results)
    
    print("\n" + "=" * 50)
    print(f"Processing complete:")
    print(f"  Total JSON files in scp: {len(json_paths)}")
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
