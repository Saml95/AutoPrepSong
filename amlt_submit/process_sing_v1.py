#!/usr/bin/env python3
import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from audioautoprep.scripts.AutoPrep.autoprep_betav1 import AutoPrepPipeline
from audioautoprep.models.utils import read_audio, load_audio_use_ffmpeg
import soundfile as sf
import librosa
from tqdm import tqdm

def read_audio_list(file_path: str, start_idx: int, num_files: int) -> List[str]:
    """
    Read a subset of audio file paths from the index file.
    
    Args:
        file_path: Path to the index file
        start_idx: Starting line/index (0-based)
        num_files: Number of files to read
        
    Returns:
        List of audio file paths
    """
    audio_paths = []
    with open(file_path, 'r') as f:
        # Skip to the starting position
        for _ in range(start_idx):
            next(f, None)
        
        # Read the required number of files
        for _ in range(num_files):
            line = next(f, None)
            if line is None:
                break
            audio_paths.append(line.strip())
    
    return audio_paths

def load_audio(audio_path: str) -> Tuple[Any, int]:
    """Load audio file and return the audio data and sample rate"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {str(e)}")
        return None, None

def get_full_audio_path(base_dir: str, relative_path: str) -> str:
    """Convert relative audio path to full path"""
    return os.path.join(base_dir, relative_path)

def get_output_json_path(output_dir: str, audio_path: str) -> str:
    """Generate output JSON path based on audio path, preserving directory structure"""
    # Get relative path parts
    # print(f"check output dir {output_dir}")
    if audio_path.startswith("/mnt/conversationhubhot/jianweiyu/datasets/sdrv1/podcast/"):
        audio_path = audio_path.replace("/mnt/conversationhubhot/jianweiyu/datasets/sdrv1/podcast/", "")

    if audio_path.startswith("/mnt/conversationhubhot/yaoyaochang/speech/data/gemini/"):
        audio_path = audio_path.replace("/mnt/conversationhubhot/yaoyaochang/speech/data/gemini/", "")

    rel_path = Path(audio_path)
    # Replace extension with .json
    json_filename = f"{rel_path.stem}.json"
    
    # Preserve the directory structure by keeping the parent directory
    
    parent_dir = rel_path.parent
    
    # Create output path that maintains the original structure
    return os.path.join(output_dir, parent_dir, json_filename)

def is_processed(output_path: str) -> bool:
    """Check if a file has already been processed"""
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0

def preprocess_thread(audio_paths: List[str], base_dir: str, output_dir: str, 
                     preprocess_queue: queue.Queue, stop_event: threading.Event):
    """Thread for loading audio files and checking if they need processing"""
    total_files = len(audio_paths)
    processed = 0
    skipped = 0
    errors = 0
    
    # Index for tracking current position in audio_paths
    i = 0
    
    # Process files until all are done or stop is requested
    while i < len(audio_paths) and not stop_event.is_set():
        rel_path = audio_paths[i]
        full_path = get_full_audio_path(base_dir, rel_path)
        output_path = get_output_json_path(output_dir, rel_path)
        # import pdb; pdb.set_trace()
        print(f"Processing {full_path} to {output_path}")
        
        # Skip if already processed
        if is_processed(output_path):
            skipped += 1
            if (i+1) % 100 == 0:
                print(f"Skipped {skipped}/{i+1} files (already processed)")
            i += 1  # Move to next file
            continue
        
        # Check if queue has room before loading audio
        if preprocess_queue.qsize() >= preprocess_queue.maxsize * 0.9:  # 90% full
            # Queue is near capacity, wait before loading more audio
            time.sleep(1.0)
            continue
            
        # Load audio
        try:
            # audio_data = read_audio(full_path, backend='librosa')
            audio_data, sr = load_audio_use_ffmpeg(full_path, resample=True, target_sr=16000)
            assert sr == 16000, f"读取后的音频采样率不等于16000 {full_path}"
            # audio_data = read_audio(full_path, backend='librosa')
            if audio_data is None:
                errors += 1
                i += 1  # Move to next file
                continue
                
            # Try to add to queue
            try:
                preprocess_queue.put((audio_data, rel_path, full_path, output_path), timeout=1)
                processed += 1
                i += 1  # Move to next file
                
                if (i+1) % 100 == 0:
                    print(f"Prepared {processed}/{i+1} files for processing (skipped {skipped}, errors {errors})")
                    
            except queue.Full:
                # Queue is full - don't advance to next file, just wait and retry
                time.sleep(0.5)
                
        except Exception as e:
            errors += 1
            print(f"Error preprocessing {full_path}: {str(e)}")
            i += 1  # Move to next file
    
    print(f"Preprocess thread finished. Total: {total_files}, Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    
    # Signal that we're done adding items
    while not stop_event.is_set():
        try:
            preprocess_queue.put(None, timeout=1)  # Sentinel to signal end of data
            break
        except queue.Full:
            time.sleep(0.1)

def process_thread(preprocess_queue: queue.Queue, save_queue: queue.Queue, 
                 autoprep: AutoPrepPipeline, stop_event: threading.Event):
    """Thread for processing audio files"""
    processed = 0
    errors = 0
    
    while not stop_event.is_set():
        try:
            item = preprocess_queue.get(timeout=1)
            if item is None:  # End of data sentinel
                preprocess_queue.task_done()
                break
                
            audio_data, rel_path, full_path, output_path = item
            
            try:
                
                # 使用数据副本进行处理
                output, diarz_info, audio = autoprep.process_audio(
                    audio_data.copy(),  # 使用副本而不是原始数据
                    output_dir=None,
                    save_output=False
                )
                output["audio_path"] = full_path.replace("/mnt/conversationhubhot/jianweiyu/datasets/sdrv1/", "/mnt/sdrgprmblob01scus/")
                import pdb; pdb.set_trace()
                # Add to save queue
                while not stop_event.is_set():
                    try:
                        save_queue.put((json.loads(json.dumps(output)), output_path), timeout=1)
                        break
                    except queue.Full:
                        time.sleep(0.1)
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed} files")
                
            except Exception as e:
                errors += 1
                print(f"Error processing file {full_path}: {str(e)}")
                
                # Add error information to save queue
                error_info = {
                    "error": str(e),
                    "audio_path": full_path
                }
                
                save_queue.put((error_info, f"{output_path}.error.json"), timeout=1)
            
            finally:
                preprocess_queue.task_done()
                
        except queue.Empty:
            continue
            
    # Signal that we're done processing
    while not stop_event.is_set():
        try:
            save_queue.put(None, timeout=1)  # Sentinel to signal end of data
            break
        except queue.Full:
            time.sleep(0.1)
            
    print(f"Process thread finished. Processed: {processed}, Errors: {errors}")

def save_thread(save_queue: queue.Queue, stop_event: threading.Event):
    """Thread for saving processed results"""
    saved = 0
    errors = 0
    
    while not stop_event.is_set():
        try:
            item = save_queue.get(timeout=1)
            if item is None:  # End of data sentinel
                save_queue.task_done()
                break
                
            result, output_path = item
            
            try:
                # Ensure output directory exists (including parent directories)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save result to file
                with open(output_path, 'w') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                # audio_length = result["audio_length"]
                # # check
                # check = json.load(open(output_path, 'r'))
                # last_segment_end = check["segments_merged"][-1]["end"]
                # if last_segment_end > audio_length+1:
                #     print(f"Error saving file due to abnormal last segment end {output_path}: {last_segment_end} != {audio_length}")
                #     errors += 1
                #     # 如果检测到异常情况，创建一个空的错误标记文件
                #     with open(f"{output_path}.error", 'w') as error_file:
                #         pass  # 创建一个空文件
                
                saved += 1
                if saved % 50 == 0:
                    print(f"Saved {saved} files")
                    
            except Exception as e:
                errors += 1
                print(f"Error saving file {output_path}: {str(e)}")
                
            finally:
                save_queue.task_done()
                
        except queue.Empty:
            continue
            
    print(f"Save thread finished. Saved: {saved} files, Errors: {errors}")

def main():
    parser = argparse.ArgumentParser(description="Process audio files in batches")
    parser.add_argument("--index_file", type=str, 
                        default="/mnt/conversationhubhot/yaoyaochang/speech/data/podcast/all_audio_index.txt",
                        help="Path to the audio index file")
    parser.add_argument("--base_dir", type=str, 
                        default="/mnt/conversationhubhot/yaoyaochang/speech/data/podcast/audio_all",
                        help="Base directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output JSON files")
    parser.add_argument("--config_path", type=str, 
                        default="config/autoprep_beta_v1_abspath.yaml",
                        help="Path to autoprep config file")
    parser.add_argument("--start_idx", type=int, required=True,
                        help="Starting index (0-based line number in index file)")
    parser.add_argument("--num_files", type=int, required=True,
                        help="Number of files to process")
    parser.add_argument("--buffer_size", type=int, default=20,
                        help="Maximum number of files to buffer in memory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for processing (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting audio processing:")
    print(f"- Starting from index: {args.start_idx}")
    print(f"- Number of files: {args.num_files}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Buffer size: {args.buffer_size}")
    
    # Read audio paths
    print("Reading audio file list...")
    audio_paths = read_audio_list(args.index_file, args.start_idx, args.num_files)
    print(f"Found {len(audio_paths)} audio files to process")
    
    # Initialize model
    print(f"Initializing AutoPrepPipeline with device: {args.device}")
    autoprep = AutoPrepPipeline(
        config_path=args.config_path,
        device=args.device
    )
    
    # Setup queues and threads
    preprocess_queue = queue.Queue(maxsize=args.buffer_size)
    save_queue = queue.Queue(maxsize=args.buffer_size)
    stop_event = threading.Event()
    
    # Start threads
    preprocess_thread_obj = threading.Thread(
        target=preprocess_thread,
        args=(audio_paths, args.base_dir, args.output_dir, preprocess_queue, stop_event)
    )
    
    process_thread_obj = threading.Thread(
        target=process_thread,
        args=(preprocess_queue, save_queue, autoprep, stop_event)
    )
    
    save_thread_obj = threading.Thread(
        target=save_thread,
        args=(save_queue, stop_event)
    )
    
    # Start execution
    try:
        print("Starting processing pipeline...")
        start_time = time.time()
        
        preprocess_thread_obj.start()
        process_thread_obj.start()
        save_thread_obj.start()
        
        # Wait for preprocess thread to finish first
        preprocess_thread_obj.join()
        print("Preprocess thread completed")
        
        # Then wait for process thread
        process_thread_obj.join()
        print("Process thread completed")
        
        # Wait for save queue to be fully processed
        save_queue.join()
        print("All items in save queue processed")
        
        # Now we can safely terminate the save thread
        stop_event.set()
        save_thread_obj.join()
        print("Save thread completed")
        
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds ({len(audio_paths) / elapsed_time:.2f} files/sec)")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping threads")
        stop_event.set()
        
        # Wait for threads to terminate
        if preprocess_thread_obj.is_alive():
            preprocess_thread_obj.join()
        if process_thread_obj.is_alive():
            process_thread_obj.join()
        if save_thread_obj.is_alive():
            save_thread_obj.join()
        
        print("All threads stopped")
        
if __name__ == "__main__":
    main()