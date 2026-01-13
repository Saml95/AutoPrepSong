#!/usr/bin/env python
"""
Simple Dataset Visualization Tool with Gradio
Visualize audio segments from JSON dataset with playback capabilities
"""

import os
import sys
import json
import random
import base64
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

import gradio as gr
import soundfile as sf
import numpy as np

# Import audio utilities
from audio_utils import load_audio_use_ffmpeg

# Try to import pydub for MP3 conversion
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("⚠️ Warning: pydub not available, falling back to WAV format")

import datetime    
def get_beijing_time():
    bj_time = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S")
    return bj_time

# Replace global print function
import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    original_print(get_beijing_time(), *args, **kwargs)
builtins.print = custom_print


# ============================================================================
# Audio Processing Functions
# ============================================================================

def clip_and_encode_audio(
    audio_data: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    segment_idx: int,
    use_mp3: bool = True
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Clip audio segment and encode to base64.
    
    Args:
        audio_data: Full audio array
        sr: Sample rate
        start_time: Start time in seconds
        end_time: End time in seconds
        segment_idx: Segment index for identification
        use_mp3: Whether to use MP3 format (smaller size)
        
    Returns:
        Tuple of (segment_idx, base64_string, error_message)
    """
    try:
        # Convert time to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return segment_idx, None, f"Invalid time range: [{start_time:.2f}s - {end_time:.2f}s]"
        
        # Extract segment
        segment_data = audio_data[start_sample:end_sample]
        
        # Convert to MP3 if pydub is available and use_mp3 is True
        if use_mp3 and HAS_PYDUB:
            try:
                # Write to WAV in memory
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, segment_data, sr, format='WAV')
                wav_buffer.seek(0)
                
                # Convert to MP3
                audio_segment = AudioSegment.from_wav(wav_buffer)
                mp3_buffer = io.BytesIO()
                audio_segment.export(mp3_buffer, format='mp3', bitrate='64k')
                mp3_buffer.seek(0)
                
                # Encode to base64
                audio_bytes = mp3_buffer.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_src = f"data:audio/mp3;base64,{audio_base64}"
                
                return segment_idx, audio_src, None
            except Exception as e:
                # Fall back to WAV on error
                print(f"MP3 conversion failed for segment {segment_idx}, using WAV: {e}")
        
        # Fall back to WAV format (no temp file, use in-memory buffer)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, segment_data, sr, format='WAV')
        wav_buffer.seek(0)
        
        audio_bytes = wav_buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_src = f"data:audio/wav;base64,{audio_base64}"
        
        return segment_idx, audio_src, None
        
    except Exception as e:
        error_msg = f"Error clipping segment {segment_idx}: {str(e)}"
        print(error_msg)
        return segment_idx, None, error_msg


def process_audio_segments_parallel(
    audio_path: str,
    segments: List[Dict],
    max_workers: int = 4,
    use_mp3: bool = True
) -> List[Tuple[int, Optional[str], Optional[str]]]:
    """
    Process multiple audio segments in parallel using threads.
    
    Args:
        audio_path: Path to audio file
        segments: List of segment dictionaries
        max_workers: Number of parallel workers
        use_mp3: Whether to use MP3 format (smaller size)
        
    Returns:
        List of tuples (segment_idx, audio_base64, error_msg)
    """
    try:
        # Load audio using ffmpeg for better format support
        print(f"📂 Loading audio file: {audio_path}")
        audio_data, sr = load_audio_use_ffmpeg(audio_path, resample=False)
        print(f"✅ Audio loaded: {len(audio_data)} samples, {sr} Hz")
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return [(i, None, f"Failed to load audio: {str(e)}") for i in range(len(segments))]
    
    # Prepare tasks
    tasks = []
    for i, seg in enumerate(segments):
        start_time = seg.get('start_time', seg.get('start', 0))
        end_time = seg.get('end_time', seg.get('end', 0))
        tasks.append((audio_data, sr, start_time, end_time, i, use_mp3))
    
    # Process in parallel using ThreadPoolExecutor
    results = []
    total_segments = len(tasks)
    completed_count = 0
    
    print(f"🚀 Starting parallel processing with {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(clip_and_encode_audio, *task): task[4] 
            for task in tasks
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                # Log progress every 50 segments or at completion
                if completed_count % 50 == 0 or completed_count == total_segments:
                    print(f"Progress: {completed_count}/{total_segments} segments processed ({completed_count*100//total_segments}%)")
            except Exception as e:
                idx = futures[future]
                results.append((idx, None, f"Processing error: {str(e)}"))
                completed_count += 1
                print(f"Progress: {completed_count}/{total_segments} segments processed ({completed_count*100//total_segments}%) - Error on segment {idx}")
    
    print(f"✅ Completed processing all {total_segments} segments")
    
    # Sort by segment index to maintain order
    results.sort(key=lambda x: x[0])
    return results


# ============================================================================
# JSON Loading Functions
# ============================================================================

def load_json_file(json_path: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """
    Load a JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Tuple of (json_data, json_filename, error_message)
    """
    try:
        if not os.path.isfile(json_path):
            return None, None, f"❌ JSON file does not exist: {json_path}"
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        json_filename = os.path.basename(json_path)
        print(f"📄 Loaded JSON file: {json_path}")
        return data, json_path, None
        
    except Exception as e:
        error_msg = f"❌ Error loading JSON: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, error_msg


def load_random_json(path: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """
    Load a JSON file from a path (can be a file, directory, or list file).
    
    Args:
        path: Can be:
            - A direct JSON file path
            - A text file with list of JSON paths (one per line)
            - A directory (will scan for JSON files)
    
    Returns:
        Tuple of (json_data, json_filename, error_message)
    """
    try:
        # Check if path is a file
        if os.path.isfile(path):
            # Check if it's a list file (text file with JSON paths)
            if path.endswith(('.txt', '.scp', '.cfg', '.list')) or (not path.endswith('.json')):
                # Try to read as list file
                with open(path, 'r', encoding='utf-8') as f:
                    json_paths = [line.strip() for line in f if line.strip()]
                
                if not json_paths:
                    return None, None, f"❌ List file is empty: {path}"
                
                # Randomly select one JSON path from list
                selected_json_path = random.choice(json_paths)
                print(f"📝 Randomly selected from list: {selected_json_path}")
                
                return load_json_file(selected_json_path)
            
            # Otherwise, treat as direct JSON file
            elif path.endswith('.json'):
                return load_json_file(path)
            else:
                return None, None, f"❌ File is not a JSON or list file: {path}"
        
        # Check if path is a directory
        elif os.path.isdir(path):
            # Scan for JSON files
            json_files = []
            for root, _, files in os.walk(path):
                for filename in files:
                    if filename.endswith('.json'):
                        json_files.append(os.path.join(root, filename))
            
            print(f"📂 Found {len(json_files)} JSON files in directory: {path}")
            
            if not json_files:
                return None, None, f"❌ No JSON files found in directory: {path}"
            
            # Randomly select one
            selected_json_path = random.choice(json_files)
            print(f"🎲 Randomly selected: {selected_json_path}")
            
            return load_json_file(selected_json_path)
        
        else:
            return None, None, f"❌ Path does not exist: {path}"
        
    except Exception as e:
        error_msg = f"❌ Error loading JSON: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, error_msg


# ============================================================================
# Data Processing Functions
# ============================================================================

def process_json_data(json_data: Dict) -> Tuple[str, List[Dict], List[str], Dict]:
    """
    Process JSON data and extract segments.
    
    Args:
        json_data: Loaded JSON data
        
    Returns:
        Tuple of (audio_path, segments, warnings, metadata)
    """
    warnings = []
    metadata = {}
    
    # Get audio path
    audio_path = json_data.get('audio_path', '')
    if not audio_path:
        warnings.append("⚠️ No audio_path found in JSON")
        return '', [], warnings, metadata
    
    if not os.path.exists(audio_path):
        warnings.append(f"⚠️ Audio file does not exist: {audio_path}")
        return audio_path, [], warnings, metadata
    
    # Get segments
    segments = json_data.get('segments', [])
    if not segments:
        warnings.append("⚠️ No segments found in JSON")
        return audio_path, [], warnings, metadata
    
    # Extract metadata
    metadata['audio_length'] = float(json_data.get('audio_length'))
    metadata['num_speakers'] = json_data.get('num_effetive_speakers')
    metadata['speaker_counter'] = json_data.get('speaker_counter_str')
    
    # Process segments
    processed_segments = []
    for i, seg in enumerate(segments):
        # Get timing info
        start_time = seg.get('start_time', seg.get('start', 0))
        end_time = seg.get('end_time', seg.get('end', 0))
        
        # Validate timing
        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            warnings.append(f"⚠️ Segment {i}: Invalid time values")
            continue
        
        if start_time >= end_time:
            warnings.append(f"⚠️ Segment {i}: Invalid time range [{start_time:.2f}s - {end_time:.2f}s]")
            continue
        
        # Extract segment info
        segment_info = {
            'start_time': start_time,
            'end_time': end_time,
            'text': seg.get('text', ''),
            'speaker': seg.get("speaker", seg.get('speaker_id', seg.get('umap_segment_labels', 'Unknown'))),
            'vad_anomaly':seg.get('vad_anomaly', False),
        }
        
        processed_segments.append(segment_info)
    
    return audio_path, processed_segments, warnings, metadata


# ============================================================================
# HTML Generation
# ============================================================================

def generate_html_output(
    json_filename: str,
    audio_path: str,
    segments: List[Dict],
    audio_results: List[Tuple[int, Optional[str], Optional[str]]],
    warnings: List[str],
    metadata: Dict
) -> str:
    """
    Generate HTML output with audio segments.
    
    Args:
        json_filename: Name of the JSON file
        audio_path: Path to audio file
        segments: List of processed segment dictionaries
        audio_results: List of (segment_idx, audio_base64, error_msg) tuples
        warnings: List of warning messages
        metadata: Dictionary with dataset metadata
        
    Returns:
        HTML string
    """
    # Theme-aware CSS
    css = """
    <style>
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --border-color: #e1e5e9;
        --border-accent: #007bff;
        --text-primary: #212529;
        --text-secondary: #6c757d;
        --text-warning: #856404;
        --bg-warning: #fff3cd;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1a202c;
            --bg-secondary: #2d3748;
            --border-color: #4a5568;
            --border-accent: #4299e1;
            --text-primary: #e2e8f0;
            --text-secondary: #a0aec0;
            --text-warning: #faf089;
            --bg-warning: #744210;
            --shadow: rgba(0, 0, 0, 0.3);
        }
    }
    
    .vis-container {
        max-height: 800px;
        overflow-y: auto;
        padding: 15px;
        background-color: var(--bg-primary);
    }
    
    .vis-header {
        margin-bottom: 20px;
        padding: 15px;
        background-color: var(--bg-secondary);
        border-radius: 8px;
        border: 2px solid var(--border-color);
    }
    
    .vis-header h3 {
        margin: 0 0 10px 0;
        color: var(--text-primary);
        font-size: 18px;
    }
    
    .vis-header p {
        margin: 5px 0;
        color: var(--text-secondary);
        font-size: 14px;
    }
    
    .vis-warnings {
        margin-bottom: 15px;
        padding: 12px;
        background-color: var(--bg-warning);
        border-radius: 6px;
        border-left: 4px solid #ffc107;
    }
    
    .vis-warnings p {
        margin: 5px 0;
        color: var(--text-warning);
        font-size: 13px;
    }
    
    .segment-item {
        margin-bottom: 20px;
        padding: 15px;
        border: 2px solid var(--border-color);
        border-radius: 8px;
        background-color: var(--bg-secondary);
        transition: all 0.3s ease;
    }
    
    .segment-item:hover {
        box-shadow: 0 4px 12px var(--shadow);
        border-color: var(--border-accent);
    }

    .segment-item-warning {
        margin-bottom: 20px;
        padding: 15px;
        border: 2px solid var(--border-color);
        border-radius: 8px;
        background-color: var(--bg-secondary);
        transition: all 0.3s ease;
    }
    .segment-item-warning:hover {
        box-shadow: 0 4px 12px var(--shadow);
        border-color: #fbc02d;
    }

    .segment-header {
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .segment-title {
        margin: 0;
        color: var(--text-primary);
        font-size: 16px;
        font-weight: 600;
    }
    
    .segment-info {
        margin-top: 10px;
    }
    
    .segment-info-row {
        margin: 8px 0;
        color: var(--text-primary);
        font-size: 14px;
        line-height: 1.6;
    }
    
    .segment-info-label {
        font-weight: 600;
        color: var(--text-secondary);
        margin-right: 8px;
        display: inline-block;
        min-width: 100px;
    }
    
    .segment-text {
        margin: 10px 0;
        padding: 10px;
        background-color: var(--bg-primary);
        border-left: 3px solid var(--border-accent);
        border-radius: 4px;
        color: var(--text-primary);
        line-height: 1.6;
    }

    .segment-text-warning {
        margin: 10px 0;
        padding: 10px;
        background-color: var(--bg-primary);
        border-left: 3px solid #fbc02d;
        border-radius: 4px;
        color: var(--text-primary);
        line-height: 1.6;
    }

    .segment-audio {
        width: 100%;
        margin-top: 12px;
        border-radius: 4px;
    }
    
    .segment-error {
        margin-top: 10px;
        padding: 10px;
        background-color: var(--bg-warning);
        border-radius: 4px;
        border-left: 4px solid #ffc107;
        color: var(--text-warning);
        font-size: 13px;
    }
    
    .speaker-a { border-left-color: #4299e1; }
    .speaker-b { border-left-color: #48bb78; }
    .speaker-c { border-left-color: #ed8936; }
    .speaker-d { border-left-color: #9f7aea; }
    </style>
    """
    
    # Start HTML
    html = css
    html += "<div class='vis-container'>"
    
    # Header
    html += "<div class='vis-header'>"
    html += f"<h3>📁 JSON File: {json_filename}</h3>"
    html += f"<p>🎵 Audio: {audio_path}</p>"
    html += f"<p>📊 Total Segments: {len(segments)}</p>"
    
    if metadata.get('audio_length'):
        html += f"<p>⏱️ Audio Length: {metadata['audio_length']:.2f}s</p>"
    if metadata.get('num_speakers'):
        html += f"<p>👥 Number of Speakers: {metadata['num_speakers']}</p>"
    if metadata.get('speaker_counter'):
        html += f"<p>🗣️ Speaker Distribution: {metadata['speaker_counter']}</p>"
    
    html += "</div>"
    
    # Warnings
    if warnings:
        html += "<div class='vis-warnings'>"
        for warning in warnings:
            html += f"<p>{warning}</p>"
        html += "</div>"
    
    # Segments
    for i, (segment, audio_result) in enumerate(zip(segments, audio_results)):
        idx, audio_src, error_msg = audio_result
        
        # Get speaker without mapping
        speaker = segment['speaker']
        if segment['vad_anomaly']:
            html += f"<div class='segment-item-warning'>"
        else:
            html += f"<div class='segment-item'>"
        
        # Header
        html += "<div class='segment-header'>"
        html += f"<h4 class='segment-title'>🔊 Segment {i}</h4>"
        html += "</div>"
        
        # Info
        html += "<div class='segment-info'>"
        
        # Time and Speaker on one line
        start = segment['start_time']
        end = segment['end_time']
        duration = end - start
        html += f"<div class='segment-info-row'>"
        html += f"<span class='segment-info-label'>⏱️ Time:</span>[{start:.2f}s - {end:.2f}s] ({duration:.2f}s)"
        html += f"<span style='margin-left: 20px;'><span class='segment-info-label'>👤 Speaker:</span>{speaker}</span>"
        html += "</div>"
        
        # Text content
        text = segment['text'].strip()
        if segment['vad_anomaly']:
            html += f"<div class='segment-text-warning'>{text if text else '<i>No text</i>'}</div>"
        else:
            html += f"<div class='segment-text'>{text if text else '<i>No text</i>'}</div>"
        
        html += "</div>"  # End segment-info
        
        # Audio player or error
        if audio_src:
            html += f"""
            <audio controls class='segment-audio' preload='metadata'>
                <source src='{audio_src}' type='audio/{('mp3' if 'audio/mp3' in audio_src else 'wav')}'>
                Your browser does not support the audio element.
            </audio>
            """
        elif error_msg:
            html += f"<div class='segment-error'>❌ {error_msg}</div>"
        else:
            html += "<div class='segment-error'>❌ Audio unavailable</div>"
        
        html += "</div>"  # End segment-item
    
    html += "</div>"  # End vis-container
    
    return html


# ============================================================================
# Main Processing Function
# ============================================================================

def process_sample(
    path: str,
    max_workers: int = 4,
    use_mp3: bool = True
) -> str:
    """
    Main function to process a sample and generate visualization.
    
    Args:
        path: JSON file path, list file path, or directory path
        max_workers: Number of parallel workers for audio processing
        use_mp3: Whether to use MP3 format (smaller size)
        
    Returns:
        HTML string with visualization
    """
    try:
        print(f"Processing path: {path}")
        
        # Load JSON (either random from list/directory or specific file)
        json_data, json_filename, error = load_random_json(path)
        if error:
            return f"<div style='padding: 20px; color: red;'>{error}</div>"
        
        # Process JSON data
        audio_path, segments, warnings, metadata = process_json_data(json_data)
        
        if not segments:
            warning_html = "<br>".join(warnings) if warnings else "No segments to display"
            return f"<div style='padding: 20px; color: orange;'>⚠️ {warning_html}</div>"
        
        # Process audio segments in parallel
        format_info = "MP3 (64kbps)" if use_mp3 and HAS_PYDUB else "WAV"
        print(f"Processing {len(segments)} segments with {max_workers} workers (format: {format_info})...")
        audio_results = process_audio_segments_parallel(
            audio_path, segments, max_workers=max_workers, use_mp3=use_mp3
        )
        
        # Generate HTML
        html_output = generate_html_output(
            json_filename,
            audio_path,
            segments,
            audio_results,
            warnings,
            metadata
        )
        
        return html_output
        
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"<div style='padding: 20px; color: red;'>{error_msg}</div>"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_gradio_interface(default_path: str = None):
    """Create and return Gradio interface.
    
    Args:
        default_path: Default path (JSON file, list file, or directory)
    """
    
    if default_path is None:
        default_path = "/data/jianwei/VibeASR/exp/MLC/240s/4o_trans_dia_v1.1/EnglishSPLITAmericanSPLIT0517_007_zero_nonspeech_mp3_chunk-0_max_240.json"
    
    with gr.Blocks(title="Simple Audio Dataset Visualization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎵 Simple Audio Dataset Visualization Tool")
        gr.Markdown("Visualize audio segments with text and speaker information")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Primary action at the top
                load_button = gr.Button("🎲 Load Sample", variant="primary", size="lg")
                gr.Markdown("## ⚙️ Configuration")
                
                # Path input
                path_input = gr.Textbox(
                    label="📁 JSON Path or List File",
                    value=default_path,
                    placeholder="Enter path to: JSON file, list file (.txt/.scp/.list), or directory",
                    lines=2,
                    info="Supports: 1) Direct JSON file, 2) List file with JSON paths, 3) Directory with JSON files"
                )
                
                # Max workers
                max_workers_slider = gr.Slider(
                    minimum=1,
                    maximum=max(os.cpu_count() or 4, 64),
                    value=max(os.cpu_count() or 4, 64),
                    step=1,
                    label="🔧 Parallel Workers",
                    info=f"Number of parallel workers for audio processing (CPU cores: {os.cpu_count() or 'Unknown'})"
                )
                
                # Audio format selection
                use_mp3_checkbox = gr.Checkbox(
                    label="🎵 Use MP3 format (smaller size, faster)",
                    value=HAS_PYDUB,
                    interactive=HAS_PYDUB,
                    info="MP3 reduces HTML size significantly" if HAS_PYDUB else "Install pydub to enable MP3"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Visualization")
                
                # Output
                output_html = gr.HTML(
                    label="Segments Visualization",
                    value="<div style='padding: 20px; text-align: center; color: gray;'>Click 'Load Sample' to start</div>"
                )
        
        # Event handler
        def on_load_click(path, max_workers, use_mp3):
            return process_sample(
                path=path,
                max_workers=int(max_workers),
                use_mp3=use_mp3
            )
        
        load_button.click(
            fn=on_load_click,
            inputs=[path_input, max_workers_slider, use_mp3_checkbox],
            outputs=output_html
        )
        
        # Instructions
        gr.Markdown("## 📋 Instructions")
        gr.Markdown("""
        1. **Configure Settings**: 
           - Enter one of the following:
             * **Direct JSON file path** - will load that specific file
             * **List file path** (.txt/.scp/.list) - will randomly select from JSON paths listed (one per line)
             * **Directory path** - will randomly select a JSON file from the directory
           - Adjust parallel workers for audio processing (default: CPU cores)
           - Choose MP3 or WAV format for audio segments
        
        2. **Load Sample**: Click "Load Sample" to load and visualize the JSON file
        
        3. **Review Results**:
           - Each segment shows timestamp, speaker, and text
           - Audio player allows direct playback
           - Speaker segments are color-coded for easy identification
        
        **Expected JSON Structure**:
        ```json
        {
          "audio_path": "/path/to/audio.mp3",
          "audio_length": 240.0,
          "num_effetive_speakers": 2,
          "speaker_counter_str": "Speaker A: 20, Speaker B: 30",
          "segments": [
            {
              "text": "Hello world",
              "start": 0.0,
              "end": 1.5,
              "speaker_id": "A",
              "start_time": 0.0,
              "end_time": 1.5
            }
          ]
        }
        ```
        """)
    
    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simple Dataset Visualization Tool")
    parser.add_argument(
        "--path",
        type=str,
        default="local/luoxue_test_final.scp",
        help="Default path (JSON file, list file, or directory)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7862,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    demo = create_gradio_interface(default_path=args.path)
    
    print(f"🚀 Starting Simple Dataset Visualization Tool...")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    
    launch_kwargs = {
        "share": args.share,
        "show_error": False,
    }
    if args.share:
        launch_kwargs["server_name"] = "0.0.0.0"
    else:
        launch_kwargs["server_name"] = args.host
        launch_kwargs["server_port"] = args.port
    
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()