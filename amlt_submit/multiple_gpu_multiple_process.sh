#!/bin/bash

# Default values
START_IDX=0
TOTAL_FILES=1000
NUM_GPUS=4
PROCESSES_PER_GPU=2
BUFFER_SIZE=30
OUTPUT_DIR="./output"
# CONFIG_PATH="config/autoprep_beta_v1_abspath.yaml"
CONFIG_PATH="config/autoprep_beta_v1_abspath_sft.yaml" 
INDEX_FILE="/mnt/conversationhubhot/yaoyaochang/speech/data/podcast/all_audio_index.txt"
BASE_DIR="/mnt/conversationhubhot/yaoyaochang/speech/data/podcast/audio_all"
# Parse command line arguments
export PYTHONPATH=$PYTHONPATH:$PWD/..
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --start_idx)
        START_IDX="$2"
        shift 2
        ;;
        --total_files)
        TOTAL_FILES="$2"
        shift 2
        ;;
        --num_gpus)
        NUM_GPUS="$2"
        shift 2
        ;;
        --processes_per_gpu)
        PROCESSES_PER_GPU="$2"
        shift 2
        ;;
        --buffer_size)
        BUFFER_SIZE="$2"
        shift 2
        ;;
        --output_dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
        --config_path)
        CONFIG_PATH="$2"
        shift 2
        ;;
        --index_file)
        INDEX_FILE="$2"
        shift 2
        ;;
        --base_dir)
        BASE_DIR="$2"
        shift 2
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Calculate total number of processes
TOTAL_PROCESSES=$((NUM_GPUS * PROCESSES_PER_GPU))

# Calculate files per process (distribute evenly)
FILES_PER_PROCESS=$((TOTAL_FILES / TOTAL_PROCESSES))
REMAINDER=$((TOTAL_FILES % TOTAL_PROCESSES))

echo "Starting distributed processing with:"
echo "- Index file: $INDEX_FILE"
echo "- Starting index: $START_IDX"
echo "- Total files: $TOTAL_FILES"
echo "- GPUs: $NUM_GPUS"
echo "- Processes per GPU: $PROCESSES_PER_GPU"
echo "- Total processes: $TOTAL_PROCESSES"
echo "- Base files per process: $FILES_PER_PROCESS"
echo "- Remainder files: $REMAINDER"
echo "- Output directory: $OUTPUT_DIR"

# Create logs directory
# LOGS_DIR="${OUTPUT_DIR}/logs"
# mkdir -p "$LOGS_DIR"

# Start processes with appropriate GPU assignments and workload divisions
CURRENT_IDX=$START_IDX
PROCESS_COUNT=0

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((proc=0; proc<PROCESSES_PER_GPU; proc++)); do
        # Calculate files for this process (distribute remainder 1 by 1)
        THIS_PROCESS_FILES=$FILES_PER_PROCESS
        if [ $PROCESS_COUNT -lt $REMAINDER ]; then
            THIS_PROCESS_FILES=$((FILES_PER_PROCESS + 1))
        fi
        
        # Skip if no files to process
        if [ $THIS_PROCESS_FILES -le 0 ]; then
            continue
        fi
        
        # Start the process in background with CUDA_VISIBLE_DEVICES set to use specific GPU
        CUDA_VISIBLE_DEVICES=$gpu python process_sing_v1.py \
            --index_file "$INDEX_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --start_idx $CURRENT_IDX \
            --num_files $THIS_PROCESS_FILES \
            --buffer_size $BUFFER_SIZE \
            --config_path "$CONFIG_PATH" \
            --base_dir "$BASE_DIR" \
            --device "cuda"  &
        
        # Update index for next process
        CURRENT_IDX=$((CURRENT_IDX + THIS_PROCESS_FILES))
        PROCESS_COUNT=$((PROCESS_COUNT + 1))

        # Optional: sleep to avoid overwhelming the system
        sleep 10
    done
done

# echo "Started $PROCESS_COUNT processes. Logs available in $LOGS_DIR"
# echo "Use 'tail -f ${LOGS_DIR}/*.log' to monitor progress"

# Wait for all background processes to complete
echo "Waiting for all processes to complete..."
wait
echo "All processes completed."
