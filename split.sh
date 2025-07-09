#!/bin/bash

# Parse command-line arguments using long options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --is_cot) IS_COT="$2"; shift 2 ;;
        --cot_answer_extract) COT_ANSWER_EXTRACT="$2"; shift 2 ;;
        --log_file) LOG_FILE="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --cot_prompt_type) COT_PROMPT_TYPE="$2"; shift 2 ;;
        --num_sequences) NUM_SEQUENCES="$2"; shift 2 ;;
        --top_p) TOP_P="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --instance) INSTANCE="$2"; shift 2 ;;
        --api_version) API_VERSION="$2"; shift 2 ;;
        --n_proc) N_PROC="$2"; shift 2 ;;
        --splits_count) SPLITS_COUNT="$2"; shift 2 ;;
        --splits_index) SPLITS_INDEX="$2"; shift 2 ;;
        *) echo "Unknown option: $1" && exit 1 ;;
    esac
done

# Set default values if arguments are not provided
MODEL_PATH=${MODEL_PATH:-"/mnt/longcontext/models/siyuan/llama3/llama-3.1-8B-instruct"}
IS_COT=${IS_COT:-false}
COT_ANSWER_EXTRACT=${COT_ANSWER_EXTRACT:-true}
LOG_FILE=${LOG_FILE:-"vllm_serve_output.log"}
TEMPERATURE=${TEMPERATURE:-"0.1"}
SAVE_DIR=${SAVE_DIR:-"/mnt/longcontext/models/siyuan/test_code/LongBench-v2/results"}
NUM_GPUS=${NUM_GPUS:-4}
COT_PROMPT_TYPE=${COT_PROMPT_TYPE:-"default"}
NUM_SEQUENCES=${NUM_SEQUENCES:-1}
TOP_P=${TOP_P:-1.0}
MODEL_TYPE=${MODEL_TYPE:-"vllm"}
INSTANCE=${INSTANCE:-"gcr/shared"}
API_VERSION=${API_VERSION:-"2024-10-21"}
N_PROC=${N_PROC:-1}
SPLITS_COUNT=${SPLITS_COUNT:-50}
SPLITS_INDEX=${SPLITS_INDEX:? "splits_index is required"}

# Kill all other vllm processes before starting
pids=$(ps auxww | grep vllm | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "No vllm processes found to kill."
else
    echo "Killing the following vllm processes: $pids"
    echo "$pids" | xargs kill
    if [ $? -eq 0 ]; then
        echo "Successfully killed vllm processes."
    else
        echo "Failed to kill some vllm processes. Please check permissions or process status."
    fi
fi

# Start the backend server in the background and redirect output to the log file
mkdir -p "$(dirname "$LOG_FILE")"
vllm serve $MODEL_PATH --api-key token-abc123 --tensor-parallel-size ${NUM_GPUS} --gpu-memory-utilization 0.95 --max_model_len 131072 --trust-remote-code --port 8000 --max_num_seqs 1 --seed 42 | tee "$LOG_FILE" > /dev/null 2>&1 &

# Wait for the server to fully start
sleep 400

# Prepare CoT arguments
COT_ARG=""
if [ "$IS_COT" == "true" ]; then
    COT_ARG="--cot"
fi

if [ "$COT_ANSWER_EXTRACT" == "true" ]; then
    COT_ARG="$COT_ARG --cot_answer_extract"
fi

# Echo parameters for logging
echo "========================="
echo "Model Path: $MODEL_PATH"
echo "Is CoT: $IS_COT"
echo "CoT Answer Extract: $COT_ANSWER_EXTRACT"
echo "Log File: $LOG_FILE"
echo "Temperature: $TEMPERATURE"
echo "Save Directory: $SAVE_DIR"
echo "COT Prompt Type: $COT_PROMPT_TYPE"
echo "Splits Count: $SPLITS_COUNT"
echo "Splits Index: $SPLITS_INDEX"
echo "========================="

# Run prediction script
python split.py --model_path $MODEL_PATH $COT_ARG --n_proc $N_PROC \
    --save_dir $SAVE_DIR \
    --temperature $TEMPERATURE \
    --cot_prompt_type $COT_PROMPT_TYPE \
    --num_sequences $NUM_SEQUENCES \
    --top_p $TOP_P \
    --model_type $MODEL_TYPE \
    --instance $INSTANCE \
    --api_version $API_VERSION \
    --splits_count $SPLITS_COUNT \
    --splits_index $SPLITS_INDEX

echo "Prediction script done..."
