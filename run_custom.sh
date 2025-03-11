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
vllm serve $MODEL_PATH --api-key token-abc123 --tensor-parallel-size ${NUM_GPUS} --gpu-memory-utilization 0.95 --max_model_len 131072 --trust-remote-code  --port 8000 > "$LOG_FILE" 2>&1 &

# Wait for the server to fully start
sleep 75

# Prepare the additional argument for CoT if IS_COT is true
COT_ARG=""
if [ "$IS_COT" == "true" ]; then
    COT_ARG="--cot"
fi

if [ "$COT_ANSWER_EXTRACT" == "true" ]; then
    COT_ARG="$COT_ARG --cot_answer_extract"
fi

# echo all key parameters
echo "========================="
echo "Model Path: $MODEL_PATH"
echo "Is CoT: $IS_COT"
echo "CoT Answer Extract: $COT_ANSWER_EXTRACT"
echo "Log File: $LOG_FILE"
echo "Temperature: $TEMPERATURE"
echo "Save Directory: $SAVE_DIR"
echo "COT Prompt Type: $COT_PROMPT_TYPE"
echo "========================="


# Run the prediction script with the specified model path and CoT argument
python pred.py --model_path $MODEL_PATH $COT_ARG --n_proc 1 \
    --save_dir $SAVE_DIR \
    --temperature $TEMPERATURE \
    --cot_prompt_type $COT_PROMPT_TYPE 
echo "Prediction script done..."

# if cot is true but cot answer extract is false, then run the following command for answer extract
if [ "$COT_ANSWER_EXTRACT" == "false" ]; then
    echo "Starting answer extract script..."
    echo "========================="
    echo "Model Path: $MODEL_PATH"
    echo "Is CoT: $IS_COT"
    echo "CoT Answer Extract: $COT_ANSWER_EXTRACT"
    echo "Log File: $LOG_FILE"
    echo "Judge Model: $JUDGE_MODEL"
    echo "Temperature: $TEMPERATURE"
    echo "Save Directory: $SAVE_DIR"
    echo "========================="
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
    # add a log file name suffix, e.g., vllm_serve_output.log -> vllm_serve_output_cot_extract.log
    LOG_FILE="${LOG_FILE%.log}_cot_extract.log"
    # Start the backend server in the background and redirect output to the log file
    mkdir -p "$(dirname "$LOG_FILE")"
    # serve the judge model
    JUDGE_MODEL="/mnt/longcontext/models/siyuan/llama3/Qwen2.5-7B-Instruct"
    vllm serve $JUDGE_MODEL --api-key token-abc123 --tensor-parallel-size ${NUM_GPUS} --gpu-memory-utilization 0.95 --max_model_len 32768 --trust-remote-code  --port 8000 > "$LOG_FILE" 2>&1 &
    # Wait for the server to fully start
    sleep 75

    # Run the answer extract script

    python answer_extract.py --model_path $MODEL_PATH --n_proc 1 \
        --save_dir $SAVE_DIR    \
        --judge_model_path $JUDGE_MODEL  \
        --temperature $TEMPERATURE
fi