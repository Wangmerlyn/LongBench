#!/bin/bash

# Kill all other vllm processes before starting
pids=$(ps aux | grep vllm | grep -v grep | awk '{print $2}')

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

# Define the model path (default value can be overridden by the first script argument)
MODEL_PATH=${1:-"/mnt/longcontext/models/siyuan/llama3/llama-3.1-8B-instruct"}

# Define the IS_COT flag (default value can be overridden by the second script argument)
IS_COT=${2:-false}

# Define the log file for the backend server output
LOG_FILE=${3:-"vllm_serve_output.log"}

# Start the backend server in the background and redirect output to the log file
vllm serve $MODEL_PATH --api-key token-abc123 --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --max_model_len 131072 --trust-remote-code > "$LOG_FILE" 2>&1 &

# Wait for the server to fully start
sleep 75

# Prepare the additional argument for CoT if IS_COT is true
COT_ARG=""
if [ "$IS_COT" == "true" ]; then
    COT_ARG="--cot"
fi

# Run the prediction script with the specified model path and CoT argument
python pred.py --model_path $MODEL_PATH $COT_ARG --n_proc 1 

