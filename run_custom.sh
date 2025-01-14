#!/bin/bash

# Define the model path (default value can be overridden by the first script argument)
MODEL_PATH=${1:-"/mnt/longcontext/models/siyuan/llama3/llama-3.1-8B-instruct"}

# Start the backend server in the background
vllm serve $MODEL_PATH --api-key token-abc123 --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --max_model_len 131072 --trust-remote-code &

# Wait for the server to fully start
sleep 30

# Run the prediction script with the specified model path
python pred.py --model_path $MODEL_PATH
