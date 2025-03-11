cot_prompt_type="train_align"
model_paths=(
    "/mnt/longcontext/models/siyuan/rl_ckpts/uk2_llama3_4k+1k_8k+2k_no_math_20ksamples_10epochs_end-step100"
    "/mnt/longcontext/models/siyuan/rl_ckpts/uk2_llama3_4k+1k_8k+2k_no_math_20ksamples_10epochs_end-step50"
)

# iterate over the model paths
for model_path in "${model_paths[@]}"; do
    # extract the model name from the path
    model_name=$(basename "$model_path")
    # create a directory for logs if it doesn't exist
    mkdir -p "/mnt/longcontext/models/siyuan/test_code/LongBench-v2/logs"
    # run the command with the extracted model name
    bash run_custom.sh --model_path "$model_path" \
        --save_dir ~/results \
        --temperature 0.6 \
        --is_cot true \
        --log_file "/mnt/longcontext/models/siyuan/test_code/LongBench-v2/logs/${model_name}.log" \
        --num_gpus 4 \
        --cot_prompt_type $cot_prompt_type 
done
