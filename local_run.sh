cot_prompt_type="default"
model_paths=(
    "/mnt/longcontext/models/siyuan/rl_ckpts/orzboxed_llama3_15steps_fix_8k_2k_math_hard_2wiki_mathqachoice_niah_sentence_fix_128bsz_40ksamples_10epochs_end-step15"
)

# get the path of the current script
current_script_path=$(realpath "$0")
current_script_folder=$(dirname "$current_script_path")

# iterate over the model paths
for model_path in "${model_paths[@]}"; do
    # extract the model name from the path
    model_name=$(basename "$model_path")
    # create a directory for logs if it doesn't exist
    mkdir -p "/mnt/longcontext/models/siyuan/test_code/LongBench-v2/logs"
    # run the command with the extracted model name
    bash run_custom.sh --model_path "$model_path" \
        --save_dir $current_script_folder/results \
        --temperature 0.6 \
        --cot_answer_extract false \
        --is_cot true \
        --log_file "/mnt/longcontext/models/siyuan/test_code/LongBench-v2/logs/${model_name}.log" \
        --num_gpus 4 \
        --cot_prompt_type $cot_prompt_type 
done

