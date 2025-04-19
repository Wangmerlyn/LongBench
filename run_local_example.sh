model_name="/mnt/longcontext/models/siyuan/rl_ckpts/debug_llama3_8k_2k_math_hard_2wiki_mathqachoice_128bsz_30ksamples_10epochs_end-step140"

bash run_custom.sh \
    --model_path $model_name \
    --is_cot true \
    --cot_answer_extract true \
    --log_file logs/debug_llama3_8k_2k_math_hard_2wiki_mathqachoice_128bsz_30ksamples_10epochs_end-step140.log \
    --temperature 0.6 \
    --save_dir ./results