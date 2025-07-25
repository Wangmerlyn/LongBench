import os, csv, json
import argparse
import shutil
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

TIMESTAMP = os.getenv("TIMESTAMP", time.strftime("%Y%m%d_%H%M%S"))

model_map = json.loads(open("config/model2path.json", encoding="utf-8").read())
maxlen_map = json.loads(
    open("config/model2maxlen.json", encoding="utf-8").read()
)

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
template_rag = open("prompts/0shot_rag.txt", encoding="utf-8").read()
template_no_context = open(
    "prompts/0shot_no_context.txt", encoding="utf-8"
).read()
template_0shot = open("prompts/0shot.txt", encoding="utf-8").read()
template_0shot_cot = open("prompts/0shot_cot.txt", encoding="utf-8").read()
template_0shot_cot_train_align = open(
    "prompts/0shot_cot_train_align.txt", encoding="utf-8"
).read()
template_0shot_cot_qwq_choice = open(
    "prompts/0shot_cot_qwq_choice.txt", encoding="utf-8"
).read()
template_0shot_cot_ans = open(
    "prompts/0shot_cot_ans.txt", encoding="utf-8"
).read()

# Main function updated with timestamp appended to output file
def main():
    os.makedirs(args.save_dir, exist_ok=True)
    if args.model_path is not None:
        args.model = args.model_path
        model_map[args.model] = args.model_path
        maxlen_map[args.model] = maxlen_map["Llama-3.1-8B-Instruct"]

    if args.rag > 0:
        base_out_file = args.model.split("/")[-1] + f"_rag_{str(args.rag)}"
    elif args.no_context:
        base_out_file = args.model.split("/")[-1] + f"_temp{args.temperature}_no_context"
    elif args.cot:
        base_out_file = args.model.split("/")[-1] + f"_temp{args.temperature}_cot"
    else:
        base_out_file = args.model.split("/")[-1] + f"_temp{args.temperature}"

    out_file = os.path.join(args.save_dir, f"{base_out_file}-{TIMESTAMP}.jsonl")
    print(args)

    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    data_all = [{k: item[k] for k in item if k != "retrieved_context"} for item in dataset]

    if args.use_cache and os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
        fout = open(out_file, "a", encoding="utf-8")
        data = [item for item in data_all if item["_id"] not in has_data]
    else:
        fout = open(out_file, "w", encoding="utf-8")
        data = data_all

    data_subsets = [data[i :: args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    fout.close()

    if os.path.exists(out_file):
        os.makedirs("./results/", exist_ok=True)
        target_file = os.path.basename(out_file)
        target_path = os.path.join("./results/", target_file)

        shutil.copy(out_file, target_path)
        print(f"Output file copied to {target_path}")
    else:
        print(f"Output file {out_file} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--cot", "-cot", action="store_true")
    parser.add_argument("--cot_prompt_type", type=str, default="default")
    parser.add_argument("--cot_answer_extract", action="store_true")
    parser.add_argument("--no_context", "-nc", action="store_true")
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--use_cache", "-c", action="store_true")
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--num_sequences", "-n_seq", type=int, default=1)
    parser.add_argument("--top_p", "-p", type=float, default=1.0)
    parser.add_argument("--model_type", type=str, default="vllm")
    parser.add_argument("--instance", type=str, default="gcr/shared")
    parser.add_argument("--api_version", type=str, default="2024-10-21")
    args = parser.parse_args()
    main()
