#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pred.py  ·  Modified to support dataset splitting for parallel runs.

Changes (★):
  • Added CLI args --splits_count (default 50) and --splits_index.
  • After loading the dataset, keep only the slice [start:end] for this worker.
  • After the original out_file selection logic, append -{splits_count}_{splits_index}
    to the filename so each job writes a unique file.
"""

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
from typing import List

# ---------------------------------------------------------
# Model / prompt configs
# ---------------------------------------------------------
model_map = json.loads(open("config/model2path.json", encoding="utf-8").read())
maxlen_map = json.loads(open("config/model2maxlen.json", encoding="utf-8").read())

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"

template_rag                  = open("prompts/0shot_rag.txt",               encoding="utf-8").read()
template_no_context           = open("prompts/0shot_no_context.txt",        encoding="utf-8").read()
template_0shot                = open("prompts/0shot.txt",                   encoding="utf-8").read()
template_0shot_cot            = open("prompts/0shot_cot.txt",               encoding="utf-8").read()
template_0shot_cot_train_align= open("prompts/0shot_cot_train_align.txt",   encoding="utf-8").read()
template_0shot_cot_qwq_choice = open("prompts/0shot_cot_qwq_choice.txt",    encoding="utf-8").read()
template_0shot_cot_ans        = open("prompts/0shot_cot_ans.txt",           encoding="utf-8").read()

# ---------------------------------------------------------
# Client helpers
# ---------------------------------------------------------
def setup_trapi(model_path=None, instance="gcr/shared", api_version="2024-10-21"):
    import re
    from azure.identity import (
        DefaultAzureCredential,
        ChainedTokenCredential,
        AzureCliCredential,
        get_bearer_token_provider,
    )
    from openai import AzureOpenAI

    setup_config = {
        "scope"        : "api://trapi/.default",
        "api_version"  : api_version,
        "model_name"   : "gpt-4o",
        "model_version": "2024-11-20",
        "instance"     : instance,
    }

    scope = setup_config["scope"]
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential            = True,
                exclude_environment_credential    = True,
                exclude_shared_token_cache_credential = True,
                exclude_developer_cli_credential  = True,
                exclude_powershell_credential     = True,
                exclude_interactive_browser_credential = True,
                exclude_visual_studio_code_credentials = True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            ),
        ),
        scope,
    )

    deployment_name = re.sub(r"[^a-zA-Z0-9-_]", "", f"{setup_config['model_name']}_{setup_config['model_version']}")
    if model_path is not None:
        deployment_name = re.sub(r"[^a-zA-Z0-9-_]", "", model_path)

    endpoint = f"https://trapi.research.microsoft.com/{setup_config['instance']}"

    client = AzureOpenAI(
        azure_endpoint        = endpoint,
        azure_ad_token_provider = credential,
        api_version           = setup_config["api_version"],
    )
    return client


def setup_openai():
    return OpenAI(
        api_key="sk-LYbQPilBE0kZMK0Jva15Wlhuih95WEvTkVpBEEXyikzIQAkG",
        base_url="https://www.dmxapi.com/v1",
    )


def setup_vllm_openai():
    return OpenAI(api_key=API_KEY, base_url=URL)


def setup_client(args):
    if args.model_type == "trapi":
        return setup_trapi(model_path=args.model_path, instance=args.instance, api_version=args.api_version)
    elif args.model_type == "openai":
        return setup_openai()
    elif args.model_type == "vllm":
        return setup_vllm_openai()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


# ---------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------
def query_llm(
    prompt: str,
    model: str,
    tokenizer,
    client=None,
    temperature=0.5,
    max_new_tokens=128,
    top_p=1.0,
    stop: List[str] = None,
    num_sequences=1,
):
    max_len = maxlen_map[model]
    if not any(x in model for x in ("gpt", "o1", "o3")):
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
            prompt = tokenizer.decode(input_ids)

    tries = 0
    if model in model_map:
        model = model_map[model]

    while tries < 100:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model        = model,
                messages     = [{"role": "user", "content": prompt}],
                temperature  = temperature,
                max_tokens   = max_new_tokens,
                n            = num_sequences,
                top_p        = top_p,
            )
            return [choice.message.content for choice in completion.choices]
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'Error: "{e}"  – retrying ...')
            time.sleep(10)

    print("Max retries reached; returning empty result.")
    return [""] * num_sequences


def extract_answer(response: str):
    response = response.replace("*", "")
    m = re.search(r"The\s+correct\s+answer\s+is\s+\(([A-D])\)", response)
    if m:
        return m.group(1)
    m = re.search(r"The\s+correct\s+answer\s+is\s+([A-D])", response)
    return m.group(1) if m else None


def get_pred(data, args, fout):
    model = args.model
    if any(x in model for x in ("gpt", "o1", "o3")):
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    elif "deepseek" in model.lower() and "distill" not in model.lower():
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    elif "qwen" in model.lower():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)

    client = setup_client(args)

    for item in tqdm(data, disable=bool(os.getenv("DISABLE_TQDM"))):
        context = item["context"]

        if args.rag > 0:
            template   = template_rag
            retrieved  = sorted(item["retrieved_context"][: args.rag], key=lambda x: x["c_idx"])
            context    = "\n\n".join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])

        elif args.no_context:
            template = template_no_context
        elif args.cot:
            if args.cot_prompt_type == "default":
                template = template_0shot_cot
            elif args.cot_prompt_type == "train_align":
                template = template_0shot_cot_train_align
            elif args.cot_prompt_type == "qwq_choice":
                template = template_0shot_cot_qwq_choice
            else:
                raise ValueError(f"Invalid cot_prompt_type: {args.cot_prompt_type}")
        else:
            template = template_0shot

        prompt = (
            template.replace("$DOC$", context.strip())
                    .replace("$Q$",  item["question"].strip())
                    .replace("$C_A$", item["choice_A"].strip())
                    .replace("$C_B$", item["choice_B"].strip())
                    .replace("$C_C$", item["choice_C"].strip())
                    .replace("$C_D$", item["choice_D"].strip())
        )

        max_tokens = 8192 if args.cot else 128
        output_list = query_llm(prompt, model, tokenizer, client,
                                temperature=args.temperature,
                                max_new_tokens=max_tokens,
                                num_sequences=args.num_sequences,
                                top_p=args.top_p)

        for index, output in enumerate(output_list):
            if not output:
                continue
            if args.cot and args.cot_answer_extract:
                # Step-two extraction
                response_cot = output.strip()
                item["response_cot"] = response_cot
                prompt2 = (
                    template_0shot_cot_ans.replace("$DOC$", context.strip())
                                          .replace("$Q$",  item["question"].strip())
                                          .replace("$C_A$", item["choice_A"].strip())
                                          .replace("$C_B$", item["choice_B"].strip())
                                          .replace("$C_C$", item["choice_C"].strip())
                                          .replace("$C_D$", item["choice_D"].strip())
                                          .replace("$COT$", response_cot)
                )
                output = query_llm(prompt2, model, tokenizer, client,
                                   temperature=args.temperature,
                                   max_new_tokens=8192,
                                   num_sequences=1)[0]
                if not output:
                    continue
            elif args.cot:
                item["response_cot"] = output.strip()

            response = output.strip()
            item["response_index"] = index
            item["response"]       = response
            item["pred"]           = extract_answer(response)
            item["judge"]          = item["pred"] == item["answer"]
            item["context"]        = context[:1000]

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_path is not None:
        args.model = args.model_path
        model_map[args.model]  = args.model_path
        maxlen_map[args.model] = maxlen_map["Llama-3.1-8B-Instruct"]

    # --- original out_file logic -------------------------------------------------
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{args.rag}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_temp{args.temperature}_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_temp{args.temperature}_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_temp{args.temperature}.jsonl")

    # ★ Append split-index suffix so each parallel run writes a unique file ------
    if args.splits_index is not None:
        dirname, fname = os.path.split(out_file)
        stem, ext      = os.path.splitext(fname)
        out_file       = os.path.join(dirname, f"{stem}-{args.splits_count}_{args.splits_index}{ext}")

    print("Arguments:", args)
    print("Output file:", out_file)

    # ---------------------------------------------------------------------------
    # Load dataset
    dataset = load_dataset("THUDM/LongBench-v2", split="train")

    data_all = [
        {
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "answer": item["answer"],
            "context": item["context"],
        }
        for item in dataset
    ]

    # ★ Keep only this split -----------------------------------------------------
    if args.splits_index is not None:
        total        = len(data_all)
        split_size   = total // args.splits_count
        start_idx    = args.splits_index * split_size
        end_idx      = (args.splits_index + 1) * split_size if args.splits_index < args.splits_count - 1 else total
        data_all     = data_all[start_idx:end_idx]
        print(f"Processing split {args.splits_index}/{args.splits_count} "
              f"→ indices [{start_idx}:{end_idx})  ({len(data_all)} items)")

    # ---------------------------------------------------------------------------
    # Cache handling
    if args.use_cache and os.path.exists(out_file):
        existing_ids = {json.loads(line)["_id"] for line in open(out_file, encoding="utf-8")}
        data         = [item for item in data_all if item["_id"] not in existing_ids]
        fout         = open(out_file, "a", encoding="utf-8")
    else:
        data         = data_all
        fout         = open(out_file, "w", encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Multi-processing
    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    fout.close()

    # Copy to ./results/ for convenience
    if os.path.exists(out_file):
        os.makedirs("./results/", exist_ok=True)
        shutil.copy(out_file, os.path.join("./results/", os.path.basename(out_file)))
        print("Copied result to ./results/")
    else:
        print("Warning: output file not found.")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
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
    parser.add_argument("--model_type", type=str, default="vllm")  # trapi, openai, vllm
    parser.add_argument("--instance", type=str, default="gcr/shared")  # for trapi
    parser.add_argument("--api_version", type=str, default="2024-10-21")  # for trapi

    # ★ NEW ARGS ----------------------------------------------------------------
    parser.add_argument("--splits_count", type=int, default=50,
                        help="Total number of dataset splits (default 50).")
    parser.add_argument("--splits_index", type=int, default=None,
                        help="Index (0-based) of the split to process. "
                             "If omitted, the full dataset is used.")

    args = parser.parse_args()
    main()
