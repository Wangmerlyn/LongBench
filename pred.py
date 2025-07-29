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

# ----------------------------------------------------------------
# NEW: global timestamp (falls back to current time if env missing)
# ----------------------------------------------------------------
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
        "scope": "api://trapi/.default",
        "api_version": api_version,
        "model_name": "gpt-4o",
        "model_version": "2024-11-20",
        "instance": instance,
    }

    scope = setup_config["scope"]
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
                # Azure ML Compute jobs that has the client id of the
                # user-assigned managed identity in it.
                # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
                # In case it is not set the ManagedIdentityCredential will
                # default to using the system-assigned managed identity, if any.
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            ),
        ),
        scope,
    )

    api_version = setup_config["api_version"]  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
    model_name = setup_config['model_name']  # Ensure this is a valid model name
    model_version = setup_config['model_version']  # Ensure this is a valid model version
    deployment_name = re.sub(
        r"[^a-zA-Z0-9-_]", "", f"{model_name}_{model_version}"
    )  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
    if model_path is not None:
        deployment_name = re.sub(
            r"[^a-zA-Z0-9-_]", "", model_path
        )
    instance = setup_config['instance']  # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly)
    endpoint = f"https://trapi.research.microsoft.com/{instance}"

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )
    return client

def setup_openai():
    return OpenAI(
        api_key="sk-LYbQPilBE0kZMK0Jva15Wlhuih95WEvTkVpBEEXyikzIQAkG",
        base_url="https://www.dmxapi.com/v1",
    )

def setup_vllm_openai():
    return OpenAI(
        api_key=API_KEY,
        base_url=URL,
    )

def setup_client(args):
    if args.model_type == "trapi":
        return setup_trapi(model_path=args.model_path, 
                           instance=args.instance, 
                           api_version=args.api_version)
    elif args.model_type == "openai":
        return setup_openai()
    elif args.model_type == "vllm":
        return setup_vllm_openai()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def query_llm(
    prompt,
    model,
    tokenizer,
    client=None,
    temperature=0.5,
    max_new_tokens=128,
    top_p=1.0,
    stop=None,
    num_sequences=1,
):
    # truncate
    max_len = maxlen_map[model]
    if not ("gpt" in model or "o1" in model or "o3" in model):
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
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=num_sequences,
                top_p=top_p,
            )
            # return completion.choices[0].message.content
            return [choice.message.content for choice in completion.choices]
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print('Error Occurs: "%s"        Retry ...' % (str(e)))
            time.sleep(10)
    else:
        print("Max tries. Failed.")
        return ""


def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        else:
            return None


def get_pred(data, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model or "o3" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    elif 'deepseek' in model.lower() and "distill" not in model.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1", trust_remote_code=True
        )
    elif "qwen" in model.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_map[model], trust_remote_code=True
        )
    # client = OpenAI(base_url=URL, api_key=API_KEY)
    client = setup_client(args)
    for item in tqdm(data):
        context = item["context"]
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][: args.rag]
            retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
            context = "\n\n".join(
                [
                    f"Retrieved chunk {idx+1}: {x['content']}"
                    for idx, x in enumerate(retrieved)
                ]
            )
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
                raise ValueError(
                    f"Invalid cot_prompt_type, got {args.cot_prompt_type} which is unknown."
                )
        else:
            template = template_0shot
        prompt = (
            template.replace("$DOC$", context.strip())
            .replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip())
            .replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip())
            .replace("$C_D$", item["choice_D"].strip())
        )
        if args.cot:
            output_list = query_llm(
                prompt,
                model,
                tokenizer,
                client,
                temperature=args.temperature,
                max_new_tokens=8192,
                num_sequences=args.num_sequences,
            )
        else:
            output_list = query_llm(
                prompt,
                model,
                tokenizer,
                client,
                temperature=args.temperature,
                max_new_tokens=128,
                num_sequences=args.num_sequences,
            )
        for index, output in enumerate(output_list):
            if output == "":
                continue
            if args.cot:  # extract answer from COT
                if args.cot_answer_extract:
                    response = output.strip()
                    item["response_cot"] = response
                    prompt = (
                        template_0shot_cot_ans.replace("$DOC$", context.strip())
                        .replace("$Q$", item["question"].strip())
                        .replace("$C_A$", item["choice_A"].strip())
                        .replace("$C_B$", item["choice_B"].strip())
                        .replace("$C_C$", item["choice_C"].strip())
                        .replace("$C_D$", item["choice_D"].strip())
                        .replace("$COT$", response)
                    )
                    output = query_llm(
                        prompt,
                        model,
                        tokenizer,
                        client,
                        temperature=args.temperature,
                        max_new_tokens=8192,
                        num_sequences=1,
                    )[0]
                    if output == "":
                        continue
                else:
                    response = output.strip()
                    item["response_cot"] = response
                    output = "not extracted"
            response = output.strip()
            item["response_index"] = index
            item["response"] = response
            item["pred"] = extract_answer(response)
            item["judge"] = item["pred"] == item["answer"]
            item["context"] = context[:1000]
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    if args.model_path is not None:
        # overwrite model
        args.model = args.model_path
        model_map[args.model] = args.model_path
        maxlen_map[args.model] = maxlen_map["Llama-3.1-8B-Instruct"]
    if args.rag > 0:
        out_file = os.path.join(
            args.save_dir,
            args.model.split("/")[-1] + f"_rag_{str(args.rag)}-{TIMESTAMP}.jsonl",
        )
    elif args.no_context:
        out_file = os.path.join(
            args.save_dir,
            args.model.split("/")[-1]
            + f"_temp{args.temperature}_no_context-{TIMESTAMP}.jsonl",
        )
    elif args.cot:
        out_file = os.path.join(
            args.save_dir,
            args.model.split("/")[-1] + f"_temp{args.temperature}_cot-{TIMESTAMP}.jsonl",
        )
    else:
        out_file = os.path.join(
            args.save_dir,
            args.model.split("/")[-1] + f"_temp{args.temperature}-{TIMESTAMP}.jsonl",
        )
    print(args)

    dataset = load_dataset(
        "zai-org/LongBench-v2", split="train"
    )  # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
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

    # cache
    if args.use_cache:
        has_data = {}
        if os.path.exists(out_file):
            with open(out_file, encoding="utf-8") as f:
                has_data = {json.loads(line)["_id"]: 0 for line in f}
        fout = open(out_file, "a", encoding="utf-8")
        data = []
        for item in data_all:
            if item["_id"] not in has_data:
                data.append(item)
    else:
        has_data = {}
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

    # after all the processes are done, close the file, and copy the output file to "./results/"
    fout.close()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            os.makedirs("./results/", exist_ok=True)

            target_path = os.path.join("./results/", os.path.basename(out_file))

            shutil.copy(out_file, target_path)
            print(f"Output file copied to {target_path}")
    else:
        print(f"Output file {out_file} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--cot", "-cot", action="store_true"
    )  # set to True if using COT
    parser.add_argument(
        "--cot_prompt_type", type=str, default="default"
    )  # set to default or train_align
    parser.add_argument(
        "--cot_answer_extract", action="store_true"
    )  # set to True if using COT answer extraction
    parser.add_argument(
        "--no_context", "-nc", action="store_true"
    )  # set to True if using no context (directly measuring memorization)
    parser.add_argument(
        "--rag", "-rag", type=int, default=0
    )  # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--use_cache", "-c", action="store_true")
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--num_sequences", "-n_seq", type=int, default=1)
    parser.add_argument("--top_p", "-p", type=float, default=1.0)
    parser.add_argument("--model_type", type=str, default="vllm")  # trapi, openai, vllm
    parser.add_argument("--instance", type=str, default="gcr/shared")  # for trapi
    parser.add_argument(
        "--api_version", type=str, default="2024-10-21"
    )  # for trapi, e.g., 2024-10-21
    args = parser.parse_args()
    main()
