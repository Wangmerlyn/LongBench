import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"

template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    max_len = maxlen_map[model]
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
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
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\" Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response, args):
    response = response.replace('*', '')
    if args.boxed:
        matches = re.findall(r'\\boxed\s*{\s*\(?\s*([A-Da-d])\)?\s*}', response)
        if matches:
            return matches[-1].strip().upper()
        else:
            return None
    else:
        match = re.search(r'The correct answer is\s*\(?([A-Da-d])\)?', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        else:
            return None

def main():
    global args

    if args.model_path is not None:
        args.model = args.model_path
        model_map[args.model] = args.model_path
        maxlen_map[args.model] = maxlen_map["Llama-3.1-8B-Instruct"]
    model_map[args.judge_model_path] = args.judge_model_path
    maxlen_map[args.judge_model_path] = maxlen_map["Llama-3.1-8B-Instruct"]
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path, trust_remote_code=True)

    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )

    out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_temp{args.temperature}_cot.jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train')

    with open(out_file, 'r', encoding='utf-8') as f:
        output_file_data = [json.loads(line) for line in f.readlines()]

    for output_item in tqdm(output_file_data):
        context = " "
        prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', output_item['question'].strip()).replace('$C_A$', output_item['choice_A'].strip()).replace('$C_B$', output_item['choice_B'].strip()).replace('$C_C$', output_item['choice_C'].strip()).replace('$C_D$', output_item['choice_D'].strip()).replace('$COT$', output_item['response_cot'])
        output = query_llm(prompt, args.judge_model_path, tokenizer, client, temperature=0, max_new_tokens=8192)
        if output == '':
            assert False, "error extracting answer, model output is empty"
        response = output.strip()
        pred = extract_answer(response, args)
        output_item['response'] = response
        output_item['pred'] = pred
        output_item['judge'] = output_item['pred'] == output_item['answer']

    with open(out_file, 'w', encoding='utf-8') as f:
        for output_item in output_file_data:
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--judge_model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--boxed", action="store_true", help="Extract the last boxed choice (A-D) from the response")
    global args
    args = parser.parse_args()

    main()
