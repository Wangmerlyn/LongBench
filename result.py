import os
import json
import re
import pandas as pd

def extract_final_answer(text):
    match = re.search(r'final answer:\s*\(([A-Za-z])\)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    pattern = re.compile(r'answer:\s*([A-Za-z])', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1)
    pattern = re.compile(r'answer:\s*\*\*([A-Za-z])\*\*\.?', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1)
    pattern = re.compile(r'answer:\s*\(([A-Za-z])\)', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None

def original_extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def last_boxed_only_string(string: str):
    idx = string.rfind("boxed")
    if idx == -1:
        idx = string.rfind("\\fbox")
    if idx == -1:
        return None
    i = idx
    num_left_braces_open = 0
    right_brace_idx = None
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]

def remove_boxed(boxed_str):
    if boxed_str.startswith('boxed{') and boxed_str.endswith('}'):
        return boxed_str[len('boxed{'):-1].strip()
    if boxed_str.startswith('\\fbox{') and boxed_str.endswith('}'):
        return boxed_str[len('\\fbox{'):-1].strip()
    return boxed_str.strip()

def extract_boxed_answer(text):
    text = text.replace("\\boxed{}", "")
    boxed_content = last_boxed_only_string(text)
    if boxed_content:
        inner_text = remove_boxed(boxed_content)
        match = re.search(r"\((\w)\)", inner_text)
        if match:
            return match.group(1).upper()
    return None

def evaluate(mode='standard'):
    files = os.listdir('results')
    columns = ['Model', 'Overall', 'Easy', 'Hard', 'Short', 'Medium', 'Long',
               'Long In-context Learning', 'Long Structured Data Understanding', 'Code Repository Understanding',
               'Single-Document QA', 'Long-dialogue History Understanding', 'Multi-Document QA']
    results_df = pd.DataFrame(columns=columns)

    all_domains = set()
    for file in files:
        pred_data = [json.loads(line) for line in open(os.path.join('results', file), encoding='utf-8')]
        for pred in pred_data:
            all_domains.add(pred['domain'])

    for file in files:
        pred_data = [json.loads(line) for line in open(os.path.join('results', file), encoding='utf-8')]

        metrics = {'easy': [0, 0], 'hard': [0, 0], 'short': [0, 0], 'medium': [0, 0], 'long': [0, 0]}
        domain_dict = {domain: [0, 0] for domain in all_domains}

        for pred in pred_data:
            pred_option = None
            if mode == 'boxed':
                pred_option = extract_boxed_answer(pred.get("response_cot", ""))
            elif mode == 'standard':
                if pred["pred"] is None and "response_cot" in pred:
                    pred_option = extract_final_answer(pred["response_cot"])
                else:
                    pred_option = pred['pred']
            elif mode == 'mix':
                pred_option = extract_boxed_answer(pred.get("response_cot", ""))
                if not pred_option:
                    if pred["pred"] is None:
                        pred_option = extract_final_answer(pred["response_cot"])
                    else:
                        pred_option = pred['pred']
            elif mode == 'original':
                pred_option = original_extract_answer(pred.get("response", ""))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            pred['judge'] = pred_option == pred['answer']

            difficulty, length, domain = pred["difficulty"], pred["length"], pred["domain"]
            metrics[difficulty][0] += pred['judge']
            metrics[difficulty][1] += 1
            metrics[length][0] += pred['judge']
            metrics[length][1] += 1

            domain_dict[domain][0] += pred['judge']
            domain_dict[domain][1] += 1

        total_acc = sum([v[0] for v in metrics.values()])
        total_qs = sum([v[1] for v in metrics.values()])
        model_results = {
            'Model': '.'.join(file.split('.')[:-1]),
            'Overall': round(100 * total_acc / total_qs, 1) if total_qs else 0,
            'Easy': round(100 * metrics['easy'][0] / metrics['easy'][1], 1) if metrics['easy'][1] else 0,
            'Hard': round(100 * metrics['hard'][0] / metrics['hard'][1], 1) if metrics['hard'][1] else 0,
            'Short': round(100 * metrics['short'][0] / metrics['short'][1], 1) if metrics['short'][1] else 0,
            'Medium': round(100 * metrics['medium'][0] / metrics['medium'][1], 1) if metrics['medium'][1] else 0,
            'Long': round(100 * metrics['long'][0] / metrics['long'][1], 1) if metrics['long'][1] else 0,
        }

        for domain in all_domains:
            domain_accuracy = round(100 * domain_dict[domain][0] / domain_dict[domain][1], 1) if domain_dict[domain][1] else 0.0
            model_results[domain] = domain_accuracy

        results_df = pd.concat([results_df, pd.DataFrame([model_results])], ignore_index=True)

    filename = f'result_{mode}.csv'
    results_df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Using mode '{mode}', results saved to {filename}")
    print(f"Results:\n{results_df}")

if __name__ == "__main__":
    evaluate(mode='standard')
    evaluate(mode='boxed')
    evaluate(mode='mix')
    evaluate(mode='original')
