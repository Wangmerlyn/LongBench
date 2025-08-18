import os
import json
import re
import pandas as pd
import smtplib
from email.message import EmailMessage
from typing import Optional

email_title="LongBench-v2 Evaluation Results Model {model_name}"
email_body="\n"

def send_mail(subject: str, body: str, to_csv: Optional[str] = None):
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")  # Gmail
    port = int(os.getenv("SMTP_PORT", "465"))  # 465=SSL, 587=STARTTLS
    user = os.getenv("SMTP_USER", "sywang0227@gmail.com")
    password = os.getenv("SMTP_PASS")
    if not user or not password:
        print("SMTP_USER or SMTP_PASS not set, skipping email sending.")
        return
    sender = os.getenv("SMTP_FROM", user)
    tos = [
        x.strip()
        for x in (to_csv or os.getenv("MAIL_TO", "wsy0227@sjtu.edu.cn")).split(",")
        if x.strip()
    ]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(tos)
    msg.set_content(body)

    if port == 465:
        with smtplib.SMTP_SSL(host, port) as s:
            if user and password:
                s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            if user and password:
                s.login(user, password)
            s.send_message(msg)

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
    global email_body
    global email_title
    all_domains = set()
    all_models = [os.path.basename(file).split('.')[0] for file in files]
    email_title = email_title.format(model_name=','.join(all_models))
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
        model_name = '.'.join(file.split('.')[:-1])
        overall_score = round(100 * total_acc / total_qs, 1) if total_qs else 0

        print(f"{model_name}: {overall_score}")
        email_body += f"Model: {model_name}\nMode: {mode}\nOverall Accuracy: {overall_score}\n"

        # safe write to file for mix mode 
        if mode == 'mix':
            try:
                out_dir = "/mnt/longcontext/models/siyuan/test_code/LongBench-v2/score_mix"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{model_name}.jsonl")
                with open(out_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"model": model_name, "overall_score": overall_score}) + "\n")
            except Exception as e:
                print(f"[Warning] Failed to write score for {model_name} in mix mode: {e}")

if __name__ == "__main__":
    evaluate(mode='standard')
    evaluate(mode='boxed')
    evaluate(mode='mix')
    evaluate(mode='original')
    send_mail(email_title, email_body)
