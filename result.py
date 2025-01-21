import os
import json
import re
import pandas as pd

def extract_final_answer(text):
    match = re.search(r'final answer:\s*\(([A-Za-z])\)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None  

# Directory containing result files
files = os.listdir('results')

# Initialize an empty DataFrame to store results
columns = ['Model', 'Overall', 'Easy', 'Hard', 'Short', 'Medium', 'Long']
results_df = pd.DataFrame(columns=columns)

# Initialize a dictionary for domain stats

compensated = False

# Collect all possible domains
all_domains = set()

for file in files:
    filename = os.path.join('results', file)
    
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    
    # Initialize counts and accuracy variables for each dataset
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    domain_dict = {}
    
    # Track domain statistics and collect all domains
    for pred in pred_data:
        if pred["pred"] is None and "response_cot" in pred:
            extra_match_answer = extract_final_answer(pred["response_cot"])
            pred['judge'] = True if extra_match_answer == pred['answer'] else False
        
        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        
        # Difficulty-based categorization
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        # Length-based categorization
        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc
        
        # Update domain statistics
        if pred['domain'] not in domain_dict:
            domain_dict[pred['domain']] = {'total': 0, 'correct': 0}
        domain_dict[pred['domain']]['total'] += 1
        domain_dict[pred['domain']]['correct'] += acc
        
        # Collect all unique domains
        all_domains.add(pred['domain'])

    # Calculate accuracies
    model_name = '.'.join(file.split('.')[:-1])
    overall_accuracy = round(100 * (easy_acc + hard_acc) / len(pred_data), 1)
    easy_accuracy = round(100 * easy_acc / easy, 1) if easy > 0 else 0
    hard_accuracy = round(100 * hard_acc / hard, 1) if hard > 0 else 0
    short_accuracy = round(100 * short_acc / short, 1) if short > 0 else 0
    medium_accuracy = round(100 * medium_acc / medium, 1) if medium > 0 else 0
    long_accuracy = round(100 * long_acc / long, 1) if long > 0 else 0

    # Prepare a row for the DataFrame
    model_results = {
        'Model': model_name,
        'Overall': overall_accuracy,
        'Easy': easy_accuracy,
        'Hard': hard_accuracy,
        'Short': short_accuracy,
        'Medium': medium_accuracy,
        'Long': long_accuracy,
    }

    # Calculate domain accuracies and add them to the row
    for domain in all_domains:
        domain_accuracy = round(100 * domain_dict[domain]['correct'] / domain_dict[domain]['total'], 1)
        model_results[domain] = domain_accuracy

    # Use pd.concat() to add the results to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([model_results])], ignore_index=True)

# Save the results to a CSV file with domain accuracies included
results_df.to_csv('result_with_domain_as_columns.csv', index=False, encoding='utf-8')

# Optionally, print the results to the console for review
print(results_df)
