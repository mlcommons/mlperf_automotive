import argparse
import json
import os
import datasets
from tqdm import tqdm

def format_mmlu_prompt(example):
    """
    Formats an MMLU example into a prompt for Llama 3.2 Instruct.
    """
    options = ["A", "B", "C", "D"]
    question = example['question']
    choices = example['choices']
    
    prompt_text = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt_text += f"{options[i]}. {choice}\n"
    
    prompt_text += "Answer:"
    
    # Llama 3.2 Instruct Format
    # Generalized system prompt for the entire dataset
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer the multiple choice question by providing only the single letter (A, B, C, or D) corresponding to the correct answer."},
        {"role": "user", "content": prompt_text}
    ]
    
    return messages, example['answer']

def fetch_samples(subsets, split, target_count, output_file):
    print(f"\nFetching {target_count if target_count else 'ALL'} samples from '{split}' split...")
    all_samples = []
    
    if target_count:
        samples_per_category = max(1, target_count // len(subsets))
        print(f"Limiting to approx {samples_per_category} samples per category.")
    else:
        samples_per_category = float('inf')
    
    for subset in tqdm(subsets, desc=f"Processing {split}"):
        try:
            ds = datasets.load_dataset("cais/mmlu", subset, split=split, trust_remote_code=True)
            
            if target_count:
                limit = min(len(ds), samples_per_category)
                ds = ds.select(range(limit))
            
            for item in ds:
                formatted_msgs, correct_idx = format_mmlu_prompt(item)
                
                all_samples.append({
                    "messages": formatted_msgs,
                    "correct_answer": ["A", "B", "C", "D"][correct_idx],
                    "subset": subset
                })
                
                if target_count and len(all_samples) >= target_count:
                    break
        except Exception as e:
            print(f"Warning: Could not process {subset}: {e}")
            continue
        
        if target_count and len(all_samples) >= target_count:
            break
            
    print(f"Saving {len(all_samples)} items to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)

def download_and_save(output_file="mmlu_full.json", sample_count=None, cal_output="mmlu_cal.json", cal_count=200):
    print("Fetching available MMLU configurations...")
    try:
        # Dynamically get all 57 subsets (math, history, law, etc.)
        subsets = datasets.get_dataset_config_names("cais/mmlu")
        # Filter out 'all' if it exists to avoid duplication
        subsets = [s for s in subsets if s != 'all']
    except Exception as e:
        print(f"Error fetching config names: {e}")
        return

    print(f"Found {len(subsets)} categories.")
    
    # 1. Fetch Calibration Data (from validation split)
    if cal_count > 0:
        fetch_samples(subsets, "validation", cal_count, cal_output)
        
    # 2. Fetch Evaluation Data (from test split)
    fetch_samples(subsets, "test", sample_count, output_file)
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mmlu_full.json", help="Output file path for evaluation dataset")
    parser.add_argument("--count", type=int, default=None, help="Total number of samples to generate (Default: None = Download Everything)")
    parser.add_argument("--cal_output", default="mmlu_cal.json", help="Output file path for calibration dataset")
    parser.add_argument("--cal_count", type=int, default=200, help="Number of calibration samples (Default: 200, set to 0 to disable)")
    args = parser.parse_args()
    
    download_and_save(args.output, args.count, args.cal_output, args.cal_count)