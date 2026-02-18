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

def download_and_save(output_file="mmlu_full.json", sample_count=None):
    print("Fetching available MMLU configurations...")
    try:
        # Dynamically get all 57 subsets (math, history, law, etc.)
        subsets = datasets.get_dataset_config_names("cais/mmlu")
        # Filter out 'all' if it exists to avoid duplication, though usually it's just the topics
        subsets = [s for s in subsets if s != 'all']
    except Exception as e:
        print(f"Error fetching config names: {e}")
        return

    print(f"Found {len(subsets)} categories. Starting download...")
    
    all_samples = []
    
    # If sample_count is provided, we distribute it across categories
    if sample_count:
        samples_per_category = max(1, sample_count // len(subsets))
        print(f"Limiting to approximately {samples_per_category} samples per category (Total target: {sample_count})")
    else:
        samples_per_category = float('inf')
        print("Downloading ALL samples from all categories.")
    
    for subset in tqdm(subsets, desc="Processing Categories"):
        try:
            # Download specific subset
            ds = datasets.load_dataset("cais/mmlu", subset, split="test", trust_remote_code=True)
            
            # Select a chunk if we are limiting samples
            if sample_count:
                limit = min(len(ds), samples_per_category)
                ds = ds.select(range(limit))
            
            for item in ds:
                formatted_msgs, correct_idx = format_mmlu_prompt(item)
                
                all_samples.append({
                    "messages": formatted_msgs,
                    "correct_answer": ["A", "B", "C", "D"][correct_idx],
                    "subset": subset
                })
                
                # Global break if we strictly hit the count
                if sample_count and len(all_samples) >= sample_count:
                    break
        except Exception as e:
            print(f"Warning: Could not process {subset}: {e}")
            continue
        
        if sample_count and len(all_samples) >= sample_count:
            break
            
    # Save to disk
    print(f"Saving {len(all_samples)} items to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mmlu_full.json", help="Output file path")
    parser.add_argument("--count", type=int, default=None, help="Total number of samples to generate (Default: None = Download Everything)")
    args = parser.parse_args()
    
    download_and_save(args.output, args.count)