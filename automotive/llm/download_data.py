import argparse
import json
import os
import datasets
from tqdm import tqdm

SUBSETS = [
    'global_facts', 'management', 'marketing', 'miscellaneous', 'nutrition',
    'professional_accounting', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_microeconomics', 'high_school_psychology', 'public_relations',
    'security_studies', 'sociology', 'us_foreign_policy', 'virology',
    'world_religions'
]

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
    messages = [
        {"role": "system", "content": "You are a helpful automotive infotainment assistant. Answer the multiple choice question by providing only the single letter (A, B, C, or D) corresponding to the correct answer."},
        {"role": "user", "content": prompt_text}
    ]
    
    return messages, example['answer']

def download_and_save(output_file="mmlu_automotive.json", sample_count=100):
    print(f"Downloading MMLU dataset ({sample_count} samples across {len(SUBSETS)} categories)...")
    
    all_samples = []
    samples_per_category = max(1, sample_count // len(SUBSETS))
    
    for subset in tqdm(SUBSETS, desc="Processing Categories"):
        try:
            # Download specific subset
            ds = datasets.load_dataset("cais/mmlu", subset, split="test", trust_remote_code=True)
            
            # Select a chunk
            limit = min(len(ds), samples_per_category)
            ds = ds.select(range(limit))
            
            for item in ds:
                formatted_msgs, correct_idx = format_mmlu_prompt(item)
                
                # We save the formatted messages directly to avoid re-processing at runtime
                all_samples.append({
                    "messages": formatted_msgs,
                    "correct_answer": ["A", "B", "C", "D"][correct_idx],
                    "subset": subset
                })
                
                if len(all_samples) >= sample_count:
                    break
        except Exception as e:
            print(f"Warning: Could not process {subset}: {e}")
            continue
        
        if len(all_samples) >= sample_count:
            break
            
    # Save to disk
    print(f"Saving {len(all_samples)} items to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mmlu_automotive.json", help="Output file path")
    parser.add_argument("--count", type=int, default=100, help="Total number of samples to generate")
    args = parser.parse_args()
    
    download_and_save(args.output, args.count)

