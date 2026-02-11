import argparse
import json
import os
import sys

def check_accuracy(log_file, dataset_file):
    print(f"Reading logs from: {log_file}")
    print(f"Reading dataset from: {dataset_file}")

    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found.")
        sys.exit(1)

    # 1. Load Ground Truth
    with open(dataset_file, 'r') as f:
        ground_truth_data = json.load(f)
    ground_truth_map = {i: item["correct_answer"] for i, item in enumerate(ground_truth_data)}

    # 2. Load MLPerf Accuracy Log
    with open(log_file, 'r') as f:
        results = json.load(f)

    correct = 0
    total = 0
    
    # Iterate through results to calculate totals
    for item in results:
        qsl_idx = item["qsl_idx"]
        hex_data = item["data"]
        
        try:
            # 1. Convert hex string back to bytes
            bytes_data = bytes.fromhex(hex_data)
            
            # 2. Decode bytes to utf-8 string
            predicted_char = bytes_data.decode('utf-8')
            
        except Exception as e:
            # print(f"Error parsing sample {qsl_idx}: {e}")
            predicted_char = "ERR"

        expected_char = ground_truth_map.get(qsl_idx, "N/A")
        is_match = predicted_char == expected_char
        
        if is_match:
            correct += 1
        total += 1

    if total == 0:
        print("No results found in log file.")
        return

    accuracy = (correct / total) * 100
    print("-" * 60)
    print(f"Total Samples: {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {accuracy:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default="results/mlperf_log_accuracy.json", help="Path to MLPerf accuracy log")
    parser.add_argument("--dataset_file", default="mmlu_automotive.json", help="Path to source dataset")
    args = parser.parse_args()

    check_accuracy(args.log_file, args.dataset_file)