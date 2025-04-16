import pickle
import random
import argparse
import csv


def read_pkl_and_select_random_items(file_path, num_items):
    # Load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Ensure the data is a list
    if not isinstance(data, list):
        raise ValueError("The pickle file does not contain a list.")

    # Filter items with a minimum value of 20
    filtered_data = [item for item in data if item >= 20]

    # Randomly select items from the filtered list
    selected_items = random.sample(
        filtered_data, min(
            num_items, len(filtered_data)))
    return selected_items


def combine_lists_to_dict(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    return {i: (list1[i], list2[i]) for i in range(len(list1))}


def filter_scene_length(input_dict, min_value):
    return {key: value for key, value in input_dict.items()
            if value[1] >= min_value}


def write_tuples_to_csv(tuples_list, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start", "Length"])  # Header row
        for index, (start, length) in enumerate(tuples_list):
            writer.writerow([start, length])


def main():
    parser = argparse.ArgumentParser(
        description="Select items from pickle files.")
    parser.add_argument(
        "--scene_lengths",
        type=str,
        help="Path to the scene lengths pickle file")
    parser.add_argument(
        "--scene_starts",
        type=str,
        help="Path to the scene starts pickle file")
    parser.add_argument(
        "--num_items",
        type=int,
        default=10,
        help="Number of scenes to select")
    args = parser.parse_args()

    try:
        with open(args.scene_lengths, 'rb') as f:
            scene_lengths = pickle.load(f)
        with open(args.scene_starts, 'rb') as f:
            scene_starts = pickle.load(f)[:-1]
        scene_dict = combine_lists_to_dict(scene_starts, scene_lengths)
        scene_dict = filter_scene_length(scene_dict, 20)
        random.seed(97)
        sample = random.sample(list(scene_dict.keys()), args.num_items)
        calibration_list = [scene_dict[i] for i in sample]
        write_tuples_to_csv(calibration_list, "calibration_list.csv")
        print(calibration_list)

    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
