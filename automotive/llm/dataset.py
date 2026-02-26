import os
import json

class MMLU_QSL:
    def __init__(self, dataset_path="mmlu_automotive.json"):
        self.dataset_path = dataset_path
        # Load all data into a backing store (simulating disk storage)
        self.all_data = self.load_data()
        # The lookup table acts as the "active memory" cache.
        # It will only contain samples explicitly loaded by LoadGen.
        self.qsl_lookup = {}
        self.count = len(self.all_data)

    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                "Please run 'python download_data.py' first."
            )
        
        print(f"Loading QSL source from {self.dataset_path}...")
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        print(f"Source loaded with {len(data)} samples.")
        return data

    def load_query_samples(self, sample_list):
        """
        Called by LoadGen to load samples into RAM.
        sample_list: A list of integer indices to load.
        """
        # For a large dataset, you would perform disk I/O here.
        # Since we have the data in self.all_data, we populate the lookup cache.
        for sample_index in sample_list:
            if sample_index not in self.qsl_lookup:
                self.qsl_lookup[sample_index] = self.all_data[sample_index]
        
        # print(f"Loaded {len(sample_list)} samples.")

    def unload_query_samples(self, sample_list):
        """
        Called by LoadGen to remove samples from RAM.
        sample_list: A list of integer indices to unload.
        """
        for sample_index in sample_list:
            if sample_index in self.qsl_lookup:
                del self.qsl_lookup[sample_index]
        
        # print(f"Unloaded {len(sample_list)} samples.")