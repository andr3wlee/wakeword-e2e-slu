import json
import os
import random

import numpy as np


class Loader_HeySnips:
    def __init__(self, json_path, mfccs_path, batch_size=32):
        """
        Data loader for the HeySnips dataset.

        Args:
            json_path: Path to the JSON file with sample information
            mfccs_path: Path to the directory containing MFCC files
            batch_size: Number of samples per batch
        """
        self.snips_json = json.load(open(json_path, "r"))

        # Ensure mfccs_path ends with a slash
        if not mfccs_path.endswith('/'):
            mfccs_path = mfccs_path + '/'

        # Create sample paths with proper formatting
        self.samples_json = []
        for item in self.snips_json:
            mfcc_file = f"{item['id']}.mfcc.npy"
            mfcc_path = os.path.join(mfccs_path, mfcc_file)

            if os.path.exists(mfcc_path):
                self.samples_json.append({
                    "mfcc_path": mfcc_path,
                    "label": item["is_hotword"]
                })

        print(f"Loaded {len(self.samples_json)} valid samples out of {len(self.snips_json)} from {json_path}")

        if len(self.samples_json) == 0:
            raise ValueError(f"No valid MFCC files found for samples in {json_path}")

        self.batch_size = batch_size
        self.batch_idx = 0
        self.num_batches = max(1, len(self.samples_json) // batch_size)

        # Shuffle initially
        self.shuffle_samples()

    def get_batch(self):
        """
        Get a batch of samples.

        Returns:
            Tuple of (mfccs, labels) where mfccs is a padded array of MFCC features
            and labels is an array of hotword indicators
        """
        # Check if we need to reshuffle
        if self.batch_idx >= self.num_batches:
            self.shuffle_samples()

        # Get batch indices
        start_idx = self.batch_size * self.batch_idx
        end_idx = min(start_idx + self.batch_size, len(self.samples_json))
        batch = self.samples_json[start_idx:end_idx]

        # Load MFCCs and labels
        mfccs_list = []
        labels = []

        for sample in batch:
            try:
                mfcc_data = np.load(sample["mfcc_path"])
                mfccs_list.append(mfcc_data)
                labels.append(sample["label"])
            except Exception as e:
                print(f"Error loading {sample['mfcc_path']}: {str(e)}")

        if not mfccs_list:
            raise RuntimeError("No valid samples in the current batch")

        # Pad and create batch
        mfccs = self.padd_concat_mfccs(mfccs_list)
        labels = np.array(labels)

        self.batch_idx += 1
        return mfccs, labels

    def padd_concat_mfccs(self, mfccs_list):
        """
        Pad MFCC arrays to the same length and concatenate them into a batch.

        Args:
            mfccs_list: List of MFCC arrays

        Returns:
            Padded array of shape [batch_size, max_length, num_features]
        """
        max_length = max([mfcc.shape[0] for mfcc in mfccs_list])
        mfccs_array = np.zeros((len(mfccs_list), max_length, mfccs_list[0].shape[-1]), dtype=np.float32)

        for idx, mfcc in enumerate(mfccs_list):
            mfccs_array[idx, :mfcc.shape[0], :] = mfcc

        return mfccs_array

    def shuffle_samples(self):
        """Shuffle samples and reset batch index"""
        random.shuffle(self.samples_json)
        self.batch_idx = 0