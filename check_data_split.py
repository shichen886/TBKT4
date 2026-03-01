import pandas as pd
import torch
from random import shuffle, seed
import numpy as np

def get_data_v1(df, max_length, train_split=0.8, randomize=True):
    """Version from train_tsakt_linear_final.py"""
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    # Chunk sequences
    lists = (item_ids, skill_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    if randomize:
        shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def get_data_v2(df, max_length, train_split=0.8, randomize=True):
    """Version from train_tsakt_linear_nopos.py"""
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    item_ids = chunk(item_ids)
    skill_ids = chunk(skill_ids)
    labels = chunk(labels)

    if randomize:
        zipped = list(zip(item_ids, skill_ids, labels))
        shuffle(zipped)
        item_ids, skill_ids, labels = zip(*zipped)

    split = int(train_split * len(item_ids))
    train_data = list(zip(item_ids[:split], skill_ids[:split], labels[:split]))
    val_data = list(zip(item_ids[split:], skill_ids[split:], labels[split:]))

    return train_data, val_data


def check_data_split(dataset_name):
    """Check if two versions produce the same data split."""
    print(f"\n{'='*60}")
    print(f"Checking data split for {dataset_name}")
    print(f"{'='*60}")

    data_path = f'data/{dataset_name}/preprocessed_data.csv'
    df = pd.read_csv(data_path, sep="\t")

    print(f"Total rows: {len(df)}")
    print(f"Total users: {len(df.groupby('user_id'))}")

    # Test with fixed seed
    seed(42)
    train1, val1 = get_data_v1(df, max_length=200, train_split=0.8, randomize=True)

    seed(42)
    train2, val2 = get_data_v2(df, max_length=200, train_split=0.8, randomize=True)

    print(f"\nVersion 1 (train_tsakt_linear_final.py):")
    print(f"  Train samples: {len(train1)}")
    print(f"  Val samples: {len(val1)}")

    print(f"\nVersion 2 (train_tsakt_linear_nopos.py):")
    print(f"  Train samples: {len(train2)}")
    print(f"  Val samples: {len(val2)}")

    # Check if splits are the same
    print(f"\n{'='*60}")
    print("Split comparison:")
    print(f"{'='*60}")

    if len(train1) == len(train2) and len(val1) == len(val2):
        print("✓ Train and val sizes are the same")
    else:
        print("✗ Train and val sizes are DIFFERENT!")
        print(f"  Train: {len(train1)} vs {len(train2)}")
        print(f"  Val: {len(val1)} vs {len(val2)}")

    # Check if the actual data is the same
    # Convert to comparable format
    train1_ids = set([tuple(t[0].tolist()) for t in train1])
    train2_ids = set([tuple(t[0].tolist()) for t in train2])

    val1_ids = set([tuple(t[0].tolist()) for t in val1])
    val2_ids = set([tuple(t[0].tolist()) for t in val2])

    train_overlap = len(train1_ids & train2_ids) / len(train1_ids) * 100
    val_overlap = len(val1_ids & val2_ids) / len(val1_ids) * 100

    print(f"\nData overlap:")
    print(f"  Train overlap: {train_overlap:.2f}%")
    print(f"  Val overlap: {val_overlap:.2f}%")

    if train_overlap == 100 and val_overlap == 100:
        print("✓ Data splits are IDENTICAL")
    else:
        print("✗ Data splits are DIFFERENT!")
        print("  This could cause significant performance differences!")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    datasets = ['assistments12', 'assistments15']

    for dataset in datasets:
        check_data_split(dataset)