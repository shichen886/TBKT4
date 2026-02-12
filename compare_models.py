import argparse
import math
import os

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence

from model_sakt import SAKT
from model_tsakt import TSAKT
from utils import *


def get_data(df, max_length):
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    return data


def prepare_batches(data, batch_size):
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          if (seqs[0] is not None) else None for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)
        batches.append([*inputs_and_ids, labels])
    return batches


def test_model(model, test_batches, device):
    model.eval()
    test_preds = np.empty(0)
    test_labels = np.empty(0)

    with torch.no_grad():
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in test_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)

            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            preds = torch.sigmoid(preds).cpu()
            
            batch_preds = preds[labels >= 0].flatten().numpy()
            batch_labels = labels[labels >= 0].float().numpy()
            
            test_preds = np.concatenate([test_preds, batch_preds])
            test_labels = np.concatenate([test_labels, batch_labels])

    test_auc = roc_auc_score(test_labels, test_preds)
    test_rmse = np.sqrt(((test_preds - test_labels) ** 2).mean())

    return test_auc, test_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare SAKT and TSAKT models.')
    parser.add_argument('--dataset', type=str, default='assistments09')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    test_data = get_data(test_df, args.max_length)
    test_batches = prepare_batches(test_data, args.batch_size)

    num_items = int(test_df["item_id"].max() + 1)
    num_skills = int(test_df["skill_id"].max() + 1)

    results = {}

    print("\n" + "="*70)
    print("Testing SAKT Model...")
    print("="*70)

    sakt_model = SAKT(num_items, num_skills, 40, 2, 5, False, 10, 0.2).to(device)
    sakt_path = os.path.join('save/sakt', f'{args.dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=10')
    print(f"Loading model from: {sakt_path}")
    loaded_model = torch.load(sakt_path, map_location=device, weights_only=False)
    sakt_model.load_state_dict(loaded_model.state_dict())

    sakt_auc, sakt_rmse = test_model(sakt_model, test_batches, device)
    results['SAKT'] = {'AUC': sakt_auc, 'RMSE': sakt_rmse}

    print(f"SAKT Test AUC: {sakt_auc:.4f}")
    print(f"SAKT Test RMSE: {sakt_rmse:.4f}")

    print("\n" + "="*70)
    print("Testing TSAKT Model...")
    print("="*70)

    tsakt_model = TSAKT(num_items, num_skills, 60, 2, 5, False, 5, 0.2, 3).to(device)
    tsakt_path = os.path.join('save/tsakt', f'{args.dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
    print(f"Loading model from: {tsakt_path}")
    loaded_model = torch.load(tsakt_path, map_location=device, weights_only=False)
    tsakt_model.load_state_dict(loaded_model.state_dict())

    tsakt_auc, tsakt_rmse = test_model(tsakt_model, test_batches, device)
    results['TSAKT'] = {'AUC': tsakt_auc, 'RMSE': tsakt_rmse}

    print(f"TSAKT Test AUC: {tsakt_auc:.4f}")
    print(f"TSAKT Test RMSE: {tsakt_rmse:.4f}")

    print("\n" + "="*70)
    print("Model Comparison Results")
    print("="*70)
    print(f"{'Model':<10} {'AUC':<10} {'RMSE':<10} {'AUC Improvement':<20} {'RMSE Improvement':<20}")
    print("-"*70)
    print(f"{'SAKT':<10} {sakt_auc:.4f}    {sakt_rmse:.4f}    {'-':<20} {'-':<20}")
    print(f"{'TSAKT':<10} {tsakt_auc:.4f}    {tsakt_rmse:.4f}    {f'+{tsakt_auc - sakt_auc:.4f}':<20} {f'-{sakt_rmse - tsakt_rmse:.4f}':<20}")
    print("="*70)

    if tsakt_auc > sakt_auc:
        print("\n✓ TSAKT outperforms SAKT in AUC!")
    if tsakt_rmse < sakt_rmse:
        print("✓ TSAKT outperforms SAKT in RMSE!")
