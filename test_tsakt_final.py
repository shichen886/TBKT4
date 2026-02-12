import argparse
import math
import os

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.nn.utils.rnn import pad_sequence

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


def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def compute_rmse(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    rmse = np.sqrt(((preds - labels) ** 2).mean())
    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TSAKT.')
    parser.add_argument('--savedir', type=str, default='save/tsakt')
    parser.add_argument('--dataset', type=str, default='assistments09')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=60)
    parser.add_argument('--num_attn_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=5)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tensor_rank', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    test_data = get_data(test_df, args.max_length)
    test_batches = prepare_batches(test_data, args.batch_size)

    num_items = int(test_df["item_id"].max() + 1)
    num_skills = int(test_df["skill_id"].max() + 1)

    model = TSAKT(num_items, num_skills, args.embed_size, args.num_attn_layers, args.num_heads,
                  args.encode_pos, args.max_pos, args.drop_prob, args.tensor_rank).to(device)

    model_path = os.path.join(args.savedir, f'{args.dataset},batch_size=128,max_length=200,encode_pos=False,max_pos={args.max_pos},tensor_rank={args.tensor_rank}')
    print(f"Loading model from: {model_path}")
    loaded_model = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded_model.state_dict())

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

    print("\n" + "="*50)
    print(f"TSAKT Model Test Results on {args.dataset}")
    print("="*50)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print("="*50)
