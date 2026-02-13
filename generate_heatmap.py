import argparse
import math
import os

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence

from model_sakt_draw import SAKT
from utils import *
from draw_heatmap import heatmap


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate attention heatmaps.')
    parser.add_argument('--savedir', type=str, default='save/sakt')
    parser.add_argument('--dataset', type=str, default='assistments09')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=40)
    parser.add_argument('--num_attn_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=10)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_heatmaps', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    test_data = get_data(test_df, args.max_length)
    test_batches = prepare_batches(test_data, args.batch_size)

    num_items = int(test_df["item_id"].max() + 1)
    num_skills = int(test_df["skill_id"].max() + 1)

    model = SAKT(num_items, num_skills, args.embed_size, args.num_attn_layers, args.num_heads,
                  args.encode_pos, args.max_pos, args.drop_prob).to(device)

    model_path = os.path.join(args.savedir, f'{args.dataset},batch_size=128,max_length=200,encode_pos=False,max_pos={args.max_pos}')
    print(f"Loading model from: {model_path}")
    loaded_model = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded_model.state_dict())

    os.makedirs('heatmaps', exist_ok=True)

    model.eval()
    i = 1
    with torch.no_grad():
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in test_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)

            preds, weights = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            for j in range(min(weights.size(0), args.num_heatmaps)):
                weight = weights[j].cpu().numpy()
                heatmap(weight, i)
                print(f"Generated heatmap {i}")
                i += 1
            
            if i > args.num_heatmaps:
                break

    print(f"Successfully generated {args.num_heatmaps} attention heatmaps in 'heatmaps' directory!")
