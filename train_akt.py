import argparse
import pandas as pd
from random import shuffle

import torch.optim.lr_scheduler
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np
import os

from model_akt import AKT
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_data(df, max_length, train_split=0.8, randomize=True):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training
    """
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

    # Chunk sequences
    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    if randomize:
        shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def prepare_batches(data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch
    Output:
        batches (list of lists of torch Tensor)
    """
    if randomize:
        shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          if (seqs[0] is not None) else None for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches


def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def compute_rmse(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    rmse = math.sqrt(torch.mean((preds - labels) ** 2, dim=0, keepdim=False))
    return rmse


def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)


def train(train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, grad_clip, scheduler):
    """Train AKT model.

    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0

    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in train_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)

            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            loss = compute_loss(preds, labels.to(device), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc = compute_auc(preds, labels)
            train_rmse = compute_rmse(preds, labels)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(f"Step {step}, Train Loss: {loss.item():.4f}, Train AUC: {train_auc:.4f}, Train RMSE: {train_rmse:.4f}")

        # Validation
        val_losses = []
        val_aucs = []
        val_rmses = []
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)

            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            loss = compute_loss(preds, labels.to(device), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            val_auc = compute_auc(preds, labels)
            val_rmse = compute_rmse(preds, labels)

            val_losses.append(loss.item())
            val_aucs.append(val_auc)
            val_rmses.append(val_rmse)

        avg_val_loss = np.mean(val_losses)
        avg_val_auc = np.mean(val_aucs)
        avg_val_rmse = np.mean(val_rmses)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val AUC: {avg_val_auc:.4f}, Val RMSE: {avg_val_rmse:.4f}\n")

        if scheduler:
            scheduler.step(avg_val_auc)

        stop = saver.save(avg_val_auc, model)
        if stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model


def main():
    parser = argparse.ArgumentParser(description='Train AKT model')
    parser.add_argument('--dataset', type=str, default='assistments09',
                       help='Dataset name')
    parser.add_argument('--embed_size', type=int, default=200,
                       help='Embedding size')
    parser.add_argument('--num_attn_layers', type=int, default=2,
                       help='Number of attention layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--drop_prob', type=float, default=0.2,
                       help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--max_seq_len', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                       help='Gradient clipping')
    parser.add_argument('--max_pos', type=int, default=10,
                       help='Maximum position for positional encoding')
    parser.add_argument('--savedir', type=str, default='save/akt',
                       help='Save directory')

    args = parser.parse_args()

    # Load data
    data_file = f'data/{args.dataset}/preprocessed_data.csv'
    if not os.path.exists(data_file):
        data_file = f'data/{args.dataset}/preprocessed_train_data.csv'
    
    df = pd.read_csv(data_file, sep='\t')
    print(f"Loaded {len(df)} records from {data_file}")

    # Get data
    train_data, val_data = get_data(df, args.max_seq_len)

    # Model parameters
    # 因为在get_data中item_id和skill_id被加了1，所以需要调整嵌入层大小
    num_items = int(df['item_id'].max() + 2)  # +2 因为原始是0开始，加1后变成1开始
    num_skills = int(df['skill_id'].max() + 2)

    print(f"Num items: {num_items}, Num skills: {num_skills}")

    # Create model
    model = AKT(
        num_items=num_items,
        num_skills=num_skills,
        embed_size=args.embed_size,
        num_attn_layers=args.num_attn_layers,
        num_heads=args.num_heads,
        drop_prob=args.drop_prob,
        max_pos=args.max_pos
    ).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Saver
    param_str = f"{args.dataset},batch_size={args.batch_size},max_length={args.max_seq_len},max_pos={args.max_pos}"
    saver = Saver(args.savedir, param_str, patience=10)

    # Train
    print("\nStarting training...")
    model = train(train_data, val_data, model, optimizer, None, saver,
                args.num_epochs, args.batch_size, args.grad_clip, scheduler)

    print("\nTraining completed!")


if __name__ == '__main__':
    main()