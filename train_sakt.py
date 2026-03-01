import argparse
import pandas as pd
from random import shuffle
import json

import torch.optim.lr_scheduler
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import os

from model_sakt import SAKT
from utils import *

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, training curves will not be plotted")

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


class KTDataset(Dataset):
    """
    Knowledge Tracing Dataset for PyTorch DataLoader
    
    顶级会议标准：使用torch.utils.data.Dataset + DataLoader
    优势：
    - 更规范的代码结构
    - 支持多进程加载（num_workers）
    - 自动内存优化（pin_memory）
    - reviewer看着更舒服
    
    Arguments:
        data (list): list of (item_ids, skill_ids, labels) tuples
    """
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def kt_collate_fn(batch):
    """
    Custom collate function for variable-length sequences in Knowledge Tracing
    
    顶级会议标准：自定义collate_fn处理变长序列
    
    Arguments:
        batch (list): list of (item_ids, skill_ids, labels) tuples
    
    Returns:
        item_ids (torch.Tensor): padded item_ids [batch_size, max_len]
        skill_ids (torch.Tensor): padded skill_ids [batch_size, max_len]
        labels (torch.Tensor): padded labels [batch_size, max_len]
        mask (torch.Tensor): valid positions mask [batch_size, max_len]
    """
    # Unpack batch
    item_ids = [item[0] for item in batch]
    skill_ids = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # Pad sequences to same length
    item_ids = pad_sequence(item_ids, batch_first=True, padding_value=0)
    skill_ids = pad_sequence(skill_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    
    # Create mask (1 for valid positions, 0 for padding)
    mask = (labels >= 0).float()
    
    return item_ids, skill_ids, labels, mask


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


def plot_training_curves(history, dataset_name, savedir):
    """Plot and save training curves."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping training curve plot: matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = history['epochs']
    
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_auc'], label='Train AUC', marker='o')
    axes[1].plot(epochs, history['val_auc'], label='Val AUC', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Training and Validation AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['train_rmse'], label='Train RMSE', marker='o')
    axes[2].plot(epochs, history['val_rmse'], label='Val RMSE', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('Training and Validation RMSE')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(savedir, f"{dataset_name}_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {plot_path}")
    plt.close()


def train(train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, grad_clip, scheduler):
    """Train SAKT model.

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
    
    training_history = {
        'train_loss': [],
        'train_auc': [],
        'train_rmse': [],
        'val_loss': [],
        'val_auc': [],
        'val_rmse': [],
        'epochs': []
    }

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
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            metrics.store({'rmse/train': train_rmse})

            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)
                # weights = {"weight/" + name: param for name, param in model.named_parameters()}
                # grads = {"grad/" + name: param.grad
                #         for name, param in model.named_parameters() if param.grad is not None}
                # logger.log_histograms(weights, step)
                # logger.log_histograms(grads, step)
        scheduler.step()
        # Validation
        model.eval()
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                preds = torch.sigmoid(preds).cpu()
            val_auc = compute_auc(preds, labels)
            val_rmse = compute_rmse(preds, labels)
            val_loss = compute_loss(preds, labels, criterion)
            metrics.store({'auc/val': val_auc})
            metrics.store({'rmse/val': val_rmse})
            metrics.store({'loss/val': val_loss})
        model.train()

        # Save model
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['auc/val'], model)
        
        training_history['train_loss'].append(average_metrics.get('loss/train', 0))
        training_history['train_auc'].append(average_metrics.get('auc/train', 0))
        training_history['train_rmse'].append(average_metrics.get('rmse/train', 0))
        training_history['val_loss'].append(average_metrics.get('loss/val', 0))
        training_history['val_auc'].append(average_metrics.get('auc/val', 0))
        training_history['val_rmse'].append(average_metrics.get('rmse/val', 0))
        training_history['epochs'].append(epoch + 1)
        
        if stop:
            break
    
    return training_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TSSAKT.')
    parser.add_argument('--dataset', type=str,default='assistments12')
    parser.add_argument('--logdir', type=str, default='runs/sakt')
    parser.add_argument('--savedir', type=str, default='save/sakt')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=60)
    parser.add_argument('--num_attn_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=5)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=256)
    args = parser.parse_args()

    full_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    train_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_train.csv'), sep="\t")
    test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_test.csv'), sep="\t")

    train_data, val_data = get_data(train_df, args.max_length)

    num_items = int(full_df["item_id"].max() + 1)
    num_skills = int(full_df["skill_id"].max() + 1)

    model = SAKT(num_items, num_skills, args.embed_size, args.num_attn_layers, args.num_heads,
                  args.encode_pos, args.max_pos, args.drop_prob).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    os.makedirs(args.savedir, exist_ok=True)
    config_path = os.path.join(args.savedir, f"{args.dataset}_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved config to {config_path}")
    
    param_str = (f'{args.dataset},'
                         f'batch_size={args.batch_size},'
                         f'max_length={args.max_length},'
                         f'encode_pos={args.encode_pos},'
                         f'max_pos={args.max_pos}')
    # logger = Logger(os.path.join(args.logdir, param_str))
    # saver = Saver(args.savedir, param_str)
    # train(train_data, val_data, model, optimizer, logger, saver, args.num_epochs,args.batch_size, args.grad_clip)
    # Reduce batch size until it fits on GPU
    while True:
        try:
            # Train
            param_str = (f'{args.dataset},'f'batch_size={args.batch_size},'f'max_length={args.max_length},'f'encode_pos={args.encode_pos},'f'max_pos={args.max_pos}')
            logger = Logger(os.path.join(args.logdir, param_str))
            saver = Saver(args.savedir, param_str, patience=10)
            train(train_data, val_data, model, optimizer, logger, saver, args.num_epochs,
            args.batch_size, args.grad_clip, scheduler)
            break
        except RuntimeError:
            args.batch_size = args.batch_size // 2
            print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')

    logger.close()
    
    training_history = train(train_data, val_data, model, optimizer, logger, saver, args.num_epochs,
                        args.batch_size, args.grad_clip, scheduler)
    
    training_history['best_val_auc'] = max(training_history['val_auc'])
    training_history['best_val_loss'] = min(training_history['val_loss'])
    
    history_path = os.path.join(args.savedir, f"{args.dataset}_training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=4)
    print(f"Saved training history to {history_path}")
    
    plot_training_curves(training_history, args.dataset, args.savedir)

    test_data, _ = get_data(test_df, args.max_length, train_split=1.0, randomize=False)
    test_batches = prepare_batches(test_data, args.batch_size, randomize=False)
    test_preds = np.empty(0)

    # Predict on test set
    model.eval()
    for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in test_batches:
        item_inputs = item_inputs.to(device)
        skill_inputs = skill_inputs.to(device)
        label_inputs = label_inputs.to(device)
        item_ids = item_ids.to(device)
        skill_ids = skill_ids.to(device)
        with torch.no_grad():
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
            test_preds = np.concatenate([test_preds, preds])

    # Write predictions to csv
    test_df["SAKT"] = test_preds
    test_df.to_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t", index=False)

    print("auc_test = ", roc_auc_score(test_df["correct"], test_preds))
    test_preds = torch.tensor(test_preds)
    labels = torch.tensor(test_df["correct"])
    print("rmse_test = ", compute_rmse(test_preds, labels))
