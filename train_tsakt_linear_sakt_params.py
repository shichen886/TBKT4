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
import math
import numpy as np
import os
from tqdm import tqdm

from model_tsakt_linear import TSAKT_Linear
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
        
        # Pad sequences
        item_ids = pad_sequence([seq for seq in seq_lists[0]], batch_first=True, padding_value=0)
        skill_ids = pad_sequence([seq for seq in seq_lists[1]], batch_first=True, padding_value=0)
        labels = pad_sequence([seq for seq in seq_lists[2]], batch_first=True, padding_value=-1)
        
        # Create mask
        mask = (labels >= 0).float()
        
        batches.append((item_ids, skill_ids, labels, mask))

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


def train_epoch(model, batches, optimizer, criterion):
    model.train()
    total_loss = 0
    total_auc = 0
    total_rmse = 0
    total_count = 0
    
    pbar = tqdm(batches, desc='Training')
    for batch in pbar:
        item_ids, skill_ids, labels, mask = batch
        
        item_ids = item_ids.to(device)
        skill_ids = skill_ids.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(item_ids, skill_ids, mask)
        
        loss = criterion(outputs.squeeze(-1), labels, mask)
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute metrics
        preds = torch.sigmoid(outputs.squeeze(-1))
        auc = compute_auc(preds.detach().cpu(), labels.cpu())
        rmse = compute_rmse(preds.detach().cpu(), labels.cpu())
        
        total_auc += auc
        total_rmse += rmse
        total_count += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'auc': f'{auc:.4f}', 'rmse': f'{rmse:.4f}'})
    
    return total_loss / total_count, total_auc / total_count, total_rmse / total_count


def validate(model, batches, criterion):
    model.eval()
    total_loss = 0
    total_auc = 0
    total_rmse = 0
    total_count = 0
    
    with torch.no_grad():
        for batch in tqdm(batches, desc='Validation'):
            item_ids, skill_ids, labels, mask = batch
            
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            outputs = model(item_ids, skill_ids, mask)
            
            loss = criterion(outputs.squeeze(-1), labels, mask)
            
            total_loss += loss.item()
            
            # Compute metrics
            preds = torch.sigmoid(outputs.squeeze(-1))
            auc = compute_auc(preds.detach().cpu(), labels.cpu())
            rmse = compute_rmse(preds.detach().cpu(), labels.cpu())
            
            total_auc += auc
            total_rmse += rmse
            total_count += 1
    
    return total_loss / total_count, total_auc / total_count, total_rmse / total_count


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


def main(args):
    print(f"Training TSAKT-Linear on {args.dataset}")
    print(f"Device: {device}")
    print(f"Parameters: embed_size={args.embed_size}, num_layers={args.num_layers}, num_heads={args.num_heads}, tensor_rank={args.tensor_rank}")
    
    data_path = os.path.join('data', args.dataset, 'preprocessed_data.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path, sep="\t")
    num_items = int(df["item_id"].max() + 1)
    num_skills = int(df["skill_id"].max() + 1)
    
    print(f"Num items: {num_items}, Num skills: {num_skills}")
    print(f"Total sequences: {len(df.groupby('user_id'))}")
    
    # Prepare data
    train_data, val_data = get_data(df, max_length=args.max_seq_len, train_split=0.8, randomize=True)
    
    train_batches = prepare_batches(train_data, batch_size=args.batch_size, randomize=True)
    val_batches = prepare_batches(val_data, batch_size=args.batch_size, randomize=False)
    
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")
    
    # Create model with SAKT-compatible parameters
    model = TSAKT_Linear(num_items, num_skills, embed_size=args.embed_size, 
                       num_layers=args.num_layers, num_heads=args.num_heads,
                       tensor_rank=args.tensor_rank, max_len=args.max_seq_len, 
                       drop_prob=args.drop_prob).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    def masked_bce_loss(preds, labels, mask):
        loss = criterion(preds, labels.float())
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    os.makedirs(args.savedir, exist_ok=True)
    
    config_path = os.path.join(args.savedir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved config to {config_path}")
    
    best_val_loss = float('inf')
    best_val_auc = 0
    
    training_history = {
        'train_loss': [],
        'train_auc': [],
        'train_rmse': [],
        'val_loss': [],
        'val_auc': [],
        'val_rmse': [],
        'epochs': []
    }
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        train_loss, train_auc, train_rmse = train_epoch(model, train_batches, optimizer, masked_bce_loss)
        val_loss, val_auc, val_rmse = validate(model, val_batches, masked_bce_loss)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val RMSE: {val_rmse:.4f}")
        
        training_history['train_loss'].append(train_loss)
        training_history['train_auc'].append(train_auc)
        training_history['train_rmse'].append(train_rmse)
        training_history['val_loss'].append(val_loss)
        training_history['val_auc'].append(val_auc)
        training_history['val_rmse'].append(val_rmse)
        training_history['epochs'].append(epoch + 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            save_path = os.path.join(args.savedir, f'{args.dataset}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_rmse': val_rmse,
            }, save_path)
            print(f"Saved best model to {save_path} (Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f})")
    
    print(f"\nTraining completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}, Best Val AUC: {best_val_auc:.4f}")
    
    training_history['best_val_loss'] = best_val_loss
    training_history['best_val_auc'] = best_val_auc
    
    history_path = os.path.join(args.savedir, f"{args.dataset}_training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=4)
    print(f"Saved training history to {history_path}")
    
    plot_training_curves(training_history, args.dataset, args.savedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TSAKT-Linear with SAKT-compatible parameters.')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--savedir', type=str, default='save/tsakt-linear-sakt-params')
    parser.add_argument('--embed_size', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--tensor_rank', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    
    args = parser.parse_args()
    
    main(args)
