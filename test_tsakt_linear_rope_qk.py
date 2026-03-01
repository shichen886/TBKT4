import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from model_tsakt_linear_rope_qk import TSAKT_Linear_RoPE_QK


# ------------------------------
# Dataset
# ------------------------------
class KTDataset(Dataset):
    def __init__(self, data_path, max_seq_len=200):
        self.data = pd.read_csv(data_path, sep='\t')
        self.max_seq_len = max_seq_len
        
        # Group by user_id
        self.user_data = self.data.groupby('user_id').apply(
            lambda x: x.sort_values('timestamp').reset_index(drop=True)
        ).reset_index(drop=True)
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        
        for _, user_seq in tqdm(self.user_data.groupby('user_id'), desc='Creating sequences'):
            items = user_seq['item_id'].values
            skills = user_seq['skill_id'].values
            labels = user_seq['correct'].values
            
            # Create sliding windows
            for i in range(1, len(items)):
                seq_len = min(i, self.max_seq_len)
                
                item_seq = items[max(0, i-seq_len):i]
                skill_seq = skills[max(0, i-seq_len):i]
                label_seq = labels[max(0, i-seq_len):i]
                
                # Pad if necessary
                if len(item_seq) < self.max_seq_len:
                    pad_len = self.max_seq_len - len(item_seq)
                    item_seq = np.pad(item_seq, (pad_len, 0), constant_values=0)
                    skill_seq = np.pad(skill_seq, (pad_len, 0), constant_values=0)
                    label_seq = np.pad(label_seq, (pad_len, 0), constant_values=-1)
                
                sequences.append({
                    'item_ids': item_seq,
                    'skill_ids': skill_seq,
                    'labels': label_seq,
                    'target': labels[i],
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'item_ids': torch.LongTensor(seq['item_ids']),
            'skill_ids': torch.LongTensor(seq['skill_ids']),
            'labels': torch.FloatTensor(seq['labels']),
            'target': torch.FloatTensor([seq['target']]),
        }


# ------------------------------
# Collate function
# ------------------------------
def collate_fn(batch):
    item_ids = torch.stack([item['item_ids'] for item in batch])
    skill_ids = torch.stack([item['skill_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    # Create mask (1 = valid, 0 = pad)
    mask = (labels >= 0).float()
    
    return {
        'item_ids': item_ids,
        'skill_ids': skill_ids,
        'labels': labels,
        'targets': targets,
        'mask': mask,
    }


# ------------------------------
# Compute AUC
# ------------------------------
def compute_auc(preds, labels):
    """
    Compute AUC with single-class fallback
    
    Args:
        preds: predictions [N]
        labels: labels [N]
    
    Returns:
        auc: AUC score
    """
    # Filter out padding labels
    valid_mask = labels >= 0
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
    # Check if we have both classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # Single class: return 0.5 (random guessing)
        return 0.5
    
    try:
        auc = roc_auc_score(labels, preds)
        return auc
    except:
        return 0.5


# ------------------------------
# Compute RMSE
# ------------------------------
def compute_rmse(preds, labels):
    """
    Compute RMSE
    
    Args:
        preds: predictions [N]
        labels: labels [N]
    
    Returns:
        rmse: RMSE score
    """
    # Filter out padding labels
    valid_mask = labels >= 0
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
    mse = torch.mean((preds - labels) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


# ------------------------------
# Test
# ------------------------------
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            item_ids = batch['item_ids'].to(device)
            skill_ids = batch['skill_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            
            outputs = model(item_ids, skill_ids, mask)
            
            # Get predictions for last position
            preds = outputs[:, -1, :]
            
            # Get target labels (last valid position)
            batch_size = labels.shape[0]
            target_labels = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                valid_positions = torch.where(labels[i] >= 0)[0]
                if len(valid_positions) > 0:
                    target_labels[i] = labels[i, valid_positions[-1]]
            
            loss = criterion(preds.squeeze(), target_labels)
            total_loss += loss.item()
            
            all_preds.append(preds.squeeze().cpu().numpy())
            all_labels.append(target_labels.cpu().numpy())
    
    # Compute global AUC
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = compute_auc(all_preds, all_labels)
    rmse = compute_rmse(torch.tensor(all_preds), torch.tensor(all_labels))
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, auc, rmse


# ------------------------------
# Main
# ------------------------------
def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    data_dir = os.path.join('data', args.dataset)
    test_path = os.path.join(data_dir, 'preprocessed_data_test.csv')
    
    print(f'Loading data from {data_dir}')
    
    # Create dataset
    test_dataset = KTDataset(test_path, max_seq_len=args.max_seq_len)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    
    print(f'Test samples: {len(test_dataset)}')
    
    # Get number of items and skills
    train_path = os.path.join(data_dir, 'preprocessed_data_train.csv')
    train_data = pd.read_csv(train_path, sep='\t')
    num_items = train_data['item_id'].max()
    num_skills = train_data['skill_id'].max()
    
    print(f'Number of items: {num_items}')
    print(f'Number of skills: {num_skills}')
    
    # Create model
    model = TSAKT_Linear_RoPE_QK(
        num_items=num_items,
        num_skills=num_skills,
        embed_size=args.embed_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tensor_rank=args.tensor_rank,
        max_len=args.max_seq_len,
        drop_prob=args.drop_prob,
    ).to(device)
    
    # Load best model
    model_path = os.path.join(args.savedir, f'{args.dataset}_best.pt')
    model.load_state_dict(torch.load(model_path))
    print(f'Loaded model from {model_path}')
    
    # Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Test
    print(f'\nTesting...')
    test_loss, test_auc, test_rmse = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test RMSE: {test_rmse:.4f}')
    
    # Save results
    results = {
        'dataset': args.dataset,
        'model': 'TSAKT-Linear-RoPE-QK',
        'test_loss': test_loss,
        'test_auc': test_auc,
        'test_rmse': test_rmse,
        'embed_size': args.embed_size,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'tensor_rank': args.tensor_rank,
        'max_seq_len': args.max_seq_len,
        'batch_size': args.batch_size,
    }
    
    results_path = os.path.join(args.savedir, f'{args.dataset}_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'\nResults saved to {results_path}')
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test TSAKT-Linear with RoPE on Q/K')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--savedir', type=str, default='save/tsakt-linear-rope-qk')
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--tensor_rank', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    
    args = parser.parse_args()
    
    results = main(args)
