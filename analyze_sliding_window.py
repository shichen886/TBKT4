import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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
# Validate
# ------------------------------
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
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
    
    # Analyze training history
    history_path = os.path.join(args.savedir, f'{args.dataset}_training_history.json')
    print(f'\nAnalyzing training history from {history_path}')
    
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    val_auc = history_data['val_auc'][:args.epochs]    
    # Calculate sliding averages for multiple window sizes
    window_sizes = [2, 3, 5, 7, 10]
    results = {}
    
    for window_size in window_sizes:
        sliding_avgs = []
        for i in range(len(val_auc)):
            if i + 1 >= window_size:
                sliding_avg = np.mean(val_auc[i+1-window_size:i+1])
            else:
                sliding_avg = val_auc[i]
            sliding_avgs.append(sliding_avg)
        
        # Find best sliding average
        best_sliding_avg = max(sliding_avgs)
        best_sliding_epoch = sliding_avgs.index(best_sliding_avg) + 1  # 1-indexed
        
        results[window_size] = {
            'best_sliding_avg': best_sliding_avg,
            'best_sliding_epoch': best_sliding_epoch,
            'sliding_avgs': sliding_avgs,
            'val_auc_at_best_sliding_epoch': val_auc[best_sliding_epoch-1]
        }
    
    # Print results for all window sizes
    print(f'\nSliding Window Analysis for Multiple Window Sizes:')
    print('=' * 80)
    for window_size in window_sizes:
        result = results[window_size]
        print(f'\nWindow Size: {window_size}')
        print(f'  Best sliding average: {result["best_sliding_avg"]:.4f}')
        print(f'  Best sliding epoch: {result["best_sliding_epoch"]}')
        print(f'  Val AUC at best sliding epoch: {result["val_auc_at_best_sliding_epoch"]:.4f}')
    
    # Use the specified window size for detailed analysis
    window_size = args.window_size
    sliding_avgs = results[window_size]['sliding_avgs']
    best_sliding_avg = results[window_size]['best_sliding_avg']
    best_sliding_epoch = results[window_size]['best_sliding_epoch']
    
    print(f'\n\nDetailed Analysis for Window Size: {window_size}')
    print('=' * 80)
    
    # Compare with original best
    with open(os.path.join(args.savedir, 'config.json'), 'r') as f:
        original_config = json.load(f)
    
    original_best_val_auc = original_config.get('best_val_auc', 0)
    original_best_epoch = original_config.get('best_epoch', 0)
    original_test_auc = original_config.get('test_auc', 0)
    original_gap = original_config.get('generalization_gap', 0)
    
    print(f'\nOriginal Best (Single Epoch):')
    print(f'Best Val AUC: {original_best_val_auc:.4f} (Epoch {original_best_epoch})')
    print(f'Test AUC: {original_test_auc:.4f}')
    print(f'Generalization Gap: {original_gap:.4f}')
    
    # Load data
    data_dir = os.path.join('data', args.dataset)
    train_path = os.path.join(data_dir, 'preprocessed_data_train.csv')
    test_path = os.path.join(data_dir, 'preprocessed_data_test.csv')
    
    print(f'\nLoading data from {data_dir}')
    
    # Create datasets
    train_dataset = KTDataset(train_path, max_seq_len=args.max_seq_len)
    test_dataset = KTDataset(test_path, max_seq_len=args.max_seq_len)
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create dataloaders
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    
    # Get number of items and skills
    train_data = pd.read_csv(train_path, sep='\t')
    num_items = train_data['item_id'].max()
    num_skills = train_data['skill_id'].max()
    
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
    
    # Load model from best sliding epoch
    # Note: We need to retrain or save checkpoints for each epoch
    # For now, we'll use the original best model and compare
    
    print(f'\nNote: To use the model from epoch {best_sliding_epoch}, we need to save checkpoints for each epoch.')
    print(f'For now, we will evaluate the original best model and compare the validation performance.')
    
    # Load original best model
    model.load_state_dict(torch.load(os.path.join(args.savedir, f'{args.dataset}_best.pt')))
    
    # Evaluate on validation set
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_auc_eval, val_rmse = validate(model, val_loader, criterion, device)
    print(f'\nOriginal Best Model on Validation Set:')
    print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc_eval:.4f}, Val RMSE: {val_rmse:.4f}')
    
    # Evaluate on test set
    test_loss, test_auc, test_rmse = validate(model, test_loader, criterion, device)
    print(f'\nOriginal Best Model on Test Set:')
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test RMSE: {test_rmse:.4f}')
    
    # Save analysis results
    analysis_path = os.path.join(args.savedir, 'sliding_window_analysis.json')
    analysis = {
        'window_sizes': window_sizes,
        'results': results,
        'selected_window_size': args.window_size,
        'best_sliding_avg': best_sliding_avg,
        'best_sliding_epoch': best_sliding_epoch,
        'val_auc_at_best_sliding_epoch': val_auc[best_sliding_epoch-1],
        'original_best_val_auc': original_best_val_auc,
        'original_best_epoch': original_best_epoch,
        'original_test_auc': original_test_auc,
        'original_gap': original_gap,
        'sliding_avgs': sliding_avgs,
        'val_auc_history': val_auc,
    }
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=4)
    
    print(f'\nAnalysis results saved to {analysis_path}')
    
    # Print recommendations
    print(f'\nRecommendations:')
    
    # Find most recommended window size
    recommended_window_size = None
    recommended_epoch = None
    best_overall_avg = 0
    
    for window_size in window_sizes:
        result = results[window_size]
        if result['best_sliding_avg'] > best_overall_avg:
            best_overall_avg = result['best_sliding_avg']
            recommended_window_size = window_size
            recommended_epoch = result['best_sliding_epoch']
    
    print(f'- Recommended window size: {recommended_window_size} (best sliding avg: {best_overall_avg:.4f} at epoch {recommended_epoch})')
    
    if recommended_epoch < original_best_epoch:
        print(f'- Using window size {recommended_window_size} suggests stopping earlier (epoch {recommended_epoch} vs {original_best_epoch})')
        print(f'- This could reduce overfitting and improve generalization')
    elif recommended_epoch > original_best_epoch:
        print(f'- Using window size {recommended_window_size} suggests training longer (epoch {recommended_epoch} vs {original_best_epoch})')
        print(f'- This could potentially improve performance')
    else:
        print(f'- The sliding window and single epoch methods agree on the best epoch')
    
    # Compare different window sizes
    print(f'\nComparison of Window Sizes:')
    print(f'{"Window Size":<15} {"Best Sliding Avg":<18} {"Best Epoch":<12} {"Diff from Original":<20}')
    print('-' * 65)
    for window_size in window_sizes:
        result = results[window_size]
        epoch_diff = result['best_sliding_epoch'] - original_best_epoch
        print(f'{window_size:<15} {result["best_sliding_avg"]:<18.4f} {result["best_sliding_epoch"]:<12} {epoch_diff:+20}')
    
    # Print sliding average table for selected window size
    print(f'\nSliding Average Table (window_size={args.window_size}):')
    print(f'{"Epoch":<8} {"Val AUC":<12} {"Sliding Avg":<12} {"Diff":<10}')
    print('-' * 42)
    for i, (va, sa) in enumerate(zip(val_auc, sliding_avgs), 1):
        diff = sa - va
        print(f'{i:<8} {va:<12.4f} {sa:<12.4f} {diff:<10.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze training history with sliding window early stopping')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--savedir', type=str, default='save/tsakt-linear-rope-qk-regularized')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--tensor_rank', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=5, help='Sliding window size for early stopping (default: 5)')
    parser.add_argument('--drop_prob', type=float, default=0.3)
    
    args = parser.parse_args()
    
    main(args)
