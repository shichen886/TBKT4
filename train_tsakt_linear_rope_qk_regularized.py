import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
# Train epoch
# ------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc='Training'):
        item_ids = batch['item_ids'].to(device)
        skill_ids = batch['skill_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
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
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        all_preds.append(preds.squeeze().detach().cpu().numpy())
        all_labels.append(target_labels.cpu().numpy())
    
    # Compute global AUC
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = compute_auc(all_preds, all_labels)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, auc


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
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    data_dir = os.path.join('data', args.dataset)
    train_path = os.path.join(data_dir, 'preprocessed_data_train.csv')
    test_path = os.path.join(data_dir, 'preprocessed_data_test.csv')
    
    print(f'Loading data from {data_dir}')
    
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
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
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
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Get number of items and skills
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
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping
    best_val_auc = 0
    patience_counter = 0
    best_epoch = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss': [],
        'val_auc': [],
        'val_rmse': [],
    }
    
    # Create save directory
    os.makedirs(args.savedir, exist_ok=True)
    
    # Training loop
    print(f'\nTraining for {args.num_epochs} epochs...')
    print(f'Increased regularization: dropout={args.drop_prob}, weight_decay={args.weight_decay}')
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch + 1}/{args.num_epochs}')
        
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        
        # Validate
        val_loss, val_auc, val_rmse = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_rmse'].append(val_rmse)
        
        print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val RMSE: {val_rmse:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.savedir, f'{args.dataset}_best.pt'))
            print(f'New best model saved! Val AUC: {best_val_auc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.savedir, f'{args.dataset}_best.pt')))
    
    # Test
    print(f'\nTesting...')
    test_loss, test_auc, test_rmse = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test RMSE: {test_rmse:.4f}')
    
    # Save training history
    history_path = os.path.join(args.savedir, f'{args.dataset}_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save config
    config_path = os.path.join(args.savedir, 'config.json')
    with open(config_path, 'w') as f:
        config = vars(args)
        config['test_auc'] = test_auc
        config['test_rmse'] = test_rmse
        config['best_val_auc'] = best_val_auc
        config['best_epoch'] = best_epoch
        config['generalization_gap'] = best_val_auc - test_auc
        json.dump(config, f, indent=4)
    
    print(f'\nTraining completed!')
    print(f'Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch})')
    print(f'Test AUC: {test_auc:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'Generalization Gap: {best_val_auc - test_auc:.4f}')
    print(f'Results saved to {args.savedir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TSAKT-Linear with RoPE on Q/K (Increased Regularization)')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--savedir', type=str, default='save/tsakt-linear-rope-qk-regularized')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--tensor_rank', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)  # Increased from 0.0001 to 0.001
    parser.add_argument('--drop_prob', type=float, default=0.3)  # Increased from 0.1 to 0.3
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    
    main(args)
