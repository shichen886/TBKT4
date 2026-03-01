import argparse
import pandas as pd
from random import shuffle
import json
import random

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


def set_seed(seed=42):
    """
    Fix random seed for reproducibility
    论文级必须：确保实验可复现
    
    注意：deterministic=True 会显著降低训练速度，但确保结果可复现
    论文中需要说明：We enabled deterministic CUDA operations for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(df, max_length, train_split=0.8, test_split=0.5, randomize=True, seed=42):
    """
    Extract sequences from dataframe with proper train/val/test split.
    
    论文级修复：先按user_id划分train/val/test，再做chunk
    避免同一用户的不同chunk同时出现在不同split中
    
    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training (default: 0.8)
        test_split (float): proportion of val data to use for test (default: 0.5)
                        val size = (1 - train_split) * test_split
        randomize (bool): whether to shuffle data
        seed (int): random seed for reproducibility
    """
    # Group by user_id
    user_ids = df["user_id"].unique()
    
    # Shuffle users (not chunks!)
    if randomize:
        np.random.seed(seed)
        np.random.shuffle(user_ids)
    
    # Split users into train and (val+test)
    train_size = int(train_split * len(user_ids))
    train_user_ids = user_ids[:train_size]
    val_test_user_ids = user_ids[train_size:]
    
    # Split val+test into val and test
    test_size = int(test_split * len(val_test_user_ids))
    val_user_ids = val_test_user_ids[:test_size]
    test_user_ids = val_test_user_ids[test_size:]
    
    # Extract sequences for train users
    train_item_ids = []
    train_skill_ids = []
    train_labels = []
    
    for user_id in train_user_ids:
        u_df = df[df["user_id"] == user_id]
        train_item_ids.append(torch.tensor(u_df["item_id"].values, dtype=torch.long))
        train_skill_ids.append(torch.tensor(u_df["skill_id"].values, dtype=torch.long))
        train_labels.append(torch.tensor(u_df["correct"].values, dtype=torch.long))
    
    # Extract sequences for val users
    val_item_ids = []
    val_skill_ids = []
    val_labels = []
    
    for user_id in val_user_ids:
        u_df = df[df["user_id"] == user_id]
        val_item_ids.append(torch.tensor(u_df["item_id"].values, dtype=torch.long))
        val_skill_ids.append(torch.tensor(u_df["skill_id"].values, dtype=torch.long))
        val_labels.append(torch.tensor(u_df["correct"].values, dtype=torch.long))
    
    # Extract sequences for test users
    test_item_ids = []
    test_skill_ids = []
    test_labels = []
    
    for user_id in test_user_ids:
        u_df = df[df["user_id"] == user_id]
        test_item_ids.append(torch.tensor(u_df["item_id"].values, dtype=torch.long))
        test_skill_ids.append(torch.tensor(u_df["skill_id"].values, dtype=torch.long))
        test_labels.append(torch.tensor(u_df["correct"].values, dtype=torch.long))
    
    def chunk_list(list_data):
        if len(list_data) == 0:
            return []
        list_data = [torch.split(elem, max_length) for elem in list_data]
        return [elem for sublist in list_data for elem in sublist]
    
    # Chunk sequences (after splitting users!)
    train_item_ids = chunk_list(train_item_ids)
    train_skill_ids = chunk_list(train_skill_ids)
    train_labels = chunk_list(train_labels)
    
    val_item_ids = chunk_list(val_item_ids)
    val_skill_ids = chunk_list(val_skill_ids)
    val_labels = chunk_list(val_labels)
    
    test_item_ids = chunk_list(test_item_ids)
    test_skill_ids = chunk_list(test_skill_ids)
    test_labels = chunk_list(test_labels)
    
    # Create tuples
    train_data = list(zip(train_item_ids, train_skill_ids, train_labels))
    val_data = list(zip(val_item_ids, val_skill_ids, val_labels))
    test_data = list(zip(test_item_ids, test_skill_ids, test_labels))
    
    # Shuffle chunks within each split
    if randomize:
        shuffle(train_data)
        shuffle(val_data)
        shuffle(test_data)
    
    return train_data, val_data, test_data


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
    
    # Pad sequences to the same length
    item_ids = pad_sequence(item_ids, batch_first=True, padding_value=0)
    skill_ids = pad_sequence(skill_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    
    # Create mask (1 for valid positions, 0 for padding)
    mask = (labels >= 0).float()
    
    return item_ids, skill_ids, labels, mask


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
    """
    Compute AUC score.
    
    顶级会议标准：明确AUC计算类型
    - KT领域默认：interaction-level AUC (micro AUC)
    - 计算方式：将所有interactions展平后计算AUC
    - 论文中必须明确说明："We report interaction-level ROC-AUC."
    
    ⚠️ 论文写作要求（非代码问题）：
    虽然代码实现的是interaction-level AUC，但论文里必须明确写：
    "We report interaction-level ROC-AUC, which is computed by flattening
    all student-item interactions and calculating AUC across all predictions.
    This is the standard metric in knowledge tracing literature."
    
    否则审稿人可能问：
    - user-level AUC？
    - skill-level AUC？
    - macro 还是 micro？
    
    Args:
        preds: predicted probabilities
        labels: ground truth labels
    
    Returns:
        float: AUC score
    """
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = 0.5  # 论文标准：单类时AUC=0.5
    else:
        auc = roc_auc_score(labels, preds)  # 默认为micro AUC（interaction-level）
    return auc


def compute_rmse(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    rmse = math.sqrt(torch.mean((preds - labels) ** 2, dim=0, keepdim=False))
    return rmse


def train_epoch(model, loader, optimizer, criterion):
    """
    Training with global AUC calculation (论文级正确做法)
    
    关键修复：收集整个训练集的所有预测，再统一算一次AUC
    确保训练AUC和验证AUC的统计方式一致
    
    Args:
        model: TSAKT model
        loader: DataLoader
        optimizer: optimizer
        criterion: loss function
    
    Returns:
        tuple: (avg_loss, auc, rmse)
    """
    model.train()
    total_loss = 0
    total_count = 0
    
    # 收集所有预测和标签用于全局AUC计算
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for item_ids, skill_ids, labels, mask in pbar:
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
        total_count += 1
        
        # 收集预测和标签（只收集有效位置）
        preds = torch.sigmoid(outputs.squeeze(-1))
        valid_mask = (labels >= 0)
        
        all_preds.append(preds[valid_mask].cpu())
        all_labels.append(labels[valid_mask].cpu())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 全局AUC计算（论文级正确）
    # 注意：需要先detach()再调用numpy()，因为preds是计算图的一部分
    all_preds = torch.cat(all_preds).detach().numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    if len(np.unique(all_labels)) == 1:  # Only one class
        auc = 0.5  # 论文标准：单类时AUC=0.5
    else:
        auc = roc_auc_score(all_labels, all_preds)
    
    # 全局RMSE计算
    rmse = math.sqrt(np.mean((all_preds - all_labels) ** 2))
    
    return total_loss / total_count, auc, rmse


def validate(model, loader, criterion):
    """
    Validation with global AUC calculation (论文级正确做法)
    
    关键修复：收集整个验证集的所有预测，再统一算一次AUC
    避免batch-wise AUC average导致的系统性偏差
    
    Args:
        model: TSAKT model
        loader: DataLoader
        criterion: loss function
    
    Returns:
        tuple: (avg_loss, auc, rmse)
    """
    model.eval()
    total_loss = 0
    total_count = 0
    
    # 收集所有预测和标签用于全局AUC计算
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for item_ids, skill_ids, labels, mask in tqdm(loader, desc='Validation'):
            
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            outputs = model(item_ids, skill_ids, mask)
            
            loss = criterion(outputs.squeeze(-1), labels, mask)
            total_loss += loss.item()
            total_count += 1
            
            # 收集预测和标签（只收集有效位置）
            preds = torch.sigmoid(outputs.squeeze(-1))
            valid_mask = (labels >= 0)
            
            all_preds.append(preds[valid_mask].cpu())
            all_labels.append(labels[valid_mask].cpu())
    
    # 全局AUC计算（论文级正确）
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    if len(np.unique(all_labels)) == 1:  # Only one class
        auc = 0.5  # 论文标准：单类时AUC=0.5（与train_epoch保持一致）
    else:
        auc = roc_auc_score(all_labels, all_preds)
    
    # 全局RMSE计算
    rmse = math.sqrt(np.mean((all_preds - all_labels) ** 2))
    
    return total_loss / total_count, auc, rmse


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
    # 修复3：固定随机种子（论文级必须）
    set_seed(args.seed)
    
    print(f"Training TSAKT-Linear on {args.dataset}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed} (论文级：确保可复现)")
    
    data_path = os.path.join('data', args.dataset, 'preprocessed_data.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path, sep="\t")
    num_items = int(df["item_id"].max() + 1)
    num_skills = int(df["skill_id"].max() + 1)
    
    print(f"Num items: {num_items}, Num skills: {num_skills}")
    print(f"Total users: {len(df.groupby('user_id'))}")
    
    # 顶级会议标准：train/val/test三划分
    train_data, val_data, test_data = get_data(df, max_length=args.max_seq_len, 
                                              train_split=0.8, test_split=0.5,
                                              randomize=True, seed=args.seed)
    
    # 顶级会议标准：使用DataLoader替代prepare_batches
    # 优势：更规范、支持多进程、自动内存优化
    
    # 工程级稳定性修复：Windows + 单GPU下num_workers=4可能卡死
    # 顶级会议代码常见做法：Windows下num_workers=0
    num_workers = 0 if os.name == 'nt' else 4
    
    train_dataset = KTDataset(train_data)
    val_dataset = KTDataset(val_data)
    test_dataset = KTDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,  # Windows下=0，Linux/Mac下=4
        pin_memory=(num_workers > 0),  # 多进程时才用pin_memory
        collate_fn=kt_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        collate_fn=kt_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        collate_fn=kt_collate_fn
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    model = TSAKT_Linear(num_items, num_skills, embed_size=args.embed_size, 
                       num_layers=args.num_layers, num_heads=args.num_heads,
                       tensor_rank=args.tensor_rank, max_len=args.max_seq_len, 
                       drop_prob=args.drop_prob).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 修复4：early stopping基于AUC而不是loss（更符合KT论文标准）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
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
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 10  # 论文标准：best AUC不提升10轮则停止
    
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
        
        train_loss, train_auc, train_rmse = train_epoch(model, train_loader, optimizer, masked_bce_loss)
        val_loss, val_auc, val_rmse = validate(model, val_loader, masked_bce_loss)
        
        # 修复4：基于AUC调整学习率
        scheduler.step(val_auc)
        
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val RMSE: {val_rmse:.4f}")
        
        training_history['train_loss'].append(train_loss)
        training_history['train_auc'].append(train_auc)
        training_history['train_rmse'].append(train_rmse)
        training_history['val_loss'].append(val_loss)
        training_history['val_auc'].append(val_auc)
        training_history['val_rmse'].append(val_rmse)
        training_history['epochs'].append(epoch + 1)
        
        # 修复4：基于AUC保存最佳模型 + Early Stopping
        # 顶级会议标准：添加微小阈值避免浮点数抖动导致频繁保存
        improvement_threshold = 1e-4  # 只改进超过0.0001才保存
        
        if val_auc >= best_val_auc + improvement_threshold:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            save_path = os.path.join(args.savedir, f'{args.dataset}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_rmse': val_rmse,
            }, save_path)
            print(f"Saved best model to {save_path} (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s) (best: {best_val_auc:.4f} at epoch {best_epoch+1})")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered! Best AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
                print(f"No improvement for {early_stop_patience} epochs.")
                break
    
    # 顶级会议标准：在test set上评估最佳模型
    print(f"\n{'='*80}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*80}\n")
    
    # 加载最佳模型
    best_model_path = os.path.join(args.savedir, f'{args.dataset}_best.pt')
    
    # 顶级会议标准：添加map_location参数确保跨设备兼容性
    # 这是真正reviewer会注意的reproducibility细节
    # 否则：CPU / GPU 切换可能加载失败
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from {best_model_path}")
    
    # 在test set上评估
    test_loss, test_auc, test_rmse = validate(model, test_loader, masked_bce_loss)
    
    print(f"\nTest Set Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    
    # 保存test结果到training_history
    training_history['test_loss'] = test_loss
    training_history['test_auc'] = test_auc
    training_history['test_rmse'] = test_rmse
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"\nFinal Results:")
    print(f"  Best Val AUC: {best_val_auc:.4f} (epoch {best_epoch+1})")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Generalization Gap: {best_val_auc - test_auc:.4f}")
    
    training_history['best_val_loss'] = best_val_loss
    training_history['best_val_auc'] = best_val_auc
    training_history['best_epoch'] = best_epoch
    
    history_path = os.path.join(args.savedir, f"{args.dataset}_training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=4)
    print(f"Saved training history to {history_path}")
    
    # 更新config.json，添加test结果
    config_path = os.path.join(args.savedir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config['test_auc'] = test_auc
    config['test_rmse'] = test_rmse
    config['generalization_gap'] = best_val_auc - test_auc
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Updated config with test results to {config_path}")
    
    plot_training_curves(training_history, args.dataset, args.savedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TSAKT-Linear (论文级修复版)')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--savedir', type=str, default='save/tsakt-linear')
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--tensor_rank', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    main(args)
