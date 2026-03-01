import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score
import math
import time
from tqdm import tqdm

from model_sakt import SAKT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_data(df, max_length, train_split=0.8, randomize=True):
    """Extract sequences from dataframe."""
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


def prepare_batches(data, batch_size, randomize=False):
    """Prepare batches grouping padded sequences."""
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
    if len(torch.unique(labels)) == 1:
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def compute_rmse(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    rmse = math.sqrt(torch.mean((preds - labels) ** 2, dim=0, keepdim=False))
    return rmse


def evaluate_sakt(model, batches):
    """Evaluate SAKT model on given batches."""
    model.eval()
    total_loss = 0
    total_auc = 0
    total_rmse = 0
    total_count = 0
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    with torch.no_grad():
        for batch in tqdm(batches, desc='Evaluating SAKT'):
            item_ids, skill_ids, labels, mask = batch
            
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            # SAKT requires different input format
            item_inputs = torch.cat((torch.zeros(item_ids.shape[0], 1, dtype=torch.long, device=device), item_ids[:, :-1]), dim=1)
            skill_inputs = torch.cat((torch.zeros(skill_ids.shape[0], 1, dtype=torch.long, device=device), skill_ids[:, :-1]), dim=1)
            label_inputs = torch.cat((torch.zeros(labels.shape[0], 1, dtype=torch.long, device=device), labels[:, :-1]), dim=1)
            
            outputs = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            # Compute loss
            loss = criterion(outputs.squeeze(-1), labels.float())
            loss = (loss * mask).sum() / mask.sum()
            total_loss += loss.item()
            
            # Compute metrics
            preds = torch.sigmoid(outputs.squeeze(-1))
            auc = compute_auc(preds.cpu(), labels.cpu())
            rmse = compute_rmse(preds.cpu(), labels.cpu())
            
            total_auc += auc
            total_rmse += rmse
            total_count += 1
    
    avg_loss = total_loss / total_count
    avg_auc = total_auc / total_count
    avg_rmse = total_rmse / total_count
    
    return {
        'loss': avg_loss,
        'auc': avg_auc,
        'rmse': avg_rmse
    }


def main(args):
    print(f"\n{'='*100}")
    print(f"评估SAKT模型在{args.dataset}数据集上")
    print(f"{'='*100}")
    
    # Load data
    data_path = f'data/{args.dataset}/preprocessed_data.csv'
    df = pd.read_csv(data_path, sep="\t")
    num_items = int(df["item_id"].max() + 1)
    num_skills = int(df["skill_id"].max() + 1)
    
    print(f"\n数据集信息:")
    print(f"  题目数量: {num_items}")
    print(f"  技能数量: {num_skills}")
    print(f"  总序列数: {len(df.groupby('user_id'))}")
    
    # Prepare validation data
    _, val_data = get_data(df, max_length=args.max_seq_len, train_split=0.8, randomize=True)
    val_batches = prepare_batches(val_data, batch_size=args.batch_size, randomize=False)
    
    print(f"  验证批次: {len(val_batches)}")
    
    # Load SAKT model with correct parameters
    print(f"\n加载SAKT模型...")
    
    # 根据数据集选择不同的参数配置
    if args.dataset == 'assistments09':
        sakt_model = SAKT(num_items, num_skills, embed_size=40, num_attn_layers=2, num_heads=5,
                         encode_pos=False, max_pos=10, drop_prob=0.1).to(device)
    elif args.dataset == 'assistments12':
        sakt_model = SAKT(num_items, num_skills, embed_size=40, num_attn_layers=2, num_heads=5,
                         encode_pos=False, max_pos=10, drop_prob=0.1).to(device)
    elif args.dataset == 'assistments15':
        # assistments15使用embed_size=80
        sakt_model = SAKT(num_items, num_skills, embed_size=80, num_attn_layers=2, num_heads=5,
                         encode_pos=False, max_pos=10, drop_prob=0.1).to(device)
    else:
        sakt_model = SAKT(num_items, num_skills, embed_size=40, num_attn_layers=2, num_heads=5,
                         encode_pos=False, max_pos=10, drop_prob=0.1).to(device)
    
    sakt_path = f'save/sakt/{args.dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=10'
    if os.path.exists(sakt_path):
        sakt_checkpoint = torch.load(sakt_path, map_location=device, weights_only=False)
        sakt_model.load_state_dict(sakt_checkpoint.state_dict())
        print(f"  SAKT模型已加载: {sakt_path}")
    else:
        print(f"  警告: SAKT模型未找到: {sakt_path}")
        return
    
    # Evaluate SAKT
    print(f"\n{'='*100}")
    print(f"评估SAKT模型")
    print(f"{'='*100}")
    sakt_results = evaluate_sakt(sakt_model, val_batches)
    
    print(f"\n{'='*100}")
    print(f"SAKT模型评估结果")
    print(f"{'='*100}")
    print(f"Val Loss: {sakt_results['loss']:.4f}")
    print(f"Val AUC: {sakt_results['auc']:.4f}")
    print(f"Val RMSE: {sakt_results['rmse']:.4f}")


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description='Evaluate SAKT model.')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    
    main(args)
