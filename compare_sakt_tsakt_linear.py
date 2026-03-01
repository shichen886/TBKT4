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
from model_tsakt_linear import TSAKT_Linear

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


def evaluate_model(model, batches, model_type='sakt'):
    """Evaluate model on given batches."""
    model.eval()
    total_loss = 0
    total_auc = 0
    total_rmse = 0
    total_count = 0
    
    inference_times = []
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    with torch.no_grad():
        for batch in tqdm(batches, desc=f'Evaluating {model_type}'):
            item_ids, skill_ids, labels, mask = batch
            
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            if model_type == 'sakt':
                # SAKT requires different input format
                item_inputs = torch.cat((torch.zeros(item_ids.shape[0], 1, dtype=torch.long, device=device), item_ids[:, :-1]), dim=1)
                skill_inputs = torch.cat((torch.zeros(skill_ids.shape[0], 1, dtype=torch.long, device=device), skill_ids[:, :-1]), dim=1)
                label_inputs = torch.cat((torch.zeros(labels.shape[0], 1, dtype=torch.long, device=device), labels[:, :-1]), dim=1)
                outputs = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            else:
                # TSAKT-Linear
                outputs = model(item_ids, skill_ids, mask)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Compute loss
            if model_type == 'sakt':
                loss = criterion(outputs.squeeze(-1), labels.float())
            else:
                loss = criterion(outputs.squeeze(-1), labels.float())
            
            loss = (loss * mask).sum() / mask.sum()
            total_loss += loss.item()
            
            # Compute metrics
            if model_type == 'sakt':
                preds = torch.sigmoid(outputs.squeeze(-1))
            else:
                preds = torch.sigmoid(outputs.squeeze(-1))
            
            auc = compute_auc(preds.cpu(), labels.cpu())
            rmse = compute_rmse(preds.cpu(), labels.cpu())
            
            total_auc += auc
            total_rmse += rmse
            total_count += 1
    
    avg_loss = total_loss / total_count
    avg_auc = total_auc / total_count
    avg_rmse = total_rmse / total_count
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return {
        'loss': avg_loss,
        'auc': avg_auc,
        'rmse': avg_rmse,
        'inference_time': avg_inference_time
    }


def main(args):
    print(f"\n{'='*100}")
    print(f"对比SAKT vs TSAKT-Linear在{args.dataset}数据集上")
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
    
    # Load SAKT model
    print(f"\n加载SAKT模型...")
    sakt_model = SAKT(num_items, num_skills, embed_size=128, num_attn_layers=2, num_heads=4,
                     encode_pos=False, max_pos=5, drop_prob=0.1).to(device)
    
    sakt_path = f'save/sakt/{args.dataset}_best.pt'
    if os.path.exists(sakt_path):
        sakt_checkpoint = torch.load(sakt_path, map_location=device)
        sakt_model.load_state_dict(sakt_checkpoint['model_state_dict'])
        print(f"  SAKT模型已加载: {sakt_path}")
    else:
        print(f"  警告: SAKT模型未找到: {sakt_path}")
    
    # Load TSAKT-Linear model
    print(f"\n加载TSAKT-Linear模型...")
    tsakt_model = TSAKT_Linear(num_items, num_skills, embed_size=128, num_layers=2, num_heads=4,
                            tensor_rank=32, max_len=args.max_seq_len, drop_prob=0.1).to(device)
    
    tsakt_path = f'save/tsakt-linear/{args.dataset}_best.pt'
    if os.path.exists(tsakt_path):
        tsakt_checkpoint = torch.load(tsakt_path, map_location=device)
        tsakt_model.load_state_dict(tsakt_checkpoint['model_state_dict'])
        print(f"  TSAKT-Linear模型已加载: {tsakt_path}")
    else:
        print(f"  警告: TSAKT-Linear模型未找到: {tsakt_path}")
    
    # Evaluate SAKT
    print(f"\n{'='*100}")
    print(f"评估SAKT模型")
    print(f"{'='*100}")
    sakt_results = evaluate_model(sakt_model, val_batches, model_type='sakt')
    
    # Evaluate TSAKT-Linear
    print(f"\n{'='*100}")
    print(f"评估TSAKT-Linear模型")
    print(f"{'='*100}")
    tsakt_results = evaluate_model(tsakt_model, val_batches, model_type='tsakt-linear')
    
    # Compare results
    print(f"\n{'='*100}")
    print(f"对比结果")
    print(f"{'='*100}")
    print(f"\n{'指标':<15} {'SAKT':<15} {'TSAKT-Linear':<15} {'改进':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {sakt_results['auc']:<15.4f} {tsakt_results['auc']:<15.4f} {(tsakt_results['auc']-sakt_results['auc'])/sakt_results['auc']*100:+.2f}%")
    print(f"{'RMSE':<15} {sakt_results['rmse']:<15.4f} {tsakt_results['rmse']:<15.4f} {(sakt_results['rmse']-tsakt_results['rmse'])/sakt_results['rmse']*100:+.2f}%")
    print(f"{'推理时间(ms)':<15} {sakt_results['inference_time']*1000:<15.2f} {tsakt_results['inference_time']*1000:<15.2f} {(tsakt_results['inference_time']-sakt_results['inference_time'])/sakt_results['inference_time']*100:+.2f}%")
    print("-" * 60)
    
    # Calculate parameter counts
    sakt_params = sum(p.numel() for p in sakt_model.parameters())
    tsakt_params = sum(p.numel() for p in tsakt_model.parameters())
    
    print(f"\n{'参数量':<15} {sakt_params:<15,} {tsakt_params:<15,} {(tsakt_params-sakt_params)/sakt_params*100:+.2f}%")
    
    # Memory comparison
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test SAKT memory
        with torch.no_grad():
            for batch in val_batches[:1]:  # Test on first batch
                item_ids, skill_ids, labels, mask = batch
                item_ids = item_ids.to(device)
                skill_ids = skill_ids.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                
                item_inputs = torch.cat((torch.zeros(item_ids.shape[0], 1, dtype=torch.long, device=device), item_ids[:, :-1]), dim=1)
                skill_inputs = torch.cat((torch.zeros(skill_ids.shape[0], 1, dtype=torch.long, device=device), skill_ids[:, :-1]), dim=1)
                label_inputs = torch.cat((torch.zeros(labels.shape[0], 1, dtype=torch.long, device=device), labels[:, :-1]), dim=1)
                outputs = sakt_model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
        
        sakt_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test TSAKT-Linear memory
        with torch.no_grad():
            for batch in val_batches[:1]:  # Test on first batch
                item_ids, skill_ids, labels, mask = batch
                item_ids = item_ids.to(device)
                skill_ids = skill_ids.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                
                outputs = tsakt_model(item_ids, skill_ids, mask)
        
        tsakt_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"\n{'显存占用(MB)':<15} {sakt_memory:<15.2f} {tsakt_memory:<15.2f} {(tsakt_memory-sakt_memory)/sakt_memory*100:+.2f}%")
    
    print(f"\n{'='*100}")
    print(f"结论")
    print(f"{'='*100}")
    
    if tsakt_results['auc'] >= sakt_results['auc']:
        print(f"✅ TSAKT-Linear在AUC上优于或等于SAKT")
    else:
        print(f"❌ TSAKT-Linear在AUC上低于SAKT")
    
    if tsakt_results['rmse'] <= sakt_results['rmse']:
        print(f"✅ TSAKT-Linear在RMSE上优于或等于SAKT")
    else:
        print(f"❌ TSAKT-Linear在RMSE上高于SAKT")
    
    if torch.cuda.is_available():
        if tsakt_memory < sakt_memory:
            print(f"✅ TSAKT-Linear在显存占用上优于SAKT，节省{(sakt_memory-tsakt_memory)/sakt_memory*100:.2f}%")
        else:
            print(f"❌ TSAKT-Linear在显存占用上高于SAKT")


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description='Compare SAKT vs TSAKT-Linear.')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    
    main(args)
