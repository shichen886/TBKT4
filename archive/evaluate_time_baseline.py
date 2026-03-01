import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from model_dkt1 import DKT1
from model_akt import AKT
from model_sakt import SAKT
from model_tsakt import TSAKT

def truncate_sequences(data, max_length):
    """截断序列到指定长度"""
    truncated_data = []
    for item_inputs, skill_inputs, item_ids, skill_ids, labels in data:
        if item_inputs is not None:
            item_inputs = item_inputs[-max_length:] if len(item_inputs) > max_length else item_inputs
        if skill_inputs is not None:
            skill_inputs = skill_inputs[-max_length:] if len(skill_inputs) > max_length else skill_inputs
        if item_ids is not None:
            item_ids = item_ids[-max_length:] if len(item_ids) > max_length else item_ids
        if skill_ids is not None:
            skill_ids = skill_ids[-max_length:] if len(skill_ids) > max_length else skill_ids
        if labels is not None:
            labels = labels[-max_length:] if len(labels) > max_length else labels
        
        truncated_data.append((item_inputs, skill_inputs, item_ids, skill_ids, labels))
    return truncated_data

def evaluate_dkt(dataset, model_path, max_length=200):
    """评估DKT模型"""
    print(f"\nEvaluating DKT on {dataset}...")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_dkt1 import get_data, prepare_batches
    train_data, val_data = get_data(train_df, item_in=False, skill_in=True, item_out=True, skill_out=False, skill_separate=False)
    
    val_data = truncate_sequences(val_data, max_length)
    val_batches = prepare_batches(val_data, batch_size=1)
    
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    model.half()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for item_inputs, skill_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.to(device) if item_inputs is not None else None
            skill_inputs = skill_inputs.to(device) if skill_inputs is not None else None
            item_ids = item_ids.to(device) if item_ids is not None else None
            skill_ids = skill_ids.to(device) if skill_ids is not None else None
            labels = labels.to(device)
            
            preds, _ = model(item_inputs, skill_inputs)
            
            preds = torch.sigmoid(preds).float()
            
            if skill_ids is not None:
                mask = labels >= 0
                valid_preds = preds[mask]
                valid_skill_ids = skill_ids[mask]
                selected_preds = valid_preds[torch.arange(valid_preds.size(0)), valid_skill_ids]
                valid_labels = labels[mask]
                
                if len(selected_preds) > 0:
                    all_preds.extend(selected_preds.cpu().numpy().tolist())
                    all_labels.extend(valid_labels.cpu().numpy().tolist())
            else:
                mask = labels >= 0
                valid_preds = preds[mask]
                valid_labels = labels[mask]
                
                if len(valid_preds) > 0:
                    all_preds.extend(valid_preds.cpu().numpy().tolist())
                    all_labels.extend(valid_labels.cpu().numpy().tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"DKT: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        return {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    return None

def evaluate_akt(dataset, model_path, max_length=200, max_pos=5):
    """评估AKT模型"""
    print(f"\nEvaluating AKT on {dataset}...")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_akt import get_data as get_data_akt, prepare_batches
    train_data, val_data = get_data_akt(train_df, max_length, train_split=0.8)
    val_batches = prepare_batches(val_data, batch_size=8)
    
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"AKT: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        return {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    return None

def evaluate_sakt(dataset, model_path, max_length=200, max_pos=5):
    """评估SAKT模型"""
    print(f"\nEvaluating SAKT on {dataset}...")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_sakt import get_data as get_data_sakt, prepare_batches
    train_data, val_data = get_data_sakt(train_df, max_length)
    val_batches = prepare_batches(val_data, batch_size=8)
    
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"SAKT: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        return {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    return None

def evaluate_tsakt(dataset, model_path, max_length=200, max_pos=200):
    """评估TSAKT模型"""
    print(f"\nEvaluating TSAKT on {dataset}...")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_sakt import get_data as get_data_sakt, prepare_batches
    train_data, val_data = get_data_sakt(train_df, max_length)
    val_batches = prepare_batches(val_data, batch_size=8)
    
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"TSAKT: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        return {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    return None

def main():
    print("=" * 80)
    print("时间依赖数据集基线对比评估")
    print("=" * 80)
    
    results = {}
    datasets = ['assistments09_time_1h', 'assistments09_time_2h', 'assistments09_time_4h']
    
    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 80}")
        
        results[dataset] = {}
        max_length = 200
        
        dkt_path = os.path.join('save', 'dkt-time', f'{dataset},batch_size=6,item_in=False,skill_in=True,item_out=True,skill_out=False,skill_separate=False')
        if os.path.exists(dkt_path):
            try:
                results[dataset]['dkt'] = evaluate_dkt(dataset, dkt_path, max_length=max_length)
            except torch.cuda.OutOfMemoryError:
                print(f"DKT evaluation failed: CUDA out of memory")
                results[dataset]['dkt'] = None
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"DKT evaluation failed: {e}")
                results[dataset]['dkt'] = None
        
        akt_path = os.path.join('save', 'akt-time', f'{dataset},batch_size=128,max_length=200,max_pos=5')
        if os.path.exists(akt_path):
            try:
                results[dataset]['akt'] = evaluate_akt(dataset, akt_path, max_length=max_length, max_pos=5)
            except Exception as e:
                print(f"AKT evaluation failed: {e}")
                results[dataset]['akt'] = None
        
        sakt_path = os.path.join('save', 'sakt-time', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
        if os.path.exists(sakt_path):
            try:
                results[dataset]['sakt'] = evaluate_sakt(dataset, sakt_path, max_length=max_length, max_pos=5)
            except Exception as e:
                print(f"SAKT evaluation failed: {e}")
                results[dataset]['sakt'] = None
        
        tsakt_ful_path = os.path.join('save', 'tsakt-ful-time', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        if os.path.exists(tsakt_ful_path):
            try:
                results[dataset]['tsakt_ful'] = evaluate_tsakt(dataset, tsakt_ful_path, max_length=max_length, max_pos=200)
            except Exception as e:
                print(f"TSAKT evaluation failed: {e}")
                results[dataset]['tsakt_ful'] = None
        
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    for dataset in results:
        print(f"\n{dataset}:")
        print(f"{'Model':<20} {'AUC':<10} {'RMSE':<10} {'ACC':<10}")
        print("-" * 50)
        
        for model_name in ['dkt', 'akt', 'sakt', 'tsakt_ful']:
            if model_name in results[dataset] and results[dataset][model_name]:
                metrics = results[dataset][model_name]
                print(f"{model_name:<20} {metrics['auc']:<10.4f} {metrics['rmse']:<10.4f} {metrics['acc']:<10.4f}")
    
    with open('time_baseline_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to time_baseline_results.json")
    return results

if __name__ == "__main__":
    results = main()
