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

def evaluate_dkt(dataset, model_path):
    print(f"\nEvaluating DKT on {dataset}...")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_dkt1 import get_data, prepare_batches
    
    batch_size = 8 if dataset == 'assistments09' else 128
    train_data, val_data = get_data(train_df, item_in=False, skill_in=True, item_out=True, skill_out=False, skill_separate=False)
    val_batches = prepare_batches(val_data, batch_size=batch_size)
    
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    
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
            
            preds = torch.sigmoid(preds)
            
            if skill_ids is not None:
                mask = labels >= 0
                valid_preds = preds[mask]
                valid_skill_ids = skill_ids[mask]
                valid_labels = labels[mask]
                
                if len(valid_preds) > 0:
                    selected_preds = valid_preds[torch.arange(valid_preds.size(0)), valid_skill_ids]
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
    val_batches = prepare_batches(val_data, batch_size=128)
    
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
    val_batches = prepare_batches(val_data, batch_size=128)
    
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
    val_batches = prepare_batches(val_data, batch_size=128)
    
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
    print("正常数据集基线对比评估")
    print("=" * 80)
    
    results = {}
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 80}")
        
        results[dataset] = {}
        
        dkt_path = os.path.join('save', 'dkt1', f'{dataset},batch_size=128,item_in=False,skill_in=True,item_out=True,skill_out=False,skill_separate=False')
        if os.path.exists(dkt_path):
            try:
                results[dataset]['dkt'] = evaluate_dkt(dataset, dkt_path)
            except Exception as e:
                print(f"DKT evaluation failed: {e}")
                results[dataset]['dkt'] = None
        
        akt_path = os.path.join('save', 'akt', f'{dataset},batch_size=128,max_length=200,max_pos=10')
        if os.path.exists(akt_path):
            results[dataset]['akt'] = evaluate_akt(dataset, akt_path, max_length=200, max_pos=10)
        
        sakt_path = os.path.join('save', 'sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=10')
        if os.path.exists(sakt_path):
            results[dataset]['sakt'] = evaluate_sakt(dataset, sakt_path, max_length=200, max_pos=10)
        
        tsakt_ful_path = os.path.join('save', 'tsakt-ful-v3', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        if not os.path.exists(tsakt_ful_path):
            tsakt_ful_path = os.path.join('save', 'tsakt-ful-v2', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        if os.path.exists(tsakt_ful_path):
            results[dataset]['tsakt_ful'] = evaluate_tsakt(dataset, tsakt_ful_path, max_length=200, max_pos=200)
        
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
    
    with open('normal_baseline_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to normal_baseline_results.json")
    return results

if __name__ == "__main__":
    results = main()
