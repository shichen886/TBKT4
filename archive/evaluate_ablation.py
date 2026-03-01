import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from model_sakt import SAKT
from model_tsakt import TSAKT

def evaluate_sakt(dataset, model_path, max_length=200):
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

def evaluate_tsakt(dataset, model_path, max_length=200):
    """评估TSAKT模型"""
    print(f"\nEvaluating TSAKT on {dataset}...")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_tsakt import get_data, prepare_batches
    train_data, val_data = get_data(train_df, max_length)
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
    print("消融实验评估")
    print("=" * 80)
    
    results = {}
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 80}")
        
        results[dataset] = {}
        max_length = 200
        
        sakt_path = os.path.join('save', 'sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
        if os.path.exists(sakt_path):
            try:
                results[dataset]['sakt'] = evaluate_sakt(dataset, sakt_path, max_length=max_length)
            except Exception as e:
                print(f"SAKT evaluation failed: {e}")
                results[dataset]['sakt'] = None
        
        tsakt_wo_pos_path = os.path.join('save', 'tsakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
        if os.path.exists(tsakt_wo_pos_path):
            try:
                results[dataset]['tsakt_wo_pos'] = evaluate_tsakt(dataset, tsakt_wo_pos_path, max_length=max_length)
            except Exception as e:
                print(f"TSAKT-w/o-Pos evaluation failed: {e}")
                results[dataset]['tsakt_wo_pos'] = None
        
        tsakt_ful_path = os.path.join('save', 'tsakt-ful-v2', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        if os.path.exists(tsakt_ful_path):
            try:
                results[dataset]['tsakt_ful'] = evaluate_tsakt(dataset, tsakt_ful_path, max_length=max_length)
            except Exception as e:
                print(f"TSAKT-Ful evaluation failed: {e}")
                results[dataset]['tsakt_ful'] = None
        
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("消融实验1：架构有效性（SAKT vs TSAKT-w/o-Pos）")
    print("=" * 80)
    
    for dataset in results:
        print(f"\n{dataset}:")
        print(f"{'Model':<25} {'AUC':<10} {'ACC':<10} {'RMSE':<10} {'Improvement':<15}")
        print("-" * 70)
        
        if 'sakt' in results[dataset] and results[dataset]['sakt'] and 'tsakt_wo_pos' in results[dataset] and results[dataset]['tsakt_wo_pos']:
            sakt_auc = results[dataset]['sakt']['auc']
            tsakt_wo_pos_auc = results[dataset]['tsakt_wo_pos']['auc']
            sakt_acc = results[dataset]['sakt']['acc']
            tsakt_wo_pos_acc = results[dataset]['tsakt_wo_pos']['acc']
            sakt_rmse = results[dataset]['sakt']['rmse']
            tsakt_wo_pos_rmse = results[dataset]['tsakt_wo_pos']['rmse']
            
            auc_diff = tsakt_wo_pos_auc - sakt_auc
            acc_diff = tsakt_wo_pos_acc - sakt_acc
            rmse_diff = tsakt_wo_pos_rmse - sakt_rmse
            
            print(f"{'SAKT':<25} {sakt_auc:<10.4f} {sakt_acc:<10.4f} {sakt_rmse:<10.4f} {'-':<15}")
            print(f"{'TSAKT-w/o-Pos':<25} {tsakt_wo_pos_auc:<10.4f} {tsakt_wo_pos_acc:<10.4f} {tsakt_wo_pos_rmse:<10.4f} {auc_diff:+.4f}")
            
            if auc_diff > 0:
                print(f"{'结论':<25} TSAKT-w/o-Pos更优 (AUC提升{auc_diff:.4f})")
            elif auc_diff < 0:
                print(f"{'结论':<25} SAKT更优 (AUC下降{abs(auc_diff):.4f})")
            else:
                print(f"{'结论':<25} 性能相当")
    
    print("\n" + "=" * 80)
    print("消融实验2：位置编码有效性（TSAKT-w/o-Pos vs TSAKT-Ful）")
    print("=" * 80)
    
    for dataset in results:
        print(f"\n{dataset}:")
        print(f"{'Model':<25} {'AUC':<10} {'ACC':<10} {'RMSE':<10} {'Improvement':<15}")
        print("-" * 70)
        
        if 'tsakt_wo_pos' in results[dataset] and results[dataset]['tsakt_wo_pos'] and 'tsakt_ful' in results[dataset] and results[dataset]['tsakt_ful']:
            tsakt_wo_pos_auc = results[dataset]['tsakt_wo_pos']['auc']
            tsakt_ful_auc = results[dataset]['tsakt_ful']['auc']
            tsakt_wo_pos_acc = results[dataset]['tsakt_wo_pos']['acc']
            tsakt_ful_acc = results[dataset]['tsakt_ful']['acc']
            tsakt_wo_pos_rmse = results[dataset]['tsakt_wo_pos']['rmse']
            tsakt_ful_rmse = results[dataset]['tsakt_ful']['rmse']
            
            auc_diff = tsakt_ful_auc - tsakt_wo_pos_auc
            acc_diff = tsakt_ful_acc - tsakt_wo_pos_acc
            rmse_diff = tsakt_ful_rmse - tsakt_wo_pos_rmse
            
            print(f"{'TSAKT-w/o-Pos':<25} {tsakt_wo_pos_auc:<10.4f} {tsakt_wo_pos_acc:<10.4f} {tsakt_wo_pos_rmse:<10.4f} {'-':<15}")
            print(f"{'TSAKT-Ful':<25} {tsakt_ful_auc:<10.4f} {tsakt_ful_acc:<10.4f} {tsakt_ful_rmse:<10.4f} {auc_diff:+.4f}")
            
            if auc_diff > 0:
                print(f"{'结论':<25} TSAKT-Ful更优 (AUC提升{auc_diff:.4f})")
            elif auc_diff < 0:
                print(f"{'结论':<25} TSAKT-w/o-Pos更优 (AUC下降{abs(auc_diff):.4f})")
            else:
                print(f"{'结论':<25} 性能相当")
    
    with open('ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to ablation_results.json")
    return results

if __name__ == "__main__":
    results = main()
