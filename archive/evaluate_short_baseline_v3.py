import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from model_dkt1 import DKT1
from model_akt import AKT
from model_sakt import SAKT
from model_tsakt import TSAKT

def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

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

def evaluate_dkt(dataset, model_path, max_length=50):
    print(f"\nEvaluating DKT on {dataset}...")
    print(f"Max sequence length: {max_length}")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_dkt1 import get_data, prepare_batches
    train_data, val_data = get_data(train_df, item_in=False, skill_in=True, item_out=True, skill_out=False, skill_separate=False)
    
    print(f"Original val_data size: {len(val_data)}")
    
    val_data = truncate_sequences(val_data, max_length)
    print(f"Truncated val_data size: {len(val_data)}")
    
    val_batches = prepare_batches(val_data, batch_size=1)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = DKT1(**checkpoint.get('model_args', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (item_inputs, skill_inputs, item_ids, skill_ids, labels) in enumerate(val_batches):
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
            
            del item_inputs, skill_inputs, item_ids, skill_ids, labels, preds
            
            if batch_idx % 5 == 0:
                print(f"Processed {batch_idx + 1} batches...")
                clear_gpu_memory()
    
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

def evaluate_akt(dataset, model_path, max_length=50, max_pos=5):
    print(f"\nEvaluating AKT on {dataset}...")
    print(f"Max sequence length: {max_length}")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_akt import get_data as get_data_akt, prepare_batches
    train_data, val_data = get_data_akt(train_df, max_length, train_split=0.8)
    
    print(f"Val data size: {len(val_data)}")
    
    val_batches = prepare_batches(val_data, batch_size=16)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = AKT(**checkpoint.get('model_args', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint
    
    model = model.to(device)
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
            
            del item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels, preds
            clear_gpu_memory()
    
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

def evaluate_sakt(dataset, model_path, max_length=50, max_pos=5):
    print(f"\nEvaluating SAKT on {dataset}...")
    print(f"Max sequence length: {max_length}")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_sakt import get_data as get_data_sakt, prepare_batches
    train_data, val_data = get_data_sakt(train_df, max_length)
    
    print(f"Val data size: {len(val_data)}")
    
    val_batches = prepare_batches(val_data, batch_size=16)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = SAKT(**checkpoint.get('model_args', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint
    
    model = model.to(device)
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
            
            del item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels, preds
            clear_gpu_memory()
    
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

def evaluate_tsakt(dataset, model_path, max_length=50, max_pos=200):
    print(f"\nEvaluating TSAKT on {dataset}...")
    print(f"Max sequence length: {max_length}")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_sakt import get_data as get_data_sakt, prepare_batches
    train_data, val_data = get_data_sakt(train_df, max_length)
    
    print(f"Val data size: {len(val_data)}")
    
    val_batches = prepare_batches(val_data, batch_size=16)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = TSAKT(**checkpoint.get('model_args', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint
    
    model = model.to(device)
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
            
            del item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels, preds
            clear_gpu_memory()
    
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
    print("短序列数据集基线对比评估（极致显存优化版本）")
    print("=" * 80)
    
    results = {}
    datasets = ['assistments09_short_50', 'assistments09_short_100']
    
    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 80}")
        
        results[dataset] = {}
        max_length = 50 if '50' in dataset else 100
        
        dkt_path = os.path.join('save', 'dkt-short', f'{dataset},batch_size=100,item_in=False,skill_in=True,item_out=True,skill_out=False,skill_separate=False')
        if os.path.exists(dkt_path):
            try:
                results[dataset]['dkt'] = evaluate_dkt(dataset, dkt_path, max_length=max_length)
            except Exception as e:
                print(f"DKT evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                results[dataset]['dkt'] = None
            finally:
                clear_gpu_memory()
        
        akt_path = os.path.join('save', 'akt-short', f'{dataset},batch_size=128,max_length={max_length},max_pos=5')
        if os.path.exists(akt_path):
            try:
                results[dataset]['akt'] = evaluate_akt(dataset, akt_path, max_length=max_length, max_pos=5)
            except Exception as e:
                print(f"AKT evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                results[dataset]['akt'] = None
            finally:
                clear_gpu_memory()
        
        sakt_path = os.path.join('save', 'sakt-short', f'{dataset},batch_size=128,max_length={max_length},encode_pos=False,max_pos=5')
        if os.path.exists(sakt_path):
            try:
                results[dataset]['sakt'] = evaluate_sakt(dataset, sakt_path, max_length=max_length, max_pos=5)
            except Exception as e:
                print(f"SAKT evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                results[dataset]['sakt'] = None
            finally:
                clear_gpu_memory()
        
        tsakt_ful_path = os.path.join('save', 'tsakt-ful-short', f'{dataset},batch_size=128,max_length={max_length},encode_pos=True,max_pos={max_length},tensor_rank=3')
        if os.path.exists(tsakt_ful_path):
            try:
                results[dataset]['tsakt_ful'] = evaluate_tsakt(dataset, tsakt_ful_path, max_length=max_length, max_pos=max_length)
            except Exception as e:
                print(f"TSAKT evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                results[dataset]['tsakt_ful'] = None
            finally:
                clear_gpu_memory()
    
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
    
    with open('short_baseline_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to short_baseline_results.json")
    return results

if __name__ == "__main__":
    results = main()
