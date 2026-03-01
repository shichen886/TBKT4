import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 导入模型
from model_dkt1 import DKT1
from model_akt import AKT
from model_sakt import SAKT
from model_tsakt import TSAKT

# 导入工具函数
from utils import Metrics

def evaluate_dkt(dataset, model_path, max_length=200):
    """评估DKT模型"""
    print(f"\nEvaluating DKT on {dataset}...")
    
    # 读取数据
    train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    test_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_test.csv'), sep="\t")
    
    # 准备数据
    from train_dkt1 import get_data, prepare_batches
    train_data, val_data = get_data(train_df, item_in=False, skill_in=True, item_out=True, skill_out=False, skill_separate=False)
    
    # 准备批次
    val_batches = prepare_batches(val_data, batch_size=128)
    
    # 加载模型
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    
    # 评估
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
            
            # 获取预测值
            if skill_ids is not None:
                preds = preds[torch.arange(preds.size(0)), skill_ids]
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 只收集有效的预测值
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"DKT: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        return {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    return None

def evaluate_akt(dataset, model_path, max_length=200, max_pos=5):
    """评估AKT模型"""
    print(f"\nEvaluating AKT on {dataset}...")
    
    # 读取数据
    df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    test_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_test.csv'), sep="\t")
    
    # 准备数据
    from train_akt import get_data as get_data_akt, prepare_batches
    train_data, val_data = get_data_akt(train_df, max_length, train_split=0.8)
    
    # 准备批次
    val_batches = prepare_batches(val_data, batch_size=128)
    
    # 加载模型
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    
    # 评估
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
            
            # 处理输出形状
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 只收集有效的预测值
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"AKT: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        return {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    return None

def evaluate_sakt(dataset, model_path, max_length=200, max_pos=5):
    """评估SAKT模型"""
    print(f"\nEvaluating SAKT on {dataset}...")
    
    # 读取数据
    df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    test_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_test.csv'), sep="\t")
    
    # 准备数据
    from train_sakt import get_data as get_data_sakt, prepare_batches
    train_data, val_data = get_data_sakt(train_df, max_length)
    
    # 准备批次
    val_batches = prepare_batches(val_data, batch_size=128)
    
    # 加载模型
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    
    # 评估
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
            
            # 处理输出形状
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 只收集有效的预测值
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"SAKT: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        return {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    return None

def evaluate_tsakt(dataset, model_path, max_length=200, max_pos=200):
    """评估TSAKT模型"""
    print(f"\nEvaluating TSAKT on {dataset}...")
    
    # 读取数据
    df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    test_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_test.csv'), sep="\t")
    
    # 准备数据
    from train_sakt import get_data as get_data_sakt, prepare_batches
    train_data, val_data = get_data_sakt(train_df, max_length)
    
    # 准备批次
    val_batches = prepare_batches(val_data, batch_size=128)
    
    # 加载模型
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    
    # 评估
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
            
            # 处理输出形状
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 只收集有效的预测值
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"TSAKT: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        return {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    return None

def main():
    print("=" * 80)
    print("Baseline Comparison on Short Sequence and Time-Dependent Datasets")
    print("=" * 80)
    
    results = {}
    
    # 1. 短序列数据集
    short_datasets = ['assistments09_short_50']
    for dataset in short_datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 80}")
        
        results[dataset] = {}
        
        # 评估DKT
        dkt_path = os.path.join('save', 'dkt-short', f'{dataset},batch_size=100,item_in=False,skill_in=True,item_out=True,skill_out=False,skill_separate=False')
        if os.path.exists(dkt_path):
            results[dataset]['dkt'] = evaluate_dkt(dataset, dkt_path, max_length=50)
        
        # 评估AKT
        akt_path = os.path.join('save', 'akt-short', f'{dataset},batch_size=128,max_length=50,max_pos=5')
        if os.path.exists(akt_path):
            results[dataset]['akt'] = evaluate_akt(dataset, akt_path, max_length=50, max_pos=5)
        
        # 评估SAKT
        sakt_path = os.path.join('save', 'sakt-short', f'{dataset},batch_size=128,max_length=50,encode_pos=False,max_pos=5')
        if os.path.exists(sakt_path):
            results[dataset]['sakt'] = evaluate_sakt(dataset, sakt_path, max_length=50, max_pos=5)
        
        # 评估TSAKT-Ful
        tsakt_ful_path = os.path.join('save', 'tsakt-ful-short', f'{dataset},batch_size=128,max_length=50,encode_pos=True,max_pos=200,tensor_rank=3')
        if os.path.exists(tsakt_ful_path):
            results[dataset]['tsakt_ful'] = evaluate_tsakt(dataset, tsakt_ful_path, max_length=50, max_pos=200)
        
        # 评估TSAKT-w/o-Pos
        tsakt_wo_pos_path = os.path.join('save', 'tsakt-wo-pos-short', f'{dataset},batch_size=128,max_length=50,encode_pos=True,max_pos=5,tensor_rank=3')
        if os.path.exists(tsakt_wo_pos_path):
            results[dataset]['tsakt_wo_pos'] = evaluate_tsakt(dataset, tsakt_wo_pos_path, max_length=50, max_pos=5)
    
    # 2. 时间依赖数据集
    time_datasets = ['assistments09_time_1h']
    for dataset in time_datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 80}")
        
        results[dataset] = {}
        
        # 评估DKT
        dkt_path = os.path.join('save', 'dkt-time', f'{dataset},batch_size=6,item_in=False,skill_in=True,item_out=True,skill_out=False,skill_separate=False')
        if os.path.exists(dkt_path):
            results[dataset]['dkt'] = evaluate_dkt(dataset, dkt_path, max_length=200)
        
        # 评估AKT
        akt_path = os.path.join('save', 'akt-time', f'{dataset},batch_size=128,max_length=200,max_pos=5')
        if os.path.exists(akt_path):
            results[dataset]['akt'] = evaluate_akt(dataset, akt_path, max_length=200, max_pos=5)
        
        # 评估SAKT
        sakt_path = os.path.join('save', 'sakt-time', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
        if os.path.exists(sakt_path):
            results[dataset]['sakt'] = evaluate_sakt(dataset, sakt_path, max_length=200, max_pos=5)
        
        # 评估TSAKT-Ful
        tsakt_ful_path = os.path.join('save', 'tsakt-ful-time', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        if os.path.exists(tsakt_ful_path):
            results[dataset]['tsakt_ful'] = evaluate_tsakt(dataset, tsakt_ful_path, max_length=200, max_pos=200)
        
        # 评估TSAKT-w/o-Pos
        tsakt_wo_pos_path = os.path.join('save', 'tsakt-wo-pos-time', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=5,tensor_rank=3')
        if os.path.exists(tsakt_wo_pos_path):
            results[dataset]['tsakt_wo_pos'] = evaluate_tsakt(dataset, tsakt_wo_pos_path, max_length=200, max_pos=5)
    
    # 打印结果表格
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    for dataset in results:
        print(f"\n{dataset}:")
        print(f"{'Model':<20} {'AUC':<10} {'RMSE':<10} {'ACC':<10}")
        print("-" * 50)
        
        for model_name in ['dkt', 'akt', 'sakt', 'tsakt_ful', 'tsakt_wo_pos']:
            if model_name in results[dataset] and results[dataset][model_name]:
                metrics = results[dataset][model_name]
                print(f"{model_name:<20} {metrics['auc']:<10.4f} {metrics['rmse']:<10.4f} {metrics['acc']:<10.4f}")
    
    return results

if __name__ == "__main__":
    results = main()