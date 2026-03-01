import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from model_tsakt import TSAKT
from train_tsakt import get_data, prepare_batches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集
dataset = 'assistments09_short_50'
max_length = 50

print("=" * 80)
print(f"Evaluating TSAKT-Ful vs TSAKT-w/o-Pos on {dataset}")
print("=" * 80)

# 加载数据
data_path = os.path.join('data', dataset, 'preprocessed_data_train.csv')
df = pd.read_csv(data_path, sep="\t")

# 获取数据
train_data, val_data = get_data(df, max_length=max_length, train_split=0.8)
val_batches = prepare_batches(val_data, batch_size=128)

# 评估TSAKT-Ful（带位置编码，max_pos=200）
tsakt_ful_path = os.path.join('save', 'tsakt-ful-short', f'{dataset},batch_size=128,max_length={max_length},encode_pos=True,max_pos=200,tensor_rank=3')
if os.path.exists(tsakt_ful_path):
    print(f"\nEvaluating TSAKT-Ful (max_pos=200) on {dataset}...")
    tsakt_ful = torch.load(tsakt_ful_path, map_location=device, weights_only=False).to(device)
    tsakt_ful.eval()
    
    # 评估
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_batches:
            item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels = batch
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            
            preds = tsakt_ful(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            # 处理输出形状
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 只收集有效的预测值（使用mask）
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"TSAKT-Ful: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        tsakt_ful_results = {'auc': auc, 'rmse': rmse, 'acc': acc}
else:
    print(f"TSAKT-Ful model not found: {tsakt_ful_path}")
    tsakt_ful_results = None

# 评估TSAKT-w/o-Pos（不带位置编码）
tsakt_wo_pos_path = os.path.join('save', 'tsakt-wo-pos-short', f'{dataset},batch_size=128,max_length={max_length},encode_pos=True,max_pos=5,tensor_rank=3')
if os.path.exists(tsakt_wo_pos_path):
    print(f"\nEvaluating TSAKT-w/o-Pos on {dataset}...")
    tsakt_wo_pos = torch.load(tsakt_wo_pos_path, map_location=device, weights_only=False).to(device)
    tsakt_wo_pos.eval()
    
    # 评估
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_batches:
            item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels = batch
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            labels = labels.to(device)
            
            preds = tsakt_wo_pos(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            # 处理输出形状
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            preds = torch.sigmoid(preds).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 只收集有效的预测值（使用mask）
            mask = labels >= 0
            valid_preds = preds[mask]
            valid_labels = labels[mask]
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.tolist())
                all_labels.extend(valid_labels.tolist())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        
        print(f"TSAKT-w/o-Pos: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
        tsakt_wo_pos_results = {'auc': auc, 'rmse': rmse, 'acc': acc}
else:
    print(f"TSAKT-w/o-Pos model not found: {tsakt_wo_pos_path}")
    tsakt_wo_pos_results = None

# 打印对比结果
if tsakt_ful_results and tsakt_wo_pos_results:
    print(f"\n{'-' * 60}")
    print(f"Short Sequence Ablation Study Results for {dataset}:")
    print(f"{'-' * 60}")
    print(f"{'Model':<25} {'AUC':<10} {'ACC':<10} {'RMSE':<10}")
    print(f"{'-' * 60}")
    
    print(f"{'TSAKT-w/o-Pos':<25} {tsakt_wo_pos_results['auc']:<10.4f} {tsakt_wo_pos_results['acc']:<10.4f} {tsakt_wo_pos_results['rmse']:<10.4f}")
    print(f"{'TSAKT-Ful':<25} {tsakt_ful_results['auc']:<10.4f} {tsakt_ful_results['acc']:<10.4f} {tsakt_ful_results['rmse']:<10.4f}")
    
    # 计算差异
    auc_diff = tsakt_ful_results['auc'] - tsakt_wo_pos_results['auc']
    acc_diff = tsakt_ful_results['acc'] - tsakt_wo_pos_results['acc']
    rmse_diff = tsakt_ful_results['rmse'] - tsakt_wo_pos_results['rmse']
    
    print(f"{'Difference':<25} {auc_diff:+10.4f} {acc_diff:+10.4f} {rmse_diff:+10.4f}")
    
    # 保存结果
    import json
    results = {
        'dataset': dataset,
        'max_length': max_length,
        'tsakt_wo_pos': tsakt_wo_pos_results,
        'tsakt_ful': tsakt_ful_results
    }
    
    with open('short_sequence_ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to short_sequence_ablation_results.json")

print("\n" + "=" * 80)
print("Short Sequence Ablation Study Completed!")
print("=" * 80)