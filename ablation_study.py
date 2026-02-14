import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from model_tsakt import TSAKT
from train_tsakt import get_data, prepare_batches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集列表
datasets = ['assistments09', 'assistments12', 'assistments15']

print("=" * 80)
print("Ablation Study: TSAKT-w/o-Pos vs TSAKT-Ful")
print("=" * 80)

results = {}

for dataset in datasets:
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset}")
    print(f"{'=' * 80}")
    
    results[dataset] = {}
    
    # 加载数据
    data_path = os.path.join('data', dataset, 'preprocessed_data_train.csv')
    df = pd.read_csv(data_path, sep="\t")
    
    # 获取数据
    train_data, val_data = get_data(df, max_length=200, train_split=0.8)
    val_batches = prepare_batches(val_data, batch_size=128)
    
    # 评估TSAKT-w/o-Pos（不带位置编码）
    tsakt_wo_pos_path = os.path.join('save', 'tsakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
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
            results[dataset]['tsakt_wo_pos'] = {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    # 评估TSAKT-Ful（带位置编码，max_pos=200, num_epochs=256）
    tsakt_ful_path = os.path.join('save', 'tsakt-ful-v2', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
    if os.path.exists(tsakt_ful_path):
        print(f"\nEvaluating TSAKT-Ful (max_pos=200, num_epochs=256) on {dataset}...")
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
            results[dataset]['tsakt_ful'] = {'auc': auc, 'rmse': rmse, 'acc': acc}
    
    # 打印对比结果
    if 'tsakt_wo_pos' in results[dataset] and 'tsakt_ful' in results[dataset]:
        print(f"\n{'-' * 60}")
        print(f"Ablation Study Results for {dataset}:")
        print(f"{'-' * 60}")
        print(f"{'Model':<25} {'AUC':<10} {'ACC':<10} {'RMSE':<10}")
        print(f"{'-' * 60}")
        
        wo_pos = results[dataset]['tsakt_wo_pos']
        ful = results[dataset]['tsakt_ful']
        
        print(f"{'TSAKT-w/o-Pos':<25} {wo_pos['auc']:<10.4f} {wo_pos['acc']:<10.4f} {wo_pos['rmse']:<10.4f}")
        print(f"{'TSAKT-Ful':<25} {ful['auc']:<10.4f} {ful['acc']:<10.4f} {ful['rmse']:<10.4f}")
        
        # 计算差异
        auc_diff = ful['auc'] - wo_pos['auc']
        acc_diff = ful['acc'] - wo_pos['acc']
        rmse_diff = ful['rmse'] - wo_pos['rmse']
        
        print(f"{'Difference':<25} {auc_diff:+10.4f} {acc_diff:+10.4f} {rmse_diff:+10.4f}")

# 保存结果
import json
with open('ablation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("Ablation Study Completed!")
print("=" * 80)
print("Results saved to ablation_results.json")