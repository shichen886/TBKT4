import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from model_tsakt import TSAKT
from train_tsakt import get_data, prepare_batches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据
dataset = 'assistments09'
data_path = os.path.join('data', dataset, 'preprocessed_data_train.csv')
df = pd.read_csv(data_path, sep="\t")

# 获取数据
train_data, val_data = get_data(df, max_length=200, train_split=0.8)
val_batches = prepare_batches(val_data, batch_size=128)

# 加载TSAKT-Ful-v2模型（max_pos=200, num_epochs=256）
tsakt_ful_v2_path = os.path.join('save', 'tsakt-ful-v2', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
tsakt_ful_v2 = torch.load(tsakt_ful_v2_path, map_location=device, weights_only=False).to(device)
tsakt_ful_v2.eval()

# 加载TSAKT-Ful-v3模型（max_pos=200, num_epochs=512）
tsakt_ful_v3_path = os.path.join('save', 'tsakt-ful-v3', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
tsakt_ful_v3 = torch.load(tsakt_ful_v3_path, map_location=device, weights_only=False).to(device)
tsakt_ful_v3.eval()

# 加载SAKT模型
sakt_path = os.path.join('save', 'sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
from model_sakt import SAKT
sakt = torch.load(sakt_path, map_location=device, weights_only=False).to(device)
sakt.eval()

# 测试前向传播
with torch.no_grad():
    batch = val_batches[0]
    item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels = batch
    item_inputs = item_inputs.to(device)
    skill_inputs = skill_inputs.to(device)
    label_inputs = label_inputs.to(device)
    item_ids = item_ids.to(device)
    skill_ids = skill_ids.to(device)
    labels = labels.to(device)
    
    # 测试TSAKT-Ful-v2
    tsakt_ful_v2_preds = tsakt_ful_v2(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
    print(f"\nTSAKT-Ful-v2 (max_pos=200, num_epochs=256) predictions shape: {tsakt_ful_v2_preds.shape}")
    print(f"TSAKT-Ful-v2 predictions range: [{tsakt_ful_v2_preds.min():.4f}, {tsakt_ful_v2_preds.max():.4f}]")
    
    # 测试TSAKT-Ful-v3
    tsakt_ful_v3_preds = tsakt_ful_v3(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
    print(f"\nTSAKT-Ful-v3 (max_pos=200, num_epochs=512) predictions shape: {tsakt_ful_v3_preds.shape}")
    print(f"TSAKT-Ful-v3 predictions range: [{tsakt_ful_v3_preds.min():.4f}, {tsakt_ful_v3_preds.max():.4f}]")
    
    # 测试SAKT
    sakt_preds = sakt(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
    print(f"\nSAKT predictions shape: {sakt_preds.shape}")
    print(f"SAKT predictions range: [{sakt_preds.min():.4f}, {sakt_preds.max():.4f}]")
    
    # 比较预测值
    tsakt_ful_v2_sigmoid = torch.sigmoid(tsakt_ful_v2_preds).cpu().numpy()
    tsakt_ful_v3_sigmoid = torch.sigmoid(tsakt_ful_v3_preds).cpu().numpy()
    sakt_sigmoid = torch.sigmoid(sakt_preds).cpu().numpy()
    
    mask = labels.cpu().numpy() >= 0
    tsakt_ful_v2_valid = tsakt_ful_v2_sigmoid[mask]
    tsakt_ful_v3_valid = tsakt_ful_v3_sigmoid[mask]
    sakt_valid = sakt_sigmoid[mask]
    labels_valid = labels.cpu().numpy()[mask]
    
    print(f"\nValid predictions count: {len(tsakt_ful_v2_valid)}")
    print(f"TSAKT-Ful-v2 mean prediction: {tsakt_ful_v2_valid.mean():.4f}")
    print(f"TSAKT-Ful-v3 mean prediction: {tsakt_ful_v3_valid.mean():.4f}")
    print(f"SAKT mean prediction: {sakt_valid.mean():.4f}")
    print(f"Labels mean: {labels_valid.mean():.4f}")
    
    # 计算指标
    tsakt_ful_v2_auc = roc_auc_score(labels_valid, tsakt_ful_v2_valid) if len(set(labels_valid)) > 1 else 0.5
    tsakt_ful_v3_auc = roc_auc_score(labels_valid, tsakt_ful_v3_valid) if len(set(labels_valid)) > 1 else 0.5
    sakt_auc = roc_auc_score(labels_valid, sakt_valid) if len(set(labels_valid)) > 1 else 0.5
    
    tsakt_ful_v2_acc = accuracy_score(labels_valid, tsakt_ful_v2_valid >= 0.5)
    tsakt_ful_v3_acc = accuracy_score(labels_valid, tsakt_ful_v3_valid >= 0.5)
    sakt_acc = accuracy_score(labels_valid, sakt_valid >= 0.5)
    
    print(f"\nTSAKT-Ful-v2 AUC: {tsakt_ful_v2_auc:.4f}, ACC: {tsakt_ful_v2_acc:.4f}")
    print(f"TSAKT-Ful-v3 AUC: {tsakt_ful_v3_auc:.4f}, ACC: {tsakt_ful_v3_acc:.4f}")
    print(f"SAKT AUC: {sakt_auc:.4f}, ACC: {sakt_acc:.4f}")
    
    # 比较性能
    print(f"\nPerformance Comparison:")
    print(f"TSAKT-Ful-v2 vs SAKT: AUC diff = {tsakt_ful_v2_auc - sakt_auc:.4f}, ACC diff = {tsakt_ful_v2_acc - sakt_acc:.4f}")
    print(f"TSAKT-Ful-v3 vs SAKT: AUC diff = {tsakt_ful_v3_auc - sakt_auc:.4f}, ACC diff = {tsakt_ful_v3_acc - sakt_acc:.4f}")
    print(f"TSAKT-Ful-v3 vs TSAKT-Ful-v2: AUC diff = {tsakt_ful_v3_auc - tsakt_ful_v2_auc:.4f}, ACC diff = {tsakt_ful_v3_acc - tsakt_ful_v2_acc:.4f}")