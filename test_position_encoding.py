import torch
import pandas as pd
import os

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

# 加载TSAKT-Ful模型
tsakt_ful_path = os.path.join('save', 'tsakt-ful', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=5,tensor_rank=3')
tsakt_ful = torch.load(tsakt_ful_path, map_location=device, weights_only=False).to(device)
tsakt_ful.eval()

# 加载SAKT模型
sakt_path = os.path.join('save', 'sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
from model_sakt import SAKT
sakt = torch.load(sakt_path, map_location=device, weights_only=False).to(device)
sakt.eval()

# 测试位置编码
print("\nTesting position encoding...")
print(f"TSAKT-Ful encode_pos: {tsakt_ful.encode_pos}")
print(f"TSAKT-Ful pos_key_embeds: {tsakt_ful.pos_key_embeds}")
print(f"TSAKT-Ful pos_value_embeds: {tsakt_ful.pos_value_embeds}")

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
    
    # 测试TSAKT-Ful
    tsakt_ful_preds = tsakt_ful(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
    print(f"\nTSAKT-Ful predictions shape: {tsakt_ful_preds.shape}")
    print(f"TSAKT-Ful predictions range: [{tsakt_ful_preds.min():.4f}, {tsakt_ful_preds.max():.4f}]")
    
    # 测试SAKT
    sakt_preds = sakt(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
    print(f"\nSAKT predictions shape: {sakt_preds.shape}")
    print(f"SAKT predictions range: [{sakt_preds.min():.4f}, {sakt_preds.max():.4f}]")
    
    # 比较预测值
    tsakt_ful_sigmoid = torch.sigmoid(tsakt_ful_preds).cpu().numpy()
    sakt_sigmoid = torch.sigmoid(sakt_preds).cpu().numpy()
    
    mask = labels.cpu().numpy() >= 0
    tsakt_ful_valid = tsakt_ful_sigmoid[mask]
    sakt_valid = sakt_sigmoid[mask]
    labels_valid = labels.cpu().numpy()[mask]
    
    print(f"\nValid predictions count: {len(tsakt_ful_valid)}")
    print(f"TSAKT-Ful mean prediction: {tsakt_ful_valid.mean():.4f}")
    print(f"SAKT mean prediction: {sakt_valid.mean():.4f}")
    print(f"Labels mean: {labels_valid.mean():.4f}")
    
    # 计算指标
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    tsakt_ful_auc = roc_auc_score(labels_valid, tsakt_ful_valid) if len(set(labels_valid)) > 1 else 0.5
    sakt_auc = roc_auc_score(labels_valid, sakt_valid) if len(set(labels_valid)) > 1 else 0.5
    
    tsakt_ful_acc = accuracy_score(labels_valid, tsakt_ful_valid >= 0.5)
    sakt_acc = accuracy_score(labels_valid, sakt_valid >= 0.5)
    
    print(f"\nTSAKT-Ful AUC: {tsakt_ful_auc:.4f}, ACC: {tsakt_ful_acc:.4f}")
    print(f"SAKT AUC: {sakt_auc:.4f}, ACC: {sakt_acc:.4f}")
    
    # 检查位置编码的影响
    print("\nChecking position encoding influence...")
    # 创建一个没有位置编码的TSAKT模型
    tsakt_wo_pos = TSAKT(
        num_items=int(df['item_id'].max() + 2),
        num_skills=int(df['skill_id'].max() + 2),
        embed_size=60,
        num_attn_layers=2,
        num_heads=5,
        encode_pos=False,
        max_pos=5,
        drop_prob=0.2,
        tensor_rank=3
    ).to(device)
    
    # 加载TSAKT-Ful的权重（除了位置编码）
    tsakt_ful_state = tsakt_ful.state_dict()
    tsakt_wo_pos_state = tsakt_wo_pos.state_dict()
    
    # 复制权重（跳过位置编码相关的权重）
    for key in tsakt_wo_pos_state:
        if key in tsakt_ful_state:
            tsakt_wo_pos_state[key] = tsakt_ful_state[key]
    
    tsakt_wo_pos.load_state_dict(tsakt_wo_pos_state)
    tsakt_wo_pos.eval()
    
    # 测试没有位置编码的TSAKT
    with torch.no_grad():
        tsakt_wo_pos_preds = tsakt_wo_pos(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
        tsakt_wo_pos_sigmoid = torch.sigmoid(tsakt_wo_pos_preds).cpu().numpy()
        tsakt_wo_pos_valid = tsakt_wo_pos_sigmoid[mask]
        
        tsakt_wo_pos_auc = roc_auc_score(labels_valid, tsakt_wo_pos_valid) if len(set(labels_valid)) > 1 else 0.5
        tsakt_wo_pos_acc = accuracy_score(labels_valid, tsakt_wo_pos_valid >= 0.5)
        
        print(f"TSAKT-w/o-Pos AUC: {tsakt_wo_pos_auc:.4f}, ACC: {tsakt_wo_pos_acc:.4f}")
        print(f"\nPosition encoding impact:")
        print(f"AUC improvement: {tsakt_ful_auc - tsakt_wo_pos_auc:.4f}")
        print(f"ACC improvement: {tsakt_ful_acc - tsakt_wo_pos_acc:.4f}")