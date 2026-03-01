import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import json

# 导入模型
from model_dkt1 import DKT1 as DKT
from model_akt import AKT
from model_sakt import SAKT
from model_tsakt import TSAKT

# 导入数据处理函数
from train_dkt1 import get_data as get_data_dkt, get_preds
from train_akt import get_data as get_data_akt
from train_sakt import get_data as get_data_sakt
from train_tsakt import get_data as get_data_tsakt, prepare_batches

# 计算指标
def compute_auc(preds, labels):
    mask = labels >= 0
    preds = preds[mask]
    labels = labels[mask]
    if len(np.unique(labels)) == 1:
        return 0.5
    return roc_auc_score(labels, preds)

def compute_rmse(preds, labels):
    mask = labels >= 0
    preds = preds[mask]
    labels = labels[mask]
    return np.sqrt(np.mean((preds - labels) ** 2))

def compute_acc(preds, labels):
    mask = labels >= 0
    preds = (preds >= 0.5).astype(int)
    labels = labels[mask]
    preds = preds[mask]
    if len(preds) == 0:
        return 0.0
    return np.mean(preds == labels)

# 评估DKT模型
def evaluate_dkt(model_path, dataset):
    print(f"\nEvaluating DKT on {dataset}...")
    
    # 加载数据
    data_path = os.path.join('data', dataset, 'preprocessed_data_train.csv')
    df = pd.read_csv(data_path, sep="\t")
    
    # 获取数据
    train_data, val_data = get_data_dkt(df, item_in=False, skill_in=True, item_out=True, skill_out=False, skill_separate=False, train_split=0.8)
    val_batches = prepare_batches(val_data, batch_size=16)  # 使用更小的批次大小
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 从文件名中提取参数
    params = {}
    
    # 初始化模型
    model = DKT(
        num_items=int(df['item_id'].max() + 2),
        num_skills=int(df['skill_id'].max() + 2),
        hid_size=200,
        num_hid_layers=1,
        drop_prob=0.5,
        item_in=False,
        skill_in=True,
        item_out=True,
        skill_out=False
    ).to(device)
    
    # 加载模型权重
    loaded_obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj.to(device)
    else:
        model.load_state_dict(loaded_obj)
    model.eval()
    
    # 评估
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_batches:
            item_inputs, skill_inputs, item_ids, skill_ids, labels = batch
            item_inputs = item_inputs.to(device) if item_inputs is not None else None
            skill_inputs = skill_inputs.to(device) if skill_inputs is not None else None
            item_ids = item_ids.to(device) if item_ids is not None else None
            skill_ids = skill_ids.to(device) if skill_ids is not None else None
            labels = labels.to(device)
            
            # DKT模型的forward方法只接受两个参数
            preds, _ = model(item_inputs, skill_inputs)
            
            # 处理输出形状
            if preds.dim() == 3:
                preds = preds.squeeze(-1)
            
            # 应用get_preds函数处理预测值
            valid_preds = get_preds(preds, item_ids, skill_ids, labels)
            valid_labels = labels[labels >= 0].float()
            
            if len(valid_preds) > 0:
                all_preds.extend(valid_preds.cpu().tolist())
                all_labels.extend(valid_labels.cpu().tolist())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) == 0:
        print(f"No valid predictions for {dataset}")
        return {'auc': 0, 'rmse': 0, 'acc': 0}
    
    auc = compute_auc(all_preds, all_labels)
    rmse = compute_rmse(all_preds, all_labels)
    acc = compute_acc(all_preds, all_labels)
    
    print(f"DKT on {dataset}: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
    return {'auc': auc, 'rmse': rmse, 'acc': acc}

# 评估AKT模型
def evaluate_akt(model_path, dataset):
    print(f"\nEvaluating AKT on {dataset}...")
    
    # 加载数据
    data_path = os.path.join('data', dataset, 'preprocessed_data_train.csv')
    df = pd.read_csv(data_path, sep="\t")
    
    # 获取数据
    train_data, val_data = get_data_akt(df, max_length=200, train_split=0.8)
    val_batches = prepare_batches(val_data, batch_size=128)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = AKT(
        num_items=int(df['item_id'].max() + 2),
        num_skills=int(df['skill_id'].max() + 2),
        embed_size=128,
        num_attn_layers=2,
        num_heads=8,
        drop_prob=0.2,
        max_pos=5
    ).to(device)
    
    # 加载模型权重
    loaded_obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj.to(device)
    else:
        model.load_state_dict(loaded_obj)
    model.eval()
    
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
            
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
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
    
    auc = compute_auc(all_preds, all_labels)
    rmse = compute_rmse(all_preds, all_labels)
    acc = compute_acc(all_preds, all_labels)
    
    print(f"AKT on {dataset}: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
    return {'auc': auc, 'rmse': rmse, 'acc': acc}

# 评估SAKT模型
def evaluate_sakt(model_path, dataset):
    print(f"\nEvaluating SAKT on {dataset}...")
    
    # 加载数据
    data_path = os.path.join('data', dataset, 'preprocessed_data_train.csv')
    df = pd.read_csv(data_path, sep="\t")
    
    # 获取数据
    train_data, val_data = get_data_sakt(df, max_length=200, train_split=0.8)
    val_batches = prepare_batches(val_data, batch_size=128)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = SAKT(
        num_items=int(df['item_id'].max() + 2),
        num_skills=int(df['skill_id'].max() + 2),
        embed_size=60,
        num_attn_layers=2,
        num_heads=5,
        encode_pos=False,
        max_pos=5,
        drop_prob=0.2
    ).to(device)
    
    # 加载模型权重
    loaded_obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj.to(device)
    else:
        model.load_state_dict(loaded_obj)
    model.eval()
    
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
            
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
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
    
    if len(all_preds) == 0:
        print(f"No valid predictions for {dataset}")
        return {'auc': 0, 'rmse': 0, 'acc': 0}
    
    auc = compute_auc(all_preds, all_labels)
    rmse = compute_rmse(all_preds, all_labels)
    acc = compute_acc(all_preds, all_labels)
    
    print(f"SAKT on {dataset}: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
    return {'auc': auc, 'rmse': rmse, 'acc': acc}

# 评估TSAKT模型
def evaluate_tsakt(model_path, dataset, encode_pos=False):
    print(f"\nEvaluating TSAKT {'(w/o Pos)' if not encode_pos else '(Ful)'} on {dataset}...")
    
    # 加载数据
    data_path = os.path.join('data', dataset, 'preprocessed_data_train.csv')
    df = pd.read_csv(data_path, sep="\t")
    
    # 获取数据
    train_data, val_data = get_data_tsakt(df, max_length=200, train_split=0.8)
    val_batches = prepare_batches(val_data, batch_size=128)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    try:
        loaded_obj = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(loaded_obj, torch.nn.Module):
            model = loaded_obj.to(device)
        else:
            # 初始化模型
            model = TSAKT(
                num_items=int(df['item_id'].max() + 2),
                num_skills=int(df['skill_id'].max() + 2),
                embed_size=60,
                num_attn_layers=2,
                num_heads=5,
                encode_pos=encode_pos,
                max_pos=5,
                drop_prob=0.2,
                tensor_rank=3
            ).to(device)
            model.load_state_dict(loaded_obj)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return {'auc': 0, 'rmse': 0, 'acc': 0}
    
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
            
            try:
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                
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
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) == 0:
        print(f"No valid predictions for {dataset}")
        return {'auc': 0, 'rmse': 0, 'acc': 0}
    
    auc = compute_auc(all_preds, all_labels)
    rmse = compute_rmse(all_preds, all_labels)
    acc = compute_acc(all_preds, all_labels)
    
    print(f"TSAKT {'(w/o Pos)' if not encode_pos else '(Ful)'} on {dataset}: AUC={auc:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}")
    return {'auc': auc, 'rmse': rmse, 'acc': acc}

# 主评估函数
def evaluate_all_models():
    print("=" * 80)
    print("Evaluating All Models")
    print("=" * 80)
    
    datasets = ['assistments09', 'assistments12', 'assistments15']
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        
        # 评估DKT（使用不同的batch_size）
        dkt_batch_size = '8' if dataset == 'assistments09' else '128'
        dkt_path = os.path.join('save', 'dkt1', f'{dataset},batch_size={dkt_batch_size},item_in=False,skill_in=True,item_out=True,skill_out=False,skill_separate=False')
        if os.path.exists(dkt_path):
            results[dataset]['dkt'] = evaluate_dkt(dkt_path, dataset)
        
        # 评估AKT（使用不同的batch_size和max_length）
        akt_batch_size = '16' if dataset == 'assistments09' else '128'
        akt_max_length = '100' if dataset == 'assistments09' else '200'
        akt_max_pos = '5' if dataset == 'assistments09' else '10'
        akt_path = os.path.join('save', 'akt', f'{dataset},batch_size={akt_batch_size},max_length={akt_max_length},max_pos={akt_max_pos}')
        if os.path.exists(akt_path):
            results[dataset]['akt'] = evaluate_akt(akt_path, dataset)
        
        # 评估SAKT
        sakt_path = os.path.join('save', 'sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
        if os.path.exists(sakt_path):
            results[dataset]['sakt'] = evaluate_sakt(sakt_path, dataset)
        
        # 评估TSAKT-w/o-Pos
        tsakt_wo_pos_path = os.path.join('save', 'tsakt-wo-pos', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
        if os.path.exists(tsakt_wo_pos_path):
            results[dataset]['tsakt_wo_pos'] = evaluate_tsakt(tsakt_wo_pos_path, dataset, encode_pos=False)
        
        # 评估TSAKT-Ful（使用正确的max_pos=200）
        tsakt_ful_path = os.path.join('save', 'tsakt-ful-v2', f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        if os.path.exists(tsakt_ful_path):
            results[dataset]['tsakt_ful'] = evaluate_tsakt(tsakt_ful_path, dataset, encode_pos=True)
    
    # 保存结果
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    
    # 打印结果表格
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        print("-" * 60)
        print(f"{'Model':<20} {'AUC':<10} {'ACC':<10} {'RMSE':<10}")
        print("-" * 60)
        
        for model_name, metrics in results[dataset].items():
            model_display = {
                'dkt': 'DKT',
                'akt': 'AKT',
                'sakt': 'SAKT',
                'tsakt_wo_pos': 'TSAKT-w/o-Pos',
                'tsakt_ful': 'TSAKT-Ful'
            }.get(model_name, model_name)
            
            print(f"{model_display:<20} {metrics.get('auc', 0):<10.4f} {metrics.get('acc', 0):<10.4f} {metrics.get('rmse', 0):<10.4f}")
    
    print("\n" + "=" * 80)
    print("Results saved to evaluation_results.json")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_all_models()
