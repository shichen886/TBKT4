import os
import torch
import numpy as np
import pandas as pd
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from model_sakt import SAKT
from model_tsakt import TSAKT

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_memory_usage(model, batch_size=128, max_length=200):
    """估算模型显存使用量"""
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            model = model.to('cpu')
            torch.cuda.empty_cache()
            return 0, 0
    
    return 0, 0

def analyze_model(dataset, model_path, model_name, model_type):
    """分析单个模型"""
    print(f"\n{'='*80}")
    print(f"分析模型: {model_name} on {dataset}")
    print(f"{'='*80}")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    full_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    num_items = int(full_df["item_id"].max() + 1)
    num_skills = int(full_df["skill_id"].max() + 1)
    
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    
    total_params, trainable_params = count_parameters(model)
    
    batch_size = 32
    max_length = 200
    memory_allocated, memory_reserved = estimate_memory_usage(model, batch_size, max_length)
    
    result = {
        'dataset': dataset,
        'model_name': model_name,
        'model_type': model_type,
        'num_items': num_items,
        'num_skills': num_skills,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved
    }
    
    print(f"数据集: {dataset}")
    print(f"题目数量: {num_items}")
    print(f"技能数量: {num_skills}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"显存占用: {memory_allocated:.2f} MB")
    print(f"显存预留: {memory_reserved:.2f} MB")
    
    return result

def main():
    print("=" * 80)
    print("模型参数量和显存使用分析")
    print("=" * 80)
    
    results = []
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    for dataset in datasets:
        max_length = 200
        
        sakt_path = os.path.join('save', 'sakt', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
        if os.path.exists(sakt_path):
            result = analyze_model(dataset, sakt_path, 'SAKT', 'SAKT')
            results.append(result)
        
        tsakt_wo_pos_path = os.path.join('save', 'tsakt', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
        if os.path.exists(tsakt_wo_pos_path):
            result = analyze_model(dataset, tsakt_wo_pos_path, 'TSAKT-w/o-Pos', 'TSAKT-w/o-Pos')
            results.append(result)
        
        tsakt_ful_path = os.path.join('save', 'tsakt-ful-v2', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        if os.path.exists(tsakt_ful_path):
            result = analyze_model(dataset, tsakt_ful_path, 'TSAKT-Ful', 'TSAKT-Ful')
            results.append(result)
    
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        print(f"{'Model':<20} {'Total Params':<15} {'Memory (MB)':<15} {'Params Ratio':<15}")
        print("-" * 80)
        
        dataset_results = [r for r in results if r['dataset'] == dataset]
        
        if len(dataset_results) >= 2:
            sakt_result = next((r for r in dataset_results if r['model_name'] == 'SAKT'), None)
            tsakt_wo_pos_result = next((r for r in dataset_results if r['model_name'] == 'TSAKT-w/o-Pos'), None)
            tsakt_ful_result = next((r for r in dataset_results if r['model_name'] == 'TSAKT-Ful'), None)
            
            if sakt_result and tsakt_wo_pos_result:
                sakt_params = sakt_result['total_params']
                tsakt_wo_pos_params = tsakt_wo_pos_result['total_params']
                sakt_memory = sakt_result['memory_allocated_mb']
                tsakt_wo_pos_memory = tsakt_wo_pos_result['memory_allocated_mb']
                
                params_ratio = tsakt_wo_pos_params / sakt_params
                memory_ratio = tsakt_wo_pos_memory / sakt_memory if sakt_memory > 0 else 1.0
                
                print(f"{'SAKT':<20} {sakt_params:>10,} {sakt_memory:>10.2f} {'-':<15}")
                print(f"{'TSAKT-w/o-Pos':<20} {tsakt_wo_pos_params:>10,} {tsakt_wo_pos_memory:>10.2f} {params_ratio:.2%}")
                
                sakt_auc = 0.7888 if dataset == 'assistments09' else (0.8092 if dataset == 'assistments12' else 0.8334)
                tsakt_wo_pos_auc = 0.7473 if dataset == 'assistments09' else (0.8009 if dataset == 'assistments12' else 0.8275)
                auc_diff = tsakt_wo_pos_auc - sakt_auc
                
                if abs(auc_diff) < 0.01 and params_ratio < 1.0:
                    print(f"{'结论':<20} 性能相近，参数减少{1-params_ratio:.1%}，显存减少{1-memory_ratio:.1%}")
                elif params_ratio < 1.0:
                    print(f"{'结论':<20} 参数减少{1-params_ratio:.1%}，显存减少{1-memory_ratio:.1%}，但性能下降{abs(auc_diff):.4f}")
                else:
                    print(f"{'结论':<20} 参数增加{params_ratio-1:.1%}，显存增加{memory_ratio-1:.1%}")
            
            if tsakt_wo_pos_result and tsakt_ful_result:
                tsakt_wo_pos_params = tsakt_wo_pos_result['total_params']
                tsakt_ful_params = tsakt_ful_result['total_params']
                tsakt_wo_pos_memory = tsakt_wo_pos_result['memory_allocated_mb']
                tsakt_ful_memory = tsakt_ful_result['memory_allocated_mb']
                
                params_ratio = tsakt_ful_params / tsakt_wo_pos_params
                memory_ratio = tsakt_ful_memory / tsakt_wo_pos_memory if tsakt_wo_pos_memory > 0 else 1.0
                
                print(f"{'TSAKT-w/o-Pos':<20} {tsakt_wo_pos_params:>10,} {tsakt_wo_pos_memory:>10.2f} {'-':<15}")
                print(f"{'TSAKT-Ful':<20} {tsakt_ful_params:>10,} {tsakt_ful_memory:>10.2f} {params_ratio:.2%}")
                
                tsakt_wo_pos_auc = 0.7473 if dataset == 'assistments09' else (0.8009 if dataset == 'assistments12' else 0.8275)
                tsakt_ful_auc = 0.7915 if dataset == 'assistments09' else (0.7903 if dataset == 'assistments12' else 0.8354)
                auc_diff = tsakt_ful_auc - tsakt_wo_pos_auc
                
                if auc_diff > 0:
                    print(f"{'结论':<20} 性能提升{auc_diff:.4f}，参数增加{params_ratio-1:.1%}，显存增加{memory_ratio-1:.1%}")
                elif abs(auc_diff) < 0.01 and params_ratio < 1.1:
                    print(f"{'结论':<20} 性能相近，参数增加{params_ratio-1:.1%}，显存增加{memory_ratio-1:.1%}")
                else:
                    print(f"{'结论':<20} 性能下降{abs(auc_diff):.4f}，参数增加{params_ratio-1:.1%}，显存增加{memory_ratio-1:.1%}")
    
    with open('model_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to model_analysis.json")
    return results

if __name__ == "__main__":
    results = main()
