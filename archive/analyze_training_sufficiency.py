import os
import torch
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model_sakt import SAKT
from model_tsakt import TSAKT

def analyze_model_state(model_path, model_name):
    """分析模型状态"""
    print(f"\n{'='*80}")
    print(f"分析模型状态: {model_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    result = {
        'model_path': model_path,
        'model_name': model_name,
        'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
        'total_params': 0,
        'trainable_params': 0,
        'param_stats': {},
        'has_optimizer_state': False
    }
    
    if isinstance(model, torch.nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        result['total_params'] = total_params
        result['trainable_params'] = trainable_params
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        param_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_stats[name] = {
                    'shape': param.shape,
                    'numel': param.numel(),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }
        
        result['param_stats'] = param_stats
        
        print(f"\n参数统计:")
        print(f"{'参数名':<50} {'形状':<20} {'均值':<10} {'标准差':<10}")
        print("-" * 90)
        
        for name, stats in param_stats.items():
            print(f"{name:<50} {str(stats['shape']):<20} {stats['mean']:<10.4f} {stats['std']:<10.4f}")
    
    print(f"\n文件大小: {result['file_size_mb']:.2f} MB")
    
    return result

def compare_models(sakt_result, tsakt_result):
    """比较两个模型"""
    if not sakt_result or not tsakt_result:
        return
    
    print(f"\n{'='*80}")
    print(f"模型对比: SAKT vs TSAKT-w/o-Pos")
    print(f"{'='*80}")
    
    print(f"{'指标':<30} {'SAKT':<25} {'TSAKT-w/o-Pos':<25} {'差异':<15}")
    print("-" * 95)
    
    print(f"{'文件大小 (MB)':<30} {sakt_result['file_size_mb']:<25.2f} {tsakt_result['file_size_mb']:<25.2f} {tsakt_result['file_size_mb'] - sakt_result['file_size_mb']:<15.2f}")
    print(f"{'总参数量':<30} {sakt_result['total_params']:<25,} {tsakt_result['total_params']:<25,} {tsakt_result['total_params'] - sakt_result['total_params']:<15,}")
    
    if sakt_result['param_stats'] and tsakt_result['param_stats']:
        print(f"\n参数统计对比:")
        print(f"{'参数名':<50} {'SAKT均值':<15} {'TSAKT均值':<15} {'差异':<15}")
        print("-" * 95)
        
        sakt_params = set(sakt_result['param_stats'].keys())
        tsakt_params = set(tsakt_result['param_stats'].keys())
        
        common_params = sakt_params & tsakt_params
        
        for param_name in sorted(common_params):
            sakt_mean = sakt_result['param_stats'][param_name]['mean']
            tsakt_mean = tsakt_result['param_stats'][param_name]['mean']
            diff = tsakt_mean - sakt_mean
            
            print(f"{param_name:<50} {sakt_mean:<15.4f} {tsakt_mean:<15.4f} {diff:<15.4f}")

def main():
    print("=" * 80)
    print("TSAKT-w/o-Pos 训练充分性分析")
    print("=" * 80)
    
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")
        
        sakt_path = os.path.join('save', 'sakt', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
        
        tsakt_wo_pos_path = os.path.join('save', 'tsakt', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
        
        sakt_result = analyze_model_state(sakt_path, 'SAKT')
        tsakt_result = analyze_model_state(tsakt_wo_pos_path, 'TSAKT-w/o-Pos')
        
        compare_models(sakt_result, tsakt_result)

if __name__ == "__main__":
    main()
