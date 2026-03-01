import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def read_tensorboard_logs(log_dir):
    """读取TensorBoard日志文件"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()
    
    data = {}
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    
    return data

def analyze_training(dataset, model_name, log_dir):
    """分析训练情况"""
    print(f"\n{'='*80}")
    print(f"分析训练: {model_name} on {dataset}")
    print(f"{'='*80}")
    
    if not os.path.exists(log_dir):
        print(f"日志目录不存在: {log_dir}")
        return None
    
    try:
        data = read_tensorboard_logs(log_dir)
    except Exception as e:
        print(f"读取日志失败: {e}")
        return None
    
    if not data:
        print("没有找到训练数据")
        return None
    
    result = {
        'dataset': dataset,
        'model_name': model_name,
        'log_dir': log_dir,
        'epochs': 0,
        'final_train_auc': None,
        'final_val_auc': None,
        'best_val_auc': None,
        'best_epoch': 0,
        'train_auc_history': [],
        'val_auc_history': [],
        'train_loss_history': [],
        'val_loss_history': []
    }
    
    if 'epoch_train_auc' in data:
        result['train_auc_history'] = [v for s, v in data['epoch_train_auc']]
        result['final_train_auc'] = result['train_auc_history'][-1] if result['train_auc_history'] else None
    
    if 'epoch_val_auc' in data:
        result['val_auc_history'] = [v for s, v in data['epoch_val_auc']]
        result['final_val_auc'] = result['val_auc_history'][-1] if result['val_auc_history'] else None
        result['best_val_auc'] = max(result['val_auc_history']) if result['val_auc_history'] else None
        result['best_epoch'] = result['val_auc_history'].index(result['best_val_auc']) + 1 if result['val_auc_history'] else 0
    
    if 'epoch_train_loss' in data:
        result['train_loss_history'] = [v for s, v in data['epoch_train_loss']]
    
    if 'epoch_val_loss' in data:
        result['val_loss_history'] = [v for s, v in data['epoch_val_loss']]
    
    result['epochs'] = len(result['train_auc_history']) if result['train_auc_history'] else 0
    
    print(f"训练轮数: {result['epochs']}")
    print(f"最佳验证AUC: {result['best_val_auc']:.4f} (Epoch {result['best_epoch']})")
    print(f"最终训练AUC: {result['final_train_auc']:.4f}")
    print(f"最终验证AUC: {result['final_val_auc']:.4f}")
    
    if result['epochs'] > 0:
        print(f"训练AUC范围: {min(result['train_auc_history']):.4f} - {max(result['train_auc_history']):.4f}")
        print(f"验证AUC范围: {min(result['val_auc_history']):.4f} - {max(result['val_auc_history']):.4f}")
    
    return result

def main():
    print("=" * 80)
    print("TSAKT-w/o-Pos 训练情况分析")
    print("=" * 80)
    
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")
        
        TSAKT_wo_pos_log_dir = os.path.join('runs', 'tsakt', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
        
        TSAKT_ful_log_dir = os.path.join('runs', 'tsakt-ful-v2', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        
        sakt_log_dir = os.path.join('runs', 'sakt', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
        
        TSAKT_wo_pos_result = analyze_training(dataset, 'TSAKT-w/o-Pos', TSAKT_wo_pos_log_dir)
        TSAKT_ful_result = analyze_training(dataset, 'TSAKT-Ful', TSAKT_ful_log_dir)
        sakt_result = analyze_training(dataset, 'SAKT', sakt_log_dir)
        
        if TSAKT_wo_pos_result and sakt_result:
            print(f"\n{'-'*80}")
            print(f"对比分析: TSAKT-w/o-Pos vs SAKT")
            print(f"{'-'*80}")
            
            print(f"{'指标':<30} {'TSAKT-w/o-Pos':<20} {'SAKT':<20} {'差异':<15}")
            print("-" * 85)
            
            print(f"{'训练轮数':<30} {TSAKT_wo_pos_result['epochs']:<20} {sakt_result['epochs']:<20} {TSAKT_wo_pos_result['epochs'] - sakt_result['epochs']:<15}")
            print(f"{'最佳验证AUC':<30} {TSAKT_wo_pos_result['best_val_auc']:<20.4f} {sakt_result['best_val_auc']:<20.4f} {TSAKT_wo_pos_result['best_val_auc'] - sakt_result['best_val_auc']:<15.4f}")
            print(f"{'最佳Epoch':<30} {TSAKT_wo_pos_result['best_epoch']:<20} {sakt_result['best_epoch']:<20} {TSAKT_wo_pos_result['best_epoch'] - sakt_result['best_epoch']:<15}")
            print(f"{'最终训练AUC':<30} {TSAKT_wo_pos_result['final_train_auc']:<20.4f} {sakt_result['final_train_auc']:<20.4f} {TSAKT_wo_pos_result['final_train_auc'] - sakt_result['final_train_auc']:<15.4f}")
            print(f"{'最终验证AUC':<30} {TSAKT_wo_pos_result['final_val_auc']:<20.4f} {sakt_result['final_val_auc']:<20.4f} {TSAKT_wo_pos_result['final_val_auc'] - sakt_result['final_val_auc']:<15.4f}")
            
            if TSAKT_wo_pos_result['epochs'] < 10:
                print(f"\n⚠️  警告: TSAKT-w/o-Pos 训练轮数过少 ({TSAKT_wo_pos_result['epochs']} epochs)，可能训练不充分")
            
            if TSAKT_wo_pos_result['best_val_auc'] < sakt_result['best_val_auc']:
                print(f"\n⚠️  警告: TSAKT-w/o-Pos 最佳验证AUC低于SAKT ({TSAKT_wo_pos_result['best_val_auc']:.4f} vs {sakt_result['best_val_auc']:.4f})")
            
            if TSAKT_wo_pos_result['best_epoch'] == TSAKT_wo_pos_result['epochs']:
                print(f"\n⚠️  警告: TSAKT-w/o-Pos 最佳Epoch在最后一轮，可能还在上升")
        
        if TSAKT_wo_pos_result and TSAKT_ful_result:
            print(f"\n{'-'*80}")
            print(f"对比分析: TSAKT-w/o-Pos vs TSAKT-Ful")
            print(f"{'-'*80}")
            
            print(f"{'指标':<30} {'TSAKT-w/o-Pos':<20} {'TSAKT-Ful':<20} {'差异':<15}")
            print("-" * 85)
            
            print(f"{'训练轮数':<30} {TSAKT_wo_pos_result['epochs']:<20} {TSAKT_ful_result['epochs']:<20} {TSAKT_wo_pos_result['epochs'] - TSAKT_ful_result['epochs']:<15}")
            print(f"{'最佳验证AUC':<30} {TSAKT_wo_pos_result['best_val_auc']:<20.4f} {TSAKT_ful_result['best_val_auc']:<20.4f} {TSAKT_wo_pos_result['best_val_auc'] - TSAKT_ful_result['best_val_auc']:<15.4f}")
            print(f"{'最佳Epoch':<30} {TSAKT_wo_pos_result['best_epoch']:<20} {TSAKT_ful_result['best_epoch']:<20} {TSAKT_wo_pos_result['best_epoch'] - TSAKT_ful_result['best_epoch']:<15}")
            print(f"{'最终训练AUC':<30} {TSAKT_wo_pos_result['final_train_auc']:<20.4f} {TSAKT_ful_result['final_train_auc']:<20.4f} {TSAKT_wo_pos_result['final_train_auc'] - TSAKT_ful_result['final_train_auc']:<15.4f}")
            print(f"{'最终验证AUC':<30} {TSAKT_wo_pos_result['final_val_auc']:<20.4f} {TSAKT_ful_result['final_val_auc']:<20.4f} {TSAKT_wo_pos_result['final_val_auc'] - TSAKT_ful_result['final_val_auc']:<15.4f}")

if __name__ == "__main__":
    main()
