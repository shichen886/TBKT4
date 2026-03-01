import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import sys
from scipy import stats


def run_training(script_name, dataset, savedir, seed, **kwargs):
    """
    运行单个seed的训练
    
    Args:
        script_name: 训练脚本名称
        dataset: 数据集名称
        savedir: 保存目录
        seed: 随机种子
        **kwargs: 其他训练参数
    
    Returns:
        dict: 训练结果
    """
    print(f"\n{'='*80}")
    print(f"Running {script_name} with seed={seed}")
    print(f"{'='*80}\n")
    
    # 构建命令
    cmd = [
        sys.executable,  # 使用当前Python解释器
        script_name,
        '--dataset', dataset,
        '--savedir', savedir,
        '--seed', str(seed),
    ]
    
    # 添加其他参数
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    # 运行训练
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running training with seed={seed}")
        print(e.stdout)
        print(e.stderr)
        return None
    
    # 读取训练结果
    config_path = os.path.join(savedir, 'config.json')
    history_path = os.path.join(savedir, f'{dataset}_training_history.json')
    
    if not os.path.exists(config_path):
        print(f"Error: config.json not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # 提取关键结果
    results = {
        'seed': seed,
        'test_auc': config.get('test_auc', history.get('test_auc', 0)),
        'test_rmse': config.get('test_rmse', 0),
        'best_val_auc': history.get('best_val_auc', 0),
        'best_epoch': history.get('best_epoch', 0),
        'generalization_gap': config.get('generalization_gap', 0),
    }
    
    return results


def run_multi_seed_training(script_name, dataset, seeds, savedir_base, **kwargs):
    """
    运行多seed训练
    
    Args:
        script_name: 训练脚本名称
        dataset: 数据集名称
        seeds: 种子列表
        savedir_base: 基础保存目录
        **kwargs: 其他训练参数
    
    Returns:
        list: 所有训练结果
    """
    all_results = []
    
    for seed in tqdm(seeds, desc=f'Running {script_name}'):
        # 为每个seed创建独立的保存目录
        savedir = os.path.join(savedir_base, f'seed_{seed}')
        os.makedirs(savedir, exist_ok=True)
        
        # 运行训练
        results = run_training(script_name, dataset, savedir, seed, **kwargs)
        
        if results is not None:
            all_results.append(results)
    
    return all_results


def compute_statistics(results):
    """
    计算统计信息（mean ± std）
    
    Args:
        results: 训练结果列表
    
    Returns:
        dict: 统计信息
    """
    if len(results) == 0:
        return {}
    
    # 提取所有AUC值
    aucs = [r['test_auc'] for r in results]
    rmses = [r['test_rmse'] for r in results]
    gaps = [r['generalization_gap'] for r in results]
    
    # 计算统计信息
    stats = {
        'num_seeds': len(results),
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'auc_min': np.min(aucs),
        'auc_max': np.max(aucs),
        'auc_values': aucs,
        'rmse_mean': np.mean(rmses),
        'rmse_std': np.std(rmses),
        'rmse_values': rmses,
        'gap_mean': np.mean(gaps),
        'gap_std': np.std(gaps),
        'gap_values': gaps,
    }
    
    return stats


def perform_t_test(results1, results2, label1='Model 1', label2='Model 2'):
    """
    执行t-test比较两个模型
    
    Args:
        results1: 模型1的结果
        results2: 模型2的结果
        label1: 模型1的标签
        label2: 模型2的标签
    
    Returns:
        dict: t-test结果
    """
    # 提取AUC值
    aucs1 = [r['test_auc'] for r in results1]
    aucs2 = [r['test_auc'] for r in results2]
    
    # 执行配对t-test
    t_stat, p_value = stats.ttest_rel(aucs1, aucs2)
    
    # 计算效应大小（Cohen's d）
    pooled_std = np.sqrt((np.var(aucs1) + np.var(aucs2)) / 2)
    cohens_d = (np.mean(aucs1) - np.mean(aucs2)) / pooled_std
    
    results = {
        'label1': label1,
        'label2': label2,
        'mean1': np.mean(aucs1),
        'mean2': np.mean(aucs2),
        'std1': np.std(aucs1),
        'std2': np.std(aucs2),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'highly_significant': p_value < 0.01,
    }
    
    return results


def save_multi_seed_results(stats, savedir, dataset, model_name):
    """
    保存多seed结果
    
    Args:
        stats: 统计信息
        savedir: 保存目录
        dataset: 数据集名称
        model_name: 模型名称
    """
    # 保存统计信息
    stats_path = os.path.join(savedir, f'{dataset}_{model_name}_multi_seed_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\nSaved multi-seed statistics to {stats_path}")
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"MULTI-SEED TRAINING RESULTS - {model_name}")
    print(f"{'='*80}\n")
    print(f"Dataset: {dataset}")
    print(f"Number of seeds: {stats['num_seeds']}")
    print(f"\nAUC Results:")
    print(f"  Mean: {stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}")
    print(f"  Min:  {stats['auc_min']:.4f}")
    print(f"  Max:  {stats['auc_max']:.4f}")
    print(f"  Values: {[f'{auc:.4f}' for auc in stats['auc_values']]}")
    print(f"\nRMSE Results:")
    print(f"  Mean: {stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}")
    print(f"\nGeneralization Gap:")
    print(f"  Mean: {stats['gap_mean']:.4f} ± {stats['gap_std']:.4f}")
    print(f"\n{'='*80}")


def main(args):
    """
    主函数：运行多seed训练
    """
    # 解析种子列表
    seeds = [int(s) for s in args.seeds.split(',')]
    
    print(f"\n{'='*80}")
    print("MULTI-SEED TRAINING")
    print(f"{'='*80}\n")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seeds: {seeds}")
    print(f"Number of seeds: {len(seeds)}")
    print(f"Base save directory: {args.savedir}")
    
    # 确定训练脚本
    script_map = {
        'tsakt-linear': 'train_tsakt_linear_final.py',
        'tsakt-linear-nopos': 'train_tsakt_linear_nopos.py',
        'tsakt-linear-rope': 'train_tsakt_linear_rope.py',
        'tsakt-linear-gate': 'train_tsakt_linear_gate.py',
    }
    
    script_name = script_map.get(args.model, args.model)
    
    # 创建基础保存目录
    model_savedir = os.path.join(args.savedir, args.model)
    os.makedirs(model_savedir, exist_ok=True)
    
    # 运行多seed训练
    results = run_multi_seed_training(
        script_name=script_name,
        dataset=args.dataset,
        seeds=seeds,
        savedir_base=model_savedir,
        embed_size=args.embed_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tensor_rank=args.tensor_rank,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        drop_prob=args.drop_prob,
    )
    
    if len(results) == 0:
        print("\nError: No successful training runs!")
        return
    
    # 计算统计信息
    stats = compute_statistics(results)
    
    # 保存结果
    save_multi_seed_results(stats, args.savedir, args.dataset, args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-seed training for TSAKT-Linear variants')
    parser.add_argument('--dataset', type=str, default='assistments12',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='tsakt-linear-nopos',
                        choices=['tsakt-linear', 'tsakt-linear-nopos', 'tsakt-linear-rope', 'tsakt-linear-gate'],
                        help='Model variant')
    parser.add_argument('--savedir', type=str, default='save/multi-seed',
                        help='Base save directory')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1011',
                        help='Comma-separated list of seeds (e.g., "42,123,456,789,1011")')
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--tensor_rank', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    
    args = parser.parse_args()
    
    main(args)
