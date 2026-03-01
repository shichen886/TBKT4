import os
import json
import numpy as np
from scipy import stats
import pandas as pd


def load_multi_seed_results(savedir, dataset, model_name):
    """
    加载多seed训练结果
    
    Args:
        savedir: 保存目录
        dataset: 数据集名称
        model_name: 模型名称
    
    Returns:
        dict: 多seed结果
    """
    stats_path = os.path.join(savedir, f'{dataset}_{model_name}_multi_seed_stats.json')
    
    if not os.path.exists(stats_path):
        print(f"Error: Stats file not found at {stats_path}")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
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
    aucs1 = results1['auc_values']
    aucs2 = results2['auc_values']
    
    # 执行配对t-test
    t_stat, p_value = stats.ttest_rel(aucs1, aucs2)
    
    # 计算效应大小（Cohen's d）
    pooled_std = np.sqrt((np.var(aucs1) + np.var(aucs2)) / 2)
    cohens_d = (np.mean(aucs1) - np.mean(aucs2)) / pooled_std
    
    # 计算置信区间
    mean_diff = np.mean(aucs1) - np.mean(aucs2)
    se_diff = np.sqrt(np.var(aucs1) / len(aucs1) + np.var(aucs2) / len(aucs2))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    results = {
        'label1': label1,
        'label2': label2,
        'mean1': np.mean(aucs1),
        'mean2': np.mean(aucs2),
        'std1': np.std(aucs1),
        'std2': np.std(aucs2),
        'mean_diff': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < 0.05,
        'highly_significant': p_value < 0.01,
        'interpretation': interpret_p_value(p_value),
    }
    
    return results


def interpret_p_value(p_value):
    """
    解释p-value
    
    Args:
        p_value: p-value
    
    Returns:
        str: 解释
    """
    if p_value < 0.001:
        return "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "Highly significant (p < 0.01)"
    elif p_value < 0.05:
        return "Significant (p < 0.05)"
    elif p_value < 0.1:
        return "Marginally significant (p < 0.1)"
    else:
        return "Not significant (p >= 0.1)"


def main():
    """
    主函数：进行t-test比较
    """
    savedir = 'save/multi-seed'
    dataset = 'assistments17_long_200'
    
    # 加载所有模型的多seed结果
    models = {
        'TSAKT-Linear-NoPos': load_multi_seed_results(savedir, dataset, 'tsakt-linear-nopos'),
        'TSAKT-Linear': load_multi_seed_results(savedir, dataset, 'tsakt-linear'),
        'TSAKT-Linear-RoPE': load_multi_seed_results(savedir, dataset, 'tsakt-linear-rope'),
        'TSAKT-Linear-Gate': load_multi_seed_results(savedir, dataset, 'tsakt-linear-gate'),
    }
    
    # 检查是否所有模型都加载成功
    for model_name, results in models.items():
        if results is None:
            print(f"Error: Failed to load results for {model_name}")
            return
    
    print(f"\n{'='*80}")
    print("T-TEST ANALYSIS - NoPos vs Others")
    print(f"{'='*80}\n")
    print(f"Dataset: {dataset}")
    print(f"Number of seeds: 5")
    
    # 以NoPos为基准，与其他模型进行比较
    baseline = models['TSAKT-Linear-NoPos']
    
    comparisons = []
    for model_name, results in models.items():
        if model_name == 'TSAKT-Linear-NoPos':
            continue
        
        # 执行t-test
        t_test_results = perform_t_test(baseline, results, 'TSAKT-Linear-NoPos', model_name)
        comparisons.append(t_test_results)
    
    # 创建DataFrame
    df = pd.DataFrame(comparisons)
    
    # 打印结果
    print(f"\n{'='*80}")
    print("T-TEST RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Baseline: TSAKT-Linear-NoPos")
    print(f"Baseline AUC: {baseline['auc_mean']:.4f} ± {baseline['auc_std']:.4f}")
    print(f"Baseline AUC Values: {[f'{auc:.4f}' for auc in baseline['auc_values']]}\n")
    
    print(f"{'='*80}")
    print("COMPARISONS")
    print(f"{'='*80}\n")
    
    for _, row in df.iterrows():
        print(f"{row['label2']} vs {row['label1']}:")
        print(f"  Mean Difference: {row['mean_diff']:+.4f}")
        print(f"  t-statistic: {row['t_statistic']:.4f}")
        print(f"  p-value: {row['p_value']:.6f}")
        print(f"  Cohen's d: {row['cohens_d']:.4f}")
        print(f"  95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
        print(f"  Significance: {row['interpretation']}")
        print()
    
    # 保存结果
    results_path = os.path.join(savedir, f'{dataset}_t_test_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'dataset': dataset,
            'baseline': baseline,
            'comparisons': [{k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in comp.items()} for comp in comparisons],
        }, f, indent=4)
    
    print(f"\nSaved t-test results to {results_path}")
    
    # 保存CSV格式
    csv_path = os.path.join(savedir, f'{dataset}_t_test_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved t-test results to {csv_path}")


if __name__ == "__main__":
    main()
