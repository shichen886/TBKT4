import os
import json
import argparse
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


def load_results_from_json(json_path: str) -> Dict:
    """
    从JSON文件加载结果
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        dict: 结果字典
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def perform_paired_t_test(aucs_model1: List[float], aucs_model2: List[float],
                          alpha: float = 0.05) -> Dict:
    """
    执行配对t检验
    
    Args:
        aucs_model1: 模型1的AUC列表
        aucs_model2: 模型2的AUC列表
        alpha: 显著性水平
    
    Returns:
        dict: t检验结果
    """
    if len(aucs_model1) != len(aucs_model2):
        raise ValueError("AUC lists must have the same length for paired t-test")
    
    # 执行配对t检验
    t_stat, p_value = stats.ttest_rel(aucs_model1, aucs_model2)
    
    # 计算置信区间
    mean_diff = np.mean(aucs_model1) - np.mean(aucs_model2)
    std_diff = np.std([a1 - a2 for a1, a2 in zip(aucs_model1, aucs_model2)], ddof=1)
    n = len(aucs_model1)
    se_diff = std_diff / np.sqrt(n)
    
    # 95%置信区间
    ci_lower = mean_diff - stats.t.ppf(1 - alpha/2, n-1) * se_diff
    ci_upper = mean_diff + stats.t.ppf(1 - alpha/2, n-1) * se_diff
    
    # Cohen's d (效应量)
    pooled_std = np.sqrt((np.var(aucs_model1, ddof=1) + np.var(aucs_model2, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohens_d': cohens_d,
        'effect_size': interpret_cohens_d(cohens_d),
    }


def perform_bootstrap_test(aucs_model1: List[float], aucs_model2: List[float],
                          n_bootstrap: int = 10000, alpha: float = 0.05) -> Dict:
    """
    执行Bootstrap检验
    
    Args:
        aucs_model1: 模型1的AUC列表
        aucs_model2: 模型2的AUC列表
        n_bootstrap: Bootstrap采样次数
        alpha: 显著性水平
    
    Returns:
        dict: Bootstrap检验结果
    """
    n1, n2 = len(aucs_model1), len(aucs_model2)
    
    # 计算原始差异
    mean_diff = np.mean(aucs_model1) - np.mean(aucs_model2)
    
    # Bootstrap采样
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # 重采样
        sample1 = np.random.choice(aucs_model1, size=n1, replace=True)
        sample2 = np.random.choice(aucs_model2, size=n2, replace=True)
        
        # 计算差异
        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # 计算置信区间
    ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    
    # 计算p值（双尾检验）
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(mean_diff))
    
    # 计算效应量
    pooled_std = np.sqrt((np.var(aucs_model1, ddof=1) + np.var(aucs_model2, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'mean_difference': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'cohens_d': cohens_d,
        'effect_size': interpret_cohens_d(cohens_d),
        'n_bootstrap': n_bootstrap,
    }


def interpret_cohens_d(cohens_d: float) -> str:
    """
    解释Cohen's d效应量
    
    Args:
        cohens_d: Cohen's d值
    
    Returns:
        str: 效应量解释
    """
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_models(model1_name: str, model1_results: Dict,
                   model2_name: str, model2_results: Dict,
                   alpha: float = 0.05) -> Dict:
    """
    比较两个模型
    
    Args:
        model1_name: 模型1名称
        model1_results: 模型1结果
        model2_name: 模型2名称
        model2_results: 模型2结果
        alpha: 显著性水平
    
    Returns:
        dict: 比较结果
    """
    # 提取AUC列表
    aucs1 = model1_results.get('all_aucs', [])
    aucs2 = model2_results.get('all_aucs', [])
    
    if len(aucs1) == 0 or len(aucs2) == 0:
        raise ValueError("AUC lists are empty")
    
    # 执行t检验
    t_test_results = perform_paired_t_test(aucs1, aucs2, alpha)
    
    # 执行Bootstrap检验
    bootstrap_results = perform_bootstrap_test(aucs1, aucs2, n_bootstrap=10000, alpha=alpha)
    
    # 汇总结果
    comparison = {
        'model1': model1_name,
        'model2': model2_name,
        'model1_mean_auc': np.mean(aucs1),
        'model1_std_auc': np.std(aucs1),
        'model2_mean_auc': np.mean(aucs2),
        'model2_std_auc': np.std(aucs2),
        'auc_difference': np.mean(aucs1) - np.mean(aucs2),
        't_test': t_test_results,
        'bootstrap_test': bootstrap_results,
        'conclusion': draw_conclusion(t_test_results, bootstrap_results, alpha),
    }
    
    return comparison


def draw_conclusion(t_test: Dict, bootstrap: Dict, alpha: float) -> str:
    """
    根据检验结果得出结论
    
    Args:
        t_test: t检验结果
        bootstrap: Bootstrap检验结果
        alpha: 显著性水平
    
    Returns:
        str: 结论
    """
    t_sig = t_test['is_significant']
    boot_sig = bootstrap['is_significant']
    effect = t_test['effect_size']
    
    if t_sig and boot_sig:
        if effect == "large":
            return f"Significant improvement (p<0.05, large effect size)"
        elif effect == "medium":
            return f"Significant improvement (p<0.05, medium effect size)"
        elif effect == "small":
            return f"Significant improvement (p<0.05, small effect size)"
        else:
            return f"Significant improvement (p<0.05, negligible effect size)"
    else:
        return "No significant difference (p>=0.05)"


def print_comparison_results(comparison: Dict):
    """
    打印比较结果
    
    Args:
        comparison: 比较结果字典
    """
    print(f"\n{'='*80}")
    print(f"SIGNIFICANCE TEST: {comparison['model1']} vs {comparison['model2']}")
    print(f"{'='*80}\n")
    
    # 打印AUC统计
    print(f"AUC Statistics:")
    print(f"  {comparison['model1']}:")
    print(f"    Mean: {comparison['model1_mean_auc']:.4f} ± {comparison['model1_std_auc']:.4f}")
    print(f"  {comparison['model2']}:")
    print(f"    Mean: {comparison['model2_mean_auc']:.4f} ± {comparison['model2_std_auc']:.4f}")
    print(f"  Difference: {comparison['auc_difference']:+.4f}")
    
    # 打印t检验结果
    print(f"\nPaired t-test:")
    t_test = comparison['t_test']
    print(f"  t-statistic: {t_test['t_statistic']:.4f}")
    print(f"  p-value: {t_test['p_value']:.6f}")
    print(f"  Significant: {'Yes' if t_test['is_significant'] else 'No'}")
    print(f"  95% CI: [{t_test['ci_lower']:.4f}, {t_test['ci_upper']:.4f}]")
    print(f"  Cohen's d: {t_test['cohens_d']:.4f} ({t_test['effect_size']})")
    
    # 打印Bootstrap检验结果
    print(f"\nBootstrap test:")
    boot = comparison['bootstrap_test']
    print(f"  p-value: {boot['p_value']:.6f}")
    print(f"  Significant: {'Yes' if boot['is_significant'] else 'No'}")
    print(f"  95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    print(f"  Cohen's d: {boot['cohens_d']:.4f} ({boot['effect_size']})")
    print(f"  Bootstrap samples: {boot['n_bootstrap']}")
    
    # 打印结论
    print(f"\nConclusion:")
    print(f"  {comparison['conclusion']}")
    print(f"\n{'='*80}")


def save_comparison_results(comparison: Dict, output_path: str):
    """
    保存比较结果到JSON
    
    Args:
        comparison: 比较结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    print(f"\nSaved comparison results to {output_path}")


def main(args):
    """
    主函数
    """
    # 加载两个模型的结果
    print(f"\n{'='*80}")
    print("SIGNIFICANCE TESTING FOR MODEL COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"Loading results from:")
    print(f"  Model 1: {args.model1_results}")
    print(f"  Model 2: {args.model2_results}\n")
    
    results1 = load_results_from_json(args.model1_results)
    results2 = load_results_from_json(args.model2_results)
    
    # 比较模型
    comparison = compare_models(
        model1_name=args.model1_name,
        model1_results=results1,
        model2_name=args.model2_name,
        model2_results=results2,
        alpha=args.alpha,
    )
    
    # 打印结果
    print_comparison_results(comparison)
    
    # 保存结果
    if args.output:
        save_comparison_results(comparison, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Significance testing for model comparison')
    parser.add_argument('--model1_results', type=str, required=True,
                        help='Path to model 1 results JSON file')
    parser.add_argument('--model2_results', type=str, required=True,
                        help='Path to model 2 results JSON file')
    parser.add_argument('--model1_name', type=str, default='Model 1',
                        help='Name of model 1')
    parser.add_argument('--model2_name', type=str, default='Model 2',
                        help='Name of model 2')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    main(args)