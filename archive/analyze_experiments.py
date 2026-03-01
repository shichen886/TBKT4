import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class ExperimentAnalyzer:
    def __init__(self, results_file='experiments/experiment_results.json'):
        """
        实验分析器
        
        Args:
            results_file: 实验结果文件
        """
        self.results_file = Path(results_file)
        self.results = self._load_results()
        self.df = pd.DataFrame(self.results)
        
    def _load_results(self):
        """
        加载实验结果
        """
        if not self.results_file.exists():
            print(f'实验结果文件不存在: {self.results_file}')
            return []
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    
    def filter_successful(self):
        """
        过滤成功的实验结果
        """
        successful = [r for r in self.results if r['status'] == 'success']
        return pd.DataFrame(successful)
    
    def generate_comparison_table(self):
        """
        生成模型对比表格
        """
        df = self.filter_successful()
        
        if df.empty:
            print('没有成功的实验结果！')
            return None
        
        pivot_auc = df.pivot_table(
            index='model',
            columns='dataset',
            values='auc',
            aggfunc='mean'
        )
        
        pivot_acc = df.pivot_table(
            index='model',
            columns='dataset',
            values='acc',
            aggfunc='mean'
        )
        
        pivot_rmse = df.pivot_table(
            index='model',
            columns='dataset',
            values='rmse',
            aggfunc='mean'
        )
        
        return {
            'auc': pivot_auc,
            'acc': pivot_acc,
            'rmse': pivot_rmse
        }
    
    def generate_ablation_table(self):
        """
        生成消融实验表格
        """
        df = self.filter_successful()
        
        if df.empty:
            print('没有成功的实验结果！')
            return None
        
        ablation_models = ['sakt', 'tsakt']
        df_ablation = df[df['model'].isin(ablation_models)]
        
        if df_ablation.empty:
            print('没有消融实验结果！')
            return None
        
        pivot_auc = df_ablation.pivot_table(
            index='model',
            columns='dataset',
            values='auc',
            aggfunc='mean'
        )
        
        return pivot_auc
    
    def plot_comparison_bar(self, metric='auc', save_path='experiments/comparison_bar.png'):
        """
        绘制模型对比柱状图
        
        Args:
            metric: 指标名称（auc、acc、rmse）
            save_path: 保存路径
        """
        df = self.filter_successful()
        
        if df.empty:
            print('没有成功的实验结果！')
            return
        
        plt.figure(figsize=(12, 6))
        
        pivot_df = df.pivot_table(
            index='model',
            columns='dataset',
            values=metric,
            aggfunc='mean'
        )
        
        pivot_df.plot(kind='bar', ax=plt.gca())
        
        plt.title(f'Model Comparison ({metric.upper()})', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'对比柱状图已保存到 {save_path}')
        plt.close()
    
    def plot_ablation_line(self, save_path='experiments/ablation_line.png'):
        """
        绘制消融实验折线图
        """
        df = self.filter_successful()
        
        if df.empty:
            print('没有成功的实验结果！')
            return
        
        ablation_models = ['sakt', 'tsakt']
        df_ablation = df[df['model'].isin(ablation_models)]
        
        if df_ablation.empty:
            print('没有消融实验结果！')
            return
        
        plt.figure(figsize=(12, 6))
        
        pivot_df = df_ablation.pivot_table(
            index='dataset',
            columns='model',
            values='auc',
            aggfunc='mean'
        )
        
        pivot_df.plot(kind='line', marker='o', ax=plt.gca())
        
        plt.title('Ablation Study (AUC)', fontsize=14)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('AUC', fontsize=12)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'消融实验折线图已保存到 {save_path}')
        plt.close()
    
    def plot_heatmap(self, metric='auc', save_path='experiments/heatmap.png'):
        """
        绘制热力图
        """
        df = self.filter_successful()
        
        if df.empty:
            print('没有成功的实验结果！')
            return
        
        pivot_df = df.pivot_table(
            index='model',
            columns='dataset',
            values=metric,
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': metric.upper()})
        
        plt.title(f'Model Performance Heatmap ({metric.upper()})', fontsize=14)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'热力图已保存到 {save_path}')
        plt.close()
    
    def save_latex_tables(self, output_dir='experiments/latex_tables'):
        """
        保存LaTeX表格
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        comparison_tables = self.generate_comparison_table()
        
        if comparison_tables:
            for metric, table in comparison_tables.items():
                latex_file = output_path / f'comparison_{metric}.tex'
                table.to_latex(latex_file, float_format='%.4f')
                print(f'LaTeX表格已保存到 {latex_file}')
        
        ablation_table = self.generate_ablation_table()
        
        if ablation_table is not None:
            latex_file = output_path / 'ablation.tex'
            ablation_table.to_latex(latex_file, float_format='%.4f')
            print(f'消融实验LaTeX表格已保存到 {latex_file}')
    
    def generate_report(self, output_file='experiments/experiment_report.txt'):
        """
        生成实验报告
        """
        df = self.filter_successful()
        
        if df.empty:
            print('没有成功的实验结果！')
            return
        
        report = []
        report.append('='*80)
        report.append('实验报告')
        report.append('='*80)
        report.append('')
        
        report.append('1. 实验概览')
        report.append('-'*80)
        report.append(f'总实验数: {len(self.results)}')
        report.append(f'成功实验数: {len(df)}')
        report.append(f'失败实验数: {len(self.results) - len(df)}')
        report.append('')
        
        report.append('2. 模型对比')
        report.append('-'*80)
        
        model_groups = df.groupby('model')
        for model, group in model_groups:
            avg_auc = group['auc'].mean()
            avg_acc = group['acc'].mean()
            avg_rmse = group['rmse'].mean()
            report.append(f'{model}:')
            report.append(f'  AUC: {avg_auc:.4f}')
            report.append(f'  ACC: {avg_acc:.4f}')
            report.append(f'  RMSE: {avg_rmse:.4f}')
            report.append('')
        
        report.append('3. 数据集对比')
        report.append('-'*80)
        
        dataset_groups = df.groupby('dataset')
        for dataset, group in dataset_groups:
            avg_auc = group['auc'].mean()
            avg_acc = group['acc'].mean()
            avg_rmse = group['rmse'].mean()
            report.append(f'{dataset}:')
            report.append(f'  AUC: {avg_auc:.4f}')
            report.append(f'  ACC: {avg_acc:.4f}')
            report.append(f'  RMSE: {avg_rmse:.4f}')
            report.append('')
        
        report.append('4. 最佳模型')
        report.append('-'*80)
        
        best_auc_idx = df['auc'].idxmax()
        best_auc_model = df.loc[best_auc_idx, 'model']
        best_auc_dataset = df.loc[best_auc_idx, 'dataset']
        best_auc_value = df.loc[best_auc_idx, 'auc']
        
        report.append(f'最佳AUC: {best_auc_model} on {best_auc_dataset} ({best_auc_value:.4f})')
        report.append('')
        
        report.append('='*80)
        
        report_text = '\n'.join(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f'实验报告已保存到 {output_file}')
        print(report_text)


def main():
    analyzer = ExperimentAnalyzer()
    
    if not analyzer.results:
        print('没有实验结果！请先运行实验。')
        return
    
    print('开始分析实验结果...\n')
    
    analyzer.generate_report()
    
    print('\n生成图表...')
    analyzer.plot_comparison_bar(metric='auc')
    analyzer.plot_comparison_bar(metric='acc')
    analyzer.plot_comparison_bar(metric='rmse')
    analyzer.plot_ablation_line()
    analyzer.plot_heatmap(metric='auc')
    
    print('\n生成LaTeX表格...')
    analyzer.save_latex_tables()
    
    print('\n分析完成！')


if __name__ == '__main__':
    main()