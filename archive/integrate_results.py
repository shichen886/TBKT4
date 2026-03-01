import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentIntegrator:
    def __init__(self, experiments_dir='experiments'):
        """
        实验结果整合器
        
        Args:
            experiments_dir: 实验结果目录
        """
        self.experiments_dir = Path(experiments_dir)
        self.training_results = []
        self.evaluation_results = []
        
    def load_training_results(self):
        """
        加载训练结果
        
        Returns:
            训练结果DataFrame
        """
        training_file = self.experiments_dir / 'training_results.json'
        
        if not training_file.exists():
            print(f'训练结果文件不存在: {training_file}')
            return None
        
        with open(training_file, 'r', encoding='utf-8') as f:
            self.training_results = json.load(f)
        
        df = pd.DataFrame(self.training_results)
        print(f'加载了 {len(df)} 条训练结果')
        
        return df
    
    def load_evaluation_results(self):
        """
        加载评估结果
        
        Returns:
            评估结果DataFrame
        """
        evaluation_file = self.experiments_dir / 'evaluation_results.json'
        
        if not evaluation_file.exists():
            print(f'评估结果文件不存在: {evaluation_file}')
            return None
        
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            self.evaluation_results = json.load(f)
        
        df = pd.DataFrame(self.evaluation_results)
        print(f'加载了 {len(df)} 条评估结果')
        
        return df
    
    def merge_results(self):
        """
        整合训练和评估结果
        
        Returns:
            整合后的结果DataFrame
        """
        training_df = self.load_training_results()
        evaluation_df = self.load_evaluation_results()
        
        if training_df is None and evaluation_df is None:
            print('没有可用的结果！')
            return None
        
        if training_df is not None and evaluation_df is not None:
            # 合并训练和评估结果
            merged_df = pd.merge(
                training_df[['model', 'dataset', 'params', 'status', 'timestamp']],
                evaluation_df[['model', 'dataset', 'auc', 'acc', 'rmse', 'status']],
                on=['model', 'dataset'],
                how='outer',
                suffixes=('_train', '_eval')
            )
            
            # 合并状态
            merged_df['status'] = merged_df['status_eval'].fillna(merged_df['status_train'])
            
            # 移除重复的列
            merged_df = merged_df.drop(columns=['status_train', 'status_eval'])
            
            print(f'整合了 {len(merged_df)} 条结果')
            
            return merged_df
        elif training_df is not None:
            return training_df
        else:
            return evaluation_df
    
    def generate_comparison_table(self, merged_df):
        """
        生成模型对比表格
        
        Args:
            merged_df: 整合后的结果DataFrame
        
        Returns:
            对比表格DataFrame
        """
        if merged_df is None:
            return None
        
        # 过滤成功的结果
        successful_df = merged_df[merged_df['status'] == 'success']
        
        if successful_df.empty:
            print('没有成功的实验结果！')
            return None
        
        # 生成对比表格
        pivot_auc = successful_df.pivot_table(
            index='model',
            columns='dataset',
            values='auc',
            aggfunc='mean'
        )
        
        pivot_acc = successful_df.pivot_table(
            index='model',
            columns='dataset',
            values='acc',
            aggfunc='mean'
        )
        
        pivot_rmse = successful_df.pivot_table(
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
    
    def generate_ablation_table(self, merged_df):
        """
        生成消融实验表格
        
        Args:
            merged_df: 整合后的结果DataFrame
        
        Returns:
            消融实验表格DataFrame
        """
        if merged_df is None:
            return None
        
        # 过滤成功的结果
        successful_df = merged_df[merged_df['status'] == 'success']
        
        # 只包含SAKT、AKT、TSAKT模型
        ablation_models = ['sakt', 'akt', 'tsakt']
        ablation_df = successful_df[successful_df['model'].isin(ablation_models)]
        
        if ablation_df.empty:
            print('没有消融实验结果！')
            return None
        
        # 生成消融实验表格
        pivot_auc = ablation_df.pivot_table(
            index='model',
            columns='dataset',
            values='auc',
            aggfunc='mean'
        )
        
        return pivot_auc
    
    def save_comparison_table(self, comparison_tables, output_file='experiments/comparison_table.csv'):
        """
        保存对比表格
        
        Args:
            comparison_tables: 对比表格字典
            output_file: 输出文件路径
        """
        if comparison_tables is None:
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存AUC表格
        comparison_tables['auc'].to_csv(
            output_path.parent / 'comparison_auc.csv',
            float_format='%.4f'
        )
        
        # 保存ACC表格
        comparison_tables['acc'].to_csv(
            output_path.parent / 'comparison_acc.csv',
            float_format='%.4f'
        )
        
        # 保存RMSE表格
        comparison_tables['rmse'].to_csv(
            output_path.parent / 'comparison_rmse.csv',
            float_format='%.4f'
        )
        
        print(f'对比表格已保存到 {output_path}')
    
    def save_latex_tables(self, comparison_tables, output_dir='experiments/latex_tables'):
        """
        保存LaTeX表格
        
        Args:
            comparison_tables: 对比表格字典
            output_dir: 输出目录
        """
        if comparison_tables is None:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存AUC表格
        comparison_tables['auc'].to_latex(
            output_path / 'comparison_auc.tex',
            float_format='%.4f',
            caption='Model Comparison (AUC)',
            label='tab:comparison_auc'
        )
        
        # 保存ACC表格
        comparison_tables['acc'].to_latex(
            output_path / 'comparison_acc.tex',
            float_format='%.4f',
            caption='Model Comparison (ACC)',
            label='tab:comparison_acc'
        )
        
        # 保存RMSE表格
        comparison_tables['rmse'].to_latex(
            output_path / 'comparison_rmse.tex',
            float_format='%.4f',
            caption='Model Comparison (RMSE)',
            label='tab:comparison_rmse'
        )
        
        # 保存消融实验表格
        ablation_table = self.generate_ablation_table(self.merge_results())
        if ablation_table is not None:
            ablation_table.to_latex(
                output_path / 'ablation.tex',
                float_format='%.4f',
                caption='Ablation Study (AUC)',
                label='tab:ablation'
            )
        
        print(f'LaTeX表格已保存到 {output_path}')
    
    def plot_comparison_bar(self, merged_df, metric='auc', save_path='experiments/integrated_comparison_bar.png'):
        """
        绘制模型对比柱状图
        
        Args:
            merged_df: 整合后的结果DataFrame
            metric: 指标名称
            save_path: 保存路径
        """
        if merged_df is None:
            return
        
        # 过滤成功的结果
        successful_df = merged_df[merged_df['status'] == 'success']
        
        if successful_df.empty:
            print('没有成功的实验结果！')
            return
        
        plt.figure(figsize=(12, 6))
        
        pivot_df = successful_df.pivot_table(
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
    
    def plot_ablation_line(self, merged_df, save_path='experiments/integrated_ablation_line.png'):
        """
        绘制消融实验折线图
        
        Args:
            merged_df: 整合后的结果DataFrame
            save_path: 保存路径
        """
        if merged_df is None:
            return
        
        # 过滤成功的结果
        successful_df = merged_df[merged_df['status'] == 'success']
        
        # 只包含SAKT、AKT、TSAKT模型
        ablation_models = ['sakt', 'akt', 'tsakt']
        ablation_df = successful_df[successful_df['model'].isin(ablation_models)]
        
        if ablation_df.empty:
            print('没有消融实验结果！')
            return
        
        plt.figure(figsize=(12, 6))
        
        pivot_df = ablation_df.pivot_table(
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
    
    def plot_heatmap(self, merged_df, metric='auc', save_path='experiments/integrated_heatmap.png'):
        """
        绘制热力图
        
        Args:
            merged_df: 整合后的结果DataFrame
            metric: 指标名称
            save_path: 保存路径
        """
        if merged_df is None:
            return
        
        # 过滤成功的结果
        successful_df = merged_df[merged_df['status'] == 'success']
        
        if successful_df.empty:
            print('没有成功的实验结果！')
            return
        
        pivot_df = successful_df.pivot_table(
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
    
    def generate_report(self, merged_df, output_file='experiments/integrated_report.txt'):
        """
        生成整合报告
        
        Args:
            merged_df: 整合后的结果DataFrame
            output_file: 输出文件路径
        """
        if merged_df is None:
            return
        
        # 过滤成功的结果
        successful_df = merged_df[merged_df['status'] == 'success']
        
        if successful_df.empty:
            print('没有成功的实验结果！')
            return
        
        report = []
        report.append('='*80)
        report.append('实验整合报告')
        report.append('='*80)
        report.append('')
        
        report.append('1. 实验概览')
        report.append('-'*80)
        report.append(f'总实验数: {len(merged_df)}')
        report.append(f'成功实验数: {len(successful_df)}')
        report.append(f'失败实验数: {len(merged_df) - len(successful_df)}')
        report.append('')
        
        report.append('2. 模型对比')
        report.append('-'*80)
        
        model_groups = successful_df.groupby('model')
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
        
        dataset_groups = successful_df.groupby('dataset')
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
        
        best_auc_idx = successful_df['auc'].idxmax()
        best_auc_model = successful_df.loc[best_auc_idx, 'model']
        best_auc_dataset = successful_df.loc[best_auc_idx, 'dataset']
        best_auc_value = successful_df.loc[best_auc_idx, 'auc']
        
        report.append(f'最佳AUC: {best_auc_model} on {best_auc_dataset} ({best_auc_value:.4f})')
        report.append('')
        
        report.append('='*80)
        
        report_text = '\n'.join(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f'整合报告已保存到 {output_file}')
        print(report_text)
    
    def integrate_all(self):
        """
        整合所有结果
        """
        print('='*80)
        print('开始整合实验结果')
        print('='*80)
        
        # 加载并整合结果
        merged_df = self.merge_results()
        
        if merged_df is None:
            print('没有可用的结果！')
            return
        
        # 生成对比表格
        comparison_tables = self.generate_comparison_table(merged_df)
        self.save_comparison_table(comparison_tables)
        
        # 生成LaTeX表格
        self.save_latex_tables(comparison_tables)
        
        # 绘制图表
        self.plot_comparison_bar(merged_df, metric='auc')
        self.plot_comparison_bar(merged_df, metric='acc')
        self.plot_comparison_bar(merged_df, metric='rmse')
        self.plot_ablation_line(merged_df)
        self.plot_heatmap(merged_df, metric='auc')
        
        # 生成报告
        self.generate_report(merged_df)
        
        print('\n' + '='*80)
        print('实验整合完成')
        print('='*80)


def main():
    integrator = ExperimentIntegrator()
    integrator.integrate_all()


if __name__ == '__main__':
    main()