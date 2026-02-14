import argparse
import pandas as pd
import numpy as np
import torch
import json
import os
from pathlib import Path
from datetime import datetime
import subprocess
import time

class ExperimentRunner:
    def __init__(self, datasets, models, output_dir='experiments'):
        """
        实验运行器
        
        Args:
            datasets: 数据集列表
            models: 模型列表
            output_dir: 输出目录
        """
        self.datasets = datasets
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        
    def run_single_experiment(self, dataset, model, params):
        """
        运行单个实验
        
        Args:
            dataset: 数据集名称
            model: 模型名称
            params: 模型参数
        
        Returns:
            实验结果字典
        """
        print(f'\n开始实验: {model} on {dataset}')
        print(f'参数: {params}')
        
        start_time = time.time()
        
        try:
            if model == 'dkt':
                script = 'train_dkt1.py'
            elif model == 'sakt':
                script = 'train_sakt.py'
            elif model == 'tsakt':
                script = 'train_tsakt.py'
            else:
                raise ValueError(f'未知模型: {model}')
            
            cmd = [
                'python',
                script,
                '--dataset', dataset,
                '--embed_size', str(params.get('embed_size', 200)),
                '--num_attn_layers', str(params.get('num_attn_layers', 2)),
                '--num_heads', str(params.get('num_heads', 8)),
                '--drop_prob', str(params.get('drop_prob', 0.2)),
                '--batch_size', str(params.get('batch_size', 32)),
                '--num_epochs', str(params.get('num_epochs', 10)),
                '--learning_rate', str(params.get('learning_rate', 0.001)),
                '--max_seq_len', str(params.get('max_seq_len', 200))
            ]
            
            print(f'运行命令: {" ".join(cmd)}')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if result.returncode == 0:
                print(f'✓ 实验成功完成，耗时: {elapsed_time:.2f}秒')
                
                output = result.stdout
                
                auc = self._parse_metric(output, 'AUC')
                acc = self._parse_metric(output, 'ACC')
                rmse = self._parse_metric(output, 'RMSE')
                
                result_dict = {
                    'dataset': dataset,
                    'model': model,
                    'params': params,
                    'auc': auc,
                    'acc': acc,
                    'rmse': rmse,
                    'elapsed_time': elapsed_time,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f'结果: AUC={auc:.4f}, ACC={acc:.4f}, RMSE={rmse:.4f}')
                
            else:
                print(f'✗ 实验失败')
                print(f'错误输出: {result.stderr}')
                
                result_dict = {
                    'dataset': dataset,
                    'model': model,
                    'params': params,
                    'auc': None,
                    'acc': None,
                    'rmse': None,
                    'elapsed_time': elapsed_time,
                    'status': 'failed',
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
            
            return result_dict
            
        except subprocess.TimeoutExpired:
            print(f'✗ 实验超时')
            
            result_dict = {
                'dataset': dataset,
                'model': model,
                'params': params,
                'auc': None,
                'acc': None,
                'rmse': None,
                'elapsed_time': time.time() - start_time,
                'status': 'timeout',
                'error': '实验超时',
                'timestamp': datetime.now().isoformat()
            }
            
            return result_dict
            
        except Exception as e:
            print(f'✗ 实验出错: {str(e)}')
            
            result_dict = {
                'dataset': dataset,
                'model': model,
                'params': params,
                'auc': None,
                'acc': None,
                'rmse': None,
                'elapsed_time': time.time() - start_time,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            return result_dict
    
    def _parse_metric(self, output, metric_name):
        """
        从输出中解析指标
        
        Args:
            output: 输出文本
            metric_name: 指标名称
        
        Returns:
            指标值
        """
        lines = output.split('\n')
        for line in lines:
            if f'{metric_name}:' in line:
                try:
                    value = float(line.split(f'{metric_name}:')[-1].strip())
                    return value
                except:
                    pass
        return None
    
    def run_all_experiments(self):
        """
        运行所有实验
        """
        print('='*80)
        print('开始运行所有实验')
        print('='*80)
        
        total_experiments = len(self.datasets) * len(self.models)
        current_experiment = 0
        
        for dataset in self.datasets:
            for model in self.models:
                current_experiment += 1
                print(f'\n进度: {current_experiment}/{total_experiments}')
                
                result = self.run_single_experiment(dataset, model, {})
                self.results.append(result)
        
        print('\n' + '='*80)
        print('所有实验完成')
        print('='*80)
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """
        保存实验结果
        """
        results_file = self.output_dir / 'experiment_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f'\n实验结果已保存到 {results_file}')
        
        csv_file = self.output_dir / 'experiment_results.csv'
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f'实验结果已保存到 {csv_file}')
    
    def print_summary(self):
        """
        打印实验结果摘要
        """
        print('\n' + '='*80)
        print('实验结果摘要')
        print('='*80)
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        if not successful_results:
            print('没有成功的实验！')
            return
        
        print(f'\n成功实验数: {len(successful_results)}/{len(self.results)}')
        
        print('\n按模型分组:')
        print('-'*80)
        
        model_groups = {}
        for result in successful_results:
            model = result['model']
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(result)
        
        for model, results in model_groups.items():
            avg_auc = np.mean([r['auc'] for r in results])
            avg_acc = np.mean([r['acc'] for r in results])
            avg_rmse = np.mean([r['rmse'] for r in results])
            print(f'{model}: AUC={avg_auc:.4f}, ACC={avg_acc:.4f}, RMSE={avg_rmse:.4f}')
        
        print('\n按数据集分组:')
        print('-'*80)
        
        dataset_groups = {}
        for result in successful_results:
            dataset = result['dataset']
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(result)
        
        for dataset, results in dataset_groups.items():
            avg_auc = np.mean([r['auc'] for r in results])
            avg_acc = np.mean([r['acc'] for r in results])
            avg_rmse = np.mean([r['rmse'] for r in results])
            print(f'{dataset}: AUC={avg_auc:.4f}, ACC={avg_acc:.4f}, RMSE={avg_rmse:.4f}')
        
        print('='*80)


def main():
    parser = argparse.ArgumentParser(description='运行知识追踪实验')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['assistments09', 'assistments12', 'assistments15'],
                       help='数据集列表')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['dkt', 'sakt', 'tsakt'],
                       help='模型列表')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='输出目录')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.datasets, args.models, args.output_dir)
    runner.run_all_experiments()


if __name__ == '__main__':
    main()