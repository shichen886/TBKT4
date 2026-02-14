import argparse
import subprocess
import os
import time
from datetime import datetime
import json


def train_model(model_type, dataset, params):
    """
    Train a single model
    
    Args:
        model_type: model type (dkt, akt)
        dataset: dataset name
        params: model parameters
    
    Returns:
        result dictionary
    """
    print(f'\n{"="*80}')
    print(f'Training {model_type} on {dataset}')
    print(f'{"="*80}')
    print(f'Parameters: {params}')
    
    start_time = time.time()
    
    try:
        if model_type == 'dkt':
            script = 'train_dkt1.py'
            cmd = [
                'python',
                script,
                '--dataset', dataset,
                '--hid_size', str(params.get('hid_size', 200)),
                '--num_epochs', str(params.get('num_epochs', 10)),
                '--batch_size', str(params.get('batch_size', 128)),
                '--lr', str(params.get('lr', 0.001)),
                '--skill_in',
                '--item_out'
            ]
        elif model_type == 'akt':
            script = 'train_akt.py'
            cmd = [
                'python',
                script,
                '--dataset', dataset,
                '--embed_size', str(params.get('embed_size', 200)),
                '--num_attn_layers', str(params.get('num_attn_layers', 2)),
                '--num_heads', str(params.get('num_heads', 8)),
                '--drop_prob', str(params.get('drop_prob', 0.2)),
                '--num_epochs', str(params.get('num_epochs', 10)),
                '--batch_size', str(params.get('batch_size', 128)),
                '--learning_rate', str(params.get('learning_rate', 0.0001)),
                '--max_seq_len', str(params.get('max_seq_len', 200)),
                '--max_pos', str(params.get('max_pos', 10))
            ]
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        
        print(f'Running command: {" ".join(cmd)}')
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours timeout
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if result.returncode == 0:
            print(f'✓ Training completed successfully in {elapsed_time:.2f} seconds')
            
            # Parse output for metrics
            output = result.stdout
            auc = parse_metric(output, 'AUC')
            acc = parse_metric(output, 'ACC')
            rmse = parse_metric(output, 'RMSE')
            
            result_dict = {
                'model': model_type,
                'dataset': dataset,
                'params': params,
                'auc': auc,
                'acc': acc,
                'rmse': rmse,
                'elapsed_time': elapsed_time,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            if auc is not None:
                print(f'Results: AUC={auc:.4f}, ACC={acc:.4f}, RMSE={rmse:.4f}')
            else:
                print(f'Warning: Could not parse metrics from output')
                result_dict['auc'] = None
                result_dict['acc'] = None
                result_dict['rmse'] = None
            
        else:
            print(f'✗ Training failed')
            print(f'Error output: {result.stderr}')
            
            result_dict = {
                'model': model_type,
                'dataset': dataset,
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
        print(f'✗ Training timeout')
        
        result_dict = {
            'model': model_type,
            'dataset': dataset,
            'params': params,
            'auc': None,
            'acc': None,
            'rmse': None,
            'elapsed_time': time.time() - start_time,
            'status': 'timeout',
            'error': 'Training timeout',
            'timestamp': datetime.now().isoformat()
        }
        
        return result_dict
        
    except Exception as e:
        print(f'✗ Training error: {str(e)}')
        
        result_dict = {
            'model': model_type,
            'dataset': dataset,
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


def parse_metric(output, metric_name):
    """
    Parse metric from output
    
    Args:
        output: output text
        metric_name: metric name
    
    Returns:
        metric value or None
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


def train_all_models(models, datasets, params_dict, output_dir='experiments'):
    """
    Train all models on all datasets
    
    Args:
        models: list of model types
        datasets: list of dataset names
        params_dict: dictionary of model parameters
        output_dir: output directory
    """
    results = []
    
    print('='*80)
    print('Starting batch training')
    print('='*80)
    
    total_experiments = len(models) * len(datasets)
    current_experiment = 0
    
    for model in models:
        for dataset in datasets:
            current_experiment += 1
            print(f'\n[{current_experiment}/{total_experiments}] Training {model} on {dataset}')
            
            params = params_dict.get(model, {})
            result = train_model(model, dataset, params)
            results.append(result)
    
    print('\n' + '='*80)
    print('Batch training completed')
    print('='*80)
    
    # Save results
    output_path = os.path.join(output_dir, 'training_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f'\nTraining results saved to {output_path}')
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results):
    """
    Print training summary
    
    Args:
        results: list of result dictionaries
    """
    print('\n' + '='*80)
    print('Training Summary')
    print('='*80)
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if successful_results:
        print(f'\nSuccessful trainings: {len(successful_results)}/{len(results)}')
        
        # Group by model
        model_groups = {}
        for result in successful_results:
            model = result['model']
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(result)
        
        print('\nModel Performance:')
        print('-'*80)
        for model, model_results in model_groups.items():
            avg_auc = np.mean([r['auc'] for r in model_results if r['auc'] is not None])
            avg_acc = np.mean([r['acc'] for r in model_results if r['acc'] is not None])
            avg_rmse = np.mean([r['rmse'] for r in model_results if r['rmse'] is not None])
            avg_time = np.mean([r['elapsed_time'] for r in model_results])
            
            if avg_auc is not None:
                print(f'{model}: AUC={avg_auc:.4f}, ACC={avg_acc:.4f}, RMSE={avg_rmse:.4f}, Time={avg_time:.2f}s')
            else:
                print(f'{model}: Time={avg_time:.2f}s (metrics not available)')
        
        # Group by dataset
        dataset_groups = {}
        for result in successful_results:
            dataset = result['dataset']
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(result)
        
        print('\nDataset Performance:')
        print('-'*80)
        for dataset, dataset_results in dataset_groups.items():
            avg_time = np.mean([r['elapsed_time'] for r in dataset_results])
            print(f'{dataset}: Avg Time={avg_time:.2f}s, Successful={len(dataset_results)}/{len(dataset_results)}')
    else:
        print('\nNo successful trainings!')
    
    print('='*80)


def main():
    parser = argparse.ArgumentParser(description='Train all models in batch')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['dkt', 'akt'],
                       help='Model list')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['assistments09', 'assistments12', 'assistments15'],
                       help='Dataset list')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory')
    
    # Model parameters
    parser.add_argument('--dkt_embed_size', type=int, default=200,
                       help='DKT embedding size')
    parser.add_argument('--dkt_num_epochs', type=int, default=10,
                       help='DKT number of epochs')
    parser.add_argument('--dkt_batch_size', type=int, default=128,
                       help='DKT batch size')
    parser.add_argument('--dkt_learning_rate', type=float, default=0.001,
                       help='DKT learning rate')
    
    parser.add_argument('--akt_embed_size', type=int, default=200,
                       help='AKT embedding size')
    parser.add_argument('--akt_num_attn_layers', type=int, default=2,
                       help='AKT number of attention layers')
    parser.add_argument('--akt_num_heads', type=int, default=8,
                       help='AKT number of attention heads')
    parser.add_argument('--akt_drop_prob', type=float, default=0.2,
                       help='AKT dropout probability')
    parser.add_argument('--akt_num_epochs', type=int, default=10,
                       help='AKT number of epochs')
    parser.add_argument('--akt_batch_size', type=int, default=128,
                       help='AKT batch size')
    parser.add_argument('--akt_learning_rate', type=float, default=0.0001,
                       help='AKT learning rate')
    parser.add_argument('--akt_max_pos', type=int, default=10,
                       help='AKT maximum position')
    
    args = parser.parse_args()
    
    # Build parameters dictionary
    params_dict = {
        'dkt': {
            'hid_size': args.dkt_embed_size,
            'num_epochs': args.dkt_num_epochs,
            'batch_size': args.dkt_batch_size,
            'lr': args.dkt_learning_rate
        },
        'akt': {
            'embed_size': args.akt_embed_size,
            'num_attn_layers': args.akt_num_attn_layers,
            'num_heads': args.akt_num_heads,
            'drop_prob': args.akt_drop_prob,
            'num_epochs': args.akt_num_epochs,
            'batch_size': args.akt_batch_size,
            'learning_rate': args.akt_learning_rate,
            'max_seq_len': 200,
            'max_pos': args.akt_max_pos
        }
    }
    
    # Train all models
    results = train_all_models(args.models, args.datasets, params_dict, args.output_dir)


if __name__ == '__main__':
    import numpy as np
    main()