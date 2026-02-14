import argparse
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, accuracy_score
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_model(model_type, dataset, model_path=None):
    """
    Load trained model
    
    Args:
        model_type: model type (dkt, sakt, akt, tsakt)
        dataset: dataset name
        model_path: custom model path
    
    Returns:
        model, num_items, num_skills
    """
    if model_path is None:
        if model_type == 'dkt':
            # Try different batch sizes
            possible_batch_sizes = [32, 16, 8, 4, 2, 1]
            for batch_size in possible_batch_sizes:
                model_path = f'save/dkt1/{dataset},batch_size={batch_size},item_in=False,skill_in=True,item_out=True,skill_out=False,skill_separate=False'
                if os.path.exists(model_path):
                    break
        elif model_type == 'sakt':
            model_path = f'save/sakt/{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=10'
            if not os.path.exists(model_path):
                model_path = f'save/sakt/{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5'
        elif model_type == 'akt':
            model_path = f'save/akt/{dataset},batch_size=128,max_length=200,max_pos=10'
        elif model_type == 'tsakt':
            model_path = f'save/tsakt/{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3'
        else:
            raise ValueError(f'Unknown model type: {model_type}')
    
    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        return None, None, None
    
    print(f'Loading model from: {model_path}')
    
    try:
        if model_type == 'dkt':
            from model_dkt1 import DKT1
            loaded_model = torch.load(model_path, map_location=device, weights_only=False)
            model = loaded_model.to(device)
            model.eval()
            
            # Get num_items and num_skills from model
            num_items = model.item_embeds.num_embeddings
            num_skills = model.skill_embeds.num_embeddings
            
        elif model_type == 'sakt':
            from model_sakt import SAKT
            loaded_model = torch.load(model_path, map_location=device, weights_only=False)
            model = loaded_model.to(device)
            model.eval()
            
            num_items = model.item_embeds.num_embeddings
            num_skills = model.skill_embeds.num_embeddings
            
        elif model_type == 'akt':
            from model_akt import AKT
            loaded_model = torch.load(model_path, map_location=device, weights_only=False)
            model = loaded_model.to(device)
            model.eval()
            
            num_items = model.item_embeds.num_embeddings
            num_skills = model.skill_embeds.num_embeddings
            
        elif model_type == 'tsakt':
            from model_tsakt import TSAKT
            loaded_model = torch.load(model_path, map_location=device, weights_only=False)
            model = loaded_model.to(device)
            model.eval()
            
            num_items = model.item_embeds.num_embeddings
            num_skills = model.skill_embeds.num_embeddings
        
        return model, num_items, num_skills
        
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        import traceback
        traceback.print_exc()
        return None, None, None


def load_test_data(dataset):
    """
    Load test data
    
    Args:
        dataset: dataset name
    
    Returns:
        test_data, num_items, num_skills
    """
    # Try different file formats
    data_file = None
    path1 = f'data/{dataset}/preprocessed_data.csv'
    path2 = f'data/{dataset}/preprocessed_test.csv'
    
    if os.path.exists(path1):
        data_file = path1
    elif os.path.exists(path2):
        data_file = path2
    
    if not data_file:
        print(f'Data file not found for dataset: {dataset}')
        return None, None, None
    
    df = pd.read_csv(data_file, sep='\t')
    print(f'Loaded {len(df)} records from {data_file}')
    
    num_items = int(df['item_id'].max() + 1)
    num_skills = int(df['skill_id'].max() + 1)
    
    return df, num_items, num_skills


def prepare_test_data(df, max_length=200):
    """
    Prepare test data
    
    Args:
        df: dataframe
        max_length: maximum sequence length
    
    Returns:
        test_data
    """
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    return data


def evaluate_model(model, model_type, test_data, batch_size=128):
    """
    Evaluate model on test data
    
    Args:
        model: model
        model_type: model type
        test_data: test data
        batch_size: batch size
    
    Returns:
        auc, acc, rmse
    """
    from torch.nn.utils.rnn import pad_sequence
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for k in range(0, len(test_data), batch_size):
            batch = test_data[k:k + batch_size]
            seq_lists = list(zip(*batch))
            
            if model_type == 'dkt':
                # DKT uses different input format
                item_inputs = seq_lists[0]
                skill_inputs = seq_lists[1]
                labels = seq_lists[-1]
                
                item_inputs = pad_sequence(item_inputs, batch_first=True, padding_value=0).to(device)
                skill_inputs = pad_sequence(skill_inputs, batch_first=True, padding_value=0).to(device)
                labels = pad_sequence(labels, batch_first=True, padding_value=-1)
                
                preds = model(item_inputs, skill_inputs)
                
            else:
                # SAKT, AKT, TSAKT use the same input format
                item_inputs = seq_lists[0]
                skill_inputs = seq_lists[1]
                label_inputs = seq_lists[2]
                item_ids = seq_lists[3]
                skill_ids = seq_lists[4]
                labels = seq_lists[-1]
                
                item_inputs = pad_sequence(item_inputs, batch_first=True, padding_value=0).to(device)
                skill_inputs = pad_sequence(skill_inputs, batch_first=True, padding_value=0).to(device)
                label_inputs = pad_sequence(label_inputs, batch_first=True, padding_value=0).to(device)
                item_ids = pad_sequence(item_ids, batch_first=True, padding_value=0).to(device)
                skill_ids = pad_sequence(skill_ids, batch_first=True, padding_value=0).to(device)
                labels = pad_sequence(labels, batch_first=True, padding_value=-1)
                
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            preds = torch.sigmoid(preds).cpu()
            
            # Collect predictions and labels
            valid_mask = labels >= 0
            all_preds.append(preds[valid_mask])
            all_labels.append(labels[valid_mask])
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds).flatten().numpy()
    all_labels = torch.cat(all_labels).float().numpy()
    
    # Compute metrics
    if len(np.unique(all_labels)) == 1:
        auc = accuracy_score(all_labels, all_preds.round())
    else:
        auc = roc_auc_score(all_labels, all_preds)
    
    acc = accuracy_score(all_labels, all_preds.round())
    rmse = math.sqrt(np.mean((all_preds - all_labels) ** 2))
    
    return auc, acc, rmse


def main():
    parser = argparse.ArgumentParser(description='Evaluate knowledge tracing models')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['assistments09', 'assistments12', 'assistments15'],
                       help='Dataset list')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['dkt', 'sakt', 'akt', 'tsakt'],
                       help='Model list')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory')
    
    args = parser.parse_args()
    
    results = []
    
    print('='*80)
    print('Starting model evaluation')
    print('='*80)
    
    total_experiments = len(args.datasets) * len(args.models)
    current_experiment = 0
    
    for dataset in args.datasets:
        print(f'\nDataset: {dataset}')
        print('-'*80)
        
        df, num_items, num_skills = load_test_data(dataset)
        if df is None:
            continue
        
        test_data = prepare_test_data(df)
        
        for model_type in args.models:
            current_experiment += 1
            print(f'\n[{current_experiment}/{total_experiments}] Evaluating {model_type} on {dataset}')
            
            model, model_num_items, model_num_skills = load_model(model_type, dataset)
            
            if model is None:
                print(f'Skipping {model_type} on {dataset} (model not found)')
                continue
            
            try:
                auc, acc, rmse = evaluate_model(model, model_type, test_data)
                
                result = {
                    'dataset': dataset,
                    'model': model_type,
                    'auc': auc,
                    'acc': acc,
                    'rmse': rmse,
                    'status': 'success'
                }
                
                results.append(result)
                
                print(f'Results: AUC={auc:.4f}, ACC={acc:.4f}, RMSE={rmse:.4f}')
                
            except Exception as e:
                print(f'Error evaluating {model_type} on {dataset}: {str(e)}')
                import traceback
                traceback.print_exc()
                
                result = {
                    'dataset': dataset,
                    'model': model_type,
                    'auc': None,
                    'acc': None,
                    'rmse': None,
                    'status': 'failed',
                    'error': str(e)
                }
                
                results.append(result)
    
    print('\n' + '='*80)
    print('Evaluation completed')
    print('='*80)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_file = output_path / 'evaluation_results.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {json_file}')
    
    # Save as CSV
    csv_file = output_path / 'evaluation_results.csv'
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f'Results saved to {csv_file}')
    
    # Print summary
    print('\n' + '='*80)
    print('Evaluation Summary')
    print('='*80)
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if successful_results:
        print(f'\nSuccessful evaluations: {len(successful_results)}/{len(results)}')
        
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
            avg_auc = np.mean([r['auc'] for r in model_results])
            avg_acc = np.mean([r['acc'] for r in model_results])
            avg_rmse = np.mean([r['rmse'] for r in model_results])
            print(f'{model}: AUC={avg_auc:.4f}, ACC={avg_acc:.4f}, RMSE={avg_rmse:.4f}')
        
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
            avg_auc = np.mean([r['auc'] for r in dataset_results])
            avg_acc = np.mean([r['acc'] for r in dataset_results])
            avg_rmse = np.mean([r['rmse'] for r in dataset_results])
            print(f'{dataset}: AUC={avg_auc:.4f}, ACC={avg_acc:.4f}, RMSE={avg_rmse:.4f}')
    else:
        print('\nNo successful evaluations!')
    
    print('='*80)


if __name__ == '__main__':
    main()