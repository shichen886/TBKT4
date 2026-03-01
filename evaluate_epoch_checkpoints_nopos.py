import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_tsakt_linear_nopos import TSAKT_Linear_NoPos
from train_tsakt_linear_nopos_regularized_checkpoints import KTDataset, collate_fn, validate


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    data_dir = os.path.join('data', args.dataset)
    train_path = os.path.join(data_dir, 'preprocessed_data_train.csv')
    test_path = os.path.join(data_dir, 'preprocessed_data_test.csv')
    
    print(f'Loading data from {data_dir}')
    
    # Create datasets
    train_dataset = KTDataset(train_path, max_seq_len=args.max_seq_len)
    test_dataset = KTDataset(test_path, max_seq_len=args.max_seq_len)
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create dataloaders
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Get number of items and skills
    train_data = pd.read_csv(train_path, sep='\t')
    num_items = train_data['item_id'].max()
    num_skills = train_data['skill_id'].max()
    
    print(f'Number of items: {num_items}')
    print(f'Number of skills: {num_skills}')
    
    # Load training history
    history_path = os.path.join(args.savedir, f'{args.dataset}_training_history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Load checkpoints and evaluate
    results = []
    
    for epoch_num in args.epochs:
        checkpoint_path = os.path.join(args.savedir, args.checkpoints_dir, f'epoch_{epoch_num}.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint {checkpoint_path} does not exist, skipping...')
            continue
        
        print(f'\n{"="*80}')
        print(f'Evaluating Epoch {epoch_num}')
        print(f'{"="*80}')
        
        # Create model
        model = TSAKT_Linear_NoPos(
            num_items=num_items,
            num_skills=num_skills,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            tensor_rank=args.tensor_rank,
            max_len=args.max_seq_len,
            drop_prob=args.drop_prob,
        ).to(device)
        
        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        # Get validation and test metrics from history
        val_auc = history['val_auc'][epoch_num - 1]
        val_rmse = history['val_rmse'][epoch_num - 1]
        
        # Evaluate on test set
        criterion = nn.BCEWithLogitsLoss()
        test_loss, test_auc, test_rmse = validate(model, test_loader, criterion, device)
        
        # Calculate generalization gap
        generalization_gap = val_auc - test_auc
        
        result = {
            'epoch': epoch_num,
            'val_auc': val_auc,
            'val_rmse': val_rmse,
            'test_auc': test_auc,
            'test_rmse': test_rmse,
            'generalization_gap': generalization_gap,
        }
        
        results.append(result)
        
        print(f'Val AUC: {val_auc:.4f}, Val RMSE: {val_rmse:.4f}')
        print(f'Test AUC: {test_auc:.4f}, Test RMSE: {test_rmse:.4f}')
        print(f'Generalization Gap: {generalization_gap:.4f}')
    
    # Save results
    results_path = os.path.join(args.savedir, 'epoch_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print(f'\n{"="*80}')
    print('Summary')
    print(f'{"="*80}')
    print(f'{"Epoch":<8} {"Val AUC":<12} {"Test AUC":<12} {"Gap":<12}')
    print('-' * 44)
    for result in results:
        print(f"{result['epoch']:<8} {result['val_auc']:<12.4f} {result['test_auc']:<12.4f} {result['generalization_gap']:<12.4f}")
    
    # Find best epoch based on generalization gap
    best_gap_result = min(results, key=lambda x: x['generalization_gap'])
    print(f'\nBest epoch based on generalization gap: Epoch {best_gap_result["epoch"]}')
    print(f'Generalization gap: {best_gap_result["generalization_gap"]:.4f}')
    print(f'Val AUC: {best_gap_result["val_auc"]:.4f}')
    print(f'Test AUC: {best_gap_result["test_auc"]:.4f}')
    
    # Compare with original best epoch
    original_best_epoch = args.original_best_epoch
    original_best_result = next((r for r in results if r['epoch'] == original_best_epoch), None)
    
    if original_best_result:
        print(f'\n{"="*80}')
        print(f'Comparison with Original Best Epoch ({original_best_epoch})')
        print(f'{"="*80}')
        print(f'Original Best Epoch:')
        print(f'  Val AUC: {original_best_result["val_auc"]:.4f}')
        print(f'  Test AUC: {original_best_result["test_auc"]:.4f}')
        print(f'  Generalization Gap: {original_best_result["generalization_gap"]:.4f}')
        
        print(f'\nBest Epoch (based on generalization gap):')
        print(f'  Val AUC: {best_gap_result["val_auc"]:.4f}')
        print(f'  Test AUC: {best_gap_result["test_auc"]:.4f}')
        print(f'  Generalization Gap: {best_gap_result["generalization_gap"]:.4f}')
        
        # Calculate improvement
        gap_reduction = original_best_result["generalization_gap"] - best_gap_result["generalization_gap"]
        test_auc_improvement = best_gap_result["test_auc"] - original_best_result["test_auc"]
        
        print(f'\nImprovement:')
        print(f'  Generalization gap reduction: {gap_reduction:.4f} ({gap_reduction/original_best_result["generalization_gap"]*100:.1f}%)')
        print(f'  Test AUC improvement: {test_auc_improvement:.4f}')
        print(f'  Training time saved: {original_best_epoch - best_gap_result["epoch"]} epochs ({(original_best_epoch - best_gap_result["epoch"])/original_best_epoch*100:.1f}%)')
    
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate different epoch checkpoints on test set for NoPos model')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--savedir', type=str, default='save/tsakt-linear-nopos-regularized')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--tensor_rank', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--drop_prob', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, nargs='+', default=[12, 13, 14, 15, 20])
    parser.add_argument('--original_best_epoch', type=int, default=20)
    
    args = parser.parse_args()
    
    main(args)
