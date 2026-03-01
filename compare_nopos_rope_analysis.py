import json
import numpy as np

# Load evaluation results
nopos_42 = json.load(open('save/tsakt-linear-nopos-regularized/assistments17_long_100/seed_42_checkpoints/epoch_comparison_results.json'))
nopos_123 = json.load(open('save/tsakt-linear-nopos-regularized/assistments17_long_100/seed_123_checkpoints/epoch_comparison_results.json'))
nopos_456 = json.load(open('save/tsakt-linear-nopos-regularized/assistments17_long_100/seed_456_checkpoints/epoch_comparison_results.json'))

rope_42 = json.load(open('save/tsakt-linear-rope-qk-regularized/assistments17_long_100/seed_42_checkpoints/epoch_comparison_results.json'))
rope_123 = json.load(open('save/tsakt-linear-rope-qk-regularized/assistments17_long_100/seed_123_checkpoints/epoch_comparison_results.json'))
rope_456 = json.load(open('save/tsakt-linear-rope-qk-regularized/assistments17_long_100/seed_456_checkpoints/epoch_comparison_results.json'))

# Function to find best epoch based on generalization gap
def find_best_gap_epoch(results):
    best_result = min(results, key=lambda x: x['generalization_gap'])
    return best_result

# Function to find original best epoch
def find_original_best_epoch(results):
    best_result = max(results, key=lambda x: x['val_auc'])
    return best_result

# Analyze each seed
seeds = [42, 123, 456]
nopos_results = [nopos_42, nopos_123, nopos_456]
rope_results = [rope_42, rope_123, rope_456]

print('=' * 100)
print('NoPos vs RoPE-QK Comparison Analysis (assistments17_long_100)')
print('=' * 100)

for i, seed in enumerate(seeds):
    print(f'\nSeed {seed}:')
    print('-' * 50)
    
    nopos_data = nopos_results[i]
    rope_data = rope_results[i]
    
    # Find best epochs
    nopos_best_gap = find_best_gap_epoch(nopos_data)
    rope_best_gap = find_best_gap_epoch(rope_data)
    
    nopos_original_best = find_original_best_epoch(nopos_data)
    rope_original_best = find_original_best_epoch(rope_data)
    
    # Print NoPos results
    print(f'NoPos:')
    print(f'  Original Best Epoch {nopos_original_best["epoch"]}: Val AUC={nopos_original_best["val_auc"]:.4f}, Test AUC={nopos_original_best["test_auc"]:.4f}, Gap={nopos_original_best["generalization_gap"]:.4f}')
    print(f'  Best Gap Epoch {nopos_best_gap["epoch"]}: Val AUC={nopos_best_gap["val_auc"]:.4f}, Test AUC={nopos_best_gap["test_auc"]:.4f}, Gap={nopos_best_gap["generalization_gap"]:.4f}')
    gap_reduction_nopos = nopos_original_best["generalization_gap"] - nopos_best_gap["generalization_gap"]
    test_auc_change_nopos = nopos_best_gap["test_auc"] - nopos_original_best["test_auc"]
    print(f'  Gap Reduction: {gap_reduction_nopos:.4f} ({gap_reduction_nopos/nopos_original_best["generalization_gap"]*100:.1f}%)')
    print(f'  Test AUC Change: {test_auc_change_nopos:.4f}')
    
    # Print RoPE-QK results
    print(f'\nRoPE-QK:')
    print(f'  Original Best Epoch {rope_original_best["epoch"]}: Val AUC={rope_original_best["val_auc"]:.4f}, Test AUC={rope_original_best["test_auc"]:.4f}, Gap={rope_original_best["generalization_gap"]:.4f}')
    print(f'  Best Gap Epoch {rope_best_gap["epoch"]}: Val AUC={rope_best_gap["val_auc"]:.4f}, Test AUC={rope_best_gap["test_auc"]:.4f}, Gap={rope_best_gap["generalization_gap"]:.4f}')
    gap_reduction_rope = rope_original_best["generalization_gap"] - rope_best_gap["generalization_gap"]
    test_auc_change_rope = rope_best_gap["test_auc"] - rope_original_best["test_auc"]
    print(f'  Gap Reduction: {gap_reduction_rope:.4f} ({gap_reduction_rope/rope_original_best["generalization_gap"]*100:.1f}%)')
    print(f'  Test AUC Change: {test_auc_change_rope:.4f}')
    
    # Comparison
    print(f'\nComparison:')
    print(f'  NoPos Test AUC: {nopos_best_gap["test_auc"]:.4f} (Gap: {nopos_best_gap["generalization_gap"]:.4f})')
    print(f'  RoPE-QK Test AUC: {rope_best_gap["test_auc"]:.4f} (Gap: {rope_best_gap["generalization_gap"]:.4f})')
    test_auc_diff = nopos_best_gap["test_auc"] - rope_best_gap["test_auc"]
    gap_diff = nopos_best_gap["generalization_gap"] - rope_best_gap["generalization_gap"]
    print(f'  Test AUC Difference: {test_auc_diff:.4f} ({"NoPos better" if test_auc_diff > 0 else "RoPE-QK better"})')
    print(f'  Gap Difference: {gap_diff:.4f} ({"NoPos better" if gap_diff > 0 else "RoPE-QK better"})')

# Overall summary
print('\n' + '=' * 100)
print('Overall Summary')
print('=' * 100)

# Calculate average performance across seeds
nopos_test_aucs = [find_best_gap_epoch(r)['test_auc'] for r in nopos_results]
rope_test_aucs = [find_best_gap_epoch(r)['test_auc'] for r in rope_results]

nopos_gaps = [find_best_gap_epoch(r)['generalization_gap'] for r in nopos_results]
rope_gaps = [find_best_gap_epoch(r)['generalization_gap'] for r in rope_results]

print(f'\nNoPos Average Test AUC: {np.mean(nopos_test_aucs):.4f} ± {np.std(nopos_test_aucs):.4f}')
print(f'NoPos Average Gap: {np.mean(nopos_gaps):.4f} ± {np.std(nopos_gaps):.4f}')

print(f'\nRoPE-QK Average Test AUC: {np.mean(rope_test_aucs):.4f} ± {np.std(rope_test_aucs):.4f}')
print(f'RoPE-QK Average Gap: {np.mean(rope_gaps):.4f} ± {np.std(rope_gaps):.4f}')

print(f'\nAverage Test AUC Difference: {np.mean(nopos_test_aucs) - np.mean(rope_test_aucs):.4f}')
print(f'Average Gap Difference: {np.mean(nopos_gaps) - np.mean(rope_gaps):.4f}')

# Overfitting analysis
print('\n' + '=' * 100)
print('Overfitting Analysis')
print('=' * 100)

for i, seed in enumerate(seeds):
    nopos_data = nopos_results[i]
    rope_data = rope_results[i]
    
    # Find the epoch where overfitting becomes severe (gap > 0.03)
    nopos_severe_overfit = next((r for r in nopos_data if r['generalization_gap'] > 0.03), None)
    rope_severe_overfit = next((r for r in rope_data if r['generalization_gap'] > 0.03), None)
    
    print(f'\nSeed {seed}:')
    if nopos_severe_overfit:
        print(f'  NoPos: Severe overfitting at epoch {nopos_severe_overfit["epoch"]} (Gap: {nopos_severe_overfit["generalization_gap"]:.4f})')
    else:
        print(f'  NoPos: No severe overfitting (max gap: {max(r["generalization_gap"] for r in nopos_data):.4f})')
    
    if rope_severe_overfit:
        print(f'  RoPE-QK: Severe overfitting at epoch {rope_severe_overfit["epoch"]} (Gap: {rope_severe_overfit["generalization_gap"]:.4f})')
    else:
        print(f'  RoPE-QK: No severe overfitting (max gap: {max(r["generalization_gap"] for r in rope_data):.4f})')

print('\n' + '=' * 100)
print('Conclusion')
print('=' * 100)

avg_nopos_test_auc = np.mean(nopos_test_aucs)
avg_rope_test_auc = np.mean(rope_test_aucs)
avg_nopos_gap = np.mean(nopos_gaps)
avg_rope_gap = np.mean(rope_gaps)

if avg_nopos_test_auc > avg_rope_test_auc:
    print(f'\nNoPos performs better on average:')
    print(f'  Test AUC: {avg_nopos_test_auc:.4f} vs {avg_rope_test_auc:.4f} (difference: {avg_nopos_test_auc - avg_rope_test_auc:.4f})')
    print(f'  Gap: {avg_nopos_gap:.4f} vs {avg_rope_gap:.4f} (difference: {avg_nopos_gap - avg_rope_gap:.4f})')
elif avg_rope_test_auc > avg_nopos_test_auc:
    print(f'\nRoPE-QK performs better on average:')
    print(f'  Test AUC: {avg_rope_test_auc:.4f} vs {avg_nopos_test_auc:.4f} (difference: {avg_rope_test_auc - avg_nopos_test_auc:.4f})')
    print(f'  Gap: {avg_rope_gap:.4f} vs {avg_nopos_gap:.4f} (difference: {avg_rope_gap - avg_nopos_gap:.4f})')
else:
    print(f'\nBoth models perform similarly on average:')
    print(f'  Test AUC: {avg_nopos_test_auc:.4f} vs {avg_rope_test_auc:.4f} (difference: {avg_nopos_test_auc - avg_rope_test_auc:.4f})')
    print(f'  Gap: {avg_nopos_gap:.4f} vs {avg_rope_gap:.4f} (difference: {avg_nopos_gap - avg_rope_gap:.4f})')
