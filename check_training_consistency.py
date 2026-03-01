import pandas as pd
import json
import os

def check_training_consistency(dataset_name):
    """Check if training settings are consistent between two models."""
    print(f"\n{'='*60}")
    print(f"Checking training consistency for {dataset_name}")
    print(f"{'='*60}")

    # Check config files
    config1_path = f'save/tsakt-linear/config.json'
    config2_path = f'save/tsakt-linear-nopos/config.json'

    if os.path.exists(config1_path):
        with open(config1_path, 'r') as f:
            config1 = json.load(f)
        print(f"\nTSAKT-Linear config:")
        print(json.dumps(config1, indent=2))

    if os.path.exists(config2_path):
        with open(config2_path, 'r') as f:
            config2 = json.load(f)
        print(f"\nTSAKT-Linear-NoPos config:")
        print(json.dumps(config2, indent=2))

    # Check training history
    history1_path = f'save/tsakt-linear/{dataset_name}_training_history.json'
    history2_path = f'save/tsakt-linear-nopos/{dataset_name}_training_history.json'

    if os.path.exists(history1_path) and os.path.exists(history2_path):
        with open(history1_path, 'r') as f:
            history1 = json.load(f)
        with open(history2_path, 'r') as f:
            history2 = json.load(f)

        print(f"\n{'='*60}")
        print("Training history comparison:")
        print(f"{'='*60}")

        print(f"\nTSAKT-Linear:")
        print(f"  Epochs: {len(history1['epochs'])}")
        print(f"  Best Val AUC: {history1['best_val_auc']:.4f}")
        print(f"  Best Val Loss: {history1['best_val_loss']:.4f}")
        print(f"  Final Train AUC: {history1['train_auc'][-1]:.4f}")
        print(f"  Final Val AUC: {history1['val_auc'][-1]:.4f}")

        print(f"\nTSAKT-Linear-NoPos:")
        print(f"  Epochs: {len(history2['epochs'])}")
        print(f"  Best Val AUC: {history2['best_val_auc']:.4f}")
        print(f"  Best Val Loss: {history2['best_val_loss']:.4f}")
        print(f"  Final Train AUC: {history2['train_auc'][-1]:.4f}")
        print(f"  Final Val AUC: {history2['val_auc'][-1]:.4f}")

        # Check for overfitting
        print(f"\n{'='*60}")
        print("Overfitting analysis:")
        print(f"{'='*60}")

        train_val_gap1 = history1['train_auc'][-1] - history1['val_auc'][-1]
        train_val_gap2 = history2['train_auc'][-1] - history2['val_auc'][-1]

        print(f"\nTSAKT-Linear:")
        print(f"  Train-Val AUC gap: {train_val_gap1:.4f}")
        if train_val_gap1 > 0.1:
            print(f"  ⚠️ POTENTIAL OVERFITTING (gap > 0.1)")
        else:
            print(f"  ✓ No significant overfitting")

        print(f"\nTSAKT-Linear-NoPos:")
        print(f"  Train-Val AUC gap: {train_val_gap2:.4f}")
        if train_val_gap2 > 0.1:
            print(f"  ⚠️ POTENTIAL OVERFITTING (gap > 0.1)")
        else:
            print(f"  ✓ No significant overfitting")

        # Check convergence
        print(f"\n{'='*60}")
        print("Convergence analysis:")
        print(f"{'='*60}")

        # Check if model converged (last 5 epochs)
        last5_auc1 = history1['val_auc'][-5:]
        last5_auc2 = history2['val_auc'][-5:]

        auc_std1 = pd.Series(last5_auc1).std()
        auc_std2 = pd.Series(last5_auc2).std()

        print(f"\nTSAKT-Linear:")
        print(f"  Last 5 epochs AUC std: {auc_std1:.4f}")
        if auc_std1 < 0.005:
            print(f"  ✓ Model converged (std < 0.005)")
        else:
            print(f"  ⚠️ Model may not have converged")

        print(f"\nTSAKT-Linear-NoPos:")
        print(f"  Last 5 epochs AUC std: {auc_std2:.4f}")
        if auc_std2 < 0.005:
            print(f"  ✓ Model converged (std < 0.005)")
        else:
            print(f"  ⚠️ Model may not have converged")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    datasets = ['assistments12', 'assistments15']

    for dataset in datasets:
        check_training_consistency(dataset)