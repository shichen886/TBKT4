import json
import os
import matplotlib.pyplot as plt

def plot_training_curves_from_history(history_path, dataset_name, output_dir=None):
    """Plot and save training curves from existing training_history.json file.
    
    Args:
        history_path: Path to training_history.json file
        dataset_name: Name of the dataset
        output_dir: Directory to save the plot (default: same as history file)
    """
    if not os.path.exists(history_path):
        print(f"❌ Training history file not found: {history_path}")
        return False
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if output_dir is None:
        output_dir = os.path.dirname(history_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = history['epochs']
    
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    axes[1].plot(epochs, history['train_auc'], label='Train AUC', marker='o', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_auc'], label='Val AUC', marker='s', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('AUC', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    axes[2].plot(epochs, history['train_rmse'], label='Train RMSE', marker='o', linewidth=2, markersize=6)
    axes[2].plot(epochs, history['val_rmse'], label='Val RMSE', marker='s', linewidth=2, markersize=6)
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('RMSE', fontsize=12, fontweight='bold')
    axes[2].set_title('Training and Validation RMSE', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{dataset_name}_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training curves to {plot_path}")
    plt.close()
    
    return True

def main():
    print("=" * 60)
    print("📊 从训练历史生成训练曲线图")
    print("=" * 60)
    
    savedir = "save/tsakt-linear-sakt-params"
    datasets = ["assistments09", "assistments12", "assistments15"]
    
    for dataset in datasets:
        print(f"\n🔍 处理数据集: {dataset}")
        history_path = os.path.join(savedir, f"{dataset}_training_history.json")
        
        if os.path.exists(history_path):
            success = plot_training_curves_from_history(history_path, dataset, savedir)
            if success:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                print(f"   📈 Best Val AUC: {history['best_val_auc']:.4f}")
                print(f"   📉 Best Val Loss: {history['best_val_loss']:.4f}")
        else:
            print(f"   ⚠️  训练历史文件不存在: {history_path}")
    
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()