import subprocess
import os

# 训练TSAKT-Ful（带位置编码的完整版本）
def train_tsakt_ful():
    # 数据集列表
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    print("=" * 80)
    print("Training TSAKT-Ful (with position encoding)")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\nTraining on dataset: {dataset}")
        print("-" * 60)
        
        # 构建命令
        cmd = [
            'python', 'train_tsakt.py',
            '--dataset', dataset,
            '--encode_pos', 'True',
            '--savedir', 'save/tsakt-ful',
            '--logdir', 'runs/tsakt-ful'
        ]
        
        # 执行命令
        print(f"Command: {' '.join(cmd)}")
        
        # 创建保存目录
        os.makedirs('save/tsakt-ful', exist_ok=True)
        os.makedirs('runs/tsakt-ful', exist_ok=True)
        
        # 执行训练
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully trained TSAKT-Ful on {dataset}")
        else:
            print(f"✗ Failed to train TSAKT-Ful on {dataset}")
            print(f"Error: {result.stderr[:500]}...")

if __name__ == "__main__":
    train_tsakt_ful()
