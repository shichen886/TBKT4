import os
import subprocess

# 数据集列表
datasets = ['assistments09', 'assistments12', 'assistments15']

print("=" * 80)
print("Retraining TSAKT-Ful with correct max_pos parameter")
print("=" * 80)

for dataset in datasets:
    print(f"\nTraining on dataset: {dataset}")
    print("-" * 60)
    
    # 构建命令
    cmd = [
        'python', 'train_tsakt.py',
        '--dataset', dataset,
        '--encode_pos', 'True',
        '--max_pos', '200',
        '--savedir', 'save/tsakt-ful-v2',
        '--logdir', 'runs/tsakt-ful-v2'
    ]
    
    # 执行命令
    print(f"Command: {' '.join(cmd)}")
    
    # 创建保存目录
    os.makedirs('save/tsakt-ful-v2', exist_ok=True)
    os.makedirs('runs/tsakt-ful-v2', exist_ok=True)
    
    # 执行训练
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Successfully trained TSAKT-Ful on {dataset}")
    else:
        print(f"✗ Failed to train TSAKT-Ful on {dataset}")
        print(f"Error: {result.stderr[:500]}...")

print("\n" + "=" * 80)
print("Training completed!")
print("=" * 80)