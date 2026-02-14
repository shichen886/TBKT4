import os
import subprocess

# 只测试一个数据集
dataset = 'assistments09'

# 不同的max_pos值
max_pos_values = [50, 100, 150, 200]

# 不同的训练轮数
num_epochs_values = [256, 512, 1024]

print("=" * 80)
print(f"Testing TSAKT-Ful on {dataset} with different max_pos and num_epochs")
print("=" * 80)

for max_pos in max_pos_values:
    for num_epochs in num_epochs_values:
        print(f"\nTesting: max_pos={max_pos}, num_epochs={num_epochs}")
        print("-" * 60)
        
        # 构建命令
        cmd = [
            'python', 'train_tsakt.py',
            '--dataset', dataset,
            '--encode_pos', 'True',
            '--max_pos', str(max_pos),
            '--num_epochs', str(num_epochs),
            '--savedir', 'save/tsakt-ful-v3',
            '--logdir', 'runs/tsakt-ful-v3'
        ]
        
        # 执行命令
        print(f"Command: {' '.join(cmd)}")
        
        # 创建保存目录
        os.makedirs('save/tsakt-ful-v3', exist_ok=True)
        os.makedirs('runs/tsakt-ful-v3', exist_ok=True)
        
        # 执行训练
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully trained: max_pos={max_pos}, num_epochs={num_epochs}")
        else:
            print(f"✗ Failed to train: max_pos={max_pos}, num_epochs={num_epochs}")
            print(f"Error: {result.stderr[:500]}...")

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)