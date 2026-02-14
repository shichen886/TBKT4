import os
import subprocess

# 只测试一个数据集
dataset = 'assistments09'

# 测试max_pos=200和num_epochs=512
max_pos = 200
num_epochs = 512

print("=" * 80)
print(f"Testing TSAKT-Ful on {dataset} with max_pos={max_pos} and num_epochs={num_epochs}")
print("=" * 80)

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
print("Test completed!")
print("=" * 80)