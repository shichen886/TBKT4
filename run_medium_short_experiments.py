import subprocess
import os

# 实验配置
datasets = ['assistments17_long_100', 'assistments17_long_20']
models = ['nopos', 'rope_qk']
seeds = [42, 123, 456]

# 基础配置
base_config = {
    'embed_size': 64,
    'num_heads': 4,
    'num_layers': 2,
    'tensor_rank': 32,
    'max_seq_len': 200,
    'batch_size': 32,
    'lr': 0.0001,
    'weight_decay': 0.001,
    'drop_prob': 0.3,
    'num_epochs': 50,
    'patience': 10
}

print("=" * 80)
print("开始运行实验：NoPos vs RoPE-QK")
print("数据集：assistments17_long_100 (中等长度), assistments17_long_20 (短序列)")
print("每个数据集每个模型运行3个seed: 42, 123, 456")
print("=" * 80)
print()

# 记录所有命令
all_commands = []

for dataset in datasets:
    print(f"\n{'=' * 80}")
    print(f"数据集: {dataset}")
    print(f"{'=' * 80}")
    
    for model in models:
        print(f"\n模型: {model}")
        
        for seed in seeds:
            # 构建保存目录
            if model == 'nopos':
                savedir = f'save/tsakt-linear-nopos-regularized/{dataset}/seed_{seed}'
                script = 'train_tsakt_linear_nopos_regularized.py'
            else:
                savedir = f'save/tsakt-linear-rope-qk-regularized/{dataset}/seed_{seed}'
                script = 'train_tsakt_linear_rope_qk_regularized.py'
            
            # 构建命令
            cmd = [
                'python', script,
                '--dataset', dataset,
                '--savedir', savedir,
                '--seed', str(seed),
                '--embed_size', str(base_config['embed_size']),
                '--num_heads', str(base_config['num_heads']),
                '--num_layers', str(base_config['num_layers']),
                '--tensor_rank', str(base_config['tensor_rank']),
                '--max_seq_len', str(base_config['max_seq_len']),
                '--batch_size', str(base_config['batch_size']),
                '--lr', str(base_config['lr']),
                '--weight_decay', str(base_config['weight_decay']),
                '--drop_prob', str(base_config['drop_prob']),
                '--num_epochs', str(base_config['num_epochs']),
                '--patience', str(base_config['patience'])
            ]
            
            cmd_str = ' '.join(cmd)
            all_commands.append(cmd_str)
            print(f"  Seed {seed}: {savedir}")

print("\n" + "=" * 80)
print(f"总共需要运行 {len(all_commands)} 个实验")
print("=" * 80)
print()

# 询问是否开始运行
print("准备运行所有实验...")
print("按回车键开始，或Ctrl+C取消")
input()

# 运行所有实验
for i, cmd_str in enumerate(all_commands, 1):
    print(f"\n{'=' * 80}")
    print(f"运行实验 {i}/{len(all_commands)}")
    print(f"{'=' * 80}")
    print(f"命令: {cmd_str}")
    print()
    
    try:
        result = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True)
        print("✓ 实验完成")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ 实验失败: {e}")
        print(e.stderr)

print("\n" + "=" * 80)
print("所有实验完成！")
print("=" * 80)
