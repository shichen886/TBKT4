import subprocess
import os

# 训练TSAKT的不同变体
def train_tsakt_variants():
    # 数据集列表
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    # 变体配置
    variants = [
        {'name': 'tsakt-wo-pos', 'encode_pos': False},
        {'name': 'tsakt-ful', 'encode_pos': True}
    ]
    
    print("=" * 80)
    print("Training TSAKT variants")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\nTraining on dataset: {dataset}")
        print("-" * 60)
        
        for variant in variants:
            print(f"\n[Training] {variant['name']} with encode_pos={variant['encode_pos']}")
            
            # 构建命令
            cmd = [
                'python', 'train_tsakt.py',
                '--dataset', dataset,
                '--encode_pos', str(variant['encode_pos']),
                '--savedir', f'save/{variant["name"]}',
                '--logdir', f'runs/{variant["name"]}'
            ]
            
            # 执行命令
            print(f"Command: {' '.join(cmd)}")
            
            # 创建保存目录
            os.makedirs(f'save/{variant["name"]}', exist_ok=True)
            os.makedirs(f'runs/{variant["name"]}', exist_ok=True)
            
            # 执行训练
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully trained {variant['name']} on {dataset}")
            else:
                print(f"✗ Failed to train {variant['name']} on {dataset}")
                print(f"Error: {result.stderr[:500]}...")

if __name__ == "__main__":
    train_tsakt_variants()
