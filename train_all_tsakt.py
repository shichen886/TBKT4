import argparse
import os
import subprocess
import sys

def train_model(dataset):
    print(f"\n{'='*60}")
    print(f"开始训练数据集: {dataset}")
    print(f"{'='*60}")
    
    model_path = f'save/tsakt/{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3'
    
    if os.path.exists(model_path):
        print(f"⏭️ {dataset} 模型已存在，跳过训练")
        return True
    
    cmd = [
        "C:/Users/32880/miniconda3/envs/emnist-gpu/python.exe",
        "train_tsakt.py",
        "--dataset", dataset,
        "--batch_size", "128",
        "--max_length", "200",
        "--embed_size", "60",
        "--num_attn_layers", "2",
        "--num_heads", "5",
        "--max_pos", "5",
        "--drop_prob", "0.2",
        "--tensor_rank", "3"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        print(f"✅ {dataset} 训练完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {dataset} 训练失败！")
        print("错误输出:", e.stderr)
        return False

if __name__ == "__main__":
    datasets = [
        "assistments09",
        "assistments12", 
        "assistments15",
        "algebra05",
        "assistments17",
        "bridge_algebra06"
    ]
    
    parser = argparse.ArgumentParser(description='批量训练 TSAKT 模型')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='要训练的数据集列表，如果不指定则训练所有数据集')
    args = parser.parse_args()
    
    if args.datasets:
        datasets = args.datasets
    
    print(f"🚀 开始批量训练 TSAKT 模型")
    print(f"📊 将训练以下数据集: {', '.join(datasets)}")
    print(f"📁 模型将保存到: save/tsakt/")
    
    results = {}
    for dataset in datasets:
        success = train_model(dataset)
        results[dataset] = success
    
    print(f"\n{'='*60}")
    print("📊 训练结果汇总")
    print(f"{'='*60}")
    
    for dataset, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{dataset}: {status}")
    
    successful = sum(1 for s in results.values() if s)
    total = len(results)
    
    print(f"\n总计: {successful}/{total} 个数据集训练成功")
    
    if successful == total:
        print("🎉 所有数据集训练完成！")
    else:
        print("⚠️ 部分数据集训练失败，请检查错误信息")
