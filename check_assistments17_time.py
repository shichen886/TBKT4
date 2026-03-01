import pandas as pd
import os

dataset = 'assistments17_time_1h'
data_dir = f'data/{dataset}'

print("=" * 60)
print(f"检查 {dataset} 数据集")
print("=" * 60)

# 检查文件
files_to_check = [
    'preprocessed_data.csv',
    'preprocessed_data_train.csv',
    'preprocessed_data_test.csv',
    'q_mat.npz'
]

print("\n📁 文件检查:")
for file in files_to_check:
    file_path = os.path.join(data_dir, file)
    if os.path.exists(file_path):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} (不存在)")

# 读取数据
if os.path.exists(os.path.join(data_dir, 'preprocessed_data.csv')):
    df = pd.read_csv(os.path.join(data_dir, 'preprocessed_data.csv'), sep='\t')
    print(f"\n📊 完整数据集:")
    print(f"  - 行数: {len(df):,}")
    print(f"  - 用户数: {len(df.groupby('user_id')):,}")
    print(f"  - 题目数: {int(df['item_id'].max()) + 1:,}")
    print(f"  - 技能数: {int(df['skill_id'].max()) + 1:,}")

if os.path.exists(os.path.join(data_dir, 'preprocessed_data_train.csv')):
    train_df = pd.read_csv(os.path.join(data_dir, 'preprocessed_data_train.csv'), sep='\t')
    print(f"\n📊 训练集:")
    print(f"  - 行数: {len(train_df):,}")
    print(f"  - 用户数: {len(train_df.groupby('user_id')):,}")

if os.path.exists(os.path.join(data_dir, 'preprocessed_data_test.csv')):
    test_df = pd.read_csv(os.path.join(data_dir, 'preprocessed_data_test.csv'), sep='\t')
    print(f"\n📊 测试集:")
    print(f"  - 行数: {len(test_df):,}")
    print(f"  - 用户数: {len(test_df.groupby('user_id')):,}")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)