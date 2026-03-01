import pandas as pd
import os

datasets = [
    'assistments09',
    'assistments12',
    'assistments15',
    'assistments17',
    'assistments09_time_1h',
    'assistments09_time_2h',
    'assistments09_time_4h',
    'assistments12_time_1h',
    'assistments12_time_2h',
    'assistments12_time_4h',
    'assistments15_time_1h',
    'assistments15_time_2h',
    'assistments15_time_4h',
]

print("=" * 60)
print("数据集信息检查")
print("=" * 60)

for dataset in datasets:
    data_path = f'data/{dataset}/preprocessed_data.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ {dataset}: 文件不存在")
        continue
    
    try:
        df = pd.read_csv(data_path, sep='\t')
        num_rows = len(df)
        num_users = len(df.groupby('user_id'))
        num_items = int(df['item_id'].max()) + 1
        num_skills = int(df['skill_id'].max()) + 1
        
        print(f"\n✅ {dataset}:")
        print(f"   - 行数: {num_rows:,}")
        print(f"   - 用户数: {num_users:,}")
        print(f"   - 题目数: {num_items:,}")
        print(f"   - 技能数: {num_skills:,}")
        
    except Exception as e:
        print(f"\n❌ {dataset}: 读取失败 - {e}")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)