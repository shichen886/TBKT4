import os
import pandas as pd

def create_preprocessed_file(dataset_name):
    """
    为短序列数据集创建preprocessed_data.csv文件
    
    Args:
        dataset_name: 数据集名称（如'assistments09_short_50'）
    """
    print(f"Creating preprocessed_data.csv for {dataset_name}")
    
    # 读取train和test数据
    data_dir = os.path.join('data', dataset_name)
    train_path = os.path.join(data_dir, 'preprocessed_data_train.csv')
    test_path = os.path.join(data_dir, 'preprocessed_data_test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Data files not found: {train_path}, {test_path}")
        return False
    
    # 读取数据
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    
    # 合并train和test数据
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 保存preprocessed_data.csv
    preprocessed_path = os.path.join(data_dir, 'preprocessed_data.csv')
    full_df.to_csv(preprocessed_path, sep="\t", index=False)
    
    print(f"Created preprocessed_data.csv with {len(full_df)} rows")
    return True

if __name__ == "__main__":
    # 数据集列表
    datasets = [
        'assistments09_short_50',
        'assistments09_short_100',
        'assistments12_short_50',
        'assistments12_short_100',
        'assistments15_short_50',
        'assistments15_short_100'
    ]
    
    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing {dataset}")
        print(f"{'=' * 80}")
        
        success = create_preprocessed_file(dataset)
        
        if success:
            print(f"✓ Successfully created preprocessed_data.csv for {dataset}")
        else:
            print(f"✗ Failed to create preprocessed_data.csv for {dataset}")
    
    print("\n" + "=" * 80)
    print("All preprocessed_data.csv files created!")
    print("=" * 80)