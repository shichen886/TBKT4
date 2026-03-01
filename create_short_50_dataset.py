import os
import pandas as pd
import shutil

def create_short_sequence_dataset(dataset_name, min_length=50):
    """
    创建短序列数据集
    
    Args:
        dataset_name: 数据集名称（如'assistments17'）
        min_length: 最小序列长度
    """
    print(f"Processing {dataset_name} with min_length={min_length}")
    
    # 读取原始数据集
    data_dir = os.path.join('data', dataset_name)
    train_path = os.path.join(data_dir, 'preprocessed_data_train.csv')
    test_path = os.path.join(data_dir, 'preprocessed_data_test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Data files not found: {train_path}, {test_path}")
        return False
    
    # 读取数据
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    
    print(f"Original train data shape: {train_df.shape}")
    print(f"Original test data shape: {test_df.shape}")
    
    # 统计每个用户的序列长度
    train_user_lengths = train_df.groupby('user_id').size()
    test_user_lengths = test_df.groupby('user_id').size()
    
    print(f"Original train users: {len(train_user_lengths)}")
    print(f"Original test users: {len(test_user_lengths)}")
    print(f"Train sequence length - min: {train_user_lengths.min()}, max: {train_user_lengths.max()}, mean: {train_user_lengths.mean():.2f}")
    print(f"Test sequence length - min: {test_user_lengths.min()}, max: {test_user_lengths.max()}, mean: {test_user_lengths.mean():.2f}")
    
    # 只保留序列长度 >= min_length 的学生
    short_train_users = train_user_lengths[train_user_lengths >= min_length].index
    short_test_users = test_user_lengths[test_user_lengths >= min_length].index
    
    train_df_short = train_df[train_df['user_id'].isin(short_train_users)]
    test_df_short = test_df[test_df['user_id'].isin(short_test_users)]
    
    print(f"\nFiltered train users with length >= {min_length}: {len(short_train_users)}")
    print(f"Filtered test users with length >= {min_length}: {len(short_test_users)}")
    
    print(f"Filtered train data shape: {train_df_short.shape}")
    print(f"Filtered test data shape: {test_df_short.shape}")
    
    # 创建新的数据集目录
    new_dataset_name = f"{dataset_name}_short_{min_length}"
    new_data_dir = os.path.join('data', new_dataset_name)
    os.makedirs(new_data_dir, exist_ok=True)
    
    # 保存数据集
    new_train_path = os.path.join(new_data_dir, 'preprocessed_data_train.csv')
    new_test_path = os.path.join(new_data_dir, 'preprocessed_data_test.csv')
    new_combined_path = os.path.join(new_data_dir, 'preprocessed_data.csv')
    
    train_df_short.to_csv(new_train_path, sep="\t", index=False)
    test_df_short.to_csv(new_test_path, sep="\t", index=False)
    
    # 合并train和test数据
    combined_df = pd.concat([train_df_short, test_df_short], ignore_index=True)
    combined_df.to_csv(new_combined_path, sep="\t", index=False)
    
    # 复制q_mat.npz文件
    q_mat_path = os.path.join(data_dir, 'q_mat.npz')
    if os.path.exists(q_mat_path):
        new_q_mat_path = os.path.join(new_data_dir, 'q_mat.npz')
        shutil.copy2(q_mat_path, new_q_mat_path)
        print(f"Copied q_mat.npz to {new_q_mat_path}")
    
    print(f"Created dataset: {new_dataset_name}")
    return True

if __name__ == "__main__":
    # 创建短序列数据集（min_length=50）
    print("=" * 80)
    print("Creating short-sequence dataset (min_length=50)")
    print("=" * 80)
    success = create_short_sequence_dataset('assistments17', min_length=50)
    
    print("\n" + "=" * 80)
    if success:
        print("✓ Successfully created assistments17_short_50")
    else:
        print("✗ Failed to create assistments17_short_50")
    print("=" * 80)
