import os
import pandas as pd
import shutil

def create_short_sequence_dataset(dataset_name, max_length=50):
    """
    创建短序列数据集，只保留每个学生的前N个练习
    
    Args:
        dataset_name: 数据集名称（如'assistments09'）
        max_length: 最大序列长度（如50或100）
    """
    print(f"Processing {dataset_name} with max_length={max_length}")
    
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
    
    # 按照user_id分组，只保留前max_length个练习
    train_df_short = train_df.groupby('user_id').head(max_length).reset_index(drop=True)
    test_df_short = test_df.groupby('user_id').head(max_length).reset_index(drop=True)
    
    print(f"Short sequence train data shape: {train_df_short.shape}")
    print(f"Short sequence test data shape: {test_df_short.shape}")
    
    # 创建新的数据集目录
    new_dataset_name = f"{dataset_name}_short_{max_length}"
    new_data_dir = os.path.join('data', new_dataset_name)
    os.makedirs(new_data_dir, exist_ok=True)
    
    # 保存短序列数据集
    new_train_path = os.path.join(new_data_dir, 'preprocessed_data_train.csv')
    new_test_path = os.path.join(new_data_dir, 'preprocessed_data_test.csv')
    
    train_df_short.to_csv(new_train_path, sep="\t", index=False)
    test_df_short.to_csv(new_test_path, sep="\t", index=False)
    
    # 复制q_mat.npz文件
    q_mat_path = os.path.join(data_dir, 'q_mat.npz')
    if os.path.exists(q_mat_path):
        new_q_mat_path = os.path.join(new_data_dir, 'q_mat.npz')
        shutil.copy2(q_mat_path, new_q_mat_path)
        print(f"Copied q_mat.npz to {new_q_mat_path}")
    
    print(f"Created short sequence dataset: {new_dataset_name}")
    return True

if __name__ == "__main__":
    # 数据集列表
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    # 最大序列长度
    max_lengths = [50, 100]
    
    for dataset in datasets:
        for max_length in max_lengths:
            print(f"\n{'=' * 80}")
            print(f"Creating short sequence dataset for {dataset} with max_length={max_length}")
            print(f"{'=' * 80}")
            
            success = create_short_sequence_dataset(dataset, max_length)
            
            if success:
                print(f"✓ Successfully created {dataset}_short_{max_length}")
            else:
                print(f"✗ Failed to create {dataset}_short_{max_length}")
    
    print("\n" + "=" * 80)
    print("All short sequence datasets created!")
    print("=" * 80)