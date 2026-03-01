import os
import pandas as pd
import numpy as np

def create_time_dependent_dataset(dataset_name, max_time_gap_hours=1, min_sequence_length=10):
    """
    创建具有强时间依赖的数据集，只保留学生在短时间内连续做题的序列
    
    Args:
        dataset_name: 数据集名称（如'assistments09'）
        max_time_gap_hours: 最大时间间隔（小时），超过此间隔则截断序列
        min_sequence_length: 最小序列长度
    """
    print(f"Processing {dataset_name} with max_time_gap={max_time_gap_hours}h")
    
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
    
    def filter_time_dependent(df, max_gap_hours):
        """
        过滤数据，只保留短时间内连续做题的序列
        """
        filtered_rows = []
        
        # 按user_id分组
        for user_id, group in df.groupby('user_id'):
            # 按timestamp排序
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # 计算时间间隔
            if len(group) > 1:
                time_diffs = group['timestamp'].diff()
                time_diffs_hours = time_diffs / 3600  # 转换为小时
                
                # 找到时间间隔超过阈值的断点
                break_points = time_diffs_hours > max_gap_hours
                break_points.iloc[0] = False  # 第一个元素不是断点
                
                # 按断点分割序列
                start_idx = 0
                for i, is_break in enumerate(break_points):
                    if is_break:
                        # 保存从start_idx到i的序列
                        if i - start_idx >= min_sequence_length:
                            filtered_rows.append(group.iloc[start_idx:i])
                        start_idx = i
                
                # 保存最后一个序列
                if len(group) - start_idx >= min_sequence_length:
                    filtered_rows.append(group.iloc[start_idx:])
            elif len(group) >= min_sequence_length:
                # 只有一个交互，但满足最小长度要求
                filtered_rows.append(group)
        
        if filtered_rows:
            return pd.concat(filtered_rows, ignore_index=True)
        else:
            return pd.DataFrame()
    
    # 过滤训练集和测试集
    train_df_filtered = filter_time_dependent(train_df, max_time_gap_hours)
    test_df_filtered = filter_time_dependent(test_df, max_time_gap_hours)
    
    print(f"Time-dependent train data shape: {train_df_filtered.shape}")
    print(f"Time-dependent test data shape: {test_df_filtered.shape}")
    
    if train_df_filtered.empty or test_df_filtered.empty:
        print(f"Warning: Filtered data is empty, trying larger time gap...")
        return False
    
    # 创建新的数据集目录
    new_dataset_name = f"{dataset_name}_time_{max_time_gap_hours}h"
    new_data_dir = os.path.join('data', new_dataset_name)
    os.makedirs(new_data_dir, exist_ok=True)
    
    # 保存时间依赖数据集
    new_train_path = os.path.join(new_data_dir, 'preprocessed_data_train.csv')
    new_test_path = os.path.join(new_data_dir, 'preprocessed_data_test.csv')
    
    train_df_filtered.to_csv(new_train_path, sep="\t", index=False)
    test_df_filtered.to_csv(new_test_path, sep="\t", index=False)
    
    # 复制q_mat.npz文件
    q_mat_path = os.path.join(data_dir, 'q_mat.npz')
    if os.path.exists(q_mat_path):
        new_q_mat_path = os.path.join(new_data_dir, 'q_mat.npz')
        import shutil
        shutil.copy2(q_mat_path, new_q_mat_path)
        print(f"Copied q_mat.npz to {new_q_mat_path}")
    
    # 创建preprocessed_data.csv
    full_df = pd.concat([train_df_filtered, test_df_filtered], ignore_index=True)
    preprocessed_path = os.path.join(new_data_dir, 'preprocessed_data.csv')
    full_df.to_csv(preprocessed_path, sep="\t", index=False)
    
    print(f"Created time-dependent dataset: {new_dataset_name}")
    return True

if __name__ == "__main__":
    # 数据集列表
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    # 时间间隔（小时）
    time_gaps = [1, 2, 4]  # 1小时、2小时、4小时内连续做题
    
    for dataset in datasets:
        for time_gap in time_gaps:
            print(f"\n{'=' * 80}")
            print(f"Creating time-dependent dataset for {dataset} with max_time_gap={time_gap}h")
            print(f"{'=' * 80}")
            
            success = create_time_dependent_dataset(dataset, max_time_gap_hours=time_gap)
            
            if success:
                print(f"✓ Successfully created {dataset}_time_{time_gap}h")
            else:
                print(f"✗ Failed to create {dataset}_time_{time_gap}h")
    
    print("\n" + "=" * 80)
    print("All time-dependent datasets created!")
    print("=" * 80)