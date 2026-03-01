import os
import pandas as pd
import numpy as np

def analyze_dataset(dataset_name):
    """
    分析数据集特性
    
    Args:
        dataset_name: 数据集名称（如'assistments12'）
    """
    print(f"\n{'=' * 80}")
    print(f"Analyzing {dataset_name}")
    print(f"{'=' * 80}")
    
    # 读取数据
    data_dir = os.path.join('data', dataset_name)
    train_path = os.path.join(data_dir, 'preprocessed_data_train.csv')
    test_path = os.path.join(data_dir, 'preprocessed_data_test.csv')
    
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"\nBasic Statistics:")
    print(f"  Total interactions: {len(full_df)}")
    print(f"  Unique users: {full_df['user_id'].nunique()}")
    print(f"  Unique items: {full_df['item_id'].nunique()}")
    print(f"  Unique skills: {full_df['skill_id'].nunique()}")
    print(f"  Overall accuracy: {full_df['correct'].mean():.4f}")
    
    # 1. 序列长度分布
    print(f"\n1. Sequence Length Distribution:")
    seq_lengths = full_df.groupby('user_id').size()
    print(f"  Mean sequence length: {seq_lengths.mean():.2f}")
    print(f"  Median sequence length: {seq_lengths.median():.2f}")
    print(f"  Max sequence length: {seq_lengths.max()}")
    print(f"  Min sequence length: {seq_lengths.min()}")
    print(f"  Std sequence length: {seq_lengths.std():.2f}")
    
    # 2. 时间间隔分布
    print(f"\n2. Time Interval Distribution:")
    time_intervals = []
    for user_id, group in full_df.groupby('user_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        if len(group) > 1:
            intervals = group['timestamp'].diff().dropna()
            time_intervals.extend(intervals.tolist())
    
    if time_intervals:
        time_intervals = np.array(time_intervals)
        time_intervals_hours = time_intervals / 3600  # 转换为小时
        print(f"  Mean time interval: {time_intervals_hours.mean():.2f} hours")
        print(f"  Median time interval: {np.median(time_intervals_hours):.2f} hours")
        print(f"  Max time interval: {time_intervals_hours.max():.2f} hours")
        print(f"  Min time interval: {time_intervals_hours.min():.2f} hours")
        print(f"  Std time interval: {time_intervals_hours.std():.2f} hours")
        print(f"  Percentage within 1 hour: {(time_intervals_hours < 1).sum() / len(time_intervals_hours) * 100:.2f}%")
        print(f"  Percentage within 4 hours: {(time_intervals_hours < 4).sum() / len(time_intervals_hours) * 100:.2f}%")
        print(f"  Percentage within 24 hours: {(time_intervals_hours < 24).sum() / len(time_intervals_hours) * 100:.2f}%")
    
    # 3. 题目类型分布
    print(f"\n3. Item Distribution:")
    item_counts = full_df['item_id'].value_counts()
    print(f"  Mean items per user: {item_counts.mean():.2f}")
    print(f"  Median items per user: {item_counts.median():.2f}")
    print(f"  Top 10 most frequent items:")
    for item_id, count in item_counts.head(10).items():
        print(f"    Item {item_id}: {count} times")
    
    # 4. 技能分布
    print(f"\n4. Skill Distribution:")
    skill_counts = full_df['skill_id'].value_counts()
    print(f"  Mean skills per user: {skill_counts.mean():.2f}")
    print(f"  Median skills per user: {skill_counts.median():.2f}")
    print(f"  Top 10 most frequent skills:")
    for skill_id, count in skill_counts.head(10).items():
        print(f"    Skill {skill_id}: {count} times")
    
    # 5. 正确率分布
    print(f"\n5. Accuracy Distribution:")
    user_accuracies = full_df.groupby('user_id')['correct'].mean()
    print(f"  Mean user accuracy: {user_accuracies.mean():.4f}")
    print(f"  Median user accuracy: {user_accuracies.median():.4f}")
    print(f"  Max user accuracy: {user_accuracies.max():.4f}")
    print(f"  Min user accuracy: {user_accuracies.min():.4f}")
    print(f"  Std user accuracy: {user_accuracies.std():.4f}")
    
    # 6. 序列长度与正确率的关系
    print(f"\n6. Sequence Length vs Accuracy:")
    user_stats = full_df.groupby('user_id').agg({
        'item_id': 'size',
        'correct': 'mean'
    }).rename(columns={'item_id': 'seq_length', 'correct': 'accuracy'})
    
    correlation = user_stats['seq_length'].corr(user_stats['accuracy'])
    print(f"  Correlation between sequence length and accuracy: {correlation:.4f}")
    
    # 7. 技能多样性
    print(f"\n7. Skill Diversity:")
    user_skill_diversity = full_df.groupby('user_id')['skill_id'].nunique()
    print(f"  Mean unique skills per user: {user_skill_diversity.mean():.2f}")
    print(f"  Median unique skills per user: {user_skill_diversity.median():.2f}")
    print(f"  Max unique skills per user: {user_skill_diversity.max()}")
    print(f"  Min unique skills per user: {user_skill_diversity.min()}")
    
    return {
        'total_interactions': len(full_df),
        'unique_users': full_df['user_id'].nunique(),
        'unique_items': full_df['item_id'].nunique(),
        'unique_skills': full_df['skill_id'].nunique(),
        'overall_accuracy': full_df['correct'].mean(),
        'mean_seq_length': seq_lengths.mean(),
        'median_seq_length': seq_lengths.median(),
        'max_seq_length': seq_lengths.max(),
        'mean_time_interval_hours': time_intervals_hours.mean() if len(time_intervals) > 0 else None,
        'median_time_interval_hours': np.median(time_intervals_hours) if len(time_intervals) > 0 else None,
        'percent_within_1h': (time_intervals_hours < 1).sum() / len(time_intervals_hours) * 100 if len(time_intervals) > 0 else None,
        'percent_within_4h': (time_intervals_hours < 4).sum() / len(time_intervals_hours) * 100 if len(time_intervals) > 0 else None,
        'mean_user_accuracy': user_accuracies.mean(),
        'correlation_seq_acc': correlation
    }

if __name__ == "__main__":
    # 分析所有数据集
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    results = {}
    for dataset in datasets:
        results[dataset] = analyze_dataset(dataset)
    
    # 对比分析
    print(f"\n{'=' * 80}")
    print("Comparative Analysis")
    print(f"{'=' * 80}")
    
    print(f"\n{'Metric':<30} {'assistments09':<15} {'assistments12':<15} {'assistments15':<15}")
    print(f"{'-' * 75}")
    
    metrics = [
        ('Total Interactions', 'total_interactions'),
        ('Unique Users', 'unique_users'),
        ('Unique Items', 'unique_items'),
        ('Unique Skills', 'unique_skills'),
        ('Overall Accuracy', 'overall_accuracy'),
        ('Mean Seq Length', 'mean_seq_length'),
        ('Median Seq Length', 'median_seq_length'),
        ('Max Seq Length', 'max_seq_length'),
        ('Mean Time Interval (h)', 'mean_time_interval_hours'),
        ('Median Time Interval (h)', 'median_time_interval_hours'),
        ('Within 1h (%)', 'percent_within_1h'),
        ('Within 4h (%)', 'percent_within_4h'),
        ('Mean User Accuracy', 'mean_user_accuracy'),
        ('Seq-Acc Correlation', 'correlation_seq_acc')
    ]
    
    for metric_name, metric_key in metrics:
        values = []
        for dataset in datasets:
            value = results[dataset].get(metric_key)
            if value is not None:
                if isinstance(value, float):
                    values.append(f"{value:.2f}")
                else:
                    values.append(f"{value}")
            else:
                values.append("N/A")
        
        print(f"{metric_name:<30} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
    
    print(f"\n{'=' * 80}")
    print("Analysis Completed!")
    print(f"{'=' * 80}")