import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_dataset(dataset_name, data_dir='data'):
    """
    分析单个数据集
    
    Args:
        dataset_name: 数据集名称（如 'assistments09'）
        data_dir: 数据目录
    
    Returns:
        数据集统计信息字典
    """
    dataset_path = Path(data_dir) / dataset_name
    
    if not dataset_path.exists():
        return None
    
    stats = {
        'dataset_name': dataset_name,
        'train_records': 0,
        'test_records': 0,
        'total_records': 0,
        'num_students': 0,
        'num_items': 0,
        'num_skills': 0,
        'avg_correct_rate': 0.0,
        'avg_records_per_student': 0.0,
        'avg_records_per_item': 0.0
    }
    
    try:
        train_file = dataset_path / 'preprocessed_data_train.csv'
        test_file = dataset_path / 'preprocessed_data_test.csv'
        
        if train_file.exists():
            train_df = pd.read_csv(train_file, sep='\t')
            stats['train_records'] = len(train_df)
            stats['num_students'] = train_df['user_id'].nunique()
            stats['num_items'] = train_df['item_id'].nunique()
            stats['num_skills'] = train_df['skill_id'].nunique()
            stats['avg_correct_rate'] = train_df['correct'].mean()
            stats['avg_records_per_student'] = len(train_df) / stats['num_students']
            stats['avg_records_per_item'] = len(train_df) / stats['num_items']
        
        if test_file.exists():
            test_df = pd.read_csv(test_file, sep='\t')
            stats['test_records'] = len(test_df)
        
        stats['total_records'] = stats['train_records'] + stats['test_records']
        
    except Exception as e:
        print(f'分析数据集 {dataset_name} 时出错: {str(e)}')
        return None
    
    return stats

def analyze_all_datasets(data_dir='data'):
    """
    分析所有数据集
    
    Args:
        data_dir: 数据目录
    
    Returns:
        所有数据集的统计信息列表
    """
    datasets = [
        'assistments09',
        'assistments12',
        'assistments15',
        'assistments17',
        'algebra05',
        'bridge_algebra06',
        'statics'
    ]
    
    all_stats = []
    
    for dataset in datasets:
        stats = analyze_dataset(dataset, data_dir)
        if stats:
            all_stats.append(stats)
            print(f'✓ {dataset}: {stats["total_records"]:,} 条记录, {stats["num_students"]:,} 学生, {stats["num_items"]:,} 题目, {stats["num_skills"]:,} 知识点')
    
    return all_stats

def save_stats_to_csv(stats_list, output_file='dataset_statistics.csv'):
    """
    保存统计信息到CSV文件
    
    Args:
        stats_list: 统计信息列表
        output_file: 输出文件名
    """
    df = pd.DataFrame(stats_list)
    
    # 重新排列列顺序
    columns = [
        'dataset_name',
        'train_records',
        'test_records',
        'total_records',
        'num_students',
        'num_items',
        'num_skills',
        'avg_correct_rate',
        'avg_records_per_student',
        'avg_records_per_item'
    ]
    
    df = df[columns]
    
    # 格式化输出
    df['train_records'] = df['train_records'].apply(lambda x: f'{x:,}')
    df['test_records'] = df['test_records'].apply(lambda x: f'{x:,}')
    df['total_records'] = df['total_records'].apply(lambda x: f'{x:,}')
    df['num_students'] = df['num_students'].apply(lambda x: f'{x:,}')
    df['num_items'] = df['num_items'].apply(lambda x: f'{x:,}')
    df['num_skills'] = df['num_skills'].apply(lambda x: f'{x:,}')
    df['avg_correct_rate'] = df['avg_correct_rate'].apply(lambda x: f'{x:.2%}')
    df['avg_records_per_student'] = df['avg_records_per_student'].apply(lambda x: f'{x:.1f}')
    df['avg_records_per_item'] = df['avg_records_per_item'].apply(lambda x: f'{x:.1f}')
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'\n统计信息已保存到 {output_file}')

def save_stats_to_json(stats_list, output_file='dataset_statistics.json'):
    """
    保存统计信息到JSON文件
    
    Args:
        stats_list: 统计信息列表
        output_file: 输出文件名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats_list, f, indent=2, ensure_ascii=False)
    print(f'统计信息已保存到 {output_file}')

def print_summary(stats_list):
    """
    打印统计信息摘要
    
    Args:
        stats_list: 统计信息列表
    """
    print('\n' + '='*80)
    print('数据集统计信息摘要')
    print('='*80)
    
    print(f'\n数据集数量: {len(stats_list)}')
    print(f'总记录数: {sum(s["total_records"] for s in stats_list):,}')
    print(f'总学生数: {sum(s["num_students"] for s in stats_list):,}')
    print(f'总题目数: {max(s["num_items"] for s in stats_list):,}')
    print(f'总知识点数: {max(s["num_skills"] for s in stats_list):,}')
    
    print('\n推荐用于实验的数据集:')
    print('-'*80)
    
    # 按记录数排序
    sorted_stats = sorted(stats_list, key=lambda x: x['total_records'], reverse=True)
    
    print(f'1. 主要实验集: {sorted_stats[0]["dataset_name"]} ({sorted_stats[0]["total_records"]:,} 条记录)')
    if len(sorted_stats) > 1:
        print(f'2. 验证实验集: {sorted_stats[1]["dataset_name"]} ({sorted_stats[1]["total_records"]:,} 条记录)')
    if len(sorted_stats) > 2:
        print(f'3. 补充实验集: {sorted_stats[2]["dataset_name"]} ({sorted_stats[2]["total_records"]:,} 条记录)')
    
    print('='*80)

if __name__ == '__main__':
    print('开始分析数据集...\n')
    
    stats_list = analyze_all_datasets()
    
    if stats_list:
        save_stats_to_csv(stats_list)
        save_stats_to_json(stats_list)
        print_summary(stats_list)
    else:
        print('未找到任何数据集！')