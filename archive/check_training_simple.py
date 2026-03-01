import os
import time
import pandas as pd

def get_model_info(model_path):
    """获取模型文件信息"""
    if not os.path.exists(model_path):
        return None
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    file_mtime = os.path.getmtime(model_path)
    file_mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mtime))
    
    return {
        'path': model_path,
        'size_mb': file_size,
        'modified_time': file_mtime_str,
        'exists': True
    }

def analyze_model_training(dataset, model_name, model_path):
    """分析模型训练情况"""
    print(f"\n{'='*80}")
    print(f"分析模型: {model_name} on {dataset}")
    print(f"{'='*80}")
    
    model_info = get_model_info(model_path)
    
    if not model_info:
        print(f"模型文件不存在: {model_path}")
        return None
    
    print(f"模型路径: {model_info['path']}")
    print(f"文件大小: {model_info['size_mb']:.2f} MB")
    print(f"修改时间: {model_info['modified_time']}")
    
    return model_info

def main():
    print("=" * 80)
    print("TSAKT-w/o-Pos 训练情况分析")
    print("=" * 80)
    
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")
        
        TSAKT_wo_pos_path = os.path.join('save', 'tsakt', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
        
        TSAKT_ful_path = os.path.join('save', 'tsakt-ful-v2', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3')
        
        sakt_path = os.path.join('save', 'sakt', 
            f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
        
        TSAKT_wo_pos_info = analyze_model_training(dataset, 'TSAKT-w/o-Pos', TSAKT_wo_pos_path)
        TSAKT_ful_info = analyze_model_training(dataset, 'TSAKT-Ful', TSAKT_ful_path)
        sakt_info = analyze_model_training(dataset, 'SAKT', sakt_path)
        
        if TSAKT_wo_pos_info and sakt_info:
            print(f"\n{'-'*80}")
            print(f"对比分析: TSAKT-w/o-Pos vs SAKT")
            print(f"{'-'*80}")
            
            print(f"{'指标':<30} {'TSAKT-w/o-Pos':<25} {'SAKT':<25} {'差异':<15}")
            print("-" * 95)
            
            size_diff = TSAKT_wo_pos_info['size_mb'] - sakt_info['size_mb']
            print(f"{'文件大小 (MB)':<30} {TSAKT_wo_pos_info['size_mb']:<25.2f} {sakt_info['size_mb']:<25.2f} {size_diff:<15.2f}")
            
            print(f"{'修改时间':<30} {TSAKT_wo_pos_info['modified_time']:<25} {sakt_info['modified_time']:<25}")
            
            if TSAKT_wo_pos_info['size_mb'] < 10:
                print(f"\n⚠️  警告: TSAKT-w/o-Pos 模型文件过小 ({TSAKT_wo_pos_info['size_mb']:.2f} MB)，可能训练不充分")
            
            if size_diff > 5:
                print(f"\n⚠️  警告: TSAKT-w/o-Pos 模型文件比SAKT大 {size_diff:.2f} MB，可能包含额外参数")
        
        if TSAKT_wo_pos_info and TSAKT_ful_info:
            print(f"\n{'-'*80}")
            print(f"对比分析: TSAKT-w/o-Pos vs TSAKT-Ful")
            print(f"{'-'*80}")
            
            print(f"{'指标':<30} {'TSAKT-w/o-Pos':<25} {'TSAKT-Ful':<25} {'差异':<15}")
            print("-" * 95)
            
            size_diff = TSAKT_ful_info['size_mb'] - TSAKT_wo_pos_info['size_mb']
            print(f"{'文件大小 (MB)':<30} {TSAKT_wo_pos_info['size_mb']:<25.2f} {TSAKT_ful_info['size_mb']:<25.2f} {size_diff:<15.2f}")
            
            print(f"{'修改时间':<30} {TSAKT_wo_pos_info['modified_time']:<25} {TSAKT_ful_info['modified_time']:<25}")
            
            if size_diff > 5:
                print(f"\nℹ️  信息: TSAKT-Ful 模型文件比TSAKT-w/o-Pos大 {size_diff:.2f} MB，位置编码增加了参数量")

if __name__ == "__main__":
    main()
