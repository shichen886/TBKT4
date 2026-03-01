import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from model_sakt import SAKT
from model_tsakt_linear import TSAKT_Linear


def measure_memory_and_speed_sakt(model, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, model_name, num_runs=10):
    """测量SAKT模型的内存占用和推理速度"""
    
    model.eval()
    
    # 预热
    with torch.no_grad():
        _ = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 测量推理时间
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            outputs = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # 测量内存占用
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            outputs = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
        
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved
    }


def measure_memory_and_speed_tsakt(model, item_ids, skill_ids, mask, model_name, num_runs=10):
    """测量TSAKT-Linear模型的内存占用和推理速度"""
    
    model.eval()
    
    # 预热
    with torch.no_grad():
        _ = model(item_ids, skill_ids, mask)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 测量推理时间
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            outputs = model(item_ids, skill_ids, mask)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # 测量内存占用
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            outputs = model(item_ids, skill_ids, mask)
        
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved
    }


def compare_models(dataset='assistments12', seq_lengths=[50, 100, 200, 500], batch_size=4):
    """对比SAKT和TSAKT-Linear在不同序列长度下的性能"""
    
    print(f"\n{'='*100}")
    print(f"对比SAKT和TSAKT-Linear在{dataset}数据集上的性能")
    print(f"{'='*100}")
    
    # 加载数据集信息
    full_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    num_items = int(full_df["item_id"].max() + 1)
    num_skills = int(full_df["skill_id"].max() + 1)
    
    print(f"\n数据集信息:")
    print(f"  题目数量: {num_items}")
    print(f"  技能数量: {num_skills}")
    
    # 创建模型
    sakt_model = SAKT(num_items, num_skills, embed_size=128, num_attn_layers=2, num_heads=4,
                     encode_pos=False, max_pos=5, drop_prob=0.1).to(device)
    
    tsakt_linear_model = TSAKT_Linear(num_items, num_skills, embed_size=128, num_layers=2, num_heads=4,
                                    tensor_rank=32, max_len=500, drop_prob=0.1).to(device)
    
    # 计算参数量
    sakt_params = sum(p.numel() for p in sakt_model.parameters())
    tsakt_params = sum(p.numel() for p in tsakt_linear_model.parameters())
    
    print(f"\n模型参数量:")
    print(f"  SAKT: {sakt_params:,}")
    print(f"  TSAKT-Linear: {tsakt_params:,}")
    print(f"  参数变化: {(1 - tsakt_params/sakt_params)*100:.2f}%")
    
    # 对比不同序列长度
    results = []
    
    for seq_len in seq_lengths:
        print(f"\n{'='*100}")
        print(f"序列长度: {seq_len}")
        print(f"{'='*100}")
        
        # 生成测试数据
        item_ids = torch.randint(0, num_items, (batch_size, seq_len)).to(device)
        skill_ids = torch.randint(0, num_skills, (batch_size, seq_len)).to(device)
        label_inputs = torch.randint(0, 2, (batch_size, seq_len)).to(device)
        mask = torch.ones(batch_size, seq_len).to(device)
        
        # 测试SAKT
        print(f"\n测试SAKT...")
        sakt_metrics = measure_memory_and_speed_sakt(sakt_model, item_ids, skill_ids, label_inputs, item_ids, skill_ids, "SAKT")
        
        # 测试TSAKT-Linear
        print(f"测试TSAKT-Linear...")
        tsakt_metrics = measure_memory_and_speed_tsakt(tsakt_linear_model, item_ids, skill_ids, mask, "TSAKT-Linear")
        
        # 计算改进
        time_improvement = (1 - tsakt_metrics['avg_time'] / sakt_metrics['avg_time']) * 100
        memory_improvement = (1 - tsakt_metrics['memory_allocated_mb'] / sakt_metrics['memory_allocated_mb']) * 100
        
        print(f"\n结果对比:")
        print(f"  推理时间:")
        print(f"    SAKT: {sakt_metrics['avg_time']*1000:.2f} ± {sakt_metrics['std_time']*1000:.2f} ms")
        print(f"    TSAKT-Linear: {tsakt_metrics['avg_time']*1000:.2f} ± {tsakt_metrics['std_time']*1000:.2f} ms")
        print(f"    改进: {time_improvement:.2f}%")
        
        print(f"  显存占用:")
        print(f"    SAKT: {sakt_metrics['memory_allocated_mb']:.2f} MB")
        print(f"    TSAKT-Linear: {tsakt_metrics['memory_allocated_mb']:.2f} MB")
        print(f"    改进: {memory_improvement:.2f}%")
        
        results.append({
            'seq_len': seq_len,
            'sakt_time': sakt_metrics['avg_time'],
            'tsakt_time': tsakt_metrics['avg_time'],
            'sakt_memory': sakt_metrics['memory_allocated_mb'],
            'tsakt_memory': tsakt_metrics['memory_allocated_mb'],
            'time_improvement': time_improvement,
            'memory_improvement': memory_improvement
        })
    
    # 打印总结
    print(f"\n{'='*100}")
    print(f"总结")
    print(f"{'='*100}")
    print(f"\n{'序列长度':<10} {'SAKT时间(ms)':<15} {'TSAKT时间(ms)':<15} {'时间改进(%)':<15} {'SAKT内存(MB)':<15} {'TSAKT内存(MB)':<15} {'内存改进(%)':<15}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['seq_len']:<10} "
              f"{result['sakt_time']*1000:<15.2f} "
              f"{result['tsakt_time']*1000:<15.2f} "
              f"{result['time_improvement']:<15.2f} "
              f"{result['sakt_memory']:<15.2f} "
              f"{result['tsakt_memory']:<15.2f} "
              f"{result['memory_improvement']:<15.2f}")
    
    # 计算平均改进
    avg_time_improvement = np.mean([r['time_improvement'] for r in results])
    avg_memory_improvement = np.mean([r['memory_improvement'] for r in results])
    
    print("-" * 100)
    print(f"{'平均':<10} "
          f"{'':<15} "
          f"{'':<15} "
          f"{avg_time_improvement:<15.2f} "
          f"{'':<15} "
          f"{'':<15} "
          f"{avg_memory_improvement:<15.2f}")
    
    return results


if __name__ == "__main__":
    datasets = ['assistments09', 'assistments12', 'assistments15']
    
    for dataset in datasets:
        try:
            results = compare_models(dataset, seq_lengths=[50, 100, 200, 500], batch_size=4)
            print(f"\n✅ {dataset} 对比完成")
        except Exception as e:
            print(f"\n❌ {dataset} 对比失败: {e}")
            import traceback
            traceback.print_exc()
