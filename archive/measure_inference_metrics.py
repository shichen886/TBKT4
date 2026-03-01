import os
import time
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from model_sakt import SAKT
from model_tsakt import TSAKT

def measure_inference_metrics(model, test_batches, model_name):
    """测量推理指标"""
    model.eval()
    
    memory_usage = []
    inference_times = []
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        for batch_idx, (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels) in enumerate(test_batches):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            item_inputs = item_inputs.to(device)
            skill_inputs = skill_inputs.to(device)
            label_inputs = label_inputs.to(device)
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            
            start_time = time.time()
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
                memory_usage.append({
                    'allocated_mb': memory_allocated,
                    'reserved_mb': memory_reserved
                })
            
            if batch_idx >= 99:
                break
    
    result = {
        'model_name': model_name,
        'avg_inference_time': np.mean(inference_times),
        'std_inference_time': np.std(inference_times),
        'min_inference_time': np.min(inference_times),
        'max_inference_time': np.max(inference_times),
        'total_inference_time': np.sum(inference_times),
        'avg_memory_allocated_mb': np.mean([m['allocated_mb'] for m in memory_usage]) if memory_usage else 0,
        'max_memory_allocated_mb': np.max([m['allocated_mb'] for m in memory_usage]) if memory_usage else 0,
        'avg_memory_reserved_mb': np.mean([m['reserved_mb'] for m in memory_usage]) if memory_usage else 0,
        'max_memory_reserved_mb': np.max([m['reserved_mb'] for m in memory_usage]) if memory_usage else 0,
        'num_batches': len(inference_times)
    }
    
    return result

def prepare_test_data(dataset, max_length=200, batch_size=8):
    """准备测试数据"""
    test_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_test.csv'), sep="\t")
    
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in test_df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in test_df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in test_df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list, max_length):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l, max_length) for l in lists]
    
    test_data = list(zip(*chunked_lists))
    
    test_batches = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        batch_items = list(zip(*batch))
        test_batches.append([torch.nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=0) 
                            if b[0] is not None else None for b in batch_items])
    
    return test_batches

def analyze_dataset(dataset, batch_sizes=[1, 2, 4, 8, 16, 32]):
    """分析数据集"""
    print(f"\n{'='*100}")
    print(f"Dataset: {dataset}")
    print(f"{'='*100}")
    
    results = defaultdict(list)
    
    for batch_size in batch_sizes:
        print(f"\n{'-'*100}")
        print(f"Batch Size: {batch_size}")
        print(f"{'-'*100}")
        
        try:
            test_batches = prepare_test_data(dataset, max_length=200, batch_size=batch_size)
            
            sakt_path = os.path.join('save', 'sakt', 
                f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
            tsakt_wo_pos_path = os.path.join('save', 'tsakt', 
                f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
            
            if os.path.exists(sakt_path):
                sakt_model = torch.load(sakt_path, map_location=device, weights_only=False).to(device)
                sakt_result = measure_inference_metrics(sakt_model, test_batches, 'SAKT')
                sakt_result['batch_size'] = batch_size
                results['sakt'].append(sakt_result)
                
                print(f"SAKT:")
                print(f"  平均推理时间: {sakt_result['avg_inference_time']:.4f}s")
                print(f"  平均显存占用: {sakt_result['avg_memory_allocated_mb']:.2f} MB")
                print(f"  最大显存占用: {sakt_result['max_memory_allocated_mb']:.2f} MB")
                
                del sakt_model
                torch.cuda.empty_cache()
            
            if os.path.exists(tsakt_wo_pos_path):
                tsakt_model = torch.load(tsakt_wo_pos_path, map_location=device, weights_only=False).to(device)
                tsakt_result = measure_inference_metrics(tsakt_model, test_batches, 'TSAKT-w/o-Pos')
                tsakt_result['batch_size'] = batch_size
                results['tsakt_wo_pos'].append(tsakt_result)
                
                print(f"TSAKT-w/o-Pos:")
                print(f"  平均推理时间: {tsakt_result['avg_inference_time']:.4f}s")
                print(f"  平均显存占用: {tsakt_result['avg_memory_allocated_mb']:.2f} MB")
                print(f"  最大显存占用: {tsakt_result['max_memory_allocated_mb']:.2f} MB")
                
                del tsakt_model
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Batch size {batch_size} 失败: {e}")
            continue
    
    return results

def compare_results(results):
    """比较结果"""
    print(f"\n{'='*100}")
    print(f"对比分析")
    print(f"{'='*100}")
    
    for batch_size in sorted(set([r['batch_size'] for r in results.get('sakt', []) + results.get('tsakt_wo_pos', [])])):
        sakt_result = next((r for r in results.get('sakt', []) if r['batch_size'] == batch_size), None)
        tsakt_result = next((r for r in results.get('tsakt_wo_pos', []) if r['batch_size'] == batch_size), None)
        
        if sakt_result and tsakt_result:
            print(f"\n{'-'*100}")
            print(f"Batch Size: {batch_size}")
            print(f"{'-'*100}")
            
            print(f"{'指标':<30} {'SAKT':<20} {'TSAKT-w/o-Pos':<20} {'差异':<20} {'改进':<15}")
            print("-" * 105)
            
            time_diff = tsakt_result['avg_inference_time'] - sakt_result['avg_inference_time']
            time_improvement = (sakt_result['avg_inference_time'] - tsakt_result['avg_inference_time']) / sakt_result['avg_inference_time'] * 100
            
            print(f"{'平均推理时间 (s)':<30} {sakt_result['avg_inference_time']:<20.4f} {tsakt_result['avg_inference_time']:<20.4f} {time_diff:<20.4f} {time_improvement:<15.2f}%")
            
            mem_diff = tsakt_result['avg_memory_allocated_mb'] - sakt_result['avg_memory_allocated_mb']
            mem_improvement = (sakt_result['avg_memory_allocated_mb'] - tsakt_result['avg_memory_allocated_mb']) / sakt_result['avg_memory_allocated_mb'] * 100
            
            print(f"{'平均显存占用 (MB)':<30} {sakt_result['avg_memory_allocated_mb']:<20.2f} {tsakt_result['avg_memory_allocated_mb']:<20.2f} {mem_diff:<20.2f} {mem_improvement:<15.2f}%")
            
            max_mem_diff = tsakt_result['max_memory_allocated_mb'] - sakt_result['max_memory_allocated_mb']
            max_mem_improvement = (sakt_result['max_memory_allocated_mb'] - tsakt_result['max_memory_allocated_mb']) / sakt_result['max_memory_allocated_mb'] * 100
            
            print(f"{'最大显存占用 (MB)':<30} {sakt_result['max_memory_allocated_mb']:<20.2f} {tsakt_result['max_memory_allocated_mb']:<20.2f} {max_mem_diff:<20.2f} {max_mem_improvement:<15.2f}%")

def main():
    print("=" * 100)
    print("TSAKT-w/o-Pos 运行时显存和推理时间分析")
    print("=" * 100)
    
    datasets = ['assistments09', 'assistments12', 'assistments15']
    batch_sizes = [1, 2, 4, 8]
    
    all_results = {}
    
    for dataset in datasets:
        results = analyze_dataset(dataset, batch_sizes)
        all_results[dataset] = results
        compare_results(results)
    
    print(f"\n{'='*100}")
    print(f"总结")
    print(f"{'='*100}")
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        results = all_results[dataset]
        
        sakt_results = results.get('sakt', [])
        tsakt_results = results.get('tsakt_wo_pos', [])
        
        if sakt_results and tsakt_results:
            avg_sakt_mem = np.mean([r['avg_memory_allocated_mb'] for r in sakt_results])
            avg_tsakt_mem = np.mean([r['avg_memory_allocated_mb'] for r in tsakt_results])
            mem_improvement = (avg_sakt_mem - avg_tsakt_mem) / avg_sakt_mem * 100
            
            avg_sakt_time = np.mean([r['avg_inference_time'] for r in sakt_results])
            avg_tsakt_time = np.mean([r['avg_inference_time'] for r in tsakt_results])
            time_improvement = (avg_sakt_time - avg_tsakt_time) / avg_sakt_time * 100
            
            print(f"  平均显存占用: SAKT {avg_sakt_mem:.2f} MB -> TSAKT-w/o-Pos {avg_tsakt_mem:.2f} MB (改进 {mem_improvement:.2f}%)")
            print(f"  平均推理时间: SAKT {avg_sakt_time:.4f}s -> TSAKT-w/o-Pos {avg_tsakt_time:.4f}s (改进 {time_improvement:.2f}%)")

if __name__ == "__main__":
    main()
