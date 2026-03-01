import torch
import gc

print("=" * 60)
print("GPU内存清理工具")
print("=" * 60)

def clear_gpu_memory():
    """清理GPU内存"""
    print("\n正在清理GPU内存...")
    
    # 显示清理前的内存使用情况
    if torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        reserved_before = torch.cuda.memory_reserved() / 1024**3
        print(f"清理前 - 已分配: {allocated_before:.2f} GB, 已保留: {reserved_before:.2f} GB")
    
    # 执行垃圾回收
    gc.collect()
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # 显示清理后的内存使用情况
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        reserved_after = torch.cuda.memory_reserved() / 1024**3
        freed = reserved_before - reserved_after
        print(f"清理后 - 已分配: {allocated_after:.2f} GB, 已保留: {reserved_after:.2f} GB")
        print(f"释放内存: {freed:.2f} GB")
        
        # 显示GPU信息
        print(f"\nGPU信息:")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("未检测到CUDA设备")
    
    print("\n" + "=" * 60)
    print("GPU内存清理完成！")

if __name__ == "__main__":
    clear_gpu_memory()
