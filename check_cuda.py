#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 CUDA 版本
"""

import torch

print("=" * 60)
print("📋 CUDA 版本检查")
print("=" * 60)

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 获取 CUDA 版本号
    cuda_version = torch.version.cuda
    print(f"\n💡 您的 CUDA 版本: {cuda_version}")
    
    # 根据 CUDA 版本推荐 PaddlePaddle 版本
    if "11.8" in cuda_version:
        print("✅ 推荐 PaddlePaddle GPU 版本: CUDA 11.8")
        print("   安装命令: pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
    elif "12.6" in cuda_version:
        print("✅ 推荐 PaddlePaddle GPU 版本: CUDA 12.6")
        print("   安装命令: pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
    else:
        print(f"⚠️ 未知 CUDA 版本: {cuda_version}")
        print("   建议安装 CPU 版本的 PaddlePaddle")
        print("   安装命令: pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/")
else:
    print("❌ CUDA 不可用")
    print("   建议安装 CPU 版本的 PaddlePaddle")
    print("   安装命令: pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/")

print("\n" + "=" * 60)
print("📊 检查完成")
print("=" * 60)