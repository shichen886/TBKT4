#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 Python 环境和 PyTorch 安装情况
"""

import sys
import os

print("=" * 60)
print("📋 环境检查")
print("=" * 60)

# 打印 Python 执行路径
print(f"Python 路径: {sys.executable}")
print(f"Python 版本: {sys.version}")

print("\nPython 搜索路径:")
for path in sys.path:
    print(f"  - {path}")

print("\n检查 PyTorch 安装:")
try:
    import torch
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except ImportError as e:
    print(f"❌ PyTorch 未安装: {e}")
except Exception as e:
    print(f"❌ PyTorch 检查失败: {e}")

print("\n检查其他依赖:")
deps = ['numpy', 'pandas', 'streamlit', 'paddlepaddle', 'paddleocr']
for dep in deps:
    try:
        module = __import__(dep)
        version = getattr(module, '__version__', '未知')
        print(f"✅ {dep}: {version}")
    except ImportError:
        print(f"❌ {dep}: 未安装")

print("\n" + "=" * 60)
print("📊 检查完成")
print("=" * 60)
