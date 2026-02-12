#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 PaddlePaddle 安装
"""

try:
    import paddle
    print(f"✅ PaddlePaddle 版本: {paddle.__version__}")
    print(f"✅ GPU 可用: {paddle.device.is_compiled_with_cuda()}")
    if paddle.device.is_compiled_with_cuda():
        print(f"✅ CUDA 版本: {paddle.device.cuda_device_count()} 个设备")
except ImportError as e:
    print(f"❌ PaddlePaddle 导入失败: {e}")
except Exception as e:
    print(f"❌ PaddlePaddle 检查失败: {e}")