#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除损坏的 PaddlePaddle 安装目录
"""

import shutil
import os

site_packages = 'C:\\Users\\32880\\miniconda3\\lib\\site-packages'

# 需要删除的损坏目录
bad_dirs = [
    '~addle',
    '~addlepaddle_gpu-2.6.1.dist-info',
    '~addlepaddle_gpu-2.6.2.dist-info'
]

for bad_dir in bad_dirs:
    bad_path = os.path.join(site_packages, bad_dir)
    if os.path.exists(bad_path):
        try:
            shutil.rmtree(bad_path)
            print(f"✅ 已删除: {bad_dir}")
        except Exception as e:
            print(f"❌ 删除失败: {bad_dir} - {e}")
    else:
        print(f"⚠️  不存在: {bad_dir}")

print("\n✅ 清理完成！")