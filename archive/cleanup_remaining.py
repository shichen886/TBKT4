import os
import shutil

remaining_files_to_move = [
    # 临时/测试脚本
    'cleanup_files.py',
    'evaluate_short_baseline_fixed.py',
    'train_short_sequence_test.py',
    'ablation_study.py',
    'evaluate_tsakt_variants.py',
    'compare_models.py',
    'train_dkt2.py',
]

archive_dir = 'archive'
if not os.path.exists(archive_dir):
    os.makedirs(archive_dir)

moved_count = 0
for filename in remaining_files_to_move:
    if os.path.exists(filename):
        try:
            shutil.move(filename, os.path.join(archive_dir, filename))
            print(f"✅ 移动: {filename}")
            moved_count += 1
        except Exception as e:
            print(f"❌ 错误移动 {filename}: {e}")

print(f"\n总计移动了 {moved_count} 个文件到 {archive_dir}/")
