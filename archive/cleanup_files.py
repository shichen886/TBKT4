import os
import shutil

files_to_move = [
    # TSAKT旧版本模型
    'model_tsakt_v3.py',
    'model_tsakt_v4.py',
    'model_tsakt_v5.py',
    'model_tsakt_v6.py',
    'model_tsakt_v7.py',
    'model_tsakt_v8.py',
    'model_tsakt_v9.py',
    'model_tsakt_v10.py',
    'model_tsakt_v11.py',
    'model_tsakt_v12.py',
    'model_tsakt_v13.py',
    'model_tsakt_v14.py',
    'model_tsakt.py',
    
    # 其他过时模型
    'model_ffw.py',
    'model_dkt2.py',
    
    # TSAKT旧版本训练脚本
    'train_tsakt.py',
    'train_tsakt_v2.py',
    'train_tsakt_ful.py',
    'train_tsakt_variants.py',
    'train_tsakt_linear.py',
    'train_tsakt_linear_simple.py',
    
    # TSAKT旧版本测试脚本
    'test_tsakt.py',
    'test_tsakt_single.py',
    'test_tsakt_variants.py',
    'test_tsakt_final.py',
    'test_tsakt_v3.py',
    'test_tsakt_ful.py',
    'test_tsakt_linear.py',
    'test_position_encoding.py',
    
    # 重复/过时的评估脚本
    'evaluate_short_baseline.py',
    'evaluate_short_baseline_v2.py',
    'evaluate_short_baseline_v3.py',
    'evaluate_short_baseline_v4.py',
    'evaluate_time_baseline.py',
    'evaluate_time_baseline_v2.py',
    'evaluate_time_baseline_fixed.py',
    'evaluate_normal_baseline.py',
    'evaluate_ablation.py',
    'evaluate_all_models.py',
    'evaluate_models.py',
    'evaluate_baselines.py',
    'evaluate_time_dependent.py',
    'evaluate_short_sequence.py',
    'compare_sakt_tsakt.py',
    'compare_tsakt_models.py',
    
    # 重复/过时的训练脚本
    'train_all_models.py',
    'train_all_tsakt.py',
    'train_baselines_short.py',
    'train_baselines_time.py',
    'train_short_sequence_models.py',
    'train_time_dependent_test.py',
    'train_ffw.py',
    'train_lr.py',
    'train_baselines_time_parallel.py',
    
    # 检查/分析脚本
    'check_sakt_models.py',
    'check_training_result.py',
    'check_assistments09_result.py',
    'check_training_epochs.py',
    'check_training.py',
    'check_training_simple.py',
    'check_model_weights.py',
    'check_cuda.py',
    'check_env.py',
    'analyze_training_sufficiency.py',
    'analyze_model_params.py',
    'analyze_dataset_characteristics.py',
    'analyze_datasets.py',
    'analyze_experiments.py',
    
    # 数据集创建脚本
    'create_short_sequence_datasets.py',
    'create_time_dependent_datasets.py',
    'create_preprocessed_files.py',
    
    # 临时/测试脚本
    'clear_gpu_memory.py',
    'diagnose_dkt_memory.py',
    'fix_auc.py',
    'measure_inference_metrics.py',
    'retrain_tsakt_ful.py',
    'integrate_results.py',
    'run_experiments.py',
    'learning_path.py',
    'generate_heatmap.py',
    'draw_heatmap.py',
    'draw_test.py',
    'model_sakt_draw.py',
    
    # Web/应用相关
    'app.py',
    'views.py',
    'urls.py',
    'config.py',
    'user_models.py',
    'manage.py',
    'wsgi.py',
    'chart_config.py',
    'launcher.py',
    'upload_data.py',
    'recommendation.py',
    'name_mappings.py',
    
    # 其他测试脚本
    'test.py',
    'test_ocr.py',
    'test_ocr_fix.py',
    'test_paddle.py',
    'cleanup_paddle.py',
    'prepare_data.py',
    'encode.py',
]

archive_dir = 'archive'
if not os.path.exists(archive_dir):
    os.makedirs(archive_dir)

moved_count = 0
for filename in files_to_move:
    if os.path.exists(filename):
        try:
            shutil.move(filename, os.path.join(archive_dir, filename))
            print(f"✅ 移动: {filename}")
            moved_count += 1
        except Exception as e:
            print(f"❌ 错误移动 {filename}: {e}")

print(f"\n总计移动了 {moved_count} 个文件到 {archive_dir}/")
