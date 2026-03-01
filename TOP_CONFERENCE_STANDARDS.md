# 顶级会议标准实现指南

## 概述

本文档说明如何使用顶级会议标准（train/val/test三划分、多seed平均、显著性检验）进行模型训练和评估。

---

## 1. train/val/test三划分

### 修改内容

已修改 `train_tsakt_linear_final.py`，支持三划分：

- **Train**: 80% 用户
- **Val**: 10% 用户
- **Test**: 10% 用户

### 使用方法

```bash
python train_tsakt_linear_final.py \
    --dataset assistments12 \
    --savedir save/tsakt-linear \
    --embed_size 64 \
    --num_layers 2 \
    --num_heads 4 \
    --tensor_rank 32 \
    --max_seq_len 200 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --drop_prob 0.1 \
    --seed 42
```

### 输出

训练完成后会自动在test set上评估：

```
================================================================================
EVALUATING ON TEST SET
================================================================================

Loaded best model from save/tsakt-linear/assistments12_best.pt

Test Set Results:
  Test Loss: 0.5234
  Test AUC: 0.7856
  Test RMSE: 0.4123

================================================================================
TRAINING COMPLETED
================================================================================

Final Results:
  Best Val AUC: 0.7912 (epoch 15)
  Test AUC: 0.7856
  Generalization Gap: 0.0056
```

### 保存的文件

- `config.json` - 包含test结果
- `{dataset}_training_history.json` - 训练历史
- `{dataset}_best.pt` - 最佳模型
- `{dataset}_training_curves.png` - 训练曲线

---

## 2. 多seed训练

### 使用方法

```bash
python train_tsakt_linear_multi_seed.py \
    --dataset assistments12 \
    --seeds 42,123,456,789,1011 \
    --savedir save/tsakt-linear-multi-seed \
    --embed_size 64 \
    --num_layers 2 \
    --num_heads 4 \
    --tensor_rank 32 \
    --max_seq_len 200 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --drop_prob 0.1
```

### 参数说明

- `--seeds`: 逗号分隔的种子列表（如 "42,123,456"）
- `--savedir`: 基础保存目录
- 其他参数与单seed训练相同

### 输出

```
================================================================================
MULTI-SEED TRAINING RESULTS
================================================================================

Dataset: assistments12
Number of seeds: 5

AUC Results:
  Mean: 0.7892 ± 0.0034
  Min:  0.7856
  Max:  0.7934

All AUCs: ['0.7856', '0.7912', '0.7890', '0.7934', '0.7868']

RMSE Results:
  Mean: 0.4123 ± 0.0021

Generalization Gap:
  Mean: 0.0056 ± 0.0012

================================================================================
```

### 保存的文件

- `save/tsakt-linear-multi-seed/seed_42/` - seed=42的结果
- `save/tsakt-linear-multi-seed/seed_123/` - seed=123的结果
- ...
- `save/tsakt-linear-multi-seed/assistments12_multi_seed_stats.json` - 统计汇总

### stats.json内容

```json
{
    "num_seeds": 5,
    "auc_mean": 0.7892,
    "auc_std": 0.0034,
    "auc_min": 0.7856,
    "auc_max": 0.7934,
    "rmse_mean": 0.4123,
    "rmse_std": 0.0021,
    "gap_mean": 0.0056,
    "gap_std": 0.0012,
    "all_aucs": [0.7856, 0.7912, 0.7890, 0.7934, 0.7868],
    "all_rmses": [0.4156, 0.4102, 0.4123, 0.4098, 0.4136],
    "all_gaps": [0.0056, 0.0048, 0.0062, 0.0051, 0.0063]
}
```

---

## 3. 显著性检验

### 使用方法

```bash
python significance_test.py \
    --model1_results save/tsakt-linear-multi-seed/assistments12_multi_seed_stats.json \
    --model2_results save/sakt-multi-seed/assistments12_multi_seed_stats.json \
    --model1_name "TSAKT-Linear" \
    --model2_name "SAKT" \
    --alpha 0.05 \
    --output save/comparison_results.json
```

### 参数说明

- `--model1_results`: 模型1的stats.json路径
- `--model2_results`: 模型2的stats.json路径
- `--model1_name`: 模型1名称
- `--model2_name`: 模型2名称
- `--alpha`: 显著性水平（默认0.05）
- `--output`: 输出JSON文件路径

### 输出

```
================================================================================
SIGNIFICANCE TEST: TSAKT-Linear vs SAKT
================================================================================

AUC Statistics:
  TSAKT-Linear:
    Mean: 0.7892 ± 0.0034
  SAKT:
    Mean: 0.7812 ± 0.0041
  Difference: +0.0080

Paired t-test:
  t-statistic: 3.4567
  p-value: 0.0256
  Significant: Yes
  95% CI: [0.0012, 0.0148]
  Cohen's d: 2.1234 (large)

Bootstrap test:
  p-value: 0.0234
  Significant: Yes
  95% CI: [0.0015, 0.0145]
  Cohen's d: 2.1234 (large)
  Bootstrap samples: 10000

Conclusion:
  Significant improvement (p<0.05, large effect size)

================================================================================
```

### 检验方法

- **配对t检验（Paired t-test）**:
  - 适用于同一数据集上的多次运行
  - 提供t统计量、p值、95%置信区间

- **Bootstrap检验**:
  - 非参数方法，不假设正态分布
  - 通过重采样估计置信区间
  - 默认10000次采样

- **Cohen's d（效应量）**:
  - 量化改进幅度
  - 解释：negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)

### 保存的文件

- `comparison_results.json` - 完整的比较结果

---

## 4. 完整工作流程

### 步骤1：训练模型1（多seed）

```bash
python train_tsakt_linear_multi_seed.py \
    --dataset assistments12 \
    --seeds 42,123,456,789,1011 \
    --savedir save/tsakt-linear-multi-seed \
    --embed_size 64 \
    --num_layers 2 \
    --num_heads 4 \
    --tensor_rank 32 \
    --max_seq_len 200 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --drop_prob 0.1
```

### 步骤2：训练模型2（多seed）

```bash
python train_sakt_multi_seed.py \
    --dataset assistments12 \
    --seeds 42,123,456,789,1011 \
    --savedir save/sakt-multi-seed \
    --embed_size 64 \
    --num_layers 2 \
    --num_heads 4 \
    --max_seq_len 200 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --drop_prob 0.1
```

### 步骤3：显著性检验

```bash
python significance_test.py \
    --model1_results save/tsakt-linear-multi-seed/assistments12_multi_seed_stats.json \
    --model2_results save/sakt-multi-seed/assistments12_multi_seed_stats.json \
    --model1_name "TSAKT-Linear" \
    --model2_name "SAKT" \
    --output save/comparison_results.json
```

---

## 5. 论文中的报告方式

### 示例1：单模型结果

```
We evaluated TSAKT-Linear on three datasets using train/val/test splits
(80%/10%/10%). The model was trained with early stopping (patience=10)
based on validation AUC.

Results on assistments12:
- Validation AUC: 0.7912 ± 0.0034 (5 seeds)
- Test AUC: 0.7856 ± 0.0034 (5 seeds)
- Generalization Gap: 0.0056 ± 0.0012
```

### 示例2：模型比较

```
We compared TSAKT-Linear with SAKT using paired t-tests on 5 random seeds.

On assistments12:
- TSAKT-Linear: 0.7892 ± 0.0034
- SAKT: 0.7812 ± 0.0041
- Improvement: +0.0080 (p=0.0256, Cohen's d=2.12)

The improvement is statistically significant (p<0.05) with a large effect size.
```

---

## 6. 常见问题

### Q1: 为什么需要test set？

A: Val用于early stopping和模型选择，Test用于最终评估，避免过拟合到val set。

### Q2: 多seed需要多少个？

A: 顶级会议通常要求3-5个seed。更多seed更稳健，但训练时间更长。

### Q3: 什么时候需要显著性检验？

A: 当你声称SOTA或明显优于baseline时，需要提供统计显著性证据。

### Q4: p<0.05和效应量哪个更重要？

A: 两者都重要。p<0.05说明不是偶然，效应量说明改进幅度。

---

## 7. 文件清单

### 修改的文件

- `train_tsakt_linear_final.py` - 支持train/val/test三划分

### 新增的文件

- `train_tsakt_linear_multi_seed.py` - 多seed训练脚本
- `significance_test.py` - 显著性检验脚本
- `TOP_CONFERENCE_STANDARDS.md` - 本文档

---

## 8. 下一步

### 需要修改的其他训练脚本

为了保持一致性，建议将相同的修改应用到：

- `train_tsakt_linear_nopos.py`
- `train_tsakt_linear_rope.py`
- `train_tsakt_linear_gate.py`
- `train_sakt.py`

### 可选的增强功能

- 添加学习率调度器对比
- 添加更多评估指标（Accuracy, F1-score）
- 添加模型复杂度分析（参数量、FLOPs）