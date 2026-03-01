# 论文级训练功能说明

## ✨ 核心功能

### 1. 自动保存 config.json
- 位置：`save/{model_name}/config.json`
- 内容：完整的训练参数配置
- 作用：确保实验完全可复现

### 2. 自动记录训练历史
- 位置：`save/{model_name}/{dataset}_training_history.json`
- 内容：每个epoch的训练和验证指标
- 包含：train_loss, train_auc, train_rmse, val_loss, val_auc, val_rmse
- 自动记录：best_val_loss, best_val_auc

### 3. 自动绘制训练曲线
- 位置：`save/{model_name}/{dataset}_training_curves.png`
- 包含3个子图：
  - Loss曲线（训练和验证）
  - AUC曲线（训练和验证）
  - RMSE曲线（训练和验证）
- 高分辨率：300 DPI

### 4. 自动保存最佳模型
- 位置：`save/{model_name}/{dataset}_best.pt`
- 触发条件：验证损失最低时
- 包含：模型权重、优化器状态、训练指标

---

## 📋 使用示例

### 训练TSAKT-Linear（SAKT参数）
```bash
python train_tsakt_linear_sakt_params.py \
    --dataset assistments12 \
    --embed_size 40 \
    --num_heads 5 \
    --batch_size 32 \
    --num_epochs 50
```

### 训练TSAKT-Linear（原始参数）
```bash
python train_tsakt_linear_final.py \
    --dataset assistments12 \
    --batch_size 32 \
    --num_epochs 50
```

### 训练SAKT
```bash
python train_sakt.py \
    --dataset assistments12 \
    --batch_size 32 \
    --num_epochs 50
```

---

## 📁 生成的文件结构

```
save/tsakt-linear-sakt-params/
├── config.json                          # 训练配置
├── assistments09_best.pt               # 最佳模型（assistments09）
├── assistments09_training_history.json # 训练历史
├── assistments09_training_curves.png   # 训练曲线
├── assistments12_best.pt               # 最佳模型（assistments12）
├── assistments12_training_history.json # 训练历史
├── assistments12_training_curves.png   # 训练曲线
├── assistments15_best.pt               # 最佳模型（assistments15）
├── assistments15_training_history.json # 训练历史
└── assistments15_training_curves.png   # 训练曲线
```

---

## 🔍 查看训练结果

### 1. 查看配置参数
```bash
cat save/tsakt-linear-sakt-params/config.json
```

### 2. 查看训练历史
```bash
cat save/tsakt-linear-sakt-params/assistments12_training_history.json
```

### 3. 查看最佳指标
```bash
python -c "import json; h=json.load(open('save/tsakt-linear-sakt-params/assistments12_training_history.json')); print(f\"Best AUC: {h['best_val_auc']:.4f}, Best Loss: {h['best_val_loss']:.4f}\")"
```

---

## 📊 训练历史JSON格式

```json
{
    "train_loss": [0.7110, 0.6900, ...],
    "train_auc": [0.5087, 0.5173, ...],
    "train_rmse": [0.5074, 0.4982, ...],
    "val_loss": [0.6973, 0.6884, ...],
    "val_auc": [0.5134, 0.5220, ...],
    "val_rmse": [0.5016, 0.4975, ...],
    "epochs": [1, 2, ...],
    "best_val_loss": 0.6884,
    "best_val_auc": 0.5220
}
```

---

## 🎓 论文中的使用方式

### 1. 引用配置文件
```
我们的模型使用以下配置进行训练：
- 嵌入维度：40
- 注意力头数：5
- 张量秩：32
- 批次大小：32
- 学习率：0.0001
```

### 2. 引用训练曲线
```
图1显示了训练和验证损失随epoch的变化。可以看到模型在第X轮达到最佳性能。
```

### 3. 引用最佳指标
```
在assistments12数据集上，我们的模型达到了0.7220的AUC和0.4475的RMSE。
```

---

## 🚀 一键复现实验

### 完全复现之前的实验
```bash
# 1. 查看配置
cat save/tsakt-linear-sakt-params/config.json

# 2. 使用相同参数重新训练
python train_tsakt_linear_sakt_params.py \
    --dataset assistments12 \
    --embed_size 40 \
    --num_layers 2 \
    --num_heads 5 \
    --tensor_rank 32 \
    --max_seq_len 200 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --drop_prob 0.1
```

---

## ⚠️ 注意事项

### Matplotlib依赖
- 如果matplotlib不可用，训练曲线图将不会生成
- 其他功能（config.json、training_history.json）不受影响
- 安装matplotlib：`pip install matplotlib`

### 文件覆盖
- 每次训练会覆盖之前的同名文件
- 建议为不同实验使用不同的savedir
- 例如：`--savedir save/tsakt-linear-sakt-params-experiment1`

---

## 📈 高级功能（可选）

### 自定义实验文件夹
```bash
python train_tsakt_linear_sakt_params.py \
    --dataset assistments12 \
    --savedir save/experiments/run_001 \
    --num_epochs 50
```

### 比较不同实验
```bash
# 比较实验1和实验2的最佳AUC
python -c "
import json
h1 = json.load(open('save/experiments/run_001/assistments12_training_history.json'))
h2 = json.load(open('save/experiments/run_002/assistments12_training_history.json'))
print(f'Experiment 1: AUC={h1[\"best_val_auc\"]:.4f}')
print(f'Experiment 2: AUC={h2[\"best_val_auc\"]:.4f}')
"
```

---

## 🎯 总结

**这4个功能让您的项目达到论文级标准：**

1. ✅ **自动保存 config.json** - 完全可复现
2. ✅ **自动记录训练历史** - 完整追踪
3. ✅ **自动绘制训练曲线** - 可视化分析
4. ✅ **自动保存最佳模型** - 性能保证

**论文评审最怕的问题解决了：**
- ❌ "你这个结果到底用的什么参数？" → ✅ config.json
- ❌ "你的训练过程如何？" → ✅ training_history.json
- ❌ "你的模型收敛了吗？" → ✅ training_curves.png
- ❌ "这个实验能复现吗？" → ✅ 一键复现

**老师直接加印象分！** 🎓