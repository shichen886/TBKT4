# TSAKT-Linear vs SAKT 对比结果（相同参数）

## 📊 **实验设置**

### **模型参数配置**
- **embed_size**: 40 (assistments09/12), 80 (assistments15)
- **num_layers**: 2
- **num_heads**: 5
- **tensor_rank**: 32
- **batch_size**: 32
- **max_seq_len**: 200
- **drop_prob**: 0.1

### **训练配置**
- **学习率**: 0.0001
- **权重衰减**: 0.0001
- **训练轮数**: 50 epochs
- **优化器**: Adam

---

## 📈 **对比结果**

### **1. assistments09 数据集**

| 指标 | SAKT | TSAKT-Linear | 改进 |
|------|------|--------------|------|
| **AUC** | 0.7472 | 0.7508 | **+0.48%** ✅ |
| **RMSE** | 0.4189 | 0.4172 | **+0.40%** ✅ |
| **推理时间(ms)** | 3.93 | 2.47 | **-36.98%** ✅ |
| **显存使用(MB)** | 312.90 | 288.29 | **-7.87%** ✅ |
| **参数量** | 1,076,031 | 2,176,553 | +102.28% |

**结论**: ✅ TSAKT-Linear在保持性能的同时，实现了内存节省

---

### **2. assistments12 数据集**

| 指标 | SAKT | TSAKT-Linear | 改进 |
|------|------|--------------|------|
| **AUC** | 0.7603 | 0.6769 | **-10.97%** ❌ |
| **RMSE** | 0.4411 | 0.4702 | **-6.60%** ❌ |
| **推理时间(ms)** | 8.74 | 3.46 | **-60.42%** ✅ |
| **显存使用(MB)** | 278.76 | 256.44 | **-8.01%** ✅ |
| **参数量** | 85,191 | 194,873 | +128.75% |

**结论**: ⚠️ TSAKT-Linear性能下降较多，可能需要调整tensor_rank参数

---

### **3. assistments15 数据集**

**状态**: ❌ 无法对比
- **原因**: SAKT模型参数与当前数据集不匹配
- **SAKT模型**: 173,114 items, 272 skills
- **当前数据集**: 1,223 items, 98 skills
- **TSAKT-Linear**: 已训练完成 (AUC: 0.7152)

---

## 🎯 **总体分析**

### **性能表现**
1. **assistments09**: TSAKT-Linear性能略优于SAKT
2. **assistments12**: TSAKT-Linear性能下降较多
3. **assistments15**: 无法对比

### **内存节省**
- **assistments09**: 节省 7.87% 显存
- **assistments12**: 节省 8.01% 显存
- **平均节省**: ~8% 显存

### **推理速度**
- **assistments09**: 提升 36.98%
- **assistments12**: 提升 60.42%
- **平均提升**: ~48% 推理速度

### **参数量**
- TSAKT-Linear参数量约为SAKT的2倍
- 主要由于张量分解的特征映射层

---

## 💡 **关键发现**

### **1. 数据集差异影响**
- **assistments09**: 数据集较大（52,850 items），TSAKT-Linear表现良好
- **assistments12**: 数据集较小（3,162 items），TSAKT-Linear性能下降
- **可能原因**: tensor_rank=32对于小数据集可能过大

### **2. 内存节省效果**
- 在所有数据集上都实现了8%左右的显存节省
- 推理速度提升显著（36%-60%）

### **3. 参数量权衡**
- TSAKT-Linear参数量增加约100%
- 但实际运行时内存使用减少
- 说明张量分解有效降低了运行时内存占用

---

## 🔧 **改进建议**

### **1. 调整tensor_rank参数**
```python
# 对于小数据集，降低tensor_rank
tensor_rank = 16  # 而不是32
```

### **2. 数据集自适应参数**
```python
# 根据数据集大小动态调整tensor_rank
if num_items < 5000:
    tensor_rank = 16
elif num_items < 20000:
    tensor_rank = 24
else:
    tensor_rank = 32
```

### **3. 重新训练assistments15的SAKT模型**
```bash
python train_sakt.py --dataset assistments15 --embed_size 80 --num_heads 5
```

---

## 📝 **结论**

### **成功之处**
1. ✅ 实现了8%的显存节省
2. ✅ 推理速度提升48%
3. ✅ 在大数据集上性能保持或略优于SAKT

### **待改进之处**
1. ❌ 在小数据集上性能下降较多
2. ❌ 参数量增加约100%
3. ❌ 需要针对不同数据集调整tensor_rank

### **下一步行动**
1. 调整tensor_rank参数重新训练
2. 重新训练assistments15的SAKT模型
3. 进行更全面的消融实验

---

## 📂 **相关文件**

### **训练脚本**
- `train_tsakt_linear_sakt_params.py` - 使用SAKT参数训练TSAKT-Linear

### **对比脚本**
- `compare_sakt_tsakt_same_params.py` - 对比SAKT vs TSAKT-Linear

### **模型文件**
- `save/tsakt-linear-sakt-params/` - TSAKT-Linear模型
- `save/sakt/` - SAKT模型

### **训练命令**
```bash
# 训练TSAKT-Linear（相同参数）
python train_tsakt_linear_sakt_params.py --dataset assistments12 --embed_size 40 --num_heads 5 --batch_size 32

# 对比模型
python compare_sakt_tsakt_same_params.py --dataset assistments12 --batch_size 32
```
