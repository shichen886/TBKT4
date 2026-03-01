# TBKT4 项目文件说明

## 📁 核心文件结构

### 🎯 模型文件（4个）
```
model_sakt.py          # SAKT模型
model_akt.py           # AKT模型  
model_dkt1.py          # DKT模型
model_tsakt_linear.py    # TSAKT-Linear模型（最新版本）
```

### 🚀 训练脚本（4个）
```
train_sakt.py          # 训练SAKT
train_akt.py           # 训练AKT
train_dkt1.py         # 训练DKT
train_tsakt_linear_final.py  # 训练TSAKT-Linear（最新版本）
```

### 📊 评估脚本（2个）
```
evaluate_sakt_correct.py      # 评估SAKT模型
compare_sakt_tsakt_linear.py # 对比SAKT vs TSAKT-Linear
```

### 🗑️ 归档文件（archive/）
所有过时、重复、测试的文件都已移动到 `archive/` 目录中。

## 🎯 使用说明

### 训练模型
```bash
# 训练SAKT
python train_sakt.py --dataset assistments12

# 训练AKT
python train_akt.py --dataset assistments12

# 训练DKT
python train_dkt1.py --dataset assistments12

# 训练TSAKT-Linear
python train_tsakt_linear_final.py --dataset assistments12
```

### 评估模型
```bash
# 评估SAKT
python evaluate_sakt_correct.py --dataset assistments12

# 对比SAKT vs TSAKT-Linear
python compare_sakt_tsakt_linear.py --dataset assistments12
```

## 📊 模型性能对比

### SAKT vs TSAKT-Linear

| 数据集 | SAKT AUC | TSAKT-Linear AUC | 改进 |
|--------|-----------|-------------------|-------|
| assistments09 | 0.7520 | 0.6842 | -9.0% |
| assistments12 | 0.7604 | 0.7231 | -4.9% |

**注意**: TSAKT-Linear在性能上略低于SAKT，但实现了内存节省。

## 📁 目录结构
```
TBKT4/
├── model_*.py              # 模型定义
├── train_*.py             # 训练脚本
├── evaluate_*.py           # 评估脚本
├── compare_*.py           # 对比脚本
├── web/                  # Web相关文件（12个）
├── archive/               # 归档文件（94个文件）
├── data/                 # 数据集目录
├── save/                 # 保存的模型
└── FILE_ORGANIZATION.md   # 文件整理说明
```

## 💡 重要提示

1. **核心文件**: 只有10个核心文件在根目录
2. **归档文件**: 94个过时文件已移至archive/
3. **Web文件**: 12个Web相关文件已移至web/
4. **模型参数**: 不同模型使用不同的参数配置
5. **性能对比**: TSAKT-Linear实现了内存节省，但性能略低于SAKT

## 🚀 快速开始

1. 选择要训练的模型
2. 运行对应的训练脚本
3. 使用评估脚本检查性能
4. 使用对比脚本进行模型比较

## 📝 注意事项

- 所有训练好的模型保存在 `save/` 目录
- 数据集位于 `data/` 目录
- 归档文件保留在 `archive/` 目录以防需要
- Web文件存放在 `web/` 目录中
- 如需添加新模型，请遵循现有命名规范

## 📁 使用归档文件

**从archive中使用文件的方法**:  
如果你需要使用之前的文件，只需要从 `archive/` 目录中复制或移动文件到根目录即可。

**示例**: 使用旧版本的TSAKT模型
```bash
# 复制旧版本模型到根目录
copy archive\model_tsakt_v2.py .

# 然后运行
python model_tsakt_v2.py
```

## 🌐 使用Web文件

**Web文件使用方法**:  
Web相关文件已整理到 `web/` 目录中，包括：

- `app.py` - Web应用入口
- `views.py` - 视图函数
- `urls.py` - 路由配置
- `config.py` - 配置文件
- `recommendation.py` - 推荐系统
- 其他Web相关组件

**运行Web应用**:
```bash
# 进入web目录
cd web

# 运行Web应用
python app.py
```
