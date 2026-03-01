# 论文级修复总结

## 修复的4个关键问题

### 1️⃣ AUC计算方式错误（最严重）
**问题**：使用batch-wise AUC average
```python
# 错误做法
total_auc += auc
total_count += 1
return total_auc / total_count
```

**修复**：全局AUC计算（收集所有预测后统一计算）
```python
# 正确做法
all_preds = []
all_labels = []

for batch in batches:
    all_preds.append(preds[valid_mask].cpu())
    all_labels.append(labels[valid_mask].cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
auc = roc_auc_score(all_labels, all_preds)
```

**影响**：
- AUC不是线性可加指标
- batch-wise average会导致系统性偏差
- 论文会被审稿人直接抓住

---

### 2️⃣ 数据切分存在潜在泄漏风险
**问题**：先chunk再split，同一用户的不同chunk可能同时出现在train和val
```python
# 错误做法
item_ids = [torch.tensor(u_df["item_id"].values) for _, u_df in df.groupby("user_id")]
chunked_lists = [chunk(l) for l in lists]
data = list(zip(*chunked_lists))
train_data, val_data = data[:train_size], data[train_size:]  # 同一用户的chunk可能泄露
```

**修复**：先按user_id划分train/val，再做chunk
```python
# 正确做法
user_ids = df["user_id"].unique()
np.random.shuffle(user_ids)
train_user_ids = user_ids[:train_size]
val_user_ids = user_ids[train_size:]

# 分别提取train和val用户的序列
for user_id in train_user_ids:
    # 提取train序列
for user_id in val_user_ids:
    # 提取val序列

# 然后chunk
train_item_ids = chunk_list(train_item_ids)
val_item_ids = chunk_list(val_item_ids)
```

**影响**：
- 用户历史强相关
- 会导致AUC虚高2-5%
- 论文会被质疑

---

### 3️⃣ 没有固定随机种子
**问题**：完全随机，实验不可复现
```python
# 错误做法
shuffle(data)  # 每次运行结果不同
```

**修复**：固定所有随机种子
```python
# 正确做法
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**影响**：
- 实验不可复现
- 论文直接降级
- 审稿人会质疑

---

### 4️⃣ Early stopping基于loss而不是AUC
**问题**：监控loss，但最终比较的是AUC
```python
# 不够科研的做法
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')  # 监控loss
if val_loss < best_val_loss:  # 基于loss保存模型
```

**修复**：基于AUC进行early stopping
```python
# 更符合KT论文标准
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')  # 监控AUC
scheduler.step(val_auc)  # 基于AUC调整学习率
if val_auc > best_val_auc:  # 基于AUC保存模型
```

**影响**：
- 不改也能发普刊
- 改了更像正式研究
- 更符合KT论文标准

---

## 已修复的文件

✅ train_tsakt_linear_final.py
✅ train_tsakt_linear_fixed.py
⏳ train_tsakt_linear_nopos.py
⏳ train_tsakt_linear_rope.py
⏳ train_tsakt_linear_gate.py

---

## 关键修改点

### 1. 添加随机种子固定
```python
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 2. 修改数据切分方式
```python
def get_data(df, max_length, train_split=0.8, randomize=True, seed=42):
    # 先按user_id划分
    user_ids = df["user_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(user_ids)
    train_user_ids = user_ids[:train_size]
    val_user_ids = user_ids[train_size:]
    
    # 分别提取序列
    # 然后chunk
```

### 3. 修改验证函数
```python
def validate(model, batches, criterion):
    all_preds = []
    all_labels = []
    
    for batch in batches:
        # 收集预测
        all_preds.append(preds[valid_mask].cpu())
        all_labels.append(labels[valid_mask].cpu())
    
    # 全局计算
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(all_labels, all_preds)
```

### 4. 修改early stopping
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
scheduler.step(val_auc)
if val_auc > best_val_auc:
    # 保存模型
```

### 5. 添加seed参数
```python
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
```

---

## 预期影响

### AUC变化
- 全局AUC计算可能导致AUC略有变化（±1-2%）
- 数据切分修复可能导致AUC下降（2-5%）
- 整体趋势应该保持一致

### 可复现性
- 固定种子后，相同参数应该得到完全相同的结果
- 多次运行结果应该一致

### 论文质量
- 修复后符合论文级标准
- 审稿人不会质疑实验设置
- 实验更加严谨