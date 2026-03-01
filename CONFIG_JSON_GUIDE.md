# 论文级项目标准：config.json 配置文件

## 🎯 **为什么需要 config.json？**

### **论文评审最怕的问题**
"你这个结果到底用的什么参数？"

### **有了 config.json 的好处**
- ✅ **实验完全可复现**
- ✅ **论文显得非常专业**
- ✅ **老师直接加印象分**

---

## 📋 **查看参数的三种层级**

### **1️⃣ 最快：看终端打印的 args.xxx**
```bash
$ python train_tsakt_linear_sakt_params.py --dataset assistments12
Parameters: embed_size=40, num_layers=2, num_heads=5, tensor_rank=32
```

### **2️⃣ 知道来源：看 argparse default**
```python
parser.add_argument('--embed_size', type=int, default=40)
parser.add_argument('--num_heads', type=int, default=5)
```

### **3️⃣ 论文级正确做法：把 args 保存成 config.json**
```python
import json

config_path = os.path.join(args.savedir, "config.json")
with open(config_path, "w") as f:
    json.dump(vars(args), f, indent=4)
```

---

## 📂 **生成的文件结构**

```
save/tsakt-linear-sakt-params/
├── config.json          # ⭐ 论文级配置文件
├── assistments09_best.pt
├── assistments12_best.pt
└── assistments15_best.pt
```

---

## 📄 **config.json 内容示例**

```json
{
    "dataset": "assistments12",
    "savedir": "save/tsakt-linear-sakt-params",
    "embed_size": 40,
    "num_layers": 2,
    "num_heads": 5,
    "tensor_rank": 32,
    "max_seq_len": 200,
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "drop_prob": 0.1
}
```

---

## 🔧 **已更新的训练脚本**

### **✅ 已添加 config.json 功能**
1. `train_tsakt_linear_sakt_params.py` - TSAKT-Linear（SAKT参数）
2. `train_tsakt_linear_final.py` - TSAKT-Linear（原始参数）
3. `train_sakt.py` - SAKT

### **📝 使用方法**
```bash
# 训练时自动生成 config.json
python train_tsakt_linear_sakt_params.py --dataset assistments12 --embed_size 40

# 查看配置文件
cat save/tsakt-linear-sakt-params/config.json
```

---

## 🚀 **科研里真正严谨的做法（进阶）**

### **每次训练自动建文件夹**
```
save/
├── run_001/
│    ├── model.pt
│    ├── config.json
│    └── log.txt
├── run_002/
│    ├── model.pt
│    ├── config.json
│    └── log.txt
└── run_003/
     ├── model.pt
     ├── config.json
     └── log.txt
```

### **优势**
- 每一次实验都是独立可追溯的
- 这是论文级项目的标准结构
- 便于对比不同实验结果

---

## 📊 **论文中的使用方式**

### **在论文中引用**
```
我们使用以下超参数配置：
- 嵌入维度：40
- 注意力头数：5
- 张量分解秩：32
- 批次大小：32
- 学习率：0.0001

详细的配置参数见附录中的 config.json 文件。
```

### **在附录中提供**
```
附录 A：模型配置

所有实验的详细配置参数保存在：
https://github.com/your-repo/configs/assistments12_config.json
```

---

## 💡 **最佳实践建议**

### **1. 始终保存配置**
```python
# 在训练开始时立即保存
os.makedirs(args.savedir, exist_ok=True)
config_path = os.path.join(args.savedir, "config.json")
with open(config_path, "w") as f:
    json.dump(vars(args), f, indent=4)
```

### **2. 配置文件命名规范**
```python
# 按数据集命名
config_path = os.path.join(args.savedir, f"{args.dataset}_config.json")

# 或者按实验编号命名
config_path = os.path.join(args.savedir, f"run_{experiment_id}_config.json")
```

### **3. 包含所有重要参数**
```python
# 确保包含所有影响结果的参数
important_params = {
    'dataset': args.dataset,
    'model': 'TSAKT-Linear',
    'embed_size': args.embed_size,
    'num_layers': args.num_layers,
    'num_heads': args.num_heads,
    'tensor_rank': args.tensor_rank,
    'batch_size': args.batch_size,
    'learning_rate': args.lr,
    'epochs': args.num_epochs,
    'seed': args.seed,  # 如果有随机种子
}
```

---

## 🎓 **为什么这是论文级标准？**

### **1. 可复现性**
- 其他研究者可以完全复现你的实验
- 避免了"参数不一致"的质疑

### **2. 透明度**
- 审稿人可以清楚了解实验设置
- 增加了论文的可信度

### **3. 专业性**
- 显示了严谨的科研态度
- 符合顶级会议/期刊的要求

### **4. 可追溯性**
- 便于回顾和解释实验结果
- 支持后续的改进和对比

---

## 📝 **总结**

### **现在你可以：**
1. ✅ **快速查看参数** - 直接读取 config.json
2. ✅ **完全复现实验** - 使用相同的配置
3. ✅ **专业展示结果** - 在论文中引用配置文件
4. ✅ **避免参数混淆** - 每个实验都有独立记录

### **下一步建议：**
1. 在所有训练脚本中添加 config.json 功能
2. 考虑实现 run_001, run_002 的文件夹结构
3. 在论文中明确引用配置文件的位置

---

## 🔗 **相关文件**

- `train_tsakt_linear_sakt_params.py` - TSAKT-Linear训练（已更新）
- `train_tsakt_linear_final.py` - TSAKT-Linear训练（已更新）
- `train_sakt.py` - SAKT训练（已更新）
- `save/tsakt-linear-sakt-params/config.json` - 示例配置文件

---

**记住：config.json 是论文级项目的标准配置，不要省略！** 🎯
