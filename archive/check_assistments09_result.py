import torch

# 加载assistments09的训练结果
model_path = 'save/tsakt-linear/assistments09_best.pt'
checkpoint = torch.load(model_path, map_location='cpu')

print("="*80)
print("TSAKT-Linear训练完成！")
print("="*80)
print(f"数据集: assistments09")
print(f"训练轮数: {checkpoint['epoch']+1}/50")
print(f"验证损失: {checkpoint['val_loss']:.4f}")
print(f"验证AUC: {checkpoint['val_auc']:.4f}")
print(f"验证RMSE: {checkpoint['val_rmse']:.4f}")
print("="*80)
print(f"模型文件: {model_path}")
print("="*80)
