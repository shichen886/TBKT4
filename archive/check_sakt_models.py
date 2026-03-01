import torch

# 检查SAKT模型的实际性能
sakt_models = [
    'save/sakt/assistments09,batch_size=128,max_length=200,encode_pos=False,max_pos=10',
    'save/sakt/assistments12,batch_size=128,max_length=200,encode_pos=False,max_pos=10',
    'save/sakt/assistments15,batch_size=128,max_length=200,encode_pos=False,max_pos=10'
]

print("="*80)
print("检查SAKT模型性能")
print("="*80)

for model_path in sakt_models:
    try:
        # 使用weights_only=False来加载
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"\n{model_path}")
        print(f"  类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"  键: {checkpoint.keys()}")
            if 'val_loss' in checkpoint:
                print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
            if 'val_auc' in checkpoint:
                print(f"  Val AUC: {checkpoint['val_auc']:.4f}")
            if 'val_rmse' in checkpoint:
                print(f"  Val RMSE: {checkpoint['val_rmse']:.4f}")
        else:
            print(f"  对象类型: {type(checkpoint)}")
            # 检查是否有属性
            if hasattr(checkpoint, 'val_loss'):
                print(f"  Val Loss: {checkpoint.val_loss:.4f}")
            if hasattr(checkpoint, 'val_auc'):
                print(f"  Val AUC: {checkpoint.val_auc:.4f}")
            if hasattr(checkpoint, 'val_rmse'):
                print(f"  Val RMSE: {checkpoint.val_rmse:.4f}")
    except Exception as e:
        print(f"\n{model_path}")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()

print("="*80)
