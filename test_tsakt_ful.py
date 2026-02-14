import pandas as pd
import torch
import os
from model_tsakt import TSAKT
from train_tsakt import get_data, prepare_batches, compute_auc, compute_rmse
import torch.nn.functional as F

# 测试TSAKT-Ful模型
def test_tsakt_ful():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载小数据集进行测试
    dataset = 'assistments09'
    batch_size = 32
    max_length = 50
    
    print(f"Testing on dataset: {dataset}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    
    # 加载数据
    train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    
    # 只使用前1000行数据进行测试
    train_df = train_df.head(1000)
    
    print(f"Loaded {len(train_df)} rows of data")
    
    # 获取数据
    train_data, val_data = get_data(train_df, max_length, train_split=0.8)
    
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    
    # 准备批次
    train_batches = prepare_batches(train_data, batch_size)
    val_batches = prepare_batches(val_data, batch_size)
    
    print(f"Train batches: {len(train_batches)}")
    print(f"Validation batches: {len(val_batches)}")
    
    # 获取模型参数
    num_items = int(train_df["item_id"].max() + 2)  # +2 因为item_id从0开始，且在get_data中被加1
    num_skills = int(train_df["skill_id"].max() + 2)  # 同样+2
    
    print(f"Number of items: {num_items}")
    print(f"Number of skills: {num_skills}")
    
    # 初始化模型
    model = TSAKT(
        num_items=num_items,
        num_skills=num_skills,
        embed_size=60,
        num_attn_layers=2,
        num_heads=5,
        encode_pos=True,  # 启用位置编码
        max_pos=10,
        drop_prob=0.2,
        tensor_rank=3
    ).to(device)
    
    print("Model initialized successfully")
    
    # 测试前向传播
    if train_batches:
        batch = train_batches[0]
        item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels = batch
        
        item_inputs = item_inputs.to(device)
        skill_inputs = skill_inputs.to(device)
        label_inputs = label_inputs.to(device)
        item_ids = item_ids.to(device)
        skill_ids = skill_ids.to(device)
        labels = labels.to(device)
        
        print(f"Batch shapes:")
        print(f"  item_inputs: {item_inputs.shape}")
        print(f"  skill_inputs: {skill_inputs.shape}")
        print(f"  label_inputs: {label_inputs.shape}")
        print(f"  item_ids: {item_ids.shape}")
        print(f"  skill_ids: {skill_ids.shape}")
        print(f"  labels: {labels.shape}")
        
        # 前向传播
        try:
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            print(f"Forward pass successful")
            print(f"Predictions shape: {preds.shape}")
            
            # 计算损失
            preds_flat = preds.squeeze(-1)  # 移除最后一个维度
            loss = F.binary_cross_entropy_with_logits(preds_flat, labels.float())
            print(f"Loss calculated: {loss.item()}")
            
            # 计算指标
            preds_sigmoid = torch.sigmoid(preds).squeeze(-1).detach().cpu()  # 移除最后一个维度并分离梯度
            labels_cpu = labels.float().cpu()
            auc = compute_auc(preds_sigmoid, labels_cpu)
            rmse = compute_rmse(preds_sigmoid, labels_cpu)
            print(f"AUC: {auc:.4f}")
            print(f"RMSE: {rmse:.4f}")
            
            print("\n✅ Test passed! TSAKT-Ful model is working correctly.")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error:")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("No train batches available")
        return False

if __name__ == "__main__":
    test_tsakt_ful()
