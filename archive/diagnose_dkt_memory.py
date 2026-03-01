import os
import torch
import numpy as np
import pandas as pd
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

from model_dkt1 import DKT1
from model_akt import AKT
from model_sakt import SAKT
from model_tsakt import TSAKT

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def print_memory_usage(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"{prefix} Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

def truncate_sequences(data, max_length):
    truncated_data = []
    for item_inputs, skill_inputs, item_ids, skill_ids, labels in data:
        if item_inputs is not None:
            item_inputs = item_inputs[-max_length:] if len(item_inputs) > max_length else item_inputs
        if skill_inputs is not None:
            skill_inputs = skill_inputs[-max_length:] if len(skill_inputs) > max_length else skill_inputs
        if item_ids is not None:
            item_ids = item_ids[-max_length:] if len(item_ids) > max_length else item_ids
        if skill_ids is not None:
            skill_ids = skill_ids[-max_length:] if len(skill_ids) > max_length else skill_ids
        if labels is not None:
            labels = labels[-max_length:] if len(labels) > max_length else labels
        truncated_data.append((item_inputs, skill_inputs, item_ids, skill_ids, labels))
    return truncated_data

def diagnose_dkt(dataset, model_path, max_length=50):
    print(f"\n{'='*80}")
    print(f"Diagnosing DKT on {dataset}")
    print(f"{'='*80}")
    
    try:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
    except:
        train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    
    from train_dkt1 import get_data, prepare_batches
    train_data, val_data = get_data(train_df, item_in=False, skill_in=True, item_out=True, skill_out=False, skill_separate=False)
    
    print(f"\n1. Data loading:")
    print(f"   Original val_data size: {len(val_data)}")
    
    val_data = truncate_sequences(val_data, max_length)
    print(f"   Truncated val_data size: {len(val_data)}")
    
    val_batches = prepare_batches(val_data, batch_size=1)
    print(f"   Number of batches: {len(val_batches)}")
    
    print(f"\n2. Model loading:")
    print(f"   Model path: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"   Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"   Checkpoint keys: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            model = DKT1(**checkpoint.get('model_args', {}))
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = checkpoint
    else:
        model = checkpoint
    
    print(f"   Model type: {type(model)}")
    
    print(f"\n3. Moving model to GPU:")
    clear_gpu_memory()
    print_memory_usage("   Before loading model:")
    
    model = model.to(device)
    print_memory_usage("   After loading model:")
    
    model.eval()
    
    print(f"\n4. Testing first batch:")
    clear_gpu_memory()
    print_memory_usage("   Before first batch:")
    
    for batch_idx, (item_inputs, skill_inputs, item_ids, skill_ids, labels) in enumerate(val_batches):
        print(f"   Batch {batch_idx + 1}:")
        print(f"      item_inputs shape: {item_inputs.shape if item_inputs is not None else 'None'}")
        print(f"      skill_inputs shape: {skill_inputs.shape if skill_inputs is not None else 'None'}")
        print(f"      item_ids shape: {item_ids.shape if item_ids is not None else 'None'}")
        print(f"      skill_ids shape: {skill_ids.shape if skill_ids is not None else 'None'}")
        print(f"      labels shape: {labels.shape}")
        
        print_memory_usage("      Before moving to GPU:")
        
        item_inputs = item_inputs.to(device) if item_inputs is not None else None
        skill_inputs = skill_inputs.to(device) if skill_inputs is not None else None
        item_ids = item_ids.to(device) if item_ids is not None else None
        skill_ids = skill_ids.to(device) if skill_ids is not None else None
        labels = labels.to(device)
        
        print_memory_usage("      After moving to GPU:")
        
        try:
            with torch.no_grad():
                preds, _ = model(item_inputs, skill_inputs)
                print_memory_usage("      After forward pass:")
                
                preds = torch.sigmoid(preds)
                print_memory_usage("      After sigmoid:")
                
                if skill_ids is not None:
                    mask = labels >= 0
                    valid_preds = preds[mask]
                    valid_skill_ids = skill_ids[mask]
                    valid_labels = labels[mask]
                    
                    if len(valid_preds) > 0:
                        selected_preds = valid_preds[torch.arange(valid_preds.size(0)), valid_skill_ids]
                        print(f"      Valid predictions: {len(selected_preds)}")
                
                del item_inputs, skill_inputs, item_ids, skill_ids, labels, preds
                print_memory_usage("      After cleanup:")
            
            print(f"   ✓ First batch processed successfully!")
            break
        except torch.cuda.OutOfMemoryError as e:
            print(f"   ✗ CUDA out of memory on first batch!")
            print(f"   Error: {e}")
            return False
        except Exception as e:
            print(f"   ✗ Error on first batch: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def main():
    print("=" * 80)
    print("DKT 显存诊断脚本")
    print("=" * 80)
    
    dataset = 'assistments09_short_50'
    max_length = 50
    
    dkt_path = os.path.join('save', 'dkt-short', f'{dataset},batch_size=100,item_in=False,skill_in=True,item_out=True,skill_out=False,skill_separate=False')
    
    if os.path.exists(dkt_path):
        success = diagnose_dkt(dataset, dkt_path, max_length=max_length)
        if success:
            print(f"\n✓ DKT evaluation should work!")
        else:
            print(f"\n✗ DKT evaluation will fail due to OOM")
    else:
        print(f"Model path does not exist: {dkt_path}")

if __name__ == "__main__":
    main()
