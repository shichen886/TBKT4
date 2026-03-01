import os
from tensorboard.backend.event_processing import event_accumulator

# 检查TSAKT-w/o-Pos的训练日志
datasets = ['assistments09', 'assistments12', 'assistments15']

for dataset in datasets:
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset}")
    print(f"{'=' * 80}")
    
    logdir = f'runs/tsakt/{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3'
    
    if os.path.exists(logdir):
        try:
            ea = event_accumulator.EventAccumulator(logdir)
            ea.Reload()
            
            # 获取所有可用的标量标签
            tags = ea.Tags()
            print(f"Available scalar tags: {tags['scalars']}")
            
            # 获取auc/val数据
            if 'auc/val' in tags['scalars']:
                auc_events = ea.Scalars('auc/val')
                print(f"Number of auc/val records: {len(auc_events)}")
                
                if len(auc_events) > 0:
                    last_event = auc_events[-1]
                    print(f"Last step: {last_event.step}")
                    print(f"Last auc/val: {last_event.value}")
                    
                    # 估计训练轮数
                    num_epochs = last_event.step + 1
                    print(f"Estimated epochs: {num_epochs}")
            else:
                print("auc/val not found in scalar tags")
                
        except Exception as e:
            print(f"Error reading tensorboard events: {e}")
    else:
        print(f"Log directory not found: {logdir}")