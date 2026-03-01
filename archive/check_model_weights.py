import torch

model_v2 = torch.load('save/tsakt-ful-v2/assistments09,batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3', map_location='cpu', weights_only=False)
model_v3 = torch.load('save/tsakt-ful-v3/assistments09,batch_size=128,max_length=200,encode_pos=True,max_pos=200,tensor_rank=3', map_location='cpu', weights_only=False)

state_v2 = model_v2.state_dict()
state_v3 = model_v3.state_dict()

print("Checking if model weights are identical...")
print("=" * 80)

diff_count = 0
same_count = 0

for key in state_v2:
    if key in state_v3:
        if torch.equal(state_v2[key], state_v3[key]):
            same_count += 1
        else:
            diff_count += 1
            print(f"Key {key} differs")
            print(f"  v2 shape: {state_v2[key].shape}, v3 shape: {state_v3[key].shape}")
            print(f"  v2 mean: {state_v2[key].mean():.6f}, v3 mean: {state_v3[key].mean():.6f}")
            print(f"  v2 std: {state_v2[key].std():.6f}, v3 std: {state_v3[key].std():.6f}")
    else:
        print(f"Key {key} exists in v2 but not in v3")

print("=" * 80)
print(f"Total keys: {len(state_v2)}")
print(f"Same keys: {same_count}")
print(f"Different keys: {diff_count}")
print(f"Keys are identical: {diff_count == 0}")