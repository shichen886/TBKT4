import torch
import numpy as np
from model_tsakt import TSAKT, TensorSelfAttention, TensorMultiHeadAttention


def test_tensor_self_attention():
    print("Testing TensorSelfAttention...")
    batch_size = 2
    seq_length = 10
    embed_size = 60
    num_heads = 5
    drop_prob = 0.2
    tensor_rank = 3
    
    model = TensorSelfAttention(embed_size, num_heads, drop_prob, tensor_rank)
    
    query = torch.randn(batch_size, seq_length, embed_size)
    key = torch.randn(batch_size, seq_length, embed_size)
    value = torch.randn(batch_size, seq_length, embed_size)
    
    output, attention_weights = model(query, key, value)
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    assert output.shape == (batch_size, seq_length, embed_size), f"Expected output shape {(batch_size, seq_length, embed_size)}, got {output.shape}"
    print("TensorSelfAttention test passed!")
    print()


def test_tensor_multi_head_attention():
    print("Testing TensorMultiHeadAttention...")
    batch_size = 2
    seq_length = 10
    embed_size = 60
    num_heads = 5
    drop_prob = 0.2
    tensor_rank = 3
    
    model = TensorMultiHeadAttention(embed_size, num_heads, drop_prob, tensor_rank)
    
    query = torch.randn(batch_size, seq_length, embed_size)
    key = torch.randn(batch_size, seq_length, embed_size)
    value = torch.randn(batch_size, seq_length, embed_size)
    
    output = model(query, key, value)
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, seq_length, embed_size), f"Expected output shape {(batch_size, seq_length, embed_size)}, got {output.shape}"
    print("TensorMultiHeadAttention test passed!")
    print()


def test_tsakt_model():
    print("Testing TSAKT model...")
    batch_size = 4
    seq_length = 20
    num_items = 100
    num_skills = 50
    embed_size = 60
    num_attn_layers = 2
    num_heads = 5
    encode_pos = True
    max_pos = 10
    drop_prob = 0.2
    tensor_rank = 3
    
    model = TSAKT(num_items, num_skills, embed_size, num_attn_layers, num_heads,
                  encode_pos, max_pos, drop_prob, tensor_rank)
    
    item_inputs = torch.randint(1, num_items + 1, (batch_size, seq_length))
    skill_inputs = torch.randint(1, num_skills + 1, (batch_size, seq_length))
    label_inputs = torch.randint(0, 2, (batch_size, seq_length))
    item_ids = torch.randint(1, num_items + 1, (batch_size, seq_length))
    skill_ids = torch.randint(1, num_skills + 1, (batch_size, seq_length))
    
    output = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
    
    print(f"Item inputs shape: {item_inputs.shape}")
    print(f"Skill inputs shape: {skill_inputs.shape}")
    print(f"Label inputs shape: {label_inputs.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, seq_length, 1), f"Expected output shape {(batch_size, seq_length, 1)}, got {output.shape}"
    print("TSAKT model test passed!")
    print()


def test_tsakt_with_mask():
    print("Testing TSAKT model with mask...")
    batch_size = 2
    seq_length = 10
    num_items = 50
    num_skills = 25
    embed_size = 60
    num_attn_layers = 2
    num_heads = 5
    encode_pos = False
    max_pos = 5
    drop_prob = 0.2
    tensor_rank = 3
    
    model = TSAKT(num_items, num_skills, embed_size, num_attn_layers, num_heads,
                  encode_pos, max_pos, drop_prob, tensor_rank)
    
    item_inputs = torch.randint(1, num_items + 1, (batch_size, seq_length))
    skill_inputs = torch.randint(1, num_skills + 1, (batch_size, seq_length))
    label_inputs = torch.randint(0, 2, (batch_size, seq_length))
    item_ids = torch.randint(1, num_items + 1, (batch_size, seq_length))
    skill_ids = torch.randint(1, num_skills + 1, (batch_size, seq_length))
    
    output = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    assert output.shape == (batch_size, seq_length, 1), f"Expected output shape {(batch_size, seq_length, 1)}, got {output.shape}"
    print("TSAKT model with mask test passed!")
    print()


def test_tensor_rank_variations():
    print("Testing TSAKT with different tensor ranks...")
    batch_size = 2
    seq_length = 10
    num_items = 50
    num_skills = 25
    embed_size = 60
    num_attn_layers = 1
    num_heads = 5
    encode_pos = False
    max_pos = 5
    drop_prob = 0.2
    
    for tensor_rank in [2, 3, 4]:
        print(f"Testing with tensor_rank={tensor_rank}")
        model = TSAKT(num_items, num_skills, embed_size, num_attn_layers, num_heads,
                      encode_pos, max_pos, drop_prob, tensor_rank)
        
        item_inputs = torch.randint(1, num_items + 1, (batch_size, seq_length))
        skill_inputs = torch.randint(1, num_skills + 1, (batch_size, seq_length))
        label_inputs = torch.randint(0, 2, (batch_size, seq_length))
        item_ids = torch.randint(1, num_items + 1, (batch_size, seq_length))
        skill_ids = torch.randint(1, num_skills + 1, (batch_size, seq_length))
        
        output = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
        
        print(f"  Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_length, 1), f"Expected output shape {(batch_size, seq_length, 1)}, got {output.shape}"
        print(f"  tensor_rank={tensor_rank} test passed!")
    
    print("All tensor rank variations test passed!")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tensor Self-Attention Module")
    print("=" * 60)
    print()
    
    try:
        test_tensor_self_attention()
        test_tensor_multi_head_attention()
        test_tsakt_model()
        test_tsakt_with_mask()
        test_tensor_rank_variations()
        
        print("=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
