import json
import os

def check_default_values():
    """Check if default values in training scripts match actual training parameters."""
    print(f"\n{'='*80}")
    print("CHECKING DEFAULT VALUES IN TRAINING SCRIPTS")
    print(f"{'='*80}\n")

    # Check train_tsakt_linear_final.py
    print("train_tsakt_linear_final.py:")
    with open('train_tsakt_linear_final.py', 'r', encoding='utf-8') as f:
        content = f.read()
        # Find default values
        import re
        embed_match = re.search(r"parser\.add_argument\('--embed_size'.*?default=(\d+)", content)
        heads_match = re.search(r"parser\.add_argument\('--num_heads'.*?default=(\d+)", content)
        layers_match = re.search(r"parser\.add_argument\('--num_layers'.*?default=(\d+)", content)
        rank_match = re.search(r"parser\.add_argument\('--tensor_rank'.*?default=(\d+)", content)

        if embed_match:
            print(f"  embed_size default: {embed_match.group(1)}")
        if heads_match:
            print(f"  num_heads default: {heads_match.group(1)}")
        if layers_match:
            print(f"  num_layers default: {layers_match.group(1)}")
        if rank_match:
            print(f"  tensor_rank default: {rank_match.group(1)}")

    print()

    # Check train_tsakt_linear_nopos.py
    print("train_tsakt_linear_nopos.py:")
    with open('train_tsakt_linear_nopos.py', 'r', encoding='utf-8') as f:
        content = f.read()
        embed_match = re.search(r"parser\.add_argument\('--embed_size'.*?default=(\d+)", content)
        heads_match = re.search(r"parser\.add_argument\('--num_heads'.*?default=(\d+)", content)
        layers_match = re.search(r"parser\.add_argument\('--num_layers'.*?default=(\d+)", content)
        rank_match = re.search(r"parser\.add_argument\('--tensor_rank'.*?default=(\d+)", content)

        if embed_match:
            print(f"  embed_size default: {embed_match.group(1)}")
        if heads_match:
            print(f"  num_heads default: {heads_match.group(1)}")
        if layers_match:
            print(f"  num_layers default: {layers_match.group(1)}")
        if rank_match:
            print(f"  tensor_rank default: {rank_match.group(1)}")

    print(f"\n{'='*80}")
    print("ACTUAL TRAINING PARAMETERS (from config.json)")
    print(f"{'='*80}\n")

    # Check actual saved configs
    config1_path = 'save/tsakt-linear/config.json'
    config2_path = 'save/tsakt-linear-nopos/config.json'

    if os.path.exists(config1_path):
        with open(config1_path, 'r') as f:
            config1 = json.load(f)
        print("TSAKT-Linear config:")
        print(f"  embed_size: {config1['embed_size']}")
        print(f"  num_heads: {config1['num_heads']}")
        print(f"  num_layers: {config1['num_layers']}")
        print(f"  tensor_rank: {config1['tensor_rank']}")

    if os.path.exists(config2_path):
        with open(config2_path, 'r') as f:
            config2 = json.load(f)
        print("\nTSAKT-Linear-NoPos config:")
        print(f"  embed_size: {config2['embed_size']}")
        print(f"  num_heads: {config2['num_heads']}")
        print(f"  num_layers: {config2['num_layers']}")
        print(f"  tensor_rank: {config2['tensor_rank']}")

    print(f"\n{'='*80}")
    print("ISSUE ANALYSIS")
    print(f"{'='*80}\n")

    # Check if default values match actual values
    if os.path.exists(config1_path):
        with open(config1_path, 'r') as f:
            config1 = json.load(f)

        with open('train_tsakt_linear_final.py', 'r', encoding='utf-8') as f:
            content = f.read()
            embed_match = re.search(r"parser\.add_argument\('--embed_size'.*?default=(\d+)", content)

            if embed_match:
                default_embed = int(embed_match.group(1))
                actual_embed = config1['embed_size']

                if default_embed != actual_embed:
                    print(f"⚠️ WARNING: embed_size mismatch!")
                    print(f"  Default value: {default_embed}")
                    print(f"  Actual value used: {actual_embed}")
                    print(f"  This could cause reproducibility issues!")
                    print(f"  Recommendation: Change default to {actual_embed}")
                else:
                    print(f"✓ embed_size matches: {actual_embed}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    check_default_values()