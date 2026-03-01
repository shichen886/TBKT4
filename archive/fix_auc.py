import re

def fix_auc_calculation(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_pattern = r'auc = roc_auc_score\(all_labels, all_preds\) if len\(np\.unique\(all_labels\)\) > 1 else 0\.5'
    new_code = '''try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5'''
    
    content = re.sub(old_pattern, new_code, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filename}")

fix_auc_calculation('evaluate_short_baseline.py')
fix_auc_calculation('evaluate_time_baseline.py')
print("All files fixed!")
