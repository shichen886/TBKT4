import json
import os

CONFIG_FILE = "name_mappings.json"

def load_mappings():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "user_names": {},
        "item_names": {},
        "skill_names": {}
    }

def save_mappings(mappings):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

def get_user_name(mappings, user_id, default=None):
    if default is None:
        default = f"学生{user_id}"
    return mappings["user_names"].get(str(user_id), default)

def get_item_name(mappings, item_id, default=None):
    if default is None:
        default = f"题目{item_id}"
    return mappings["item_names"].get(str(item_id), default)

def get_skill_name(mappings, skill_id, default=None):
    if default is None:
        default = f"知识点{skill_id}"
    return mappings["skill_names"].get(str(skill_id), default)

def set_user_name(mappings, user_id, name):
    mappings["user_names"][str(user_id)] = name

def set_item_name(mappings, item_id, name):
    mappings["item_names"][str(item_id)] = name

def set_skill_name(mappings, skill_id, name):
    mappings["skill_names"][str(skill_id)] = name

def auto_generate_skill_names(df):
    mappings = load_mappings()
    unique_skills = df['skill_id'].unique()
    for skill_id in unique_skills:
        if str(skill_id) not in mappings["skill_names"]:
            skill_data = df[df['skill_id'] == skill_id]
            avg_correct = skill_data['correct'].mean()
            if avg_correct >= 0.8:
                difficulty = "简单"
            elif avg_correct >= 0.6:
                difficulty = "中等"
            else:
                difficulty = "困难"
            mappings["skill_names"][str(skill_id)] = f"知识点{skill_id}（{difficulty}）"
    save_mappings(mappings)
    return mappings
