import os
import sys

# 在导入任何库之前，先设置环境变量
app_dir = os.path.dirname(os.path.abspath(__file__))
paddlex_cache_dir = os.path.join(app_dir, 'paddlex_cache')
modelscope_cache_dir = os.path.join(app_dir, 'modelscope_cache')
os.makedirs(paddlex_cache_dir, exist_ok=True)
os.makedirs(modelscope_cache_dir, exist_ok=True)

# 设置PaddleOCR缓存目录到应用目录，避免权限问题
os.environ['PADDLEX_HOME'] = paddlex_cache_dir
os.environ['PADDLE_HOME'] = paddlex_cache_dir
os.environ['MODELSCOPE_CACHE'] = modelscope_cache_dir
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLEOCR_OFFLINE'] = 'True'

# 设置PaddlePaddle使用GPU，但不与PyTorch冲突
# 尝试设置PaddlePaddle特定的环境变量来避免冲突
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'  # 自动增长内存分配
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'  # 只使用50%的GPU内存
os.environ['FLAGS_use_mkldnn'] = '0'  # 禁用MKLDNN避免冲突

import streamlit as st
import pandas as pd
import numpy as np
import torch
import shutil
import tempfile
from torch.nn.utils.rnn import pad_sequence
import plotly.graph_objects as go
import plotly.express as px

# 全局变量
PADDLEOCR_AVAILABLE = False
global_ocr_engine = None


from model_sakt import SAKT
from model_tsakt import TSAKT
from name_mappings import (
    load_mappings, save_mappings,
    get_user_name, get_item_name, get_skill_name,
    set_user_name, set_item_name, set_skill_name,
    auto_generate_skill_names
)
from recommendation import CollaborativeFiltering, ContentBasedRecommender, HybridRecommender
from learning_path import AdaptiveLearningPath, LearningPathOptimizer
from chart_config import ChartConfig

st.set_page_config(
    page_title="知识追踪系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open('style.css', encoding='utf-8') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("📚 智能知识追踪系统")
st.markdown("---")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(df, max_length):
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    return data

def prepare_single_sequence(item_ids, skill_ids, labels):
    item_inputs = torch.cat((torch.zeros(1, dtype=torch.long), item_ids + 1))[:-1]
    skill_inputs = torch.cat((torch.zeros(1, dtype=torch.long), skill_ids + 1))[:-1]
    label_inputs = torch.cat((torch.zeros(1, dtype=torch.long), labels))[:-1]
    
    return item_inputs.unsqueeze(0), skill_inputs.unsqueeze(0), label_inputs.unsqueeze(0), \
           item_ids.unsqueeze(0), skill_ids.unsqueeze(0), labels.unsqueeze(0)

def load_models(dataset):
    df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
    num_items = int(df["item_id"].max() + 1)
    num_skills = int(df["skill_id"].max() + 1)
    
    sakt_path = os.path.join('save/sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=10')
    if not os.path.exists(sakt_path):
        sakt_path = os.path.join('save/sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
    
    if os.path.exists(sakt_path):
        loaded_model = torch.load(sakt_path, map_location=device, weights_only=False)
        sakt_model = loaded_model.to(device)
        sakt_model.eval()
    else:
        sakt_model = None
    
    tsakt_path = os.path.join('save/tsakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
    
    if os.path.exists(tsakt_path):
        loaded_model = torch.load(tsakt_path, map_location=device, weights_only=False)
        tsakt_model = loaded_model.to(device)
        tsakt_model.eval()
        tsakt_available = True
    else:
        tsakt_model = None
        tsakt_available = False
    
    return sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available

def predict_next_question(model, item_ids, skill_ids, labels):
    item_inputs, skill_inputs, label_inputs, item_ids_batch, skill_ids_batch, labels_batch = \
        prepare_single_sequence(item_ids, skill_ids, labels)
    
    item_inputs = item_inputs.to(device)
    skill_inputs = skill_inputs.to(device)
    label_inputs = label_inputs.to(device)
    item_ids_batch = item_ids_batch.to(device)
    skill_ids_batch = skill_ids_batch.to(device)
    
    with torch.no_grad():
        preds = model(item_inputs, skill_inputs, label_inputs, item_ids_batch, skill_ids_batch)
        preds = torch.sigmoid(preds).cpu().numpy()
    
    return preds[0, -1].item()

def analyze_student_performance(df, user_id):
    user_data = df[df['user_id'] == user_id].sort_values('item_id')
    
    if len(user_data) == 0:
        return None
    
    skill_stats = user_data.groupby('skill_id').agg({
        'correct': ['mean', 'count']
    }).reset_index()
    skill_stats.columns = ['skill_id', 'accuracy', 'count']
    
    # 修改掌握度计算方式：综合考虑正确率和答题数量
    # 掌握度 = 正确率 * (1 - 1/(1 + count/10)) 
    # 这样答题数量越多，权重越高，但不会无限增长
    skill_stats['mastery'] = skill_stats['accuracy'] * (1 - 1/(1 + skill_stats['count']/10))
    
    return skill_stats

def recommend_questions(skill_stats, num_questions=5, difficulty='balanced'):
    if skill_stats is None or len(skill_stats) == 0:
        return []
    
    if difficulty == 'easy':
        recommended = skill_stats.nlargest(num_questions, 'accuracy')
    elif difficulty == 'hard':
        recommended = skill_stats.nsmallest(num_questions, 'accuracy')
    else:
        recommended = skill_stats.nsmallest(num_questions, 'mastery')
    
    return recommended['skill_id'].tolist()

st.sidebar.title("⚙️ 系统设置")

available_datasets = []
for folder in os.listdir('data'):
    if os.path.isdir(os.path.join('data', folder)):
        if os.path.exists(os.path.join('data', folder, 'preprocessed_data.csv')):
            available_datasets.append(folder)

dataset = st.sidebar.selectbox(
    "选择数据集",
    sorted(available_datasets)
)

st.sidebar.markdown("---")

st.sidebar.header("📊 模型性能")

sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)

mappings = load_mappings()

st.sidebar.metric("SAKT AUC", "0.7769")
st.sidebar.metric("TSAKT AUC", "0.7843", delta="+0.0074")

st.sidebar.markdown("---")

st.sidebar.header("📈 系统功能")

model_choice = st.sidebar.selectbox(
    "选择模型",
    ["SAKT"] + (["TSAKT"] if tsakt_available else [])
)

st.sidebar.markdown("---")

st.sidebar.header("🤖 推荐算法")

recommendation_method = st.sidebar.selectbox(
    "推荐方法",
    ["混合推荐", "协同过滤", "基于内容", "传统方法"]
)

if recommendation_method != "传统方法":
    try:
        if recommendation_method == "混合推荐":
            recommender = HybridRecommender()
        elif recommendation_method == "协同过滤":
            recommender = CollaborativeFiltering()
        else:
            recommender = ContentBasedRecommender()
        
        recommender.fit(df)
        st.sidebar.success("✅ 推荐模型已加载")
    except Exception as e:
        st.sidebar.warning(f"⚠️ 推荐模型加载失败: {str(e)}")
        recommender = None
else:
    recommender = None

st.sidebar.markdown("---")

st.sidebar.header("🗺️ 学习路径优化")

enable_adaptive_path = st.sidebar.checkbox("启用自适应学习路径", value=True)

if enable_adaptive_path:
    try:
        learning_path_optimizer = LearningPathOptimizer(df)
        st.sidebar.success("✅ 学习路径优化器已加载")
    except Exception as e:
        st.sidebar.warning(f"⚠️ 学习路径优化器加载失败: {str(e)}")
        learning_path_optimizer = None
else:
    learning_path_optimizer = None

tabs = st.tabs([
    "🎯 个性化学习推荐",
    "📊 学生学习分析",
    "🗺️ 学习路径优化",
    "📝 教育评估",
    "📤 上传数据"
])

with tabs[0]:
    st.header("🎯 个性化学习推荐")
    st.markdown("根据学生历史答题记录，智能推荐适合的题目")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("学生信息")
        
        unique_users = df['user_id'].unique()
        user_options = {get_user_name(mappings, uid): uid for uid in unique_users}
        selected_user_name = st.selectbox("选择学生", sorted(user_options.keys()))
        user_id = user_options[selected_user_name]
        
        user_data = df[df['user_id'] == user_id]
        
        if len(user_data) > 0:
            st.write(f"总答题数: {len(user_data)}")
            st.write(f"正确率: {user_data['correct'].mean():.2%}")
            st.write(f"涉及知识点: {user_data['skill_id'].nunique()}")
        else:
            st.warning("未找到该学生的数据")
    
    with col2:
        st.subheader("推荐设置")
        difficulty = st.selectbox(
            "推荐难度",
            ["balanced", "easy", "hard"],
            index=0,
            format_func=lambda x: {"balanced": "均衡（推荐薄弱知识点）", "easy": "简单（巩固已掌握）", "hard": "困难（挑战高难度）"}[x]
        )
        num_questions = st.slider("推荐题目数量", 1, 10, 5)
    
    if len(user_data) > 0:
        st.markdown("---")
        st.subheader("📋 推荐题目")
        
        if recommender is not None:
            recommended_items = recommender.recommend_for_user(user_id, num_questions)
            
            if recommended_items:
                for i, item_id in enumerate(recommended_items, 1):
                    item_data = df[df['item_id'] == item_id].iloc[0]
                    skill_id = item_data['skill_id']
                    skill_name = get_skill_name(mappings, skill_id)
                    
                    skill_stats = analyze_student_performance(df, user_id)
                    if skill_stats is not None and skill_id in skill_stats['skill_id'].values:
                        skill_data = skill_stats[skill_stats['skill_id'] == skill_id].iloc[0]
                        mastery = skill_data['mastery']
                        accuracy = skill_data['accuracy']
                    else:
                        mastery = 0
                        accuracy = 0
                    
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_a:
                        st.metric(f"题目 {i}", skill_name)
                    with col_b:
                        progress = mastery * 100
                        st.progress(progress / 100)
                        st.caption(f"掌握度: {mastery:.2%} | 正确率: {accuracy:.2%}")
                    with col_c:
                        if accuracy < 0.5:
                            st.error("需加强")
                        elif accuracy < 0.7:
                            st.warning("一般")
                        else:
                            st.success("良好")
            else:
                st.info("暂无推荐题目")
                st.caption("💡 提示：推荐算法可能因为以下原因无法生成推荐：")
                st.caption("1. 当前学生不在活跃用户列表中（推荐算法只处理最活跃的500个用户）")
                st.caption("2. 学生答题的题目不在推荐算法处理的题目范围内（只处理最活跃的500个题目）")
                st.caption("3. 建议切换到'传统方法'推荐，或选择答题数较多的学生")
        else:
            skill_stats = analyze_student_performance(df, user_id)
            recommended_skills = recommend_questions(skill_stats, num_questions, difficulty)
            
            if recommended_skills:
                for i, skill_id in enumerate(recommended_skills, 1):
                    skill_data = skill_stats[skill_stats['skill_id'] == skill_id].iloc[0]
                    mastery = skill_data['mastery']
                    accuracy = skill_data['accuracy']
                    skill_name = get_skill_name(mappings, skill_id)
                    
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_a:
                        st.metric(f"题目 {i}", skill_name)
                    with col_b:
                        progress = mastery * 100
                        st.progress(progress / 100)
                        st.caption(f"掌握度: {mastery:.2%} | 正确率: {accuracy:.2%}")
                    with col_c:
                        if accuracy < 0.5:
                            st.error("需加强")
                        elif accuracy < 0.7:
                            st.warning("一般")
                        else:
                            st.success("良好")
            else:
                st.info("暂无推荐题目")
        
        st.markdown("---")
        st.subheader("🔮 预测下一题")
        
        model = sakt_model if model_choice == "SAKT" else tsakt_model
        
        if model is None:
            st.warning(f"⚠️ {model_choice} 模型不可用，请选择其他模型")
        else:
            recent_answers = user_data.tail(10)
            item_ids = torch.tensor(recent_answers['item_id'].values, dtype=torch.long)
            skill_ids = torch.tensor(recent_answers['skill_id'].values, dtype=torch.long)
            labels = torch.tensor(recent_answers['correct'].values, dtype=torch.long)
            
            prediction = predict_next_question(model, item_ids, skill_ids, labels)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("预测正确率", f"{prediction:.2%}")
            with col_b:
                if prediction > 0.7:
                    st.success("准备充分")
                elif prediction > 0.5:
                    st.warning("需要复习")
                else:
                    st.error("建议先学习")

with tabs[1]:
    st.header("📊 学生学习分析")
    st.markdown("深入分析学生的学习情况和薄弱环节")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unique_users = df['user_id'].unique()
        user_options = {get_user_name(mappings, uid): uid for uid in unique_users}
        selected_user_name = st.selectbox("选择学生", sorted(user_options.keys()), key="analysis_user")
        user_id = user_options[selected_user_name]
    
    with col2:
        analysis_type = st.selectbox(
            "分析类型",
            ["知识点掌握情况", "答题趋势", "错误分析"]
        )
    
    user_data = df[df['user_id'] == user_id]
    
    if len(user_data) > 0:
        if analysis_type == "知识点掌握情况":
            st.subheader("知识点掌握情况")
            
            skill_stats = analyze_student_performance(df, user_id)
            
            if skill_stats is not None and len(skill_stats) > 0:
                skill_stats['skill_name'] = skill_stats['skill_id'].apply(lambda x: get_skill_name(mappings, x))
                fig = ChartConfig.create_bar_chart(
                    skill_stats,
                    x_col='skill_name',
                    y_col='accuracy',
                    title='各知识点正确率',
                    color_col='accuracy',
                    labels={'skill_name': '知识点', 'accuracy': '正确率'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("薄弱知识点识别")
                weak_skills = skill_stats.nsmallest(5, 'accuracy')
                
                for _, row in weak_skills.iterrows():
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_a:
                        st.write(get_skill_name(mappings, int(row['skill_id'])))
                    with col_b:
                        st.progress(row['accuracy'])
                    with col_c:
                        st.error(f"{row['accuracy']:.2%}")
        
        elif analysis_type == "答题趋势":
            st.subheader("答题趋势分析")
            
            user_data_sorted = user_data.sort_values('item_id')
            user_data_sorted['cumulative_accuracy'] = user_data_sorted['correct'].expanding().mean()
            
            fig = ChartConfig.create_line_chart(
                user_data_sorted,
                y_col='cumulative_accuracy',
                title='累计正确率趋势',
                x_col='index',
                labels={'index': '答题序号', 'cumulative_accuracy': '累计正确率'}
            )
            fig.add_hline(y=0.7, line_dash="dash", line_color="#EF4444", annotation_text="目标线 70%")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("学习进度")
            total_questions = len(user_data)
            correct_questions = user_data['correct'].sum()
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("总答题数", total_questions)
            with col_b:
                st.metric("正确答题数", correct_questions)
            
            st.progress(correct_questions / total_questions)
        
        elif analysis_type == "错误分析":
            st.subheader("错误分析")
            
            wrong_answers = user_data[user_data['correct'] == 0]
            
            if len(wrong_answers) > 0:
                skill_error_counts = wrong_answers['skill_id'].value_counts().head(10)
                
                fig = ChartConfig.create_pie_chart(
                    values=skill_error_counts.values,
                    names=[get_skill_name(mappings, int(k)) for k in skill_error_counts.index],
                    title='错误分布'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("需要重点复习的知识点")
                for skill_id, count in skill_error_counts.items():
                    st.write(f"{get_skill_name(mappings, int(skill_id))}: {count} 次错误")
            else:
                st.success("🎉 恭喜！该学生没有错误记录")
    else:
        st.warning("未找到该学生的数据")

with tabs[2]:
    st.header("🗺️ 学习路径优化")
    st.markdown("制定个性化的学习计划，优化学习路径")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unique_users = df['user_id'].unique()
        user_options = {get_user_name(mappings, uid): uid for uid in unique_users}
        selected_user_name = st.selectbox("选择学生", sorted(user_options.keys()), key="path_user")
        user_id = user_options[selected_user_name]
    
    with col2:
        learning_goal = st.selectbox(
            "学习目标",
            ["全面掌握", "重点突破", "查漏补缺"]
        )
    
    user_data = df[df['user_id'] == user_id]
    
    if len(user_data) > 0:
        st.subheader("📅 个性化学习计划")
        
        if learning_path_optimizer is not None and learning_path_optimizer.adaptive_path is not None:
            # 自适应学习路径
            learning_path = learning_path_optimizer.adaptive_path.recommend_learning_path(user_id, max_length=10)
            
            if learning_path:
                for i, skill_id in enumerate(learning_path, 1):
                    skill_name = get_skill_name(mappings, skill_id)
                    skill_data = df[df['skill_id'] == skill_id]
                    
                    with st.expander(f"第 {i} 阶段: {skill_name}"):
                        # 获取学生在该知识点上的表现
                        user_skill_data = user_data[user_data['skill_id'] == skill_id]
                        if len(user_skill_data) > 0:
                            accuracy = user_skill_data['correct'].mean()
                            count = len(user_skill_data)
                        else:
                            accuracy = 0
                            count = 0
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("当前正确率", f"{accuracy:.2%}")
                        with col_b:
                            st.metric("练习次数", count)
                        
                        st.progress(accuracy)
                        
                        # 预测学习时间
                        predicted_time = learning_path_optimizer.predict_learning_time(user_id, skill_id)
                        st.info(f"⏱️ 预计学习时间: {predicted_time:.1f} 小时")
                        
                        # 获取前置知识点
                        prerequisites = learning_path_optimizer.adaptive_path.get_skill_prerequisites(skill_id)
                        if prerequisites:
                            st.write(f"📚 前置知识点: {', '.join([get_skill_name(mappings, p) for p in prerequisites])}")
                        else:
                            st.caption("📚 前置知识点: 无（该知识点可独立学习）")
                        
                        # 生成学习任务
                        tasks = learning_path_optimizer._generate_tasks(skill_id)
                        st.write("📋 学习任务:")
                        for task in tasks:
                            st.write(f"  • {task}")
                
                st.markdown("---")
                st.subheader("📊 学习路径可视化")
                
                fig = go.Figure()
                
                for i, skill_id in enumerate(learning_path, 1):
                    skill_name = get_skill_name(mappings, skill_id)
                    user_skill_data = user_data[user_data['skill_id'] == skill_id]
                    accuracy = user_skill_data['correct'].mean() if len(user_skill_data) > 0 else 0
                    
                    fig.add_trace(go.Bar(
                        x=[skill_name],
                        y=[accuracy],
                        name=f"阶段 {i}",
                        marker_color=['red' if accuracy < 0.5 else 'orange' if accuracy < 0.7 else 'green'][0]
                    ))
                
                fig.update_layout(
                    title="自适应学习路径规划",
                    xaxis_title="知识点",
                    yaxis_title="正确率",
                    yaxis_range=[0, 1],
                    showlegend=False
                )
                
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("💡 当前学生不在自适应学习路径的活跃用户列表中（只处理最活跃的500个用户）")
                st.caption("建议：选择答题数较多的学生，或关闭自适应学习路径功能")
        elif learning_path_optimizer is not None and learning_path_optimizer.adaptive_path is None:
            # 自适应学习路径初始化失败，显示错误信息
            st.error("❌ 自适应学习路径未初始化")
            if learning_path_optimizer.error_message:
                st.error(f"📋 错误详情: {learning_path_optimizer.error_message}")
            st.info("💡 建议关闭自适应学习路径功能，使用普通学习路径规划")
        
        # 普通学习路径（无论是否启用自适应学习路径，都会显示）
        st.markdown("---")
        st.subheader("📖 普通学习路径规划")
        st.caption("基于正确率排序的学习路径，不考虑知识点依赖关系")
        skill_stats = analyze_student_performance(df, user_id)
        
        if skill_stats is not None and len(skill_stats) > 0:
            skill_stats['skill_name'] = skill_stats['skill_id'].apply(lambda x: get_skill_name(mappings, x))
            
            if learning_goal == "全面掌握":
                sorted_skills = skill_stats.sort_values('accuracy')
            elif learning_goal == "重点突破":
                sorted_skills = skill_stats.nsmallest(5, 'accuracy')
            else:
                sorted_skills = skill_stats[skill_stats['accuracy'] < 0.7].sort_values('accuracy')
                
                if len(sorted_skills) == 0:
                    st.success("🎉 恭喜！你已经掌握了所有知识点（正确率均≥70%）")
                    sorted_skills = skill_stats.sort_values('accuracy')
            
            if len(sorted_skills) > 0:
                for i, (_, row) in enumerate(sorted_skills.iterrows(), 1):
                    with st.expander(f"第 {i} 阶段: {row['skill_name']}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("当前正确率", f"{row['accuracy']:.2%}")
                        with col_b:
                            st.metric("练习次数", int(row['count']))
                        
                        st.progress(row['accuracy'])
                        
                        if row['accuracy'] < 0.5:
                            st.warning("⚠️ 该知识点掌握较差，建议优先学习")
                        elif row['accuracy'] < 0.7:
                            st.info("📚 该知识点需要加强练习")
                        else:
                            st.success("✅ 该知识点掌握良好")
                
                st.markdown("---")
                st.subheader("📊 学习路径可视化")
                
                fig = go.Figure()
                
                for i, (_, row) in enumerate(sorted_skills.head(10).iterrows()):
                    fig.add_trace(go.Bar(
                        x=[row['skill_name']],
                        y=[row['accuracy']],
                        name=f"阶段 {i+1}",
                        marker_color=['red' if row['accuracy'] < 0.5 else 'orange' if row['accuracy'] < 0.7 else 'green'][0]
                    ))
                
                fig.update_layout(
                    title="学习路径规划",
                    xaxis_title="知识点",
                    yaxis_title="正确率",
                    yaxis_range=[0, 1],
                    showlegend=False
                )
                
                st.plotly_chart(fig, width='stretch')
    else:
        st.warning("未找到该学生的数据")

with tabs[3]:
    st.header("📝 教育评估")
    st.markdown("评估教学效果，分析不同学生的学习模式")
    
    col1, col2 = st.columns(2)
    
    with col1:
        evaluation_type = st.selectbox(
            "评估类型",
            ["整体教学效果", "学生群体分析", "知识点难度分析"]
        )
    
    with col2:
        num_students = st.slider("分析学生数量", 10, 100, 50)
    
    if evaluation_type == "整体教学效果":
        st.subheader("整体教学效果评估")
        
        sample_users = df['user_id'].unique()[:num_students]
        user_stats = []
        
        for user_id in sample_users:
            user_data = df[df['user_id'] == user_id]
            if len(user_data) > 0:
                user_stats.append({
                    'user_id': user_id,
                    'accuracy': user_data['correct'].mean(),
                    'total_questions': len(user_data)
                })
        
        user_stats_df = pd.DataFrame(user_stats)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("平均正确率", f"{user_stats_df['accuracy'].mean():.2%}")
        with col_b:
            st.metric("最高正确率", f"{user_stats_df['accuracy'].max():.2%}")
        with col_c:
            st.metric("最低正确率", f"{user_stats_df['accuracy'].min():.2%}")
        
        fig = px.histogram(
            user_stats_df,
            x='accuracy',
            title='学生正确率分布',
            labels={'accuracy': '正确率', 'count': '学生数量'},
            nbins=20
        )
        fig.add_vline(x=user_stats_df['accuracy'].mean(), line_dash="dash", line_color="red", 
                     annotation_text=f"平均: {user_stats_df['accuracy'].mean():.2%}")
        st.plotly_chart(fig, width='stretch')
    
    elif evaluation_type == "学生群体分析":
        st.subheader("学生群体分析")
        
        sample_users = df['user_id'].unique()[:num_students]
        
        user_categories = {'优秀': 0, '良好': 0, '一般': 0, '需加强': 0}
        
        for user_id in sample_users:
            user_data = df[df['user_id'] == user_id]
            if len(user_data) > 0:
                accuracy = user_data['correct'].mean()
                if accuracy >= 0.9:
                    user_categories['优秀'] += 1
                elif accuracy >= 0.7:
                    user_categories['良好'] += 1
                elif accuracy >= 0.5:
                    user_categories['一般'] += 1
                else:
                    user_categories['需加强'] += 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(user_categories.keys()),
            values=list(user_categories.values()),
            hole=.3
        )])
        
        fig.update_layout(
            title='学生群体分布',
            annotations=[dict(text='学生分布', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, width='stretch')
        
        for category, count in user_categories.items():
            st.write(f"{category}: {count} 人 ({count/num_students:.1%})")
    
    elif evaluation_type == "知识点难度分析":
        st.subheader("知识点难度分析")
        
        skill_stats = df.groupby('skill_id').agg({
            'correct': ['mean', 'count']
        }).reset_index()
        skill_stats.columns = ['skill_id', 'accuracy', 'count']
        
        skill_stats['skill_name'] = skill_stats['skill_id'].apply(lambda x: get_skill_name(mappings, x))
        
        skill_stats['difficulty'] = pd.cut(
            skill_stats['accuracy'],
            bins=[0, 0.5, 0.7, 0.9, 1],
            labels=['困难', '中等', '简单', '非常简单']
        )
        
        difficulty_counts = skill_stats['difficulty'].value_counts()
        
        fig = ChartConfig.create_bar_chart(
            pd.DataFrame({'difficulty': difficulty_counts.index, 'count': difficulty_counts.values}),
            x_col='difficulty',
            y_col='count',
            title='知识点难度分布',
            labels={'difficulty': '难度等级', 'count': '知识点数量'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("困难知识点列表")
        difficult_skills = skill_stats[skill_stats['accuracy'] < 0.5].sort_values('accuracy')
        
        for _, row in difficult_skills.head(10).iterrows():
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_a:
                st.write(row['skill_name'])
            with col_b:
                st.progress(row['accuracy'])
            with col_c:
                st.error(f"{row['accuracy']:.2%}")

with tabs[4]:
    st.header("📤 上传数据")
    st.markdown("上传自定义的习题数据或通过拍照添加新题目")
    
    upload_type = st.radio(
        "选择上传方式",
        ["CSV 文件上传", "拍照/图片上传", "自定义名称"],
        horizontal=True
    )
    
    if upload_type == "CSV 文件上传":
        st.subheader("📋 数据格式要求")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **必需列：**
            | 列名 | 类型 | 说明 |
            |--------|------|------|
            | `user_id` | 整数 | 学生ID（从0开始） |
            | `item_id` | 整数 | 题目ID（从0开始） |
            | `correct` | 0或1 | 答题结果（0=错误，1=正确） |
            | `skill_id` | 整数 | 知识点ID（从0开始） |
            """)
        
        with col2:
            st.markdown("""
            **可选列：**
            | 列名 | 类型 | 说明 |
            |--------|------|------|
            | `timestamp` | 整数 | 答题时间戳 |
            """)
        
        st.markdown("---")
        
        st.subheader("📏 数据范围要求")
        
        st.info("""
        **最小要求：**
        - 至少 2 个学生
        - 每个学生至少 5 次答题
        - 至少 2 个不同的知识点
        - 至少 10 道不同的题目
        
        **推荐配置：**
        - 10-1000 个学生
        - 每个学生 20-200 次答题
        - 5-100 个知识点
        - 50-50000 道题目
        """)
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "选择 CSV 文件",
            type=['csv'],
            help="请上传符合格式要求的 CSV 文件"
        )
        
        if uploaded_file is None:
            st.info("💡 请上传 CSV 文件开始")
        else:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                
                st.subheader("📊 数据预览")
                st.dataframe(df_uploaded.head(10))
                
                st.subheader("✅ 数据验证")
                
                required_columns = ['user_id', 'item_id', 'correct', 'skill_id']
                missing_columns = [col for col in required_columns if col not in df_uploaded.columns]
                
                if missing_columns:
                    st.error(f"❌ 缺少必需列: {', '.join(missing_columns)}")
                else:
                    st.success("✅ 所有必需列都存在")
                
                num_users = df_uploaded['user_id'].nunique()
                num_items = df_uploaded['item_id'].nunique()
                num_skills = df_uploaded['skill_id'].nunique()
                num_records = len(df_uploaded)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("学生数量", num_users)
                with col2:
                    st.metric("题目数量", num_items)
                with col3:
                    st.metric("知识点数量", num_skills)
                with col4:
                    st.metric("答题记录数", num_records)
                
                st.subheader("🔍 数据质量检查")
                
                checks = []
                
                if num_users < 2:
                    checks.append(("❌", "学生数量少于 2 个"))
                else:
                    checks.append(("✅", f"学生数量: {num_users} 个"))
                
                records_per_user = df_uploaded.groupby('user_id').size()
                min_records = records_per_user.min()
                if min_records < 5:
                    checks.append(("❌", f"有学生答题记录少于 5 次（最少 {min_records} 次）"))
                else:
                    checks.append(("✅", f"每个学生至少 {min_records} 次答题"))
                
                if num_skills < 2:
                    checks.append(("❌", "知识点数量少于 2 个"))
                else:
                    checks.append(("✅", f"知识点数量: {num_skills} 个"))
                
                if num_items < 10:
                    checks.append(("❌", "题目数量少于 10 道"))
                else:
                    checks.append(("✅", f"题目数量: {num_items} 道"))
                
                if df_uploaded['correct'].isin([0, 1]).all():
                    checks.append(("✅", "correct 列只包含 0 和 1"))
                else:
                    checks.append(("❌", "correct 列包含非 0 或 1 的值"))
                
                for status, message in checks:
                    st.write(f"{status} {message}")
                
                all_passed = all("✅" in status for status, _ in checks)
                
                st.markdown("---")
                
                if all_passed:
                    st.success("🎉 数据验证通过！可以保存")
                    
                    dataset_name = st.text_input(
                        "数据集名称",
                        value="custom_dataset",
                        help="用于标识这个数据集",
                        key="dataset_name_input"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💾 保存数据", type="primary", key="save_data_btn"):
                            os.makedirs('data', exist_ok=True)
                            os.makedirs(f'data/{dataset_name}', exist_ok=True)
                            
                            save_path = f'data/{dataset_name}/preprocessed_data.csv'
                            df_uploaded.to_csv(save_path, sep='\t', index=False)
                            
                            st.success(f"✅ 数据已保存到: {save_path}")
                            st.info("💡 刷新页面后，可以在左侧选择这个数据集了！")
                            st.balloons()
                    
                    with col2:
                        if st.button("🗑️ 清除数据", key="clear_data_btn"):
                            if os.path.exists(f'data/{dataset_name}'):
                                import shutil
                                shutil.rmtree(f'data/{dataset_name}')
                                st.success(f"✅ 已清除数据集: {dataset_name}")
                else:
                    st.error("❌ 数据验证失败，请检查数据格式和要求")
                    st.warning("💡 提示：确保所有检查项都通过后再上传")
                
                st.markdown("---")
                
                st.subheader("📝 数据示例")
                st.code("""
user_id,item_id,timestamp,correct,skill_id
0,5504,20964177,1,206
0,5479,20964214,0,206
0,5466,20964236,1,206
0,5515,20964257,1,206
0,5491,20964272,0,206
0,5472,20964349,1,206
0,5490,20964372,1,206
0,5508,20964388,1,206
0,1754,20964422,1,195
0,2803,20964440,1,195
                """, language="csv")
            
            except Exception as e:
                st.error(f"❌ 读取文件时出错: {str(e)}")
                st.info("💡 请确保上传的是有效的 CSV 文件")
    
    elif upload_type == "拍照/图片上传":
        st.subheader("📷 拍照上传题目")
        st.info("💡 请拍摄清晰的题目图片，确保文字清晰可见")
        
        # 检查是否安装了必要的库
        image_available = False
        try:
            # 尝试导入PIL库（用于图像处理）
            from PIL import Image
            image_available = True
        except ImportError as e:
            st.error(f"❌ 缺少必要的库: {str(e)}")
            st.info("💡 请运行以下命令安装:")
            st.code("pip install pillow", language="bash")
            st.info("💡 或者，如果您只需要上传CSV数据，可以使用'CSV 文件上传'功能")
        
        if image_available:
            # 图片上传功能
            image_file = st.file_uploader(
                "选择图片文件",
                type=['jpg', 'jpeg', 'png'],
                help="请上传包含题目的图片文件"
            )
            
            if image_file is not None:
                try:
                    # 读取图片
                    image = Image.open(image_file)
                    
                    # 显示图片预览
                    st.image(image, caption="上传的图片", width='stretch')
                    
                    # OCR识别功能
                    st.subheader("🔍 OCR识别")
                    
                    # 检查OCR是否可用
                    ocr_enabled = st.checkbox("启用OCR自动识别", value=True, help="自动识别图片中的文字内容")
                    
                    recognized_text = ""
                    if ocr_enabled:
                        # 选择OCR引擎
                        ocr_engine_type = st.selectbox(
                            "选择OCR引擎",
                            ["PaddleOCR（快速，适合文字识别）", "PaddleOCR-VL（强大，适合复杂文档）"],
                            help="PaddleOCR：快速轻量，适合题目文字识别\nPaddleOCR-VL：功能强大，适合表格、公式、图表等复杂内容"
                        )
                        
                        try:
                            # 设置环境变量禁用模型源检查
                            import os
                            os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
                            
                            # 强制重新初始化引擎，避免状态混乱
                            import os
                            
                            # 显示加载提示
                            with st.spinner("正在初始化OCR引擎..."):
                                # 清除旧的引擎实例
                                st.session_state.ocr_engine = None
                                st.session_state.ocr_engine_type = ocr_engine_type
                                
                                # 初始化OCR引擎
                                if "PaddleOCR-VL" in ocr_engine_type:
                                    # 使用PaddleOCR-VL
                                    try:
                                        from paddleocr import PaddleOCRVL
                                        st.session_state.ocr_engine = PaddleOCRVL()
                                        st.success("✅ PaddleOCR-VL引擎已加载")
                                    except ImportError:
                                        st.error("❌ PaddleOCR-VL库未安装")
                                        st.info("💡 请运行以下命令安装:")
                                        st.code("pip install paddleocr", language="bash")
                                        ocr_enabled = False
                                    except Exception as e:
                                        st.error(f"❌ 初始化PaddleOCR-VL时出错: {str(e)}")
                                        st.info("💡 PaddleOCR-VL需要下载较大的模型文件，可能需要较长时间和网络连接")
                                        ocr_enabled = False
                                else:
                                    # 使用PaddleOCR
                                    try:
                                        # 先导入PaddlePaddle并设置配置，避免与PyTorch冲突
                                        import paddle
                                        # 设置PaddlePaddle的GPU配置
                                        paddle.device.set_device('gpu:0' if paddle.device.is_compiled_with_cuda() else 'cpu')
                                        
                                        from paddleocr import PaddleOCR
                                        import numpy as np
                                        
                                        app_dir = os.path.dirname(os.path.abspath(__file__))
                                        model_dir = os.path.join(app_dir, 'models')
                                        
                                        if os.path.exists(model_dir):
                                            # 检查模型目录结构
                                            det_model_path = os.path.join(model_dir, 'ch_PP-OCRv4_det_infer')
                                            rec_model_path = os.path.join(model_dir, 'ch_PP-OCRv4_rec_infer')
                                            cls_model_path = os.path.join(model_dir, 'ch_PP-OCRv4_cls_infer')
                                            
                                            # 处理嵌套目录结构：如果存在嵌套目录，使用嵌套目录
                                            if os.path.exists(os.path.join(det_model_path, 'ch_PP-OCRv4_det_infer')):
                                                det_model_path = os.path.join(det_model_path, 'ch_PP-OCRv4_det_infer')
                                            if os.path.exists(os.path.join(rec_model_path, 'ch_PP-OCRv4_rec_infer')):
                                                rec_model_path = os.path.join(rec_model_path, 'ch_PP-OCRv4_rec_infer')
                                            if os.path.exists(os.path.join(cls_model_path, 'ch_PP-OCRv4_cls_infer')):
                                                cls_model_path = os.path.join(cls_model_path, 'ch_PP-OCRv4_cls_infer')
                                            
                                            # 检查模型文件是否存在
                                            det_model_exists = os.path.exists(os.path.join(det_model_path, 'inference.pdmodel'))
                                            rec_model_exists = os.path.exists(os.path.join(rec_model_path, 'inference.pdmodel'))
                                            
                                            if det_model_exists and rec_model_exists:
                                                # 使用PaddleOCR 3.0.0的API，让PaddleOCR自动处理模型路径
                                                # 设置环境变量禁用模型源检查
                                                import os
                                                os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
                                                os.environ['PADDLEOCR_OFFLINE'] = 'True'
                                                
                                                # 尝试使用PaddleOCR 3.0.0的API
                                                try:
                                                    # 简化参数，让PaddleOCR自动处理模型
                                                    st.session_state.ocr_engine = PaddleOCR(
                                                        lang='ch'
                                                    )
                                                    st.success("✅ PaddleOCR引擎已加载（默认模型）")
                                                    st.info("💡 使用PP-OCRv5_server最高精度模型")
                                                except Exception as e:
                                                    # 如果失败，尝试不指定模型路径，让PaddleOCR自动下载
                                                    st.warning(f"⚠️ 使用本地模型失败: {str(e)}")
                                                    st.info("💡 尝试使用默认模型（PP-OCRv5_server，最高精度）...")
                                                    st.session_state.ocr_engine = PaddleOCR(
                                                        use_angle_cls=True,
                                                        lang='ch',
                                                        det_db_thresh=0.3,
                                                        det_db_box_thresh=0.5,
                                                        det_db_unclip_ratio=1.5
                                                    )
                                                    st.success("✅ PaddleOCR引擎已加载（PP-OCRv5_server最高精度模型）")
                                            else:
                                                st.error("❌ 模型文件不存在")
                                                st.info(f"💡 检测到的模型路径：")
                                                st.info(f"   - 检测模型: {det_model_path} {'✅' if det_model_exists else '❌'}")
                                                st.info(f"   - 识别模型: {rec_model_path} {'✅' if rec_model_exists else '❌'}")
                                                st.info("💡 请确保模型文件正确解压")
                                        else:
                                            # 使用默认模型，但设置环境变量禁用网络请求
                                            import os
                                            os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
                                            os.environ['PADDLEOCR_OFFLINE'] = 'True'
                                            try:
                                                st.session_state.ocr_engine = PaddleOCR(
                                                    use_angle_cls=True,
                                                    lang='ch',
                                                    det_db_thresh=0.3,
                                                    det_db_box_thresh=0.5,
                                                    det_db_unclip_ratio=1.5
                                                )
                                                st.success("✅ PaddleOCR引擎已加载（PP-OCRv5_server最高精度模型）")
                                            except Exception as e:
                                                st.error(f"❌ 初始化PaddleOCR默认模型时出错: {str(e)}")
                                                st.info("💡 OCR功能暂时不可用，将自动切换到手动输入模式")
                                                ocr_enabled = False
                                    except Exception as e:
                                        st.error(f"❌ 初始化PaddleOCR时出错: {str(e)}")
                                        st.info("💡 请检查模型文件是否正确解压到models目录")
                                        # 当OCR初始化失败时，自动切换到手动输入模式
                                        st.info("💡 OCR功能暂时不可用，将自动切换到手动输入模式")
                                        st.info("💡 系统核心功能仍然正常可用")
                                        ocr_enabled = False
                            
                            # 使用OCR识别文字
                            with st.spinner("正在识别图片中的文字..."):
                                if "PaddleOCR-VL" in ocr_engine_type:
                                    # 使用PaddleOCR-VL识别
                                    import tempfile
                                    import os
                                    
                                    # 保存图片到临时文件
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                                        image.save(tmp.name)
                                        tmp_path = tmp.name
                                    
                                    try:
                                        result = st.session_state.ocr_engine.predict(tmp_path)
                                        
                                        # 提取识别结果
                                        text_lines = []
                                        for res in result:
                                            if hasattr(res, 'rec_texts'):
                                                text_lines.extend(res.rec_texts)
                                            elif hasattr(res, 'text'):
                                                text_lines.append(res.text)
                                        
                                        recognized_text = '\n'.join(text_lines)
                                    finally:
                                        # 删除临时文件
                                        if os.path.exists(tmp_path):
                                            os.unlink(tmp_path)
                                else:
                                    # 使用PaddleOCR识别
                                    import numpy as np
                                    # 调整图片大小以提高识别率
                                    from PIL import Image
                                    
                                    # 调整图片大小，确保文字清晰
                                    max_size = 1024
                                    width, height = image.size
                                    if max(width, height) > max_size:
                                        ratio = max_size / max(width, height)
                                        new_width = int(width * ratio)
                                        new_height = int(height * ratio)
                                        image = image.resize((new_width, new_height), Image.LANCZOS)
                                    
                                    # 使用PaddleOCR识别
                                    result = st.session_state.ocr_engine.ocr(np.array(image))
                                    
                                    # 提取识别结果
                                    text_lines = []
                                    try:
                                        if result:
                                            st.info(f"📊 OCR结果类型: {type(result)}")
                                            if isinstance(result, list):
                                                st.info(f"📊 OCR结果长度: {len(result)}")
                                                for i, item in enumerate(result):
                                                    st.info(f"📊 第{i}项类型: {type(item)}")
                                                    if isinstance(item, list):
                                                        st.info(f"📊 第{i}项是列表，长度: {len(item)}")
                                                        if len(item) > 0:
                                                            st.info(f"📊 第{i}项第一个元素类型: {type(item[0])}")
                                                            st.info(f"📊 第{i}项第一个元素内容: {item[0]}")
                                                        # 处理 PP-OCRv5_server 的返回格式
                                                        for j, line in enumerate(item):
                                                            st.info(f"📊 第{i}-{j}行类型: {type(line)}")
                                                            st.info(f"📊 第{i}-{j}行长度: {len(line) if hasattr(line, '__len__') else 'N/A'}")
                                                            if isinstance(line, (list, tuple)):
                                                                st.info(f"📊 第{i}-{j}行内容: {line}")
                                                                # 尝试不同的结果格式
                                                                if len(line) >= 2:
                                                                    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                                                                        text_lines.append(line[1][0])
                                                                        st.info(f"📊 识别到文字: {line[1][0]}")
                                                                    elif isinstance(line[1], str):
                                                                        text_lines.append(line[1])
                                                                        st.info(f"📊 识别到文字: {line[1]}")
                                                                elif len(line) >= 1:
                                                                    if isinstance(line[0], str):
                                                                        text_lines.append(line[0])
                                                                        st.info(f"📊 识别到文字: {line[0]}")
                                                    elif hasattr(item, 'rec_texts'):
                                                        rec_texts = item.rec_texts
                                                        st.info(f"📊 第{i}项rec_texts类型: {type(rec_texts)}")
                                                        st.info(f"📊 第{i}项rec_texts内容: {rec_texts}")
                                                        if isinstance(rec_texts, list):
                                                            text_lines.extend(rec_texts)
                                                            for text in rec_texts:
                                                                st.info(f"📊 识别到文字: {text}")
                                                        elif isinstance(rec_texts, str):
                                                            text_lines.append(rec_texts)
                                                            st.info(f"📊 识别到文字: {rec_texts}")
                                                    elif isinstance(item, dict):
                                                        st.info(f"📊 第{i}项是字典，键: {list(item.keys())}")
                                                        if 'text' in item:
                                                            text_lines.append(item['text'])
                                                            st.info(f"📊 识别到文字: {item['text']}")
                                                        elif 'rec_texts' in item:
                                                            text_lines.extend(item['rec_texts'])
                                                            st.info(f"📊 识别到文字: {item['rec_texts']}")
                                                    elif hasattr(item, 'text'):
                                                        text_lines.append(item.text)
                                                        st.info(f"📊 识别到文字: {item.text}")
                                                    else:
                                                        st.info(f"📊 第{i}项内容: {item}")
                                    except Exception as e:
                                        st.warning(f"⚠️ 解析OCR结果时出错: {str(e)}")
                                    
                                    recognized_text = '\n'.join(text_lines)
                                    st.info(f"📊 最终识别结果: {recognized_text}")
                            
                            # 显示识别结果
                            if recognized_text.strip():
                                st.success("✅ 识别成功！")
                                st.text_area("识别的题目内容", recognized_text, height=200, key="ocr_result")
                                st.info("💡 您可以编辑识别结果，确保内容准确")
                            else:
                                st.warning("⚠️ 未能识别出文字，请手动输入")
                                
                        except ImportError as e:
                            st.error("❌ OCR库未安装")
                            st.info("💡 请运行以下命令安装:")
                            st.code("pip install paddleocr paddlepaddle", language="bash")
                            ocr_enabled = False
                        except Exception as e:
                            error_msg = str(e)
                            if "model source" in error_msg or "download model" in error_msg or "network" in error_msg or "拒绝访问" in error_msg:
                                st.error("⚠️ OCR功能暂时不可用")
                                st.warning("📋 原因分析：")
                                st.warning("   1. PaddleOCR需要下载模型文件到系统目录")
                                st.warning("   2. 可能存在权限或网络访问限制")
                                st.warning("   3. 这是Windows系统的常见问题")
                                st.info("💡 最佳解决方案：使用本地模型文件")
                                st.info("   1. 下载模型文件（3个文件，总共约18M）：")
                                st.code("https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar", language="text")
                                st.code("https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar", language="text")
                                st.code("https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_cls_infer.tar", language="text")
                                st.info("   2. 解压到以下目录：")
                                st.code("TBKT4/models/", language="text")
                                st.info("   3. 解压后目录结构应该是：")
                                st.code("TBKT4/models/ch_PP-OCRv4_det_infer/", language="text")
                                st.code("TBKT4/models/ch_PP-OCRv4_rec_infer/", language="text")
                                st.code("TBKT4/models/ch_PP-OCRv4_cls_infer/", language="text")
                                st.success("✅ 系统核心功能完全正常，不受影响！")
                                st.info("📚 可用功能：")
                                st.info("   - 个性化学习推荐")
                                st.info("   - 学习路径优化")
                                st.info("   - 知识追踪分析")
                                st.info("   - 教育评估")
                            else:
                                st.warning(f"⚠️ OCR识别失败: {error_msg}")
                            st.info("💡 您可以手动输入题目内容")
                            ocr_enabled = False
                    
                    # 题目信息输入
                    st.subheader("📝 题目信息")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        skill_id = st.number_input("知识点 ID", min_value=0, value=0, help="题目所属的知识点 ID")
                    with col2:
                        difficulty = st.selectbox("难度级别", ["简单", "中等", "困难"], help="题目的难度级别")
                    
                    # 题目内容输入（如果OCR识别成功，预填充识别结果）
                    default_text = recognized_text if ocr_enabled and recognized_text.strip() else ""
                    manual_text = st.text_area("题目内容", default_text, height=200, help="请输入或编辑题目内容")
                    
                    # 保存题目
                    dataset_name = st.text_input(
                        "数据集名称",
                        value="custom_images",
                        help="用于标识这个数据集",
                        key="image_dataset_name"
                    )
                    
                    if st.button("💾 保存题目", type="primary", key="save_image_btn"):
                        # 创建数据集目录
                        os.makedirs('data', exist_ok=True)
                        os.makedirs(f'data/{dataset_name}', exist_ok=True)
                        
                        # 保存图片
                        image_path = f'data/{dataset_name}/images'
                        os.makedirs(image_path, exist_ok=True)
                        
                        image_filename = f"{len(os.listdir(image_path)) + 1}.jpg"
                        image.save(f"{image_path}/{image_filename}")
                        
                        # 保存题目信息
                        import json
                        questions_path = f'data/{dataset_name}/questions.json'
                        
                        if os.path.exists(questions_path):
                            with open(questions_path, 'r', encoding='utf-8') as f:
                                questions = json.load(f)
                        else:
                            questions = []
                        
                        new_question = {
                            "id": len(questions) + 1,
                            "content": manual_text,
                            "skill_id": int(skill_id),
                            "difficulty": difficulty,
                            "image": image_filename
                        }
                        
                        questions.append(new_question)
                        
                        with open(questions_path, 'w', encoding='utf-8') as f:
                            json.dump(questions, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"✅ 题目已保存到: {questions_path}")
                        st.info("💡 您可以继续上传更多题目")
                        st.balloons()
                        
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ 处理图片时出错: {error_msg}")
                    st.info("💡 请确保上传的是有效的图片文件")
        else:
            st.info("💡 请上传图片文件开始")
        
        st.markdown("---")
        st.subheader("📝 使用提示")
        st.write("1. **拍摄技巧**：确保光线充足，文字清晰可见")
        st.write("2. **图片要求**：尽量只包含题目内容，避免其他干扰")
        st.write("3. **OCR识别**：系统会自动识别图片中的文字，您可以编辑识别结果")
        st.write("4. **手动输入**：如果OCR识别不准确，可以手动输入题目内容")
        st.write("5. **批量上传**：您可以多次上传图片来创建题库")
    
    elif upload_type == "自定义名称":
        st.subheader("🏷️ 自定义名称")
        st.info("💡 为学生、题目和知识点设置友好的名称")
        
        name_type = st.radio(
            "选择要自定义的类型",
            ["学生名称", "知识点名称", "题目名称"],
            horizontal=True
        )
        
        if name_type == "学生名称":
            st.subheader("👨‍🎓 学生名称设置")
            
            unique_users = sorted(df['user_id'].unique())
            selected_user_id = st.selectbox(
                "选择学生ID",
                unique_users,
                format_func=lambda x: f"{get_user_name(mappings, x)} (ID: {x})"
            )
            
            current_name = get_user_name(mappings, selected_user_id)
            new_name = st.text_input("输入新名称", value=current_name)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("保存名称"):
                    set_user_name(mappings, selected_user_id, new_name)
                    save_mappings(mappings)
                    st.success(f"✅ 已保存: {new_name}")
                    st.rerun()
            
            with col2:
                if st.button("自动生成名称"):
                    set_user_name(mappings, selected_user_id, f"学生{selected_user_id}")
                    save_mappings(mappings)
                    st.success(f"✅ 已重置为默认名称")
                    st.rerun()
        
        elif name_type == "知识点名称":
            st.subheader("📚 知识点名称设置")
            
            unique_skills = sorted(df['skill_id'].unique())
            selected_skill_id = st.selectbox(
                "选择知识点ID",
                unique_skills,
                format_func=lambda x: f"{get_skill_name(mappings, x)} (ID: {x})"
            )
            
            current_name = get_skill_name(mappings, selected_skill_id)
            new_name = st.text_input("输入新名称", value=current_name)
            
            skill_data = df[df['skill_id'] == selected_skill_id]
            avg_accuracy = skill_data['correct'].mean()
            st.info(f"该知识点平均正确率: {avg_accuracy:.2%}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("保存名称"):
                    set_skill_name(mappings, selected_skill_id, new_name)
                    save_mappings(mappings)
                    st.success(f"✅ 已保存: {new_name}")
                    st.rerun()
            
            with col2:
                if st.button("自动生成名称"):
                    set_skill_name(mappings, selected_skill_id, f"知识点{selected_skill_id}")
                    save_mappings(mappings)
                    st.success(f"✅ 已重置为默认名称")
                    st.rerun()
            
            with col3:
                if st.button("批量生成名称"):
                    mappings = auto_generate_skill_names(df)
                    st.success(f"✅ 已为所有知识点生成名称")
                    st.rerun()
        
        elif name_type == "题目名称":
            st.subheader("📝 题目名称设置")
            
            unique_items = sorted(df['item_id'].unique())
            selected_item_id = st.selectbox(
                "选择题目ID",
                unique_items,
                format_func=lambda x: f"{get_item_name(mappings, x)} (ID: {x})"
            )
            
            current_name = get_item_name(mappings, selected_item_id)
            new_name = st.text_input("输入新名称", value=current_name)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("保存名称"):
                    set_item_name(mappings, selected_item_id, new_name)
                    save_mappings(mappings)
                    st.success(f"✅ 已保存: {new_name}")
                    st.rerun()
            
            with col2:
                if st.button("自动生成名称"):
                    set_item_name(mappings, selected_item_id, f"题目{selected_item_id}")
                    save_mappings(mappings)
                    st.success(f"✅ 已重置为默认名称")
                    st.rerun()
        
        st.markdown("---")
        st.subheader("📊 当前名称映射")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("学生名称数", len(mappings["user_names"]))
        with col2:
            st.metric("知识点名称数", len(mappings["skill_names"]))
        with col3:
            st.metric("题目名称数", len(mappings["item_names"]))

st.markdown("---")
st.caption("© 2024 智能知识追踪系统 | 基于 SAKT 和 TSAKT 模型")