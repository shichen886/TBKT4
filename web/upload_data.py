import streamlit as st
import pandas as pd
import os
import shutil

st.set_page_config(
    page_title="上传习题数据",
    page_icon="📤",
    layout="wide"
)

st.title("📤 上传习题数据")
st.markdown("---")

st.header("📋 数据格式要求")

col1, col2 = st.columns(2)

with col1:
    st.subheader("必需列")
    st.markdown("""
    | 列名 | 类型 | 说明 |
    |--------|------|------|
    | `user_id` | 整数 | 学生ID（从0开始） |
    | `item_id` | 整数 | 题目ID（从0开始） |
    | `correct` | 0或1 | 答题结果（0=错误，1=正确） |
    | `skill_id` | 整数 | 知识点ID（从0开始） |
    """)

with col2:
    st.subheader("可选列")
    st.markdown("""
    | 列名 | 类型 | 说明 |
    |--------|------|------|
    | `timestamp` | 整数 | 答题时间戳 |
    """)

st.markdown("---")

st.header("📏 数据范围要求")

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

st.header("📤 上传数据")

uploaded_file = st.file_uploader(
    "选择 CSV 文件",
    type=['csv'],
    help="请上传符合格式要求的 CSV 文件"
)

if uploaded_file is not None:
    st.warning("⚠️ 请上传 CSV 文件")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 数据预览")
    st.dataframe(df.head(10))
    
    st.subheader("✅ 数据验证")
    
    required_columns = ['user_id', 'item_id', 'correct', 'skill_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ 缺少必需列: {', '.join(missing_columns)}")
        st.stop()
    else:
        st.success("✅ 所有必需列都存在")
    
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    num_skills = df['skill_id'].nunique()
    num_records = len(df)
    
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
    
    records_per_user = df.groupby('user_id').size()
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
    
    if df['correct'].isin([0, 1]).all():
        checks.append(("✅", "correct 列只包含 0 和 1"))
    else:
        checks.append(("❌", "correct 列包含非 0 或 1 的值"))
    
    for status, message in checks:
        st.write(f"{status} {message}")
    
    all_passed = all("✅" in status for status, _ in checks)
    
    st.markdown("---")
    
    if all_passed:
        st.success("🎉 数据验证通过！可以上传")
        
        dataset_name = st.text_input(
            "数据集名称",
            value="custom_dataset",
            help="用于标识这个数据集"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 保存数据", type="primary"):
                os.makedirs('data', exist_ok=True)
                os.makedirs(f'data/{dataset_name}', exist_ok=True)
                
                save_path = f'data/{dataset_name}/preprocessed_data.csv'
                df.to_csv(save_path, sep='\t', index=False)
                
                st.success(f"✅ 数据已保存到: {save_path}")
                st.info("💡 现在可以在主系统中使用这个数据集了！")
        
        with col2:
            if st.button("🗑️ 清除数据"):
                if os.path.exists(f'data/{dataset_name}'):
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

st.markdown("---")
st.caption("© 2024 智能知识追踪系统 | 数据上传工具")
