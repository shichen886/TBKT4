import numpy as np
import pandas as pd
from collections import defaultdict, deque
import networkx as nx

class AdaptiveLearningPath:
    def __init__(self, df):
        """
        自适应学习路径推荐
        
        Args:
            df: 学生答题数据，包含 user_id, item_id, skill_id, correct
        """
        try:
            # 不限制数据量，保留所有用户的数据
            # 限制数据量，减少内存使用（增加到500个用户）
            # top_users = df['user_id'].value_counts().head(500).index
            # df = df[df['user_id'].isin(top_users)]
            # 
            # top_items = df['item_id'].value_counts().head(500).index
            # df = df[df['item_id'].isin(top_items)]
            
            # 检查数据是否足够
            if len(df) < 10:
                raise ValueError(f"数据量不足，只有{len(df)}条记录，至少需要10条记录")
            
            # 检查必要的列是否存在
            required_columns = ['user_id', 'item_id', 'skill_id', 'correct']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"数据缺少必要的列: {', '.join(missing_columns)}")
            
            # 保存原始数据，用于查找用户数据
            self.df = df
            self.original_df = df
              
            # 构建知识点关系图
            self.skill_graph = self._build_skill_graph()
            
            # 计算知识点难度
            self.skill_difficulty = self._calculate_skill_difficulty()
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            raise Exception(f"自适应学习路径初始化失败: {str(e)}\n\n详细错误:\n{error_detail}")
        
    def _build_skill_graph(self):
        """构建知识点关系图"""
        G = nx.DiGraph()
        
        # 添加所有知识点节点
        skills = self.df['skill_id'].unique()
        for skill in skills:
            G.add_node(skill)
        
        # 方法1：基于学生答题序列构建知识点之间的依赖关系
        for user_id in self.df['user_id'].unique():
            user_data = self.df[self.df['user_id'] == user_id].sort_values('item_id')
            skill_sequence = user_data['skill_id'].values
            
            # 如果学生连续答对某个知识点，然后转向另一个知识点，可能存在依赖关系
            for i in range(len(skill_sequence) - 1):
                skill1 = skill_sequence[i]
                skill2 = skill_sequence[i + 1]
                
                # 检查学生是否掌握了第一个知识点
                skill1_correct = user_data[user_data['skill_id'] == skill1]['correct'].mean()
                
                if skill1_correct > 0.7 and skill1 != skill2:
                    # 增加边的权重
                    if G.has_edge(skill1, skill2):
                        G[skill1][skill2]['weight'] += 1
                    else:
                        G.add_edge(skill1, skill2, weight=1)
        
        # 方法2：基于知识点难度构建依赖关系（启发式规则）
        # 简单知识点 → 中等知识点 → 困难知识点
        skill_difficulties = self._calculate_skill_difficulty()
        
        # 将知识点按难度分组
        easy_skills = [s for s, d in skill_difficulties.items() if d < 0.3]
        medium_skills = [s for s, d in skill_difficulties.items() if 0.3 <= d < 0.6]
        hard_skills = [s for s, d in skill_difficulties.items() if d >= 0.6]
        
        # 建立简单 → 中等的依赖关系
        for easy_skill in easy_skills:
            for medium_skill in medium_skills:
                if not G.has_edge(easy_skill, medium_skill):
                    G.add_edge(easy_skill, medium_skill, weight=0.5)
        
        # 建立中等 → 困难的依赖关系
        for medium_skill in medium_skills:
            for hard_skill in hard_skills:
                if not G.has_edge(medium_skill, hard_skill):
                    G.add_edge(medium_skill, hard_skill, weight=0.5)
        
        # 方法3：基于学生答题顺序构建依赖关系
        # 如果大多数学生先学A再学B，则A可能是B的前置知识点
        skill_pairs = defaultdict(int)
        skill_order = defaultdict(dict)
        
        for user_id in self.df['user_id'].unique():
            user_data = self.df[self.df['user_id'] == user_id].sort_values('item_id')
            skill_sequence = user_data['skill_id'].values
            
            # 记录每个知识点首次出现的位置
            for i, skill in enumerate(skill_sequence):
                if skill not in skill_order[user_id]:
                    skill_order[user_id][skill] = i
        
        # 统计知识点顺序
        for user_id in skill_order:
            skills_in_order = sorted(skill_order[user_id].items(), key=lambda x: x[1])
            
            # 检查是否有足够的数据
            if len(skills_in_order) < 2:
                continue
            
            for i in range(len(skills_in_order)):
                for j in range(i + 1, len(skills_in_order)):
                    skill1 = skills_in_order[i][0]
                    skill2 = skills_in_order[j][0]
                    skill_pairs[(skill1, skill2)] += 1
        
        # 如果超过50%的学生先学A再学B，则建立A→B的依赖关系
        total_users = len(skill_order)
        for (skill1, skill2), count in skill_pairs.items():
            if count > total_users * 0.5 and not G.has_edge(skill1, skill2):
                G.add_edge(skill1, skill2, weight=count / total_users)
        
        return G
    
    def _calculate_skill_difficulty(self):
        """计算每个知识点的难度"""
        skill_difficulty = {}
        
        for skill in self.df['skill_id'].unique():
            skill_data = self.df[self.df['skill_id'] == skill]
            # 正确率越低，难度越高
            difficulty = 1 - skill_data['correct'].mean()
            skill_difficulty[skill] = difficulty
        
        return skill_difficulty
    
    def recommend_learning_path(self, user_id, max_length=10):
        """
        为学生推荐学习路径
        
        Args:
            user_id: 学生ID
            max_length: 学习路径的最大长度
        
        Returns:
            推荐的学习路径（知识点列表）
        """
        print(f'recommend_learning_path: user_id={user_id}, max_length={max_length}')
        
        # 使用原始数据查找用户数据
        user_data = self.original_df[self.original_df['user_id'] == user_id]
        
        if len(user_data) == 0:
            print('用户数据为空，返回空路径')
            return []
        
        # 获取学生已掌握的知识点（正确率 > 0.7）
        mastered_skills = set()
        for skill in user_data['skill_id'].unique():
            skill_correct = user_data[user_data['skill_id'] == skill]['correct'].mean()
            if skill_correct > 0.7:
                mastered_skills.add(skill)
        
        print(f'已掌握的知识点数量: {len(mastered_skills)}')
        
        # 获取学生未掌握的知识点
        all_skills = set(self.df['skill_id'].unique())
        unmastered_skills = all_skills - mastered_skills
        
        print(f'未掌握的知识点数量: {len(unmastered_skills)}')
        
        if not unmastered_skills:
            # 如果学生已经掌握了所有知识点，返回需要巩固的知识点
            # 选择正确率较低的知识点（0.5-0.7之间）
            weak_skills = set()
            for skill in user_data['skill_id'].unique():
                skill_correct = user_data[user_data['skill_id'] == skill]['correct'].mean()
                if 0.5 <= skill_correct <= 0.7:
                    weak_skills.add(skill)
            
            print(f'需要巩固的知识点数量: {len(weak_skills)}')
            
            if not weak_skills:
                # 如果没有需要巩固的知识点，返回所有知识点，按难度排序
                sorted_skills = sorted(all_skills, key=lambda s: self.skill_difficulty.get(s, 0.5))[:max_length]
                print(f'返回所有知识点，按难度排序，数量: {len(sorted_skills)}')
                return sorted_skills
            else:
                # 返回需要巩固的知识点，按难度排序
                sorted_skills = sorted(weak_skills, key=lambda s: self.skill_difficulty.get(s, 0.5))[:max_length]
                print(f'返回需要巩固的知识点，按难度排序，数量: {len(sorted_skills)}')
                return sorted_skills
        
        # 计算每个未掌握知识点的优先级
        skill_priorities = {}
        for skill in unmastered_skills:
            priority = 0
            
            # 基础性：有多少已掌握的知识点指向该知识点
            predecessors = list(self.skill_graph.predecessors(skill))
            base_count = sum(1 for p in predecessors if p in mastered_skills)
            priority += base_count * 2
            
            # 难度适配：推荐与学生当前水平相近的知识点
            if mastered_skills:
                user_avg_difficulty = np.mean([self.skill_difficulty[s] for s in mastered_skills])
            else:
                user_avg_difficulty = 0.5
            
            skill_difficulty = self.skill_difficulty.get(skill, 0.5)
            difficulty_diff = abs(skill_difficulty - user_avg_difficulty)
            priority -= difficulty_diff * 3
            
            # 关联性：与学生最近学习的知识点相关
            if len(user_data) > 0:
                recent_skills = user_data.tail(min(5, len(user_data)))['skill_id'].values
                for recent_skill in recent_skills:
                    if self.skill_graph.has_edge(recent_skill, skill):
                        priority += 1.5
            
            skill_priorities[skill] = priority
        
        # 按优先级排序
        sorted_skills = sorted(skill_priorities.items(), key=lambda x: x[1], reverse=True)
        print(f'按优先级排序的知识点数量: {len(sorted_skills)}')
        
        # 构建学习路径
        learning_path = []
        current_skills = mastered_skills.copy()
        
        for skill, priority in sorted_skills:
            if len(learning_path) >= max_length:
                break
            
            # 检查是否满足前置条件
            predecessors = list(self.skill_graph.predecessors(skill))
            prerequisites_met = all(p in current_skills for p in predecessors)
            
            if prerequisites_met:
                learning_path.append(skill)
                current_skills.add(skill)
            else:
                # 如果不满足前置条件，但前置条件为空，也添加到学习路径中
                if len(predecessors) == 0:
                    if len(learning_path) < max_length:
                        learning_path.append(skill)
                        current_skills.add(skill)
                else:
                    # 如果前置条件不满足，尝试添加前置条件到学习路径中
                    for p in predecessors:
                        if p not in current_skills and p not in learning_path and len(learning_path) < max_length:
                            learning_path.append(p)
                            current_skills.add(p)
                    # 然后添加当前知识点
                    if all(p in current_skills for p in predecessors) and len(learning_path) < max_length:
                        learning_path.append(skill)
                        current_skills.add(skill)
        
        print(f'最终学习路径数量: {len(learning_path)}')
        
        # 如果学习路径仍然为空，返回所有未掌握的知识点，按优先级排序
        if not learning_path:
            print('学习路径为空，返回所有未掌握的知识点，按优先级排序')
            learning_path = [skill for skill, _ in sorted_skills[:max_length]]
        
        # 确保学习路径长度不超过最大长度
        if len(learning_path) > max_length:
            print(f'学习路径长度超过最大长度，截取前{max_length}个知识点')
            learning_path = learning_path[:max_length]
        
        return learning_path
    
    def get_skill_prerequisites(self, skill_id):
        """
        获取知识点的前置知识点
        
        Args:
            skill_id: 知识点ID
        
        Returns:
            前置知识点列表
        """
        if skill_id not in self.skill_graph:
            return []
        
        return list(self.skill_graph.predecessors(skill_id))
    
    def get_skill_dependents(self, skill_id):
        """
        获取依赖于该知识点的后续知识点
        
        Args:
            skill_id: 知识点ID
        
        Returns:
            后续知识点列表
        """
        if skill_id not in self.skill_graph:
            return []
        
        return list(self.skill_graph.successors(skill_id))
    
    def visualize_learning_path(self, user_id, max_length=10):
        """
        可视化学习路径
        
        Args:
            user_id: 学生ID
            max_length: 学习路径的最大长度
        
        Returns:
            NetworkX图对象
        """
        learning_path = self.recommend_learning_path(user_id, max_length)
        
        if not learning_path:
            return None
        
        # 创建子图
        subgraph = self.skill_graph.subgraph(learning_path)
        
        return subgraph


class LearningPathOptimizer:
    def __init__(self, df):
        """
        学习路径优化器
        
        Args:
            df: 学生答题数据
        """
        self.df = df
        self.adaptive_path = None
        self.error_message = None
        
        try:
            self.adaptive_path = AdaptiveLearningPath(df)
        except Exception as e:
            self.error_message = str(e)
            self.adaptive_path = None
        
    def optimize_path(self, user_id, current_path, performance_data):
        """
        根据学生表现优化学习路径
        
        Args:
            user_id: 学生ID
            current_path: 当前学习路径
            performance_data: 学生在路径上的表现数据
        
        Returns:
            优化后的学习路径
        """
        if self.adaptive_path is None:
            return []
        
        if not current_path:
            return self.adaptive_path.recommend_learning_path(user_id)
        
        # 分析学生在当前路径上的表现
        weak_skills = []
        strong_skills = []
        
        for skill_id in current_path:
            if skill_id in performance_data:
                performance = performance_data[skill_id]
                if performance['accuracy'] < 0.5:
                    weak_skills.append(skill_id)
                elif performance['accuracy'] > 0.8:
                    strong_skills.append(skill_id)
        
        # 优化策略
        optimized_path = []
        
        # 1. 保留薄弱知识点，优先学习
        optimized_path.extend(weak_skills)
        
        # 2. 添加新的知识点
        new_path = self.adaptive_path.recommend_learning_path(user_id, max_length=10)
        
        for skill in new_path:
            if skill not in optimized_path and skill not in strong_skills:
                optimized_path.append(skill)
        
        # 3. 添加巩固知识点（已掌握但需要加强）
        optimized_path.extend(strong_skills[:2])
        
        return optimized_path[:10]
    
    def predict_learning_time(self, user_id, skill_id):
        """
        预测学生学习某个知识点需要的时间
        
        Args:
            user_id: 学生ID
            skill_id: 知识点ID
        
        Returns:
            预测的学习时间（小时）
        """
        if self.adaptive_path is None:
            return 2.0
        
        user_data = self.df[self.df['user_id'] == user_id]
        
        if len(user_data) == 0:
            return 2.0
        
        # 计算学生平均学习速度
        user_avg_attempts = user_data.groupby('skill_id').size().mean()
        
        # 获取知识点的难度
        skill_difficulty = self.adaptive_path.skill_difficulty.get(skill_id, 0.5)
        
        # 预测时间（基于难度和学生平均尝试次数）
        predicted_time = user_avg_attempts * skill_difficulty * 0.5
        
        return min(max(predicted_time, 0.5), 5.0)  # 限制在0.5-5小时之间
    
    def generate_study_plan(self, user_id, days=7):
        """
        生成学习计划
        
        Args:
            user_id: 学生ID
            days: 计划天数
        
        Returns:
            学习计划（字典列表）
        """
        if self.adaptive_path is None:
            return []
        
        learning_path = self.adaptive_path.recommend_learning_path(user_id, max_length=days)
        
        if not learning_path:
            return []
        
        study_plan = []
        
        for day, skill_id in enumerate(learning_path, 1):
            learning_time = self.predict_learning_time(user_id, skill_id)
            
            study_plan.append({
                'day': day,
                'skill_id': skill_id,
                'learning_time': learning_time,
                'tasks': self._generate_tasks(skill_id)
            })
        
        return study_plan
    
    def _generate_tasks(self, skill_id):
        """
        为知识点生成学习任务
        
        Args:
            skill_id: 知识点ID
        
        Returns:
            任务列表
        """
        # 获取知识点难度
        skill_difficulty = self.adaptive_path.skill_difficulty.get(skill_id, 0.5)
        
        # 根据难度生成不同的任务
        if skill_difficulty < 0.3:
            # 简单知识点
            tasks = [
                "📖 阅读知识点基础概念和定义",
                "✏️ 完成基础练习题（至少10道）",
                "🔄 复习错题，理解错误原因",
                "✅ 进行自我测试，确保正确率≥80%"
            ]
        elif skill_difficulty < 0.6:
            # 中等难度知识点
            tasks = [
                "📖 深入学习知识点概念和原理",
                "✏️ 完成中等难度练习题（至少15道）",
                "🔄 整理错题本，分析错误模式",
                "🤝 与同学讨论疑难问题",
                "✅ 进行自我测试，确保正确率≥70%"
            ]
        else:
            # 困难知识点
            tasks = [
                "📖 系统学习知识点理论和方法",
                "✏️ 完成高难度练习题（至少20道）",
                "🔄 建立错题档案，深入分析错误原因",
                "🤝 寻求老师或同学的帮助",
                "💡 尝试多种解题方法",
                "✅ 进行自我测试，确保正确率≥60%"
            ]
        
        return tasks